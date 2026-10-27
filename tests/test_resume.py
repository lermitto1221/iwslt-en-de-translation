"""Recovery checks focus on interrupted writes and immutable provenance."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mt.train import CORE_TRAINING_SOURCES, _validate_resume_sources, reconcile_training_log


def encoded(records):
    return b"".join((json.dumps(record) + "\n").encode("utf-8") for record in records)


class TrainingLogRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.log = self.run / "train.jsonl"
        self.records = [{"update": i, "loss": 4.0 / i} for i in range(1, 4)]
        self.committed = encoded(self.records)
        self.state = {"update": 3, "log_record": self.records[-1],
                      "log_sha256": hashlib.sha256(self.committed).hexdigest()}

    def test_discard_uncommitted_updates_and_torn_utf8_tail(self):
        original = self.committed + encoded([{"update": 4, "loss": 0.7}]) + b'{"update":5,"text":"\xe4\xb8'
        self.log.write_bytes(original)
        result = reconcile_training_log(self.run, self.state)
        self.assertEqual(self.log.read_bytes(), self.committed)
        self.assertEqual(Path(result["backup"]).read_bytes(), original)
        self.assertFalse(result["repaired_checkpoint_record"])

    def test_repair_only_final_checkpoint_record(self):
        self.log.write_bytes(encoded(self.records[:-1]) + b'{"update": 3, "loss":')
        result = reconcile_training_log(self.run, self.state)
        self.assertTrue(result["repaired_checkpoint_record"])
        self.assertEqual(self.log.read_bytes(), self.committed)

    def test_committed_history_corruption_is_not_silently_rewritten(self):
        corrupted = [dict(record) for record in self.records]
        corrupted[0]["loss"] = 99.0
        original = encoded(corrupted)
        self.log.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            reconcile_training_log(self.run, self.state)
        self.assertEqual(self.log.read_bytes(), original)
        self.assertFalse(list(self.run.glob("*.bak")))

    def test_missing_older_committed_records_cannot_be_fabricated(self):
        self.log.write_bytes(encoded(self.records[:1]))
        with self.assertRaisesRegex(ValueError, "missing committed updates"):
            reconcile_training_log(self.run, self.state)

    def test_duplicate_committed_update_is_rejected(self):
        self.log.write_bytes(encoded([self.records[0], self.records[0], self.records[2]]))
        with self.assertRaisesRegex(ValueError, "continuous checkpoint prefix"):
            reconcile_training_log(self.run, self.state)

    def test_matching_committed_log_is_not_rewritten(self):
        self.log.write_bytes(self.committed)
        before = self.log.stat().st_mtime_ns
        result = reconcile_training_log(self.run, self.state)
        self.assertFalse(result["changed"])
        self.assertEqual(self.log.stat().st_mtime_ns, before)


class SourceProvenanceTests(unittest.TestCase):
    def test_changed_live_code_or_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            snapshot = run / "source_snapshot"
            snapshot.mkdir()
            hashes = {}
            for name in CORE_TRAINING_SOURCES:
                content = ("# preserved " + name + "\n").encode()
                (snapshot / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            (snapshot / "manifest.json").write_text(json.dumps({"core_source_hashes": hashes}))
            (run / "source_hashes.json").write_text(json.dumps(hashes))
            state = {"source_hashes": hashes}
            _validate_resume_sources(state, hashes, run)
            with self.assertRaisesRegex(ValueError, "core training source differs"):
                _validate_resume_sources(state, dict(hashes, **{"train.py": "0" * 64}), run)
            (snapshot / "model.py").write_text("# altered snapshot\n")
            with self.assertRaisesRegex(ValueError, "snapshot is missing or changed"):
                _validate_resume_sources(state, hashes, run)


class RealCpuResumeTests(unittest.TestCase):
    def test_actual_checkpoint_interruption_preserves_optimizer_and_predictions(self):
        import sentencepiece as spm

        from mt.common import sha256, write_json
        from mt.resume_check import run_check

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            # Synthetic fixture for recovery engineering only; never reported as
            # IWSLT data or translation-quality evidence.
            examples = [("Hello world", "Hallo Welt"),
                        ("Good morning", "Guten Morgen")]
            rows = [{"id": i, "en": en, "de": de}
                    for i, (en, de) in enumerate(examples * 8)]
            manifest = {"dataset": "synthetic-resume-test-only", "files": {}}
            for split in ("train", "validation", "test"):
                path = data / (split + ".jsonl")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                encoding="utf-8")
                manifest["files"][split] = {"jsonl_sha256": sha256(path),
                                             "rows_saved": len(rows)}
            write_json(data / "manifest.json", manifest)
            corpus = data / "corpus.txt"
            corpus.write_text("Hello world\nHallo Welt\nGood morning\nGuten Morgen\n" * 20,
                              encoding="utf-8")
            spm.SentencePieceTrainer.train(input=str(corpus),
                model_prefix=str(data / "spm"), vocab_size=40, model_type="bpe",
                pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False,
                num_threads=1, minloglevel=2)
            tok = spm.SentencePieceProcessor(model_file=str(data / "spm.model"))
            write_json(data / "tokenizer.json", {
                "model_sha256": sha256(data / "spm.model"),
                "training_file_sha256": sha256(data / "train.jsonl")})
            cfg = dict(seed=42, data_dir=str(data), device="cpu",
                vocab_size=tok.vocab_size(), max_train_tokens=128,
                d_model=16, nhead=2, num_layers=1, dim_ff=32, dropout=0.1,
                label_smoothing=0.1, batch_tokens=128, accum_steps=1,
                max_updates=10, warmup_updates=2, lr=0.0005,
                validate_every=10, max_generation_tokens=4, beam_size=2,
                length_penalty=0.6, amp=False, cpu_threads=1,
                decode_batch_size=2)
            config = root / "smoke.json"
            write_json(config, cfg)
            result = run_check("cpu", config, root / "checks")
        self.assertTrue(result["model_bitwise_equal"])
        self.assertTrue(result["optimizer_equal"])
        self.assertTrue(result["data_order_equal"])
        self.assertTrue(result["log_updates_unique_and_contiguous"])
        self.assertTrue(result["source_change_rejected"])
        self.assertTrue(result["completed_run_artifacts_preserved"])


if __name__ == "__main__":
    unittest.main()
