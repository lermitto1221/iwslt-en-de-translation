"""Real CPU optimization, recovery, held-out isolation and journal checks."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import sentencepiece as spm
import torch

from mt.batching import tokenize_pairs
from mt.common import read_json, sha256, write_json
from mt.representation_batches import load_schedule, schedule_records
from mt.cv_train import (EXECUTION_SOURCES, execution_source_hashes,
                                     train, validate_completed_run)
from mt.resume_check import _equal_nested
from mt.tokenizer import load
from mt.cv_evaluate import evaluate_checkpoint, verify_results, corpus_metrics, _atomic_json
from mt.export import export_bundle
from mt.inference import Translator


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CVTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        cls.data.mkdir()
        cls.rows = [{"id": i * 3, "en": "a small cat " + "sits " * (i % 3 + 1),
                     "de": "eine kleine katze " + "sitzt " * (i % 2 + 1)} for i in range(12)]
        for split, rows in (("train", cls.rows), ("holdout", [{**row, "id": 1000 + row["id"]} for row in cls.rows[:5]])):
            jsonl(cls.data / (split + ".jsonl"), rows)
        corpus = cls.data / "spm_train.txt"
        corpus.write_text("\n".join(row[side] for row in cls.rows for side in ("en", "de")), encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(cls.data / "spm"),
            vocab_size=64, model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False, character_coverage=1.0, num_threads=1,
            shuffle_input_sentence=False, minloglevel=2)
        tok = load(cls.data)
        write_json(cls.data / "manifest.json", {"files": {split: {"jsonl_sha256": sha256(cls.data / (split + ".jsonl")), "rows_saved": len(cls.rows) if split == "train" else 5}
                                                     for split in ("train", "holdout")}})
        write_json(cls.data / "tokenizer.json", {"model_type": "bpe", "actual_vocab_size": tok.vocab_size(),
            "model_sha256": sha256(cls.data / "spm.model"), "training_file_sha256": sha256(cls.data / "train.jsonl")})
        cls.examples, _ = tokenize_pairs(cls.rows, tok)
        lengths = [{"id": identifier, "bpe_src": len(src), "bpe_tgt": len(tgt),
                    "unigram_src": len(src), "unigram_tgt": len(tgt)}
                   for src, tgt, identifier in cls.examples]
        cls.lengths = lengths
        jsonl(cls.root / "lengths.jsonl", lengths)
        common = {"ids": [row["id"] for row in cls.rows], "excluded_ids": [],
            "lengths_file": str(cls.root / "lengths.jsonl"), "lengths_sha256": sha256(cls.root / "lengths.jsonl"),
            "source_train_sha256": sha256(cls.data / "train.jsonl"), "max_tokens": 128,
            "tokenizer_sha256": {"bpe": sha256(cls.data / "spm.model"), "unigram": sha256(cls.data / "spm.model")}, "pool_size": 512}
        write_json(cls.root / "common_ids.json", common)
        cls.plan = list(schedule_records(lengths, seed=42, max_updates=4, token_budget=80, accumulation=4))
        jsonl(cls.root / "schedule.jsonl", cls.plan)
        cls.cfg = {"seed": 42, "stage": "cv", "checkpoint_selection": "fixed_last", "data_dir": str(cls.data), "run_dir": "set-per-test", "representation": "bpe",
            "vocab_size": tok.vocab_size(), "max_train_tokens": 128, "d_model": 16, "nhead": 2,
            "num_layers": 1, "dim_ff": 32, "dropout": 0.1, "label_smoothing": 0.1,
            "batch_tokens": 80, "accum_steps": 4, "max_updates": 4, "warmup_updates": 2,
            "lr": 0.0005, "validate_every": 2, "max_generation_tokens": 8, "beam_size": 4,
            "length_penalty": 0.6, "amp": False, "device": "cpu", "cpu_threads": 1,
            "decode_batch_size": 2, "deterministic": True, "save_every": 2, "log_every": 4,
            "expected_validation_samples": 2, "common_ids": str(cls.root / "common_ids.json"),
            "common_ids_sha256": sha256(cls.root / "common_ids.json"), "batch_plan": str(cls.root / "schedule.jsonl"),
            "batch_plan_sha256": sha256(cls.root / "schedule.jsonl")}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, name):
        return dict(self.cfg, run_dir=str(self.root / name))

    def run_quietly(self, cfg, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(cfg, **kwargs)

    def test_continuous_vs_interrupted_bitwise_recovery_and_readonly_reuse(self):
        continuous = self.config("continuous")
        self.run_quietly(continuous)
        full = torch.load(Path(continuous["run_dir"]) / "last.pt", weights_only=False)
        resumed = self.config("resumed")
        self.run_quietly(resumed, stop_after=2)
        run = Path(resumed["run_dir"])
        with (run / "train.jsonl").open("ab") as handle:
            handle.write(b'{"update":3,"torn":"\xe4\xb8')
        self.run_quietly(resumed, resume=run / "last.pt")
        recovered = torch.load(run / "last.pt", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "scaler"):
            self.assertTrue(_equal_nested(full[key], recovered[key]), key)
        self.assertTrue(torch.equal(full["rng"]["torch"], recovered["rng"]["torch"]))
        self.assertEqual(full["initial_model_sha256"], recovered["initial_model_sha256"])
        self.assertEqual(full["model_sha256"], recovered["model_sha256"])
        records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        full_records = [json.loads(line) for line in (Path(continuous["run_dir"]) / "train.jsonl").read_text().splitlines()]
        for left, right, planned in zip(records, full_records, self.plan):
            for key in ("loss", "lr_used", "data_order_sha256", "microbatch_sha256", "sentences", "target_tokens", "source_tokens", "padded_tokens"):
                self.assertEqual(left[key], right[key], key)
            self.assertEqual(left["data_order_sha256"], planned["data_order_sha256"])
            self.assertEqual(left["microbatch_sha256"], planned["microbatch_sha256"])
            self.assertEqual(left["target_tokens"], planned["target_tokens"]["bpe"])
        self.assertTrue(list(run.glob("train.jsonl.before-resume.*.bak")))
        self.assertEqual(set(read_json(run / "source_snapshot" / "manifest.json")["execution_source_hashes"]), set(EXECUTION_SOURCES))
        before = {str(path.relative_to(run)): (sha256(path), path.stat().st_mtime_ns) for path in run.rglob("*") if path.is_file()}
        with patch("mt.cv_train.write_json", side_effect=AssertionError("reuse attempted a write")):
            summary = self.run_quietly(resumed, resume=run / "last.pt")
        self.assertTrue(summary["complete"])
        after = {str(path.relative_to(run)): (sha256(path), path.stat().st_mtime_ns) for path in run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_schedule_changed_hash_and_changed_microbatch_boundaries_rejected(self):
        cfg = self.config("bad-plan")
        changed = json.loads(json.dumps(self.plan))
        # Preserve flattened ID order but shift the boundary by one sentence.
        first = changed[0]["micro_batches"]
        first[1].insert(0, first[0].pop())
        changed[0]["microbatch_sha256"] = hashlib.sha256(json.dumps(first).encode()).hexdigest()
        bad_plan = self.root / "bad_schedule.jsonl"
        jsonl(bad_plan, changed)
        cfg["batch_plan"] = str(bad_plan)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_quietly(cfg)
        cfg["batch_plan_sha256"] = sha256(bad_plan)
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            self.run_quietly(cfg)
        self.assertFalse(Path(cfg["run_dir"]).exists())

    def test_missing_source_snapshot_and_wrong_resume_checkpoint_rejected(self):
        cfg = self.config("resume-source")
        self.run_quietly(cfg, stop_after=2)
        run = Path(cfg["run_dir"])
        with self.assertRaisesRegex(ValueError, "only allowed"):
            self.run_quietly(cfg, resume=run / "best.pt")
        snapshot = run / "source_snapshot" / "representation_batches.py"
        snapshot.write_text("# altered shared schedule source\n")
        with self.assertRaisesRegex(ValueError, "snapshot is missing or changed"):
            self.run_quietly(cfg, resume=run / "last.pt")

    def test_wrong_seed_and_incomplete_checkpoint_rejected(self):
        cfg = self.config("resume-state")
        self.run_quietly(cfg, stop_after=2)
        run = Path(cfg["run_dir"])
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            self.run_quietly(dict(cfg, seed=43), resume=run / "last.pt")
        state = torch.load(run / "last.pt", weights_only=False)
        del state["optimizer"]
        torch.save(state, run / "last.pt")
        with self.assertRaisesRegex(ValueError, "complete training checkpoint"):
            self.run_quietly(cfg, resume=run / "last.pt")

    def test_completed_log_and_tensor_corruption_rejected(self):
        cfg = self.config("complete-corruption")
        self.run_quietly(cfg)
        run = Path(cfg["run_dir"])
        state = torch.load(run / "last.pt", weights_only=False)
        name = next(iter(state["model"]))
        state["model"][name].view(-1)[0] += 1
        torch.save(state, run / "last.pt")
        with self.assertRaisesRegex(ValueError, "tensor SHA-256"):
            validate_completed_run(cfg)

    def test_completed_log_corruption_is_rejected_without_repair(self):
        cfg = self.config("complete-log-corruption")
        self.run_quietly(cfg)
        run = Path(cfg["run_dir"])
        records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        records[0]["loss"] += 1
        jsonl(run / "train.jsonl", records)
        before = (run / "train.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_quietly(cfg, resume=run / "last.pt")
        self.assertEqual(before, (run / "train.jsonl").read_bytes())
        self.assertFalse(list(run.glob("train.jsonl.before-resume.*.bak")))

    def test_resume_rejects_run_metadata_before_log_repair(self):
        cfg = self.config("resume-metadata")
        self.run_quietly(cfg, stop_after=2)
        run = Path(cfg["run_dir"])
        with (run / "train.jsonl").open("ab") as handle:
            handle.write(b'{"update":3')
        before = (run / "train.jsonl").read_bytes()
        write_json(run / "config.json", dict(cfg, seed=44))
        with self.assertRaisesRegex(ValueError, "saved run configuration"):
            self.run_quietly(cfg, resume=run / "last.pt")
        self.assertEqual(before, (run / "train.jsonl").read_bytes())

    def test_real_schedule_indices_match_consumed_ids(self):
        schedule = load_schedule(self.cfg, self.examples)
        for item, planned in zip(schedule, self.plan):
            self.assertEqual([[self.examples[i][2] for i in group] for group in item["microbatches"]], planned["micro_batches"])
        self.assertIn("cv_train.py", execution_source_hashes())
        self.assertIn("representation_batches.py", execution_source_hashes())

    def test_fixed_last_checkpoint_never_reads_holdout_text_or_metrics(self):
        from mt.data import read_pairs
        cfg = self.config("fixed-last")
        def only_training(path):
            self.assertEqual(Path(path).name, "train.jsonl")
            return read_pairs(path)
        with patch("mt.cv_train.read_pairs", side_effect=only_training), patch(
                "mt.train.validation_bleu", side_effect=AssertionError("must not validate")):
            summary = self.run_quietly(cfg)
        run = Path(cfg["run_dir"])
        self.assertFalse((run / "best.pt").exists())
        self.assertEqual(summary["checkpoint_selection"], "fixed_last")
        self.assertEqual(summary["selected_checkpoint"], "last.pt")
        self.assertEqual(summary["updates"], cfg["max_updates"])
        records = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        self.assertFalse(any("validation_bleu_greedy" in row for row in records))
        self.assertEqual(summary["exposure"]["sentences"], sum(item["sentences"] for item in self.plan))

    def test_same_fold_candidates_share_initialization_and_exposure(self):
        a, b = self.config("candidate-zero"), dict(self.config("candidate-point1"), label_smoothing=0.1)
        a["label_smoothing"] = 0.0
        for cfg in (a, b):
            self.run_quietly(cfg)
        left, right = [torch.load(Path(cfg["run_dir"]) / "last.pt", weights_only=False) for cfg in (a, b)]
        self.assertEqual(left["initial_model_sha256"], right["initial_model_sha256"])
        self.assertNotEqual(left["model_sha256"], right["model_sha256"])
        summaries = [read_json(Path(cfg["run_dir"]) / "summary.json") for cfg in (a, b)]
        self.assertEqual(summaries[0]["exposure"], summaries[1]["exposure"])

    def test_wrong_fold_tokenizer_and_checkpoint_selection_rejected(self):
        cfg = self.config("bad-fold")
        copied = self.root / "wrong-fold-data"
        shutil.copytree(self.data, copied)
        metadata = read_json(copied / "tokenizer.json")
        metadata["training_file_sha256"] = "0" * 64
        write_json(copied / "tokenizer.json", metadata)
        with self.assertRaisesRegex(ValueError, "not fitted on this training fold"):
            self.run_quietly(dict(cfg, data_dir=str(copied)))
        with self.assertRaisesRegex(ValueError, "fixed_last"):
            self.run_quietly(dict(cfg, checkpoint_selection="best_validation"))

    def test_journal_resume_matches_full_decode_and_completed_reuse_is_readonly(self):
        cfg = self.config("eval-resume-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        full, resumed = self.root / "eval-full", self.root / "eval-resumed"
        def evaluate(out, **kwargs):
            with contextlib.redirect_stdout(io.StringIO()):
                return evaluate_checkpoint(checkpoint, "holdout", "greedy", out,
                                           decode_batch_size=2, device="cpu", **kwargs)
        complete = evaluate(full)
        partial = evaluate(resumed, stop_after_batches=1)
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["samples_completed"], 2)
        first_hash = sha256(resumed / "batches" / "batch_000000.json")
        continued = evaluate(resumed)
        self.assertEqual(sha256(resumed / "batches" / "batch_000000.json"), first_hash)
        for name in ("predictions.txt", "references.txt", "examples.jsonl"):
            self.assertEqual((full / name).read_bytes(), (resumed / name).read_bytes())
        for key in ("bleu", "chrf2", "signature", "chrf2_signature", "samples"):
            self.assertEqual(complete[key], continued[key])
        self.assertEqual(continued, verify_results(resumed))
        before = {str(p.relative_to(resumed)): (sha256(p), p.stat().st_mtime_ns) for p in resumed.rglob("*") if p.is_file()}
        with patch("mt.cv_evaluate.build_model", side_effect=AssertionError("must not generate")), patch(
                "mt.cv_evaluate._atomic_json", side_effect=AssertionError("must not write")):
            self.assertEqual(evaluate(resumed), continued)
        after = {str(p.relative_to(resumed)): (sha256(p), p.stat().st_mtime_ns) for p in resumed.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_corrupted_batch_and_changed_identity_are_rejected(self):
        cfg = self.config("eval-damaged-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        out = self.root / "eval-damaged"
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2,
                                device="cpu", stop_after_batches=1)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=3, device="cpu")
        path = out / "batches" / "batch_000000.json"
        saved = read_json(path)
        saved["examples"][0]["prediction"] += " changed"
        write_json(path, saved)
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu")

    def test_complete_metrics_and_text_corruption_are_rejected(self):
        cfg = self.config("eval-metric-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        out = self.root / "eval-metric"
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu")
        metrics = read_json(out / "metrics.json")
        write_json(out / "metrics.json", dict(metrics, bleu=metrics["bleu"] + 1))
        with self.assertRaisesRegex(ValueError, "independent corpus"):
            verify_results(out)
        write_json(out / "metrics.json", metrics)
        with (out / "predictions.txt").open("a") as handle:
            handle.write("extra line\n")
        with self.assertRaisesRegex(ValueError, "text differs"):
            verify_results(out)

    def test_incomplete_checkpoint_or_test_use_in_cv_rejected(self):
        cfg = self.config("eval-short-model")
        self.run_quietly(cfg, stop_after=2)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        with self.assertRaisesRegex(ValueError, "fixed final"):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", self.root / "short-eval", device="cpu")
        self.run_quietly(cfg, resume=checkpoint)
        with self.assertRaisesRegex(ValueError, "only the complete holdout"):
            evaluate_checkpoint(checkpoint, "test", "greedy", self.root / "bad-test-eval", device="cpu")
        with self.assertRaisesRegex(ValueError, "only the complete holdout"):
            evaluate_checkpoint(checkpoint, "holdout", "beam", self.root / "bad-beam-eval", device="cpu")

    def test_export_reload_has_identical_predictions(self):
        cfg = self.config("export-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        protocol = self.root / "fixture-protocol.json"
        write_json(protocol, {"selection_split": "validation", "beam_size": cfg["beam_size"],
                   "length_penalty": cfg["length_penalty"], "max_generation_tokens": cfg["max_generation_tokens"]})
        bundle = export_bundle(checkpoint, protocol, self.root / "export-bundle")
        original = Translator(checkpoint=checkpoint, device="cpu")
        reloaded = Translator(bundle=bundle, device="cpu")
        sources = [row["en"] for row in self.rows[:3]]
        self.assertEqual(original.translate(sources, method="greedy", batch_size=2),
                         reloaded.translate(sources, method="greedy", batch_size=2))

    def test_corpus_metrics_recomputed_not_sentence_average(self):
        predictions = ["a b c d e f", "one short sentence"]
        references = ["a b c d e f", "one completely different sentence"]
        result = corpus_metrics(predictions, references)
        individual = [corpus_metrics([p], [r])["bleu"] for p, r in zip(predictions, references)]
        self.assertNotAlmostEqual(result["bleu"], sum(individual) / len(individual))
        self.assertIn("tok:13a", result["signature"])
        self.assertIn("nc:6|nw:0", result["chrf2_signature"])

    def test_committed_batch_before_progress_write_is_recovered(self):
        cfg = self.config("eval-atomic-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        out = self.root / "eval-atomic"
        def interrupted(path, value):
            if Path(path).name == "progress.json" and value["committed_batches"] == 1:
                raise RuntimeError("simulated interruption after atomic batch commit")
            return _atomic_json(path, value)
        with patch("mt.cv_evaluate._atomic_json", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu")
        saved = sha256(out / "batches" / "batch_000000.json")
        self.assertEqual(read_json(out / "progress.json")["committed_batches"], 0)
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu")
        self.assertTrue(metrics["complete"])
        self.assertEqual(sha256(out / "batches" / "batch_000000.json"), saved)

    def test_missing_committed_batch_is_not_silently_regenerated(self):
        cfg = self.config("eval-missing-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        out = self.root / "eval-missing"
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu", stop_after_batches=2)
        (out / "batches" / "batch_000001.json").unlink()
        with self.assertRaisesRegex(ValueError, "progress differs"):
            evaluate_checkpoint(checkpoint, "holdout", "greedy", out, decode_batch_size=2, device="cpu")

    def test_long_sources_fail_explicitly_without_truncation(self):
        cfg = self.config("eval-long-model")
        self.run_quietly(cfg)
        checkpoint = Path(cfg["run_dir"]) / "last.pt"
        with patch("mt.cv_evaluate.encode", return_value=[2] + [4] * 4095 + [3]):
            with self.assertRaisesRegex(ValueError, "positional encoding capacity"):
                evaluate_checkpoint(checkpoint, "holdout", "greedy", self.root / "eval-long",
                                    decode_batch_size=2, device="cpu")



if __name__ == "__main__":
    unittest.main()
