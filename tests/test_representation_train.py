"""Real CPU optimization checks for shared-schedule training and recovery."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sentencepiece as spm
import torch

from mt.batching import tokenize_pairs
from mt.common import read_json, sha256, write_json
from mt.representation_batches import load_schedule, schedule_records
from mt.representation_train import (EXECUTION_SOURCES, execution_source_hashes,
                                     train, validate_completed_run)
from mt.resume_check import _equal_nested
from mt.tokenizer import load


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SharedScheduleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        cls.data.mkdir()
        cls.rows = [{"id": i * 3, "en": "a small cat " + "sits " * (i % 3 + 1),
                     "de": "eine kleine katze " + "sitzt " * (i % 2 + 1)} for i in range(12)]
        for split, rows in (("train", cls.rows), ("validation", cls.rows[:2]), ("test", cls.rows[2:4])):
            jsonl(cls.data / (split + ".jsonl"), rows)
        corpus = cls.data / "spm_train.txt"
        corpus.write_text("\n".join(row[side] for row in cls.rows for side in ("en", "de")), encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(cls.data / "spm"),
            vocab_size=64, model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False, character_coverage=1.0, num_threads=1,
            shuffle_input_sentence=False, minloglevel=2)
        tok = load(cls.data)
        write_json(cls.data / "manifest.json", {"files": {split: {"jsonl_sha256": sha256(cls.data / (split + ".jsonl"))}
                                                     for split in ("train", "validation", "test")}})
        write_json(cls.data / "tokenizer.json", {"model_type": "bpe", "actual_vocab_size": tok.vocab_size(),
            "model_sha256": sha256(cls.data / "spm.model"), "training_file_sha256": sha256(cls.data / "train.jsonl")})
        cls.examples, _ = tokenize_pairs(cls.rows, tok)
        lengths = [{"id": identifier, "bpe_src": len(src), "bpe_tgt": len(tgt),
                    "unigram_src": len(src) + 1, "unigram_tgt": len(tgt) + 1}
                   for src, tgt, identifier in cls.examples]
        cls.lengths = lengths
        jsonl(cls.root / "lengths.jsonl", lengths)
        common = {"ids": [row["id"] for row in cls.rows], "excluded_ids": [],
            "lengths_file": str(cls.root / "lengths.jsonl"), "lengths_sha256": sha256(cls.root / "lengths.jsonl"),
            "source_train_sha256": sha256(cls.data / "train.jsonl"), "max_tokens": 128,
            "tokenizer_sha256": {"bpe": sha256(cls.data / "spm.model"), "unigram": "fixture-unused"}, "pool_size": 512}
        write_json(cls.root / "common_ids.json", common)
        cls.plan = list(schedule_records(lengths, seed=42, max_updates=4, token_budget=80, accumulation=4))
        jsonl(cls.root / "schedule.jsonl", cls.plan)
        cls.cfg = {"seed": 42, "data_dir": str(cls.data), "run_dir": "set-per-test", "representation": "bpe",
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
        with patch("mt.representation_train.write_json", side_effect=AssertionError("reuse attempted a write")):
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
        self.assertIn("representation_train.py", execution_source_hashes())
        self.assertIn("representation_batches.py", execution_source_hashes())

    def test_same_score_retains_earliest_checkpoint(self):
        cfg = self.config("ties")
        fixed = {"validation_bleu_greedy": 3.0, "validation_signature": "fixture",
                 "validation_samples": 2, "validation_generation_limit_fraction": 0.0}
        with patch("mt.representation_train.validation_bleu", return_value=fixed):
            self.run_quietly(cfg)
        state = torch.load(Path(cfg["run_dir"]) / "best.pt", weights_only=False)
        self.assertEqual(state["update"], 2)
        self.assertTrue(validate_completed_run(cfg)["complete"])


if __name__ == "__main__":
    unittest.main()
