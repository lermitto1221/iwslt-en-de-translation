"""Checks of the real paired-schedule algorithm, without fitting a tokenizer."""
import json
import tempfile
import unittest
from pathlib import Path

from mt.common import sha256, write_json
from mt.representation_batches import common_rows, ids_digest, load_schedule, schedule_records
from mt.representation_prepare import _immutable_json, _validate_design, _verify_files, trainer_options


class SharedRepresentationScheduleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [{"id": i * 7, "bpe_src": 3 + i % 7, "bpe_tgt": 4 + i % 8,
                      "unigram_src": 4 + i % 7, "unigram_tgt": 6 + i % 8} for i in range(30)]
        self.rows.append({"id": 999, "bpe_src": 129, "bpe_tgt": 6, "unigram_src": 8, "unigram_tgt": 4})
        self.retained = common_rows(self.rows)
        self.lengths = self.root / "lengths.jsonl"
        self.write_lines(self.lengths, self.rows)
        self.meta_path = self.root / "common_ids.json"
        write_json(self.meta_path, {"ids": [r["id"] for r in self.retained], "excluded_ids": [999],
            "max_tokens": 128, "pool_size": 8, "lengths_file": "lengths.jsonl", "lengths_sha256": sha256(self.lengths)})
        self.schedule = list(schedule_records(self.retained, seed=42, max_updates=6, token_budget=80,
                                              accumulation=4, pool_size=8))
        self.batch_path = self.root / "seed42.jsonl"
        self.write_lines(self.batch_path, self.schedule)
        self.cfg = {"representation": "bpe", "seed": 42, "max_updates": 6, "batch_tokens": 80,
                    "accum_steps": 4, "batch_plan": str(self.batch_path), "batch_plan_sha256": sha256(self.batch_path),
                    "common_ids": str(self.meta_path), "common_ids_sha256": sha256(self.meta_path)}

    @staticmethod
    def write_lines(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def examples(self, rep="bpe"):
        return [([2] + [5] * (r[f"{rep}_src"] - 2) + [3],
                 [2] + [6] * (r[f"{rep}_tgt"] - 2) + [3], r["id"]) for r in self.retained]

    def rewrite_schedule(self):
        self.write_lines(self.batch_path, self.schedule)
        self.cfg["batch_plan_sha256"] = sha256(self.batch_path)

    def test_intersection_filters_either_representation_and_side(self):
        for rep in ("bpe", "unigram"):
            for side in ("src", "tgt"):
                row = dict(self.rows[0], **{f"{rep}_{side}": 129})
                self.assertEqual(common_rows([row]), [])
                row[f"{rep}_{side}"] = 128
                self.assertEqual(common_rows([row]), [row])

    def test_common_boundaries_satisfy_both_real_token_budgets(self):
        by_id = {r["id"]: r for r in self.retained}
        for row in self.schedule:
            self.assertEqual(len(row["micro_batches"]), 4)
            for group in row["micro_batches"]:
                for rep in ("bpe", "unigram"):
                    cost = len(group) * (max(by_id[i][f"{rep}_src"] for i in group)
                                         + max(by_id[i][f"{rep}_tgt"] for i in group))
                    self.assertLessEqual(cost, 80)

    def test_seed_and_epoch_schedule_reproducible(self):
        repeated = list(schedule_records(self.retained, 42, 6, 80, 4, 8))
        different = list(schedule_records(self.retained, 43, 6, 80, 4, 8))
        self.assertEqual(self.schedule, repeated)
        self.assertNotEqual(self.schedule[0]["micro_batches"], different[0]["micro_batches"])
        self.assertGreater(self.schedule[-1]["epoch"], 0)

    def test_representations_share_ids_but_need_not_share_token_counts(self):
        bpe = load_schedule(self.cfg, self.examples())
        uni = load_schedule(dict(self.cfg, representation="unigram"), self.examples("unigram"))
        self.assertEqual([r["microbatch_ids"] for r in bpe], [r["microbatch_ids"] for r in uni])
        self.assertTrue(any(r["target_tokens"]["bpe"] != r["target_tokens"]["unigram"] for r in bpe))
        self.assertEqual(bpe[0]["indices"], bpe[0]["microbatches"])

    def test_original_id_mapping_survives_example_reordering(self):
        examples = self.examples()[::-1]
        result = load_schedule(self.cfg, examples)
        for row in result:
            actual = [[examples[i][2] for i in group] for group in row["indices"]]
            self.assertEqual(actual, row["microbatch_ids"])

    def test_wrong_seed_rejected_even_with_relabelled_file_hash(self):
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            load_schedule(dict(self.cfg, seed=43), self.examples())

    def test_tampered_bytes_rejected_by_frozen_hash(self):
        self.batch_path.write_text(self.batch_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_schedule(self.cfg, self.examples())

    def test_changed_boundaries_rejected_even_if_flat_order_preserved(self):
        row = self.schedule[0]
        first, second = row["micro_batches"][:2]
        second.insert(0, first.pop())
        row["microbatch_sha256"] = ids_digest(row["micro_batches"])
        self.rewrite_schedule()
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            load_schedule(self.cfg, self.examples())

    def test_fabricated_token_count_rejected(self):
        self.schedule[0]["target_tokens"]["unigram"] += 1
        self.rewrite_schedule()
        with self.assertRaisesRegex(ValueError, "schedule differs"):
            load_schedule(self.cfg, self.examples())

    def test_missing_or_duplicate_encoded_ids_rejected(self):
        examples = self.examples()
        for invalid in (examples[:-1], examples + [examples[0]]):
            with self.assertRaisesRegex(ValueError, "duplicate IDs or omit"):
                load_schedule(self.cfg, invalid)

    def test_encoding_length_mismatch_rejected(self):
        examples = self.examples()
        examples[0][0].insert(1, 5)
        with self.assertRaisesRegex(ValueError, "encoded lengths"):
            load_schedule(self.cfg, examples)

    def test_missing_update_rejected(self):
        self.schedule.pop()
        self.rewrite_schedule()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_schedule(self.cfg, self.examples())

    def test_common_id_and_length_evidence_must_match(self):
        metadata = json.loads(self.meta_path.read_text())
        metadata["ids"].pop()
        write_json(self.meta_path, metadata)
        self.cfg["common_ids_sha256"] = sha256(self.meta_path)
        with self.assertRaisesRegex(ValueError, "common ID set"):
            load_schedule(self.cfg, self.examples())

    def test_input_tokenizer_fingerprint_mismatch_rejected(self):
        data = self.root / "data"
        data.mkdir()
        (data / "train.jsonl").write_text("original train", encoding="utf-8")
        (data / "spm.model").write_bytes(b"actual tokenizer")
        metadata = json.loads(self.meta_path.read_text())
        metadata.update(source_train_sha256=sha256(data / "train.jsonl"), tokenizer_sha256={"bpe": "bad"})
        write_json(self.meta_path, metadata)
        self.cfg.update(data_dir=str(data), common_ids_sha256=sha256(self.meta_path))
        with self.assertRaisesRegex(ValueError, "different tokenizer"):
            load_schedule(self.cfg, self.examples())


class PreparationEvidenceTests(unittest.TestCase):
    def test_immutable_evidence_identical_reuse_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            _immutable_json(path, {"seed": 42})
            original = path.read_bytes()
            _immutable_json(path, {"seed": 42})
            with self.assertRaises(FileExistsError):
                _immutable_json(path, {"seed": 43})
            self.assertEqual(path.read_bytes(), original)

    def test_completed_preparation_requires_every_saved_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").write_bytes(b"verified")
            manifest = {"files": {"data": sha256(root / "data")}}
            _verify_files(root, manifest)
            (root / "data").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                _verify_files(root, manifest)

    def test_tokenizer_training_settings_are_explicit_and_paired(self):
        bpe = trainer_options("bpe", "train.txt", "output")
        unigram = trainer_options("unigram", "train.txt", "output")
        self.assertEqual(dict(bpe, model_type="unigram"), unigram)
        self.assertEqual(bpe["vocab_size"], 8000)
        self.assertTrue(bpe["hard_vocab_limit"])
        self.assertFalse(bpe["shuffle_input_sentence"])
        self.assertEqual(bpe["input_sentence_size"], 0)
        self.assertEqual(bpe["normalization_rule_name"], "nmt_nfkc")
        self.assertEqual([bpe[k] for k in ("pad_id", "unk_id", "bos_id", "eos_id")], [0, 1, 2, 3])

    def test_missing_or_incompatible_design_is_not_accepted(self):
        with self.assertRaises((KeyError, ValueError)):
            _validate_design({})


if __name__ == "__main__":
    unittest.main()
