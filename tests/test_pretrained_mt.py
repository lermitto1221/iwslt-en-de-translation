import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mt import pretrained_mt as pm


class PretrainedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.rows = [{"id": i, "en": f"Source {i}.", "de": f"Referenz {i}."} for i in (2, 7, 11)]

    def record(self, rows, predictions=None):
        value = {"ids": [r["id"] for r in rows],
                 "predictions": predictions or [r["de"] for r in rows],
                 "source_lengths": [4] * len(rows), "generated_lengths": [4] * len(rows),
                 "limit_hits": 0, "unfinished": 0, "peak_cuda_bytes": None, "elapsed_seconds": 0.2}
        return dict(value, record_sha256=pm.digest(value))

    def write_journal(self, records, tail=b""):
        path = self.path / "batches.jsonl"
        path.write_bytes(b"".join((json.dumps(r) + "\n").encode() for r in records) + tail)
        return path

    def completed_fixture(self):
        records = [self.record(self.rows[:2]), self.record(self.rows[2:])]
        self.write_journal(records)
        predictions = [r["de"] for r in self.rows]
        for name in ("predictions.txt", "references.txt"):
            (self.path / name).write_text("\n".join(predictions) + "\n", encoding="utf-8")
        examples = [{"id": r["id"], "source": r["en"], "reference": r["de"], "prediction": p}
                    for r, p in zip(self.rows, predictions)]
        (self.path / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in examples) + "\n", encoding="utf-8")
        context = {"batch_size": 2, "split": "validation", "method": "beam", "plan_sha256": "frozen", "device": "cpu"}
        metrics = {"context": context, "samples": len(self.rows), **pm.score(predictions, predictions),
                   **pm.summarize_records(records, len(self.rows), "cpu"),
                   "file_sha256": {name: pm.sha256(self.path / name) for name in
                        ("predictions.txt", "references.txt", "examples.jsonl", "batches.jsonl")}}
        pm.write_json(self.path / "metrics.json", metrics)
        return context

    def test_exclusive_plan_never_overwrites(self):
        path = self.path / "plan.json"
        pm.exclusive_json(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            pm.exclusive_json(path, {"a": 2})
        self.assertEqual(pm.read_json(path), {"a": 1})

    def test_journal_partial_line_can_resume_only_complete_batches(self):
        path = self.write_journal([self.record(self.rows[:2])], b'{"interrupted":')
        records, end = pm.read_journal(path, self.rows, 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(path.read_bytes()[end:], b'{"interrupted":')

    def test_journal_rejects_wrong_id_order(self):
        path = self.write_journal([self.record(list(reversed(self.rows[:2])))])
        with self.assertRaisesRegex(ValueError, "ID/order"):
            pm.read_journal(path, self.rows, 2)

    def test_journal_rejects_corrupted_prediction(self):
        record = self.record(self.rows[:2])
        record["predictions"][0] = "tampered"
        path = self.write_journal([record])
        with self.assertRaisesRegex(ValueError, "Damaged"):
            pm.read_journal(path, self.rows, 2)

    def test_journal_rejects_missing_prediction(self):
        path = self.write_journal([self.record(self.rows[:2], ["one"] )])
        with self.assertRaisesRegex(ValueError, "alignment"):
            pm.read_journal(path, self.rows, 2)

    def test_complete_results_are_read_only_and_recomputed(self):
        context = self.completed_fixture()
        before = {p.name: pm.sha256(p) for p in self.path.iterdir()}
        with patch.object(pm, "atomic_json", side_effect=AssertionError("write")), \
             patch.object(pm, "load_native", side_effect=AssertionError("load model")):
            result = pm.verify_result(self.path, self.rows, context)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(before, {p.name: pm.sha256(p) for p in self.path.iterdir()})

    def test_complete_reuse_rejects_wrong_split_or_protocol(self):
        context = self.completed_fixture()
        with self.assertRaisesRegex(ValueError, "identity"):
            pm.verify_result(self.path, self.rows, dict(context, method="greedy"))

    def test_complete_reuse_rejects_score_tampering(self):
        context = self.completed_fixture()
        metrics = pm.read_json(self.path / "metrics.json")
        metrics["bleu"] += 1
        pm.write_json(self.path / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            pm.verify_result(self.path, self.rows, context)

    def test_complete_reuse_rejects_nonfinite_score(self):
        context = self.completed_fixture()
        metrics = pm.read_json(self.path / "metrics.json")
        metrics["bleu"] = float("nan")
        pm.write_json(self.path / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            pm.verify_result(self.path, self.rows, context)

    def test_complete_reuse_rejects_reference_hash_corruption(self):
        context = self.completed_fixture()
        (self.path / "references.txt").write_text("different\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            pm.verify_result(self.path, self.rows, context)

    def test_complete_reuse_rejects_cost_and_limit_tampering(self):
        context = self.completed_fixture()
        metrics = pm.read_json(self.path / "metrics.json")
        metrics["generation_limit_hits"] += 1
        pm.write_json(self.path / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "batch journal"):
            pm.verify_result(self.path, self.rows, context)

    def test_complete_reuse_checks_rows_even_if_hashes_changed_together(self):
        context = self.completed_fixture()
        examples_path = self.path / "examples.jsonl"
        values = [json.loads(x) for x in examples_path.read_text().splitlines()]
        values[0]["id"] = 999
        examples_path.write_text("\n".join(json.dumps(x) for x in values) + "\n")
        metrics = pm.read_json(self.path / "metrics.json")
        metrics["file_sha256"]["examples.jsonl"] = pm.sha256(examples_path)
        pm.write_json(self.path / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "alignment"):
            pm.verify_result(self.path, self.rows, context)

    def test_score_is_corpus_score_not_sentence_average(self):
        predictions = ["The cat is here.", "A much longer sentence with missing content."]
        references = ["The cat is here.", "A much longer sentence with all the required content."]
        import sacrebleu
        metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
        self.assertAlmostEqual(pm.score(predictions, references)["bleu"],
                               metric.corpus_score(predictions, [references]).score)
        with self.assertRaises(ValueError):
            pm.score(predictions[:-1], references)

    def test_length_audit_disables_truncation_and_preserves_ids(self):
        class Tokenizer:
            def __call__(self, values, padding, truncation):
                self.assert_false = not padding and not truncation
                return {"input_ids": [[1] * n for n in (3, 513, 4)]}
        tokenizer = Tokenizer()
        value = pm.check_lengths(self.rows, tokenizer)
        self.assertTrue(tokenizer.assert_false)
        self.assertEqual(value["oversize"], [{"id": 7, "length": 513}])

    def test_freeze_refuses_started_formal_output(self):
        pm.write_json(self.path / "config.json", {"splits": ["validation", "test"]})
        (self.path / "test").mkdir()
        with self.assertRaisesRegex(ValueError, "after formal"):
            pm.freeze(self.path)


if __name__ == "__main__":
    unittest.main()
