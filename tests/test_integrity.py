"""Regression checks for corrupted saved runs; these tests never load a model."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sacrebleu

from mt.common import read_json, sha256, write_json
from mt.examples import export_examples, select_examples
from mt.summary import (EXPECTED, FINGERPRINT_KEYS, collect, validate_saved_outputs,
                        validate_training_log)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class SavedRunIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.data = self.root / "data"
        self.sources = [
            {"id": 0, "en": "The small house is near the river.", "de": "Das kleine Haus steht neben dem Fluss."},
            {"id": 1, "en": "The birds fly over the village.", "de": "Die Vögel fliegen über das kleine Dorf."},
            {"id": 400, "en": "We read the book in the garden.", "de": "Wir lesen das Buch in unserem Garten."},
        ]
        write_jsonl(self.data / "test.jsonl", self.sources)
        self.configs = {}
        for condition in ("A", "B"):
            run = self.runs / f"{condition}42"
            cfg = {"seed": 42, "label_smoothing": 0.1 if condition == "A" else 0.0,
                   "max_updates": 3, "data_dir": str(self.data), "run_dir": str(run)}
            self.configs[condition] = cfg
            write_json(run / "config.json", cfg)
            write_json(run / "summary.json", {"updates": 3, "smoke": False, "pilot": False})
            (run / "best.pt").write_bytes(f"synthetic checkpoint {condition}".encode())
            log = [{"update": update, "data_order_sha256": f"{update:064x}",
                    "target_tokens": 12 + update, "sentences": 2} for update in range(1, 4)]
            write_jsonl(run / "train.jsonl", log)
            for method in ("greedy", "beam"):
                self.write_evaluation(condition, method)

    def write_evaluation(self, condition, method, split="test", rows=None):
        rows = self.sources if rows is None else rows
        source = self.data / f"{split}.jsonl"
        write_jsonl(source, rows)
        run = self.runs / f"{condition}42"
        output = run / (method if split == "test" else f"{split}_{method}")
        output.mkdir(parents=True, exist_ok=True)
        predictions = [row["de"] for row in rows]
        references = [row["de"] for row in rows]
        (output / "predictions.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
        (output / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
        write_jsonl(output / "examples.jsonl", [{"id": row["id"], "source": row["en"],
                    "reference": row["de"], "prediction": pred} for row, pred in zip(rows, predictions)])
        metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
        score = metric.corpus_score(predictions, [references]).score
        fingerprints = {key: "a" * 64 for key in FINGERPRINT_KEYS}
        fingerprints[split + "_sha256"] = sha256(source)
        metrics = {"split": split, "samples": len(rows), "expected_samples": len(rows),
                   "smoke": False, "pilot": False, "method": method,
                   "beam_size": 1 if method == "greedy" else 4,
                   "bleu": score, "signature": str(metric.get_signature()),
                   "sacrebleu_version": sacrebleu.__version__, "config": self.configs[condition],
                   "checkpoint": str(run / "best.pt"), "checkpoint_sha256": sha256(run / "best.pt"),
                   "initial_model_sha256": "b" * 64, "inference_source_sha256": "c" * 64,
                   "data_fingerprints": fingerprints, "reference_source": str(source),
                   "reference_sha256": sha256(source), "predictions_sha256": sha256(output / "predictions.txt"),
                   "references_sha256": sha256(output / "references.txt"),
                   "inference_precision": "float32", "inference_batch_size": 16,
                   "environment": {"gpu": "synthetic fixture"},
                   "length_penalty": 0.6, "max_generation_tokens": 256}
        write_json(output / "metrics.json", metrics)
        return output, metrics

    def evaluation(self):
        output = self.runs / "A42" / "greedy"
        return output, read_json(output / "metrics.json")

    def examples(self, output):
        return [json.loads(line) for line in (output / "examples.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_valid_paired_summary_and_export(self):
        # The fixture uses three official-shaped rows; production still requires 8079.
        with patch.dict(EXPECTED, test=3):
            result = collect(self.runs)
            self.assertEqual(result["paired_seeds_completed"], [42])
            exported = export_examples(self.runs, 42, 400, self.root / "fixed.jsonl")
        self.assertEqual([row["id"] for row in exported], [0, 400])
        self.assertEqual(set(exported[0]["predictions"]), {"A1", "A2", "B1", "B2"})

    def test_validation_cardinality_and_older_reference_hash_schema(self):
        output, metrics = self.write_evaluation("A", "greedy", "validation", self.sources[:2])
        metrics.pop("references_sha256")
        self.assertEqual(len(validate_saved_outputs(metrics, output)), 2)
        (output / "references.txt").write_text("wrong reference\nwrong reference\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            validate_saved_outputs(metrics, output)

    def test_changed_prediction_bytes_are_rejected(self):
        output, metrics = self.evaluation()
        with (output / "predictions.txt").open("a", encoding="utf-8") as stream:
            stream.write("extra sentence\n")
        with self.assertRaisesRegex(ValueError, "predictions: SHA-256 mismatch"):
            validate_saved_outputs(metrics, output)

    def test_malformed_metrics_identifies_the_file(self):
        output, _ = self.evaluation()
        (output / "metrics.json").write_text("{", encoding="utf-8")
        with patch.dict(EXPECTED, test=3), self.assertRaisesRegex(ValueError, r"cannot read JSON.*metrics\.json"):
            collect(self.runs)

    def test_rehashed_prediction_still_must_match_examples(self):
        output, metrics = self.evaluation()
        prediction = output / "predictions.txt"
        prediction.write_text("wrong text\n" + "\n".join(row["de"] for row in self.sources[1:]) + "\n", encoding="utf-8")
        metrics["predictions_sha256"] = sha256(prediction)
        with self.assertRaisesRegex(ValueError, "example/prediction mismatch at row 1"):
            validate_saved_outputs(metrics, output)

    def test_missing_example_rejected_even_when_not_selected(self):
        output, metrics = self.evaluation()
        rows = self.examples(output)
        write_jsonl(output / "examples.jsonl", [rows[0], rows[2]])
        with self.assertRaisesRegex(ValueError, "examples has 2 rows; expected 3"):
            validate_saved_outputs(metrics, output)

    def test_duplicate_or_reordered_ids_are_rejected(self):
        output, metrics = self.evaluation()
        original = self.examples(output)
        for kind in ("duplicate", "reordered"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(original)
                if kind == "duplicate":
                    rows[-1]["id"] = rows[0]["id"]
                else:
                    rows[0], rows[1] = rows[1], rows[0]
                write_jsonl(output / "examples.jsonl", rows)
                with self.assertRaisesRegex(ValueError, "duplicate id|id order mismatch"):
                    validate_saved_outputs(metrics, output)

    def test_example_source_and_reference_are_checked(self):
        output, metrics = self.evaluation()
        original = self.examples(output)
        for field in ("source", "reference"):
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[1][field] = "a different sentence"
                write_jsonl(output / "examples.jsonl", rows)
                with self.assertRaisesRegex(ValueError, f"{field} mismatch at row 2"):
                    validate_saved_outputs(metrics, output)

    def test_bleu_is_recomputed_from_saved_text(self):
        output, metrics = self.evaluation()
        metrics["bleu"] -= 1
        with self.assertRaisesRegex(ValueError, "does not match recomputed BLEU"):
            validate_saved_outputs(metrics, output)

    def test_checkpoint_replacement_and_missing_output_are_rejected(self):
        output, metrics = self.evaluation()
        checkpoint = Path(metrics["checkpoint"])
        old_bytes = checkpoint.read_bytes()
        checkpoint.write_bytes(b"a different checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint: SHA-256 mismatch"):
            validate_saved_outputs(metrics, output)
        checkpoint.write_bytes(old_bytes)
        (output / "predictions.txt").unlink()
        with self.assertRaisesRegex(ValueError, "required file is missing"):
            validate_saved_outputs(metrics, output)

    def test_matching_but_incomplete_training_logs_cannot_pass(self):
        for condition in ("A", "B"):
            path = self.runs / f"{condition}42" / "train.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            write_jsonl(path, rows[:1] + rows[2:])
        with patch.dict(EXPECTED, test=3), self.assertRaisesRegex(ValueError, "training log has 2 updates; expected 3"):
            collect(self.runs)

    def test_duplicate_updates_cannot_hide_behind_dictionary_keys(self):
        run = self.runs / "A42"
        path = run / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["update"] = 2
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "missing, duplicate, or reordered update"):
            validate_training_log(run, self.configs["A"], "A42")

    def test_paired_training_token_counts_must_match(self):
        path = self.runs / "B42" / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["target_tokens"] += 1
        write_jsonl(path, rows)
        with patch.dict(EXPECTED, test=3), self.assertRaisesRegex(ValueError, "token/sentence counts differ"):
            collect(self.runs)

    def test_fixed_selection_rejects_missing_unselected_rows(self):
        output, _ = self.evaluation()
        groups = {name: self.examples(output) for name in ("A1", "A2", "B1", "B2")}
        groups["B2"].pop(1)  # id=1 would not have appeared in the modulus=400 output.
        with self.assertRaisesRegex(ValueError, "refusing to discard samples"):
            select_examples(groups, 400)

    def test_failed_export_preserves_existing_file(self):
        output, _ = self.evaluation()
        write_jsonl(output / "examples.jsonl", self.examples(output)[:-1])
        saved = self.root / "fixed.jsonl"
        saved.write_text("previous valid export\n", encoding="utf-8")
        with patch.dict(EXPECTED, test=3), self.assertRaises(ValueError):
            export_examples(self.runs, 42, 400, saved)
        self.assertEqual(saved.read_text(), "previous valid export\n")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            export_examples(self.runs, 42, 0, saved)


if __name__ == "__main__":
    unittest.main()
