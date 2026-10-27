"""Quality report regressions use saved text fixtures and never touch CUDA."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sacrebleu

from mt.common import read_json, sha256, write_json
from mt.quality_report import (CORE_SOURCES, GROUPS, bucket_name, collect_quality, compare_configs, generate_report,
                               compare_examples, read_source_provenance, report_lines, source_lengths)
from mt.summary import EXPECTED, FINGERPRINT_KEYS


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class QualityReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline, self.long = self.root / "baseline", self.root / "long"
        self.data = self.root / "data"
        self.rows = [{"id": index, "en": f"This is the original source sentence number {index}.",
                      "de": f"Das ist der deutsche Referenzsatz mit der Nummer {index}."} for index in range(8)]
        self.lengths = [1, 20, 21, 40, 41, 80, 81, 110]
        score_protocol = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
        score_protocol.corpus_score([row["de"] for row in self.rows], [[row["de"] for row in self.rows]])
        jsonl(self.data / "test.jsonl", self.rows)
        (self.data / "spm.model").write_bytes(b"tokenizer fixture: only its hash is used")
        for run, budget, best in ((self.baseline, 2, 2), (self.long, 4, 3)):
            config = {"seed": 42, "label_smoothing": .1, "max_updates": budget,
                      "data_dir": str(self.data), "run_dir": str(run), "warmup_updates": 1,
                      "validate_every": 1}
            write_json(run / "config.json", config)
            write_json(run / "summary.json", {"complete": True, "updates": budget, "smoke": False,
                "pilot": False, "best_validation_bleu": 10 + best, "training_seconds_total": 4 * budget,
                "peak_cuda_bytes": 1024, "parameters": {"total": 100, "trainable": 100}})
            (run / "best.pt").write_bytes(f"fixture checkpoint {run.name}".encode())
            logs = [{"update": update, "data_order_sha256": f"{update:064x}", "target_tokens": 100 + update,
                     "sentences": 8, "lr_used": 1 / update, "validation_bleu_greedy": 10 + min(update, best)
                     - (1 if update > best else 0), "validation_samples": 888,
                     "validation_signature": str(score_protocol.get_signature())}
                    for update in range(1, budget + 1)]
            jsonl(run / "train.jsonl", logs)
            for method in ("greedy", "beam"):
                output = run / method
                predictions = [row["de"] if run == self.long else row["de"].replace("deutsche", "kleine")
                               for row in self.rows]
                references = [row["de"] for row in self.rows]
                jsonl(output / "examples.jsonl", [{"id": row["id"], "source": row["en"],
                    "reference": row["de"], "prediction": pred} for row, pred in zip(self.rows, predictions)])
                (output / "predictions.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
                (output / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
                bleu = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
                score = bleu.corpus_score(predictions, [references]).score
                fingerprints = {key: "a" * 64 for key in FINGERPRINT_KEYS}
                fingerprints["test_sha256"] = sha256(self.data / "test.jsonl")
                fingerprints["tokenizer_sha256"] = sha256(self.data / "spm.model")
                write_json(output / "metrics.json", {"split": "test", "samples": 8, "expected_samples": 8,
                    "smoke": False, "pilot": False, "method": method, "beam_size": 1 if method == "greedy" else 4,
                    "bleu": score, "signature": str(bleu.get_signature()), "sacrebleu_version": sacrebleu.__version__,
                    "config": config, "checkpoint": str(run / "best.pt"), "checkpoint_sha256": sha256(run / "best.pt"),
                    "checkpoint_update": best, "initial_model_sha256": "b" * 64, "inference_source_sha256": "c" * 64,
                    "training_source_hashes": {"train.py": "old" if run == self.baseline else "new"},
                    "data_fingerprints": fingerprints, "reference_source": str(self.data / "test.jsonl"),
                    "reference_sha256": sha256(self.data / "test.jsonl"), "predictions_sha256": sha256(output / "predictions.txt"),
                    "references_sha256": sha256(output / "references.txt"), "inference_precision": "float32",
                    "inference_batch_size": 16, "environment": {"gpu": "synthetic fixture"},
                    "length_penalty": .6, "max_generation_tokens": 256})
        long_hashes = self.write_source_snapshot(self.long)
        for run in (self.baseline, self.long):
            hashes = dict(long_hashes)
            if run == self.baseline:
                hashes["train.py"] = "d" * 64  # Historical 42 source change, no original snapshot.
            for method in ("greedy", "beam"):
                self.change_metric(run, method, "training_source_hashes", hashes)
        self.addCleanup(patch.stopall)
        patch.dict(EXPECTED, test=8).start()

    def collect(self, long=None, chrf=False):
        with patch("mt.quality_report.source_lengths", return_value=self.lengths):
            return collect_quality(self.baseline, long or self.long, chrf)

    def change_metric(self, run, method, key, value):
        path = run / method / "metrics.json"
        metrics = read_json(path)
        metrics[key] = value
        write_json(path, metrics)

    def test_complete_report_independently_scores_every_bucket_and_difference(self):
        result, membership, fixed = self.collect(chrf=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["covered_test_samples"], 8)
        self.assertEqual([row["count"] for row in result["buckets"]], [2, 2, 2, 2])
        self.assertEqual([row["id"] for row in membership], list(range(8)))
        self.assertEqual(result["training_prefix_controls"], "verified")
        self.assertGreater(result["long_minus_baseline"]["greedy"]["bleu"], 0)
        self.assertAlmostEqual(result["scores"]["long_greedy"]["chrf"], 100)
        for row in result["buckets"]:
            self.assertAlmostEqual(row["scores"]["long_greedy"]["bleu"], 100)
        self.assertEqual([row["id"] for row in fixed], [0])
        self.assertEqual(set(fixed[0]["predictions"]), set(GROUPS))
        self.assertEqual(result["training_source_differences"], ["train.py"])

    def test_pending_long_run_keeps_four_columns_and_nulls(self):
        result, membership, fixed = self.collect(self.root / "not_started")
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["scores"]["long_beam"])
        self.assertIsNone(result["long_minus_baseline"]["beam"]["bleu"])
        self.assertIsNone(fixed[0]["predictions"]["long_greedy"])
        self.assertEqual(len(membership), 8)
        content = "\n".join(report_lines(result))
        self.assertIn("待完成（pending）", content)
        self.assertIn("不含 BOS/EOS", content)
        self.assertIn("同一组 1 条样例", content)
        self.assertNotIn("21 条样例", content)

    def test_saved_prediction_corruption_is_not_reported_as_pending(self):
        path = self.long / "greedy" / "predictions.txt"
        path.write_text("corrupt\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "predictions: SHA-256 mismatch"):
            self.collect()

    def test_same_budget_controls_reject_changed_learning_schedule(self):
        baseline = {"max_updates": 3000, "warmup_updates": 300}
        extended = {"max_updates": 12000, "warmup_updates": 1200}
        with self.assertRaisesRegex(ValueError, "warmup_updates"):
            compare_configs(baseline, extended)

    def test_frozen_test_decoder_cannot_change_in_extension(self):
        self.change_metric(self.long, "greedy", "length_penalty", 0)
        with self.assertRaisesRegex(ValueError, "length_penalty differs"):
            self.collect()

    def test_same_initialization_is_checked(self):
        self.change_metric(self.long, "greedy", "initial_model_sha256", "d" * 64)
        with self.assertRaisesRegex(ValueError, "initial_model_sha256 differs"):
            self.collect()

    def test_model_source_change_is_not_accepted_as_a_budget_extension(self):
        self.change_metric(self.long, "greedy", "training_source_hashes", {"model.py": "changed"})
        with self.assertRaisesRegex(ValueError, "checkpoint source metadata differs"):
            self.collect()

    def test_validation_scoring_protocol_cannot_silently_change(self):
        path = self.long / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["validation_signature"] = "different scoring protocol"
        jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "validation signature differs"):
            self.collect()

    def test_checkpoint_must_match_first_best_validation_update(self):
        self.change_metric(self.long, "greedy", "checkpoint_update", 4)
        with self.assertRaisesRegex(ValueError, "not the earliest best full-validation"):
            self.collect()

    def test_shared_prefix_cannot_hide_changed_order_or_learning_rate(self):
        path = self.long / "train.jsonl"
        original = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for field, value in (("lr_used", 99), ("data_order_sha256", "f" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[0][field] = value
                jsonl(path, changed)
                with self.assertRaisesRegex(ValueError, f"training prefix differs.*{field}"):
                    self.collect()

    def test_missing_unselected_example_does_not_shrink_bucket_coverage(self):
        path = self.long / "beam" / "examples.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        jsonl(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "examples has 7 rows; expected 8"):
            self.collect()

    def test_source_bucket_boundaries_are_fixed_without_special_tokens(self):
        self.assertEqual([bucket_name(value) for value in (20, 21, 40, 41, 80, 81)],
                         ["<=20", "21–40", "21–40", "41–80", "41–80", ">80"])
        metrics = read_json(self.baseline / "greedy" / "metrics.json")
        rows = [{"source": "fixture"}]
        with patch("mt.quality_report.spm.SentencePieceProcessor") as processor:
            processor.return_value.encode.return_value = [4] * 20
            self.assertEqual(source_lengths(rows, metrics), [20])
        (self.data / "spm.model").write_bytes(b"changed tokenizer")
        with self.assertRaisesRegex(ValueError, "tokenizer: SHA-256 mismatch"):
            source_lengths(rows, metrics)

    def test_empty_bucket_is_not_a_zero_score_and_alignment_is_required(self):
        path = self.baseline / "greedy" / "examples.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        buckets, _, _ = compare_examples({"baseline_greedy": rows}, [20] * 8)
        self.assertEqual(buckets[1]["count"], 0)
        self.assertIsNone(buckets[1]["scores"]["baseline_greedy"]["bleu"])
        broken = copy.deepcopy(rows)
        broken[3]["source"] = "a different source"
        with self.assertRaisesRegex(ValueError, "alignment differs at ID 3"):
            compare_examples({"baseline_greedy": rows, "long_beam": broken}, [20] * 8)

    def test_explicit_seed43_and_seed44_match_their_own_baseline(self):
        for seed in (43, 44):
            with self.subTest(seed=seed):
                for run in (self.baseline, self.long):
                    config = read_json(run / "config.json")
                    config["seed"] = seed
                    write_json(run / "config.json", config)
                    hashes = self.write_source_snapshot(run)
                    for method in ("greedy", "beam"):
                        self.change_metric(run, method, "config", config)
                        self.change_metric(run, method, "training_source_hashes", hashes)
                with patch("mt.quality_report.source_lengths", return_value=self.lengths):
                    result, _, _ = collect_quality(self.baseline, self.long, True, seed)
                self.assertEqual(result["seed"], seed)
                self.assertEqual(result["status"], "complete")
                with self.assertRaisesRegex(ValueError, "seed or label smoothing differs"):
                    self.collect()

    def write_source_snapshot(self, run):
        snapshot = run / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        hashes = {}
        for name in CORE_SOURCES:
            (snapshot / name).write_text(f"# fixture {name}\n", encoding="utf-8")
            hashes[name] = sha256(snapshot / name)
        write_json(run / "source_hashes.json", hashes)
        write_json(snapshot / "manifest.json", {"schema_version": 1, "core_source_hashes": hashes})
        return hashes

    def test_seven_module_snapshot_verifies_common_and_data_and_preserves_historical_missing(self):
        original = read_source_provenance(self.baseline)
        self.assertEqual(original["snapshot_status"], "missing_historical_snapshot")
        with self.assertRaisesRegex(ValueError, "seven-module source snapshot is required"):
            read_source_provenance(self.baseline, require_snapshot=True)
        self.write_source_snapshot(self.baseline)
        self.assertEqual(read_source_provenance(self.baseline, True)["snapshot_status"], "verified")
        for name in ("common.py", "data.py"):
            with self.subTest(module=name):
                self.write_source_snapshot(self.baseline)
                (self.baseline / "source_snapshot" / name).write_text("# corrupt", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    read_source_provenance(self.baseline, True)

    def test_snapshot_manifest_cannot_disagree_with_recorded_hashes(self):
        hashes = self.write_source_snapshot(self.baseline)
        hashes["common.py"] = "a" * 64
        write_json(self.baseline / "source_hashes.json", hashes)
        with self.assertRaisesRegex(ValueError, "source snapshot manifest differs"):
            read_source_provenance(self.baseline, True)

    def test_completed_seed42_long_requires_snapshot_but_historical_baseline_does_not(self):
        result, _, _ = self.collect()
        self.assertEqual(result["core_source_provenance"]["baseline"]["snapshot_status"], "missing_historical_snapshot")
        self.assertEqual(result["core_source_provenance"]["long"]["snapshot_status"], "verified")
        (self.long / "source_snapshot" / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "seven-module source snapshot is required"):
            self.collect()

    def test_new_seed_cannot_use_historical_train_or_decoder_change_exception(self):
        for seed in (43, 44):
            for filename in ("train.py", "decoding.py", "common.py", "data.py"):
                with self.subTest(seed=seed, module=filename):
                    for run in (self.baseline, self.long):
                        config = read_json(run / "config.json")
                        config["seed"] = seed
                        write_json(run / "config.json", config)
                        hashes = self.write_source_snapshot(run)
                        if run == self.long:
                            (run / "source_snapshot" / filename).write_text("# changed execution core\n", encoding="utf-8")
                            hashes[filename] = sha256(run / "source_snapshot" / filename)
                            write_json(run / "source_hashes.json", hashes)
                            write_json(run / "source_snapshot" / "manifest.json", {"schema_version": 1, "core_source_hashes": hashes})
                        for method in ("greedy", "beam"):
                            self.change_metric(run, method, "config", config)
                            self.change_metric(run, method, "training_source_hashes", hashes)
                    with patch("mt.quality_report.source_lengths", return_value=self.lengths), \
                         self.assertRaisesRegex(ValueError, "unexpected training source changes"):
                        collect_quality(self.baseline, self.long, True, seed)

    def test_missing_metrics_do_not_hide_wrong_seed_config(self):
        run = self.root / "unstarted43"
        config = read_json(self.long / "config.json")
        config["seed"] = 43
        write_json(run / "config.json", config)
        with self.assertRaisesRegex(ValueError, "seed or label smoothing differs"):
            self.collect(run)

    def test_prefix_floating_point_differences_are_preserved_not_rejected(self):
        for run in (self.baseline, self.long):
            logs = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in logs:
                row["loss"] = 5.0 + row["update"] / 10 + (0.00001 if run == self.long else 0)
            jsonl(run / "train.jsonl", logs)
        result, _, _ = self.collect()
        difference = result["training_prefix_numerical_differences"]
        self.assertEqual(difference["compared_updates"], 2)
        self.assertEqual(difference["fields"]["loss"]["different_points"], 2)
        self.assertAlmostEqual(difference["fields"]["loss"]["max_absolute_difference"], .00001)
        self.assertEqual(result["training_prefix_controls"], "verified")

    def test_default_report_outputs_are_isolated_and_seed42_stays_compatible(self):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed), \
                 patch("mt.quality_report.collect_quality", return_value=({}, [], [])) as collect, \
                 patch("mt.quality_report.save_report", side_effect=lambda result, membership, fixed, out: out):
                output = generate_report(seed=seed)
                collect.assert_called_once_with(f"artifacts/runs/A{seed}", f"artifacts/runs/quality_A{seed}_12000", False, seed)
                self.assertEqual(output, "artifacts/quality_report" if seed == 42 else f"artifacts/quality_seed{seed}/report")


if __name__ == "__main__":
    unittest.main()
