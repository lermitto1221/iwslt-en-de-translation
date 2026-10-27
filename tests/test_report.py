"""Statistical regressions: incomplete pairs, within-seed differences and SD."""
import math
import tempfile
import unittest
from pathlib import Path

from mt.report import paired_aggregates, report_lines, sample_statistics, select_seeds


def result_for_seed(seed, scores):
    return {f"{group}_seed{seed}": {"bleu": score, "sentences_per_second": 20 + index,
            "mean_batch_latency_ms": 40 + index, "mean_latency_ms": 3 + index,
            "config": {"label_smoothing": .1 if group.startswith("A") else .0},
            "method": "greedy" if group.endswith("1") else "beam",
            "samples": 8079, "generation_limit_hits": 0, "checkpoint_update": 3000,
            "signature": "fixture", "environment": {"gpu": "fixture GPU", "torch": "fixture"},
            "inference_precision": "float32", "inference_batch_size": 16}
            for index, (group, score) in enumerate(zip(("A1", "A2", "B1", "B2"), scores))}


class PairedReportTests(unittest.TestCase):
    def test_overview_uses_only_complete_paired_seeds(self):
        result = result_for_seed(42, [10, 13, 9, 11])
        result.update(result_for_seed(43, [20, 21, 18, 22]))
        # A spectacular score from an incomplete seed must not affect any mean.
        result.update(result_for_seed(44, [999, 1000]))
        values = paired_aggregates(result, [42, 43, 44])
        self.assertEqual(values["paired_seeds"], [42, 43])
        self.assertEqual(values["excluded_incomplete_seeds"], [44])
        self.assertEqual(values["groups"]["A1"]["bleu"]["mean"], 15)
        self.assertAlmostEqual(values["groups"]["A1"]["bleu"]["sample_std"], math.sqrt(50))
        self.assertEqual(values["groups"]["A2"]["bleu"]["n"], 2)

    def test_contrasts_are_paired_before_standard_deviation(self):
        result = result_for_seed(42, [10, 13, 9, 11])
        result.update(result_for_seed(43, [20, 21, 18, 22]))
        contrast = paired_aggregates(result, [42, 43])["contrasts"]
        self.assertEqual(contrast["A_beam_minus_greedy"]["per_seed"], {42: 3, 43: 1})
        self.assertEqual(contrast["A_beam_minus_greedy"]["mean"], 2)
        self.assertAlmostEqual(contrast["A_beam_minus_greedy"]["sample_std"], math.sqrt(2))
        self.assertEqual(contrast["B_beam_minus_greedy"]["mean"], 3)
        self.assertAlmostEqual(contrast["greedy_smoothing_effect"]["sample_std"], math.sqrt(.5))
        self.assertEqual(contrast["beam_smoothing_effect"]["per_seed"], {42: 2, 43: -1})
        self.assertAlmostEqual(contrast["beam_smoothing_effect"]["sample_std"], math.sqrt(4.5))

    def test_single_seed_has_no_estimated_standard_deviation(self):
        values = paired_aggregates(result_for_seed(43, [15, 16, 14, 17]), [43])
        self.assertEqual(values["paired_seeds"], [43])
        self.assertIsNone(values["groups"]["A1"]["bleu"]["sample_std"])
        self.assertIsNone(values["contrasts"]["A_beam_minus_greedy"]["sample_std"])
        self.assertEqual(values["contrasts"]["A_beam_minus_greedy"]["mean"], 1)

    def test_no_complete_pair_keeps_all_aggregate_values_pending(self):
        values = paired_aggregates(result_for_seed(42, [15, 16, 14]), [42])
        self.assertEqual(values["paired_seeds"], [])
        self.assertIsNone(values["groups"]["A1"]["bleu"]["mean"])
        self.assertIsNone(values["contrasts"]["greedy_smoothing_effect"]["mean"])
        self.assertIsNone(sample_statistics([])["sample_std"])

    def test_selected_seed_does_not_silently_include_other_complete_seeds(self):
        result = result_for_seed(42, [10, 13, 9, 11])
        result.update(result_for_seed(43, [20, 21, 18, 22]))
        values = paired_aggregates(result, [43, 43])
        self.assertEqual(values["groups"]["A1"]["bleu"]["mean"], 20)
        self.assertEqual(values["groups"]["A1"]["bleu"]["n"], 1)

    def test_all_seeds_includes_progress_but_not_nonexistent_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            log = root / "A43" / "train.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text("", encoding="utf-8")
            result = result_for_seed(42, [10, 13, 9, 11])
            self.assertEqual(select_seeds(result, root, all_seeds=True), [42, 43])
            self.assertEqual(select_seeds(result, root, seed=42), [42])

    def test_markdown_is_dynamic_and_does_not_invent_sample_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary) / "runs", Path(temporary) / "report"
            result = result_for_seed(43, [10, 13, 9, 11])
            values = paired_aggregates(result, [43])
            content = "\n".join(report_lines(result, root, out, [43], values))
            self.assertIn("## 种子 43", content)
            self.assertNotIn("## 种子 42", content)
            self.assertIn("不可估计", content)
            self.assertIn("不是置信区间", content)
            self.assertIn("尚未保存该种子的固定样例文件", content)
            self.assertNotIn("21 句", content)
            sample = root.parent / "fixed_examples_seed43.jsonl"
            sample.write_text('{"id": 0}\n{"id": 400}\n', encoding="utf-8")
            content = "\n".join(report_lines(result, root, out, [43], values))
            self.assertIn("实际 2 条", content)


if __name__ == "__main__":
    unittest.main()
