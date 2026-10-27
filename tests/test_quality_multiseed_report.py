"""Paired quality statistics and controls use CPU-only, deterministic fixtures."""
import copy
import math
import unittest
from unittest.mock import patch

from mt.quality_multiseed_report import paired_aggregates, validate_cross_seed, collect_multiseed, generate_report
from mt.quality_report import CORE_SOURCES, GROUPS


def fixture(seed, complete=True):
    scores = {name: {"bleu": 10 + (seed - 42) + (5 + (seed - 42) if name.startswith("long_") else 0)
                           + (1 if name.endswith("beam") else 0),
                     "chrf": 30 + (seed - 42) + (8 + (seed - 42) if name.startswith("long_") else 0)}
              for name in GROUPS}
    metadata = {}
    for name in GROUPS:
        if not complete and name.startswith("long_"):
            scores[name] = None
            continue
        metadata[name] = {"config": {"seed": seed, "max_updates": 12000 if name.startswith("long_") else 3000,
                         "run_dir": f"run/{seed}/{name}", "validate_every": 500, "d_model": 256},
                         "data_fingerprints": {"test_sha256": "a" * 64}, "inference_source_sha256": "b" * 64,
                         "signature": "common signature", "inference_precision": "float32", "inference_batch_size": 16,
                         "length_penalty": .6, "max_generation_tokens": 256, "beam_size": 1 if name.endswith("greedy") else 4,
                         "initial_model_sha256": f"{seed:064x}"}
    return {"seed": seed, "status": "complete" if complete else "pending", "scores": scores,
            "config_controls": "verified", "training_prefix_controls": "verified" if complete else "pending",
            "core_source_provenance": {"long": {"snapshot_status": "verified" if complete else "pending",
                "core_source_hashes": {name: f"{index:064x}" for index, name in enumerate(CORE_SOURCES, 1)} if complete else {}}},
            "evaluation_metadata": metadata}


class MultiSeedQualityTests(unittest.TestCase):
    def setUp(self):
        self.results = {seed: fixture(seed) for seed in (42, 43, 44)}
        self.protocol = {"selection_split": "validation", "beam_size": 4, "length_penalty": .6,
                         "max_generation_tokens": 256}

    def test_three_seed_mean_sample_std_and_paired_differences(self):
        result = paired_aggregates(self.results, [42, 43, 44])
        self.assertEqual(result["included_seeds"], [42, 43, 44])
        self.assertEqual(result["groups"]["long_greedy"]["bleu"], {"n": 3, "mean": 17, "sample_std": 2})
        delta = result["paired_long_minus_baseline"]["beam"]["bleu"]
        self.assertEqual(delta["per_seed"], {42: 5, 43: 6, 44: 7})
        self.assertEqual(delta["mean"], 6)
        self.assertEqual(delta["sample_std"], 1)  # ddof=1, not population sqrt(2/3).

    def test_pending_seed_excluded_from_all_aggregates_without_zero_fill(self):
        self.results[43] = fixture(43, False)
        result = paired_aggregates(self.results, [42, 43, 44])
        self.assertEqual(result["included_seeds"], [42, 44])
        self.assertEqual(result["excluded_incomplete_seeds"], [43])
        self.assertEqual(result["groups"]["baseline_greedy"]["bleu"]["mean"], 11)
        self.assertAlmostEqual(result["groups"]["long_greedy"]["bleu"]["sample_std"], math.sqrt(8))
        self.assertIsNone(self.results[43]["scores"]["long_greedy"])

    def test_no_complete_seed_and_single_seed_uncertainty_are_null(self):
        result = paired_aggregates({42: fixture(42, False)}, [42])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["groups"]["long_greedy"]["bleu"]["mean"])
        one = paired_aggregates({42: fixture(42)}, [42])
        self.assertIsNone(one["groups"]["long_greedy"]["bleu"]["sample_std"])

    def test_complete_seed_with_missing_score_or_unverified_controls_is_rejected(self):
        for field in ("score", "controls"):
            with self.subTest(field=field):
                values = copy.deepcopy(self.results)
                if field == "score":
                    values[43]["scores"]["long_beam"]["chrf"] = None
                else:
                    values[43]["training_prefix_controls"] = "pending"
                with self.assertRaises(ValueError):
                    paired_aggregates(values, [42, 43, 44])

    def test_wrong_labels_duplicate_seeds_and_nonfinite_scores_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_aggregates(self.results, [42, 42])
        self.results[43]["seed"] = 44
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            paired_aggregates(self.results, [42, 43, 44])
        self.results[43] = fixture(43)
        self.results[43]["scores"]["long_greedy"]["bleu"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing bleu"):
            paired_aggregates(self.results, [42, 43, 44])

    def test_cross_seed_allows_distinct_init_and_requires_same_seed_init(self):
        controls = validate_cross_seed(self.results, self.protocol)
        self.assertEqual(len(set(controls["initialization_sha256_by_seed"].values())), 3)
        self.results[43]["evaluation_metadata"]["long_beam"]["initial_model_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "initialization differs within seed"):
            validate_cross_seed(self.results, self.protocol)

    def test_cross_seed_duplicate_initialization_is_detected(self):
        for metrics in self.results[43]["evaluation_metadata"].values():
            metrics["initial_model_sha256"] = f"{42:064x}"
        with self.assertRaisesRegex(ValueError, "different seeds unexpectedly share"):
            validate_cross_seed(self.results, self.protocol)

    def test_cross_seed_data_decoder_and_training_config_must_match(self):
        for field, value in (("data_fingerprints", {"test_sha256": "d" * 64}),
                             ("inference_source_sha256", "d" * 64), ("inference_batch_size", 32)):
            with self.subTest(field=field):
                values = copy.deepcopy(self.results)
                values[44]["evaluation_metadata"]["long_greedy"][field] = value
                with self.assertRaisesRegex(ValueError, "cross-seed"):
                    validate_cross_seed(values, self.protocol)
        self.results[44]["evaluation_metadata"]["long_greedy"]["config"]["d_model"] = 512
        with self.assertRaisesRegex(ValueError, "cross-seed long config differs"):
            validate_cross_seed(self.results, self.protocol)

    def test_all_long_seeds_require_identical_seven_module_sources(self):
        for filename in CORE_SOURCES:
            with self.subTest(module=filename):
                values = copy.deepcopy(self.results)
                values[43]["core_source_provenance"]["long"]["core_source_hashes"][filename] = "d" * 64
                with self.assertRaisesRegex(ValueError, "cross-seed long core source hashes differ"):
                    validate_cross_seed(values, self.protocol)

    def test_even_historical_long42_requires_complete_verified_provenance(self):
        for modification in ("missing", "unverified", "incomplete"):
            with self.subTest(modification=modification):
                values = copy.deepcopy(self.results)
                if modification == "missing":
                    del values[42]["core_source_provenance"]
                elif modification == "unverified":
                    values[42]["core_source_provenance"]["long"]["snapshot_status"] = "missing_historical_snapshot"
                else:
                    del values[42]["core_source_provenance"]["long"]["core_source_hashes"]["common.py"]
                with self.assertRaisesRegex(ValueError, "verified seven-module source provenance"):
                    validate_cross_seed(values, self.protocol)

    def test_fixed_protocol_and_budgets_cannot_be_changed(self):
        for path, value in (("length_penalty", 0), ("max_generation_tokens", 128), ("beam_size", 8)):
            with self.subTest(field=path):
                values = copy.deepcopy(self.results)
                values[43]["evaluation_metadata"]["long_beam"][path] = value
                with self.assertRaisesRegex(ValueError, "frozen protocol"):
                    validate_cross_seed(values, self.protocol)
        self.results[43]["evaluation_metadata"]["long_beam"]["config"]["max_updates"] = 15000
        with self.assertRaisesRegex(ValueError, "fixed budgets"):
            validate_cross_seed(self.results, self.protocol)

    def test_corrupt_saved_result_propagates_instead_of_becoming_pending(self):
        with patch("mt.quality_multiseed_report.artifact_path", return_value="fake"), \
             patch("mt.quality_multiseed_report.read_object", return_value=self.protocol), \
             patch("mt.quality_multiseed_report.collect_quality", side_effect=ValueError("saved evidence corrupt")):
            with self.assertRaisesRegex(ValueError, "saved evidence corrupt"):
                collect_multiseed(seeds=[43])


if __name__ == "__main__":
    unittest.main()
