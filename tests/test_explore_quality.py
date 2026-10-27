import copy
import unittest

from mt.explore import check_hashes
from mt.explore_quality import bin_index, corpus_scores, group_scores, join_validation


class ExplorationQualityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": 7, "en": "One sentence.", "de": "Ein Satz."},
                     {"id": 12, "en": "Another sentence.", "de": "Ein anderer Satz."}]
        self.features = [{"id": r["id"], "src_words": 2, "tgt_words": 2,
                          "src_bpe": 5, "src_rare_fraction": 0, "src_unseen_fraction": 0,
                          "src_fragmentation": 2.5} for r in self.rows]
        self.assignments = [{"split": "validation", "id": r["id"], "cluster": 0} for r in self.rows]
        self.pred = {f"{b}_{m}": [{"id": r["id"], "source": r["en"], "reference": r["de"],
                                   "prediction": r["de"]} for r in self.rows]
                     for b in ("baseline", "long") for m in ("beam", "greedy")}
        self.config = {"vectors": {"kmeans_clusters": 2}, "quality_bins": {"src_bpe": [4, 8]},
                       "methods": ["beam", "greedy"], "small_cluster_size": 30}

    def join(self):
        return join_validation(self.rows, self.features, self.assignments, self.pred, 2)

    def test_complete_join_preserves_original_order(self):
        self.assertEqual([r["id"] for r in self.join()], [7, 12])

    def test_shuffled_predictions_rejected(self):
        self.pred["baseline_beam"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            self.join()

    def test_reference_damage_rejected(self):
        self.pred["long_beam"][0]["reference"] = "wrong"
        with self.assertRaisesRegex(ValueError, "misalignment"):
            self.join()

    def test_missing_and_duplicate_assignments_rejected(self):
        self.assignments.pop()
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.join()
        self.assignments.append(self.assignments[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.join()

    def test_empty_and_small_groups_distinguished(self):
        results = group_scores(self.join(), self.config)
        empty = next(r for r in results if r["category"] == "cluster" and r["group"] == "1")
        self.assertTrue(empty["empty"])
        self.assertIsNone(empty["scores"]["long_beam"]["bleu"])
        self.assertIsNone(empty["long_minus_baseline"]["beam"]["chrf"])
        self.assertTrue(results[0]["small"])

    def test_saved_text_scores_and_paired_differences(self):
        self.pred["baseline_beam"][0]["prediction"] = "Falsch."
        joined = self.join()
        group = group_scores(joined, self.config)[0]
        direct = corpus_scores([r["prediction"] for r in self.pred["baseline_beam"]], [r["de"] for r in self.rows])
        self.assertEqual(group["scores"]["baseline_beam"], direct)
        self.assertAlmostEqual(group["long_minus_baseline"]["beam"]["chrf"],
                               group["scores"]["long_beam"]["chrf"] - direct["chrf"])

    def test_predeclared_bins_are_disjoint_inclusive_right(self):
        self.assertEqual([bin_index(v, [0, .05, .15]) for v in [0, .01, .05, .15, .2]], [0, 1, 1, 2, 3])

    def test_corrupted_empty_manifest_not_reusable(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            check_hashes({})


if __name__ == "__main__":
    unittest.main()
