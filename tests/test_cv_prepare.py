import json
from pathlib import Path
import tempfile
import unittest

from mt.cv_prepare import (assign_components, choose_grouping, components, fit_scope,
    normalize_source, partition_rows, tagged_rows, verify_raw_alignment, _verify_files)
from mt.common import sha256


class CVPreparationTests(unittest.TestCase):
    def test_group_normalization_keeps_punctuation_and_does_not_mutate(self):
        text = "  Ｈｅｌｌｏ\tSTRASSE!  "
        self.assertEqual(normalize_source(text), "hello strasse!")
        self.assertEqual(normalize_source("Straße"), "strasse")
        self.assertNotEqual(normalize_source("hello!"), normalize_source("hello"))
        self.assertEqual(text, "  Ｈｅｌｌｏ\tSTRASSE!  ")

    def test_explicit_talk_tags_and_pinned_builder_text_parsing(self):
        en = ["<doc>", "<talkid>5</talkid>", "hello &amp; world", "</doc>", "<doc>", "<talkid>9</talkid>", '<seg id="1"> Test </seg>']
        de = ["<doc>", "<talkid>5</talkid>", "hallo &amp; Welt", "</doc>", "<doc>", "<talkid>9</talkid>", '<seg id="1"> Prüfung </seg>']
        rows = list(tagged_rows(en, de))
        self.assertEqual([r["talk_id"] for r in rows], ["5", "9"])
        self.assertEqual([r["en"] for r in rows], ["hello &amp; world", "Test"])
        self.assertEqual(verify_raw_alignment(rows, [{"en": r["en"], "de": r["de"]} for r in rows]), 2)

    def test_talk_mapping_rejects_missing_mismatch_and_unequal_files(self):
        for en, de in [(["hello"], ["hallo"]), (["<talkid>2</talkid>"], ["<talkid>3</talkid>"]),
                       (["<talkid>2</talkid>", "hello"], ["<talkid>2</talkid>"]),
                       (["<talkid>2</talkid>", "<doc>", "hello"], ["<talkid>2</talkid>", "<doc>", "hallo"])]:
            with self.assertRaises(ValueError):
                list(tagged_rows(en, de))

    def test_every_raw_pair_and_order_must_match(self):
        tagged = [{"id": 0, "en": "a", "de": "b"}]
        for raw in [[], [{"en": "a ", "de": "b"}], [{"en": "a", "de": "wrong"}]]:
            with self.assertRaises(ValueError):
                verify_raw_alignment(tagged, raw)

    def test_talk_source_relation_is_transitive(self):
        rows = [{"id": 10, "en": "a", "talk_id": "x"}, {"id": 20, "en": "b", "talk_id": "x"},
                {"id": 30, "en": " B ", "talk_id": "y"}, {"id": 40, "en": "c", "talk_id": "y"},
                {"id": 50, "en": "z", "talk_id": "z"}]
        self.assertEqual(components(rows), [[10, 20, 30, 40], [50]])
        self.assertEqual(components(rows, False), [[10], [20, 30], [40], [50]])

    def test_component_assignments_reproducible_without_splitting(self):
        groups = [[i * 2, i * 2 + 1] for i in range(12)]
        assignment, sizes = assign_components(groups)
        self.assertEqual((assignment, sizes), assign_components(groups))
        self.assertEqual(sizes, [8, 8, 8])
        self.assertTrue(all(len({assignment[i] for i in g}) == 1 for g in groups))

    def test_giant_joint_component_triggers_declared_fallback(self):
        rows = [{"id": i, "en": str(i), "talk_id": "same-talk"} for i in range(30)]
        selected, assignments, audit, joint = choose_grouping(rows)
        self.assertEqual(audit["strategy"], "normalized_source_only")
        self.assertEqual(audit["joint_grouping"]["largest_component_fraction"], 1)
        self.assertEqual(audit["selected_grouping"]["fold_sizes"], [10, 10, 10])
        self.assertEqual(len(joint), 1)
        self.assertEqual(len(selected), 30)

    def test_feasible_joint_components_keep_talks_isolated(self):
        rows = [{"id": i, "en": str(i), "talk_id": str(i // 3)} for i in range(36)]
        _, assignment, audit, _ = choose_grouping(rows)
        self.assertEqual(audit["strategy"], "talk_and_normalized_source")
        self.assertTrue(all(len({assignment[i] for i in range(start, start + 3)}) == 1 for start in range(0, 36, 3)))

    def test_unpartitionable_source_component_is_not_split(self):
        rows = [{"id": i, "en": "same", "talk_id": str(i)} for i in range(30)]
        with self.assertRaises(ValueError):
            choose_grouping(rows)

    def test_partition_preserves_full_holdout_original_ids_and_text(self):
        rows = [{"id": i * 3, "en": str(i) + " " * i, "de": "x" * (i + 1)} for i in range(9)]
        assignment = {r["id"]: i % 3 for i, r in enumerate(rows)}
        train, holdout = partition_rows(rows, assignment, 1)
        self.assertEqual(holdout, [rows[1], rows[4], rows[7]])
        scope = fit_scope(train, holdout)
        self.assertFalse(set(scope["train_ids"]) & set(scope["excluded_holdout_ids"]))
        self.assertEqual(len(train) + len(holdout), len(rows))

    def test_partition_rejects_incomplete_or_leaking_assignment(self):
        rows = [{"id": i, "en": str(i)} for i in range(6)]
        with self.assertRaises(ValueError):
            partition_rows(rows, {0: 0}, 0)
        rows[1]["en"] = " 0 "
        with self.assertRaises(ValueError):
            partition_rows(rows, {i: i % 3 for i in range(6)}, 0)

    def test_tokenizer_fit_rejects_holdout_overlap(self):
        with self.assertRaises(ValueError):
            fit_scope([{"id": 7}], [{"id": 7}])
        with self.assertRaises(ValueError):
            fit_scope([{"id": 7}, {"id": 7}], [])

    def test_completed_preparation_rejects_damaged_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            path.write_text("good", encoding="utf-8")
            evidence = {"artifact_sha256": {path.name: sha256(path)}}
            _verify_files(directory, evidence)
            path.write_text("bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                _verify_files(directory, evidence)


if __name__ == "__main__":
    unittest.main()
