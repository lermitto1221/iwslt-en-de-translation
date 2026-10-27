"""Small fixtures exercise feature semantics and fail-closed corpus provenance."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mt.common import sha256, write_json
from mt.explore_data import (assert_pair_identity, audit_cleaned_data, describe,
    replay_cleaning, sentence_features, training_word_frequencies, vocabulary_summary,
    word_tokens, _input_evidence, run_data_analysis)


class ToyTokenizer:
    def pad_id(self):
        return 0

    def unk_id(self):
        return 1

    def bos_id(self):
        return 2

    def eos_id(self):
        return 3

    def encode(self, text, out_type=int):
        # A complete word maps to two ordinary pieces; no structural tokens.
        return [4, 5] * len(text.split())


class ExploreDataTests(unittest.TestCase):
    def test_words_casefold_without_punctuation_normalization(self):
        self.assertEqual(word_tokens("  STRASSE Straße, HELLO! -  "),
                         ["strasse", "strasse,", "hello!", "-"])

    def test_frequency_reference_is_training_only(self):
        train = [{"id": 2, "en": "Known known", "de": "Gut gut"}]
        valid = [{"id": 9, "en": "Known unknown unknown", "de": "Gut neu neu"}]
        frequencies = training_word_frequencies(train)
        before = copy.deepcopy(frequencies)
        row = sentence_features(valid, ToyTokenizer(), frequencies, rare_max_count=1)[0]
        self.assertEqual(frequencies, before)
        self.assertNotIn("unknown", frequencies["en"])
        self.assertEqual(row["src_unseen_fraction"], 2 / 3)
        self.assertEqual(row["src_rare_fraction"], 2 / 3)
        self.assertEqual(row["src_bpe"], 6)
        self.assertEqual(row["src_fragmentation"], 2)
        vocab = vocabulary_summary(frequencies, valid, 1)
        self.assertEqual(vocab["en"]["validation_unseen_type_fraction"], .5)
        self.assertEqual(vocab["en"]["validation_unseen_token_fraction"], 2 / 3)

    def test_training_filter_includes_two_special_tokens_and_both_sides(self):
        rows = [{"id": 0, "en": "a b c", "de": "d"},
                {"id": 1, "en": "a", "de": "b c d e"}]
        frequencies = training_word_frequencies(rows)
        result = sentence_features(rows, ToyTokenizer(), frequencies, max_train_tokens=8)
        self.assertTrue(result[0]["train_length_eligible"])
        self.assertFalse(result[1]["train_length_eligible"])
        self.assertEqual(len(result), 2)  # Feature extraction never drops long rows.
        self.assertEqual(result[0]["length_ratio"], 1 / 3)

    def test_empty_special_and_duplicate_id_fail_explicitly(self):
        rows = [{"id": 0, "en": "a", "de": "b"}]
        frequencies = training_word_frequencies(rows)
        with self.assertRaisesRegex(ValueError, "unique integer"):
            sentence_features(rows * 2, ToyTokenizer(), frequencies)
        with self.assertRaisesRegex(ValueError, "empty en"):
            sentence_features([{"id": 0, "en": " ", "de": "b"}], ToyTokenizer(), frequencies)
        class BadTokenizer:
            def encode(self, text, out_type=int):
                return [2, 4, 3]
        with self.assertRaisesRegex(ValueError, "structural token"):
            sentence_features(rows, BadTokenizer(), frequencies)

    def test_cleaning_first_match_order_and_seen_before_length_filter(self):
        raw = [{"en": " ", "de": "b"},
               {"en": "  a b c d ", "de": "x"},
               {"en": "a b c d", "de": "x"},
               {"en": "a b c", "de": "x"},
               {"en": "a b", "de": "x y"}]
        rows, removed, lengths = replay_cleaning(raw, "train", max_words=3, max_ratio=2)
        self.assertEqual(removed, {"empty": 1, "duplicate_pair": 1, "long_words": 1, "length_ratio": 1})
        self.assertEqual(rows, [{"id": 4, "en": "a b", "de": "x y"}])
        self.assertEqual(len(lengths), 5)

    def test_validation_keeps_duplicates_and_long_rows_but_rejects_empty(self):
        raw = [{"en": "a b c d", "de": "x"}] * 2
        rows, removed, _ = replay_cleaning(raw, "validation", max_words=1, max_ratio=1)
        self.assertEqual([row["id"] for row in rows], [0, 1])
        self.assertEqual(sum(removed.values()), 0)
        with self.assertRaisesRegex(ValueError, "empty validation"):
            replay_cleaning([{"en": " ", "de": "x"}], "validation")

    def test_identity_checks_order_text_and_count(self):
        rows = [{"id": 0, "en": "a", "de": "b"}, {"id": 2, "en": "c", "de": "d"}]
        assert_pair_identity(rows, copy.deepcopy(rows), "fixture")
        for damaged in (rows[::-1], rows[:1], [dict(rows[0], en="changed"), rows[1]]):
            with self.assertRaises(ValueError):
                assert_pair_identity(rows, damaged, "fixture")

    def test_statistics_use_full_population_and_save_histogram_counts(self):
        stats = describe([1, 2, 3])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], 2)
        self.assertAlmostEqual(stats["population_std"], (2 / 3) ** .5)
        self.assertEqual(sum(stats["histogram"]["counts"]), 3)
        self.assertIsNone(describe([])["mean"])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            describe([float("nan")])

    def _fixture(self, directory):
        import pyarrow as pa
        import pyarrow.parquet as pq
        directory = Path(directory)
        (directory / "raw").mkdir()
        manifest = {"files": {}, "cleaning": {}, "train_limit": None,
            "cleaning_rules": {"all_splits": {"strip_outer_whitespace": True, "require_aligned_string_pairs": True},
                "train": {"order": ["empty", "duplicate_pair", "long_words", "length_ratio"],
                          "max_words_per_side": 300, "max_length_ratio": 3.0},
                "validation_and_test": {"empty_pair": "error", "filter_length": False, "deduplicate": False, "truncate": False}}}
        for split in ("train", "validation"):
            raw = [{"en": " A b ", "de": " C d "}, {"en": "e", "de": "f"}]
            raw_path = directory / "raw" / f"iwslt2017-{split}.parquet"
            pq.write_table(pa.table({"translation": raw}), raw_path)
            pairs, removed, _ = replay_cleaning(raw, split)
            clean_path = directory / f"{split}.jsonl"
            clean_path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
            manifest["files"][split] = {"rows_raw": 2, "rows_saved": 2,
                "jsonl_sha256": sha256(clean_path), "parquet_sha256": sha256(raw_path)}
            manifest["cleaning"][split] = removed
        for name in ("spm.model", "spm.vocab", "tokenizer.json"):
            (directory / name).write_text("fixture", encoding="utf-8")
        write_json(directory / "manifest.json", manifest)
        return manifest

    def test_full_replay_verifies_inputs_and_rejects_changed_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._fixture(directory)
            rows, audit = audit_cleaned_data(directory, manifest, _input_evidence(directory))
            self.assertEqual(audit["train"]["status"], "verified")
            self.assertEqual(audit["validation"]["rows_saved"], 2)
            changed = directory / "validation.jsonl"
            changed.write_text(changed.read_text().replace('"C d"', '"changed"'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hash"):
                audit_cleaned_data(directory, manifest, _input_evidence(directory))
            manifest["files"]["validation"]["jsonl_sha256"] = sha256(changed)
            with self.assertRaisesRegex(ValueError, "ID/text/order mismatch"):
                audit_cleaned_data(directory, manifest, _input_evidence(directory))

    def test_raw_missing_is_explicit_and_training_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._fixture(directory)
            (directory / "raw" / "iwslt2017-train.parquet").unlink()
            _, audit = audit_cleaned_data(directory, manifest, _input_evidence(directory))
            self.assertEqual(audit["train"]["status"], "raw_unavailable")
            self.assertNotIn("exclusive_removals", audit["train"])
            manifest["train_limit"] = 2
            with self.assertRaisesRegex(ValueError, "full cleaned train"):
                audit_cleaned_data(directory, manifest, _input_evidence(directory))

    def test_saved_features_reusable_only_with_matching_inputs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "corpus"
            directory.mkdir()
            self._fixture(directory)
            write_json(directory / "tokenizer.json", {
                "model_sha256": sha256(directory / "spm.model"),
                "vocab_sha256": sha256(directory / "spm.vocab"),
                "training_file_sha256": sha256(directory / "train.jsonl")})
            out = Path(temporary) / "analysis"
            config = {"data_dir": str(directory), "rare_max_count": 5, "max_train_tokens": 128}
            with patch("mt.explore_data.load", return_value=ToyTokenizer()), patch("mt.explore_data._figures", return_value=[]):
                first = run_data_analysis(config, out)
            summary_path = out / "data" / "summary.json"
            unchanged = summary_path.read_bytes()
            with patch("mt.explore_data.load", side_effect=AssertionError("must reuse validated output")):
                self.assertEqual(run_data_analysis(config, out), first)
            self.assertEqual(summary_path.read_bytes(), unchanged)
            feature_path = out / "data" / "validation_features.jsonl"
            feature_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing/corrupt"):
                run_data_analysis(config, out)

    def test_cannot_write_into_original_data_by_choosing_its_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            with self.assertRaisesRegex(ValueError, "independent"):
                run_data_analysis({"data_dir": str(data)}, temporary)


if __name__ == "__main__":
    unittest.main()
