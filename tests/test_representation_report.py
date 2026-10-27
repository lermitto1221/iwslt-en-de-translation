"""Independent scoring, same-ID groups, paired statistics and rejection gates."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sacrebleu

from mt.common import sha256, write_json
from mt.representation_report import (FEATURES, METHODS, REPRESENTATIONS, SPLITS,
    _read_run, aggregate_pairs, corpus_scores, fixed_examples, grouped_scores,
    read_evaluation, stats, validate_membership, verify_pair_controls, write_report)


def pair(seed, delta):
    representations = {}
    for rep in REPRESENTATIONS:
        representations[rep] = {"status": "complete", "evaluations": {
            split + "_" + method: {"bleu": seed - 32 + (delta if rep == "unigram" else 0),
                                  "chrf": seed - 12 + (2 * delta if rep == "unigram" else 0)}
            for split in SPLITS for method in METHODS}}
    return {"seed": seed, "status": "complete", "controls": {"status": "verified"}, "representations": representations}


def fixtures():
    definitions = {"source_words_boundaries": [5, 10, 20], "source_rare_fraction_boundaries": [0, .05, .15],
                   "anchor_fragmentation_boundaries": [1.25, 1.5, 2]}
    examples = [{"id": i, "source": source, "reference": reference, "prediction": reference}
                for i, source, reference in (
                    (0, "this is a small cat", "das ist eine kleine katze"),
                    (3, "this is a small dog", "das ist ein kleiner hund"),
                    (400, "the sun shines on the house", "die sonne scheint auf das haus"),
                    (800, "we are going home after school", "wir gehen nach der schule heim"))]
    tables = {}
    for split in SPLITS:
        rows = []
        for index, example in enumerate(examples):
            values = {"source_words": [2, 5, 10, 20][index], "source_rare_fraction": [0, .1, .2, .4][index],
                      "anchor_fragmentation": [1, 1.2, 1.7, 2.3][index]}
            groups = {feature: sum(value > upper for upper in definitions[feature + "_boundaries"]) for feature, value in values.items()}
            if split == "validation":
                groups["cluster"] = index // 2
            rows.append({"id": example["id"], "source_sha256": hashlib.sha256(example["source"].encode()).hexdigest(),
                         "reference_sha256": hashlib.sha256(example["reference"].encode()).hexdigest(),
                         "features": values, "groups": groups})
        tables[split] = {"n": 4, "rows": rows, "definitions": definitions}
    sets = {}
    for seed in (42, 43):
        sets[str(seed)] = {}
        for rep in REPRESENTATIONS:
            sets[str(seed)][rep] = {}
            for split in SPLITS:
                for method in METHODS:
                    values = copy.deepcopy(examples)
                    if rep == "bpe":
                        values[0]["prediction"] = "das ist eine katze"
                        values[2]["prediction"] = "die sonne ist auf dem haus"
                    sets[str(seed)][rep][split + "_" + method] = values
    return tables, sets


class PairedRepresentationStatisticsTests(unittest.TestCase):
    def test_stats_missing_single_and_ddof_one(self):
        self.assertEqual(stats([]), {"n": 0, "mean": None, "sample_std": None})
        self.assertEqual(stats([4]), {"n": 1, "mean": 4, "sample_std": None})
        self.assertEqual(stats([1, 3, 5])["sample_std"], 2)
        with self.assertRaises(ValueError):
            stats([None])
        with self.assertRaises(ValueError):
            stats([float("nan")])

    def test_paired_differences_are_computed_before_standard_deviation(self):
        entries = {str(s): pair(s, d) for s, d in ((42, 1), (43, 3), (44, 5))}
        value = aggregate_pairs(entries)
        delta = value["paired_unigram_minus_bpe"]["test"]["beam"]["bleu"]
        self.assertEqual(delta, {"n": 3, "mean": 3, "sample_std": 2, "per_seed": {"42": 1, "43": 3, "44": 5}})
        self.assertEqual(value["conditions"]["test"]["beam"]["bpe"]["bleu"]["mean"], 11)

    def test_missing_arm_excludes_entire_seed_from_all_aggregates(self):
        entries = {"42": pair(42, 1), "43": {"seed": 43, "status": "pending"}}
        value = aggregate_pairs(entries)
        self.assertEqual(value["included_seeds"], [42])
        self.assertEqual(value["excluded_incomplete_seeds"], [43, 44])
        self.assertEqual(value["conditions"]["test"]["beam"]["bpe"]["bleu"]["sample_std"], None)
        self.assertEqual(aggregate_pairs({})["conditions"]["validation"]["greedy"]["unigram"]["chrf"]["mean"], None)

    def test_completed_bad_label_or_missing_score_rejected(self):
        entry = pair(42, 1)
        entry["seed"] = 43
        with self.assertRaisesRegex(ValueError, "identity"):
            aggregate_pairs({"42": entry})
        entry = pair(42, 1)
        entry["representations"]["unigram"]["evaluations"]["test_beam"]["chrf"] = None
        with self.assertRaisesRegex(ValueError, "invalid"):
            aggregate_pairs({"42": entry})

    def test_pair_requires_same_configuration_initialization_and_microbatch_boundaries(self):
        configs = {r: {"seed": 42, "representation": r, "data_dir": r, "run_dir": r, "max_updates": 2, "lr": .1} for r in REPRESENTATIONS}
        logs = {r: [{"update": i, "epoch": 0, "batch_cursor": i * 4, "lr_used": .01,
                     "data_order_sha256": "ids", "microbatch_sha256": "boundaries", "sentences": 8} for i in (1, 2)] for r in REPRESENTATIONS}
        runs = {r: {"initial_model_sha256": "init", "training": {"parameters": {"total": 10}}} for r in REPRESENTATIONS}
        plan = {"initial_model_sha256": {"42": "init"}}
        self.assertEqual(verify_pair_controls(42, runs, configs, logs, plan)["status"], "verified")
        logs["unigram"][1]["microbatch_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "microbatch"):
            verify_pair_controls(42, runs, configs, logs, plan)
        logs["unigram"][1]["microbatch_sha256"] = "boundaries"
        configs["unigram"]["lr"] = .2
        with self.assertRaisesRegex(ValueError, "configurations"):
            verify_pair_controls(42, runs, configs, logs, plan)
        configs["unigram"]["lr"] = .1
        runs["unigram"]["initial_model_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "initialization"):
            verify_pair_controls(42, runs, configs, logs, plan)


class SharedGroupTests(unittest.TestCase):
    def test_corpus_score_recomputed_from_concatenated_group_text(self):
        tables, sets = fixtures()
        result = grouped_scores(tables, sets, [42, 43])
        first = result["validation"]["cluster"][0]
        self.assertEqual(first["ids"], [0, 3])
        self.assertTrue(first["small"])
        selected = sets["42"]["bpe"]["validation_beam"][:2]
        expected = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False).corpus_score(
            [r["prediction"] for r in selected], [[r["reference"] for r in selected]]).score
        self.assertEqual(first["per_seed"]["42"]["bpe"]["beam"]["bleu"], expected)
        self.assertEqual(first["conditions"]["beam"]["bpe"]["bleu"]["mean"], expected)
        self.assertEqual(first["paired_unigram_minus_bpe"]["beam"]["bleu"]["sample_std"], 0)

    def test_empty_groups_are_null_and_not_zero(self):
        tables, sets = fixtures()
        result = grouped_scores(tables, sets, [42])
        empty = result["validation"]["cluster"][4]
        self.assertEqual(empty["n"], 0)
        self.assertTrue(empty["empty"])
        self.assertIsNone(empty["per_seed"]["42"]["bpe"]["beam"]["bleu"])
        self.assertIsNone(empty["conditions"]["beam"]["bpe"]["bleu"]["mean"])
        self.assertEqual(empty["paired_unigram_minus_bpe"]["beam"]["bleu"]["per_seed"], {})

    def test_group_membership_mismatch_never_rebuckets_or_reorders(self):
        tables, sets = fixtures()
        changed = sets["42"]["unigram"]["test_beam"]
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(ValueError, "ID order"):
            grouped_scores(tables, sets, [42])
        changed[0], changed[1] = changed[1], changed[0]
        changed[0]["reference"] += " changed"
        with self.assertRaisesRegex(ValueError, "reference"):
            grouped_scores(tables, sets, [42])

    def test_fixed_example_ids_are_identical_across_seeds_and_arms(self):
        tables, sets = fixtures()
        selected = fixed_examples(tables, sets, [42, 43])
        for seed in (42, 43):
            test = [r for r in selected if r["split"] == "test" and r["seed"] == seed]
            self.assertEqual([r["id"] for r in test], [0, 400, 800])
            self.assertTrue(all(set(r["translations"]) == {"bpe_greedy", "bpe_beam", "unigram_greedy", "unigram_beam"} for r in test))
        self.assertIn("not independent human annotation", selected[0]["annotation"])

    def test_full_membership_source_hash_must_match(self):
        tables, sets = fixtures()
        tables["test"]["source_file_sha256"] = "a"
        with self.assertRaisesRegex(ValueError, "different source"):
            validate_membership(tables["test"], sets["42"]["bpe"]["test_beam"], "b")

    def test_metric_definition_and_empty_input(self):
        values = corpus_scores(["the small cat is on the mat"], ["the small cat is on the mat"])
        self.assertAlmostEqual(values["bleu"], 100)
        self.assertEqual(values["chrf"], 100)
        self.assertIn("tok:13a", values["signature"])
        self.assertIn("nc:6", values["chrf_signature"])
        self.assertIn("nw:0", values["chrf_signature"])
        self.assertEqual(corpus_scores([], [])["bleu"], None)
        with self.assertRaisesRegex(ValueError, "sample count"):
            corpus_scores(["a"], [])


class SavedEvaluationRejectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.run = self.root / "data", self.root / "run"
        self.data.mkdir(); self.run.mkdir()
        self.folder = self.run / "test_beam"
        self.folder.mkdir()
        tables, sets = fixtures()
        self.table = tables["test"]
        examples = sets["42"]["bpe"]["test_beam"]
        raw = [{"id": r["id"], "en": r["source"], "de": r["reference"]} for r in examples]
        for split in ("train", "validation", "test"):
            (self.data / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in raw))
        (self.data / "manifest.json").write_text("{}")
        (self.data / "spm.model").write_bytes(b"fixture-tokenizer")
        (self.run / "best.pt").write_bytes(b"fixture-checkpoint")
        self.cfg = {"representation": "bpe", "seed": 42, "decode_batch_size": 16,
                    "data_dir": str(self.data), "run_dir": str(self.run)}
        self.protocol = {"beam_size": 4, "length_penalty": .6, "max_generation_tokens": 256}
        self.table["source_file_sha256"] = sha256(self.data / "test.jsonl")
        (self.folder / "predictions.txt").write_text("\n".join(r["prediction"] for r in examples) + "\n")
        (self.folder / "references.txt").write_text("\n".join(r["reference"] for r in examples) + "\n")
        (self.folder / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
        scores = corpus_scores([r["prediction"] for r in examples], [r["reference"] for r in examples])
        source = Path(__file__).resolve().parents[1] / "mt"
        self.meta = {"config": self.cfg, "split": "test", "method": "beam", "smoke": False, "pilot": False,
            "checkpoint_update": 3000, "checkpoint": str(self.run / "best.pt"), "checkpoint_sha256": sha256(self.run / "best.pt"),
            "inference_source_sha256": sha256(source / "decoding.py"), **self.protocol,
            "inference_precision": "float32", "inference_batch_size": 16, "samples": 4, "expected_samples": 4,
            "predictions_sha256": sha256(self.folder / "predictions.txt"), "references_sha256": sha256(self.folder / "references.txt"),
            "reference_source": str(self.data / "test.jsonl"), "reference_sha256": sha256(self.data / "test.jsonl"),
            "data_fingerprints": {"manifest_sha256": sha256(self.data / "manifest.json"), "tokenizer_sha256": sha256(self.data / "spm.model"),
                **{s + "_sha256": sha256(self.data / (s + ".jsonl")) for s in ("train", "validation", "test")}},
            "training_source_hashes": {n: sha256(source / n) for n in ("train.py", "model.py", "batching.py", "tokenizer.py", "decoding.py")},
            "environment": {"device": "cpu-fixture"}, "elapsed_seconds": 1., "sentences_per_second": 4.,
            "mean_latency_ms": 250., "mean_batch_latency_ms": 1000., "bleu": scores["bleu"],
            "signature": scores["signature"], "sacrebleu_version": sacrebleu.__version__}
        self.save()

    def save(self):
        write_json(self.folder / "metrics.json", self.meta)

    def read(self):
        return read_evaluation(self.folder, self.cfg, "test", "beam", self.protocol, 3000, self.table)

    def test_real_saved_metrics_and_text_recompute(self):
        value, examples = self.read()
        self.assertEqual(value["n"], 4)
        self.assertEqual(value["bleu"], self.meta["bleu"])
        self.assertEqual(len(examples), 4)
        self.assertTrue(value["chrf"] > 0)

    def test_wrong_seed_label_checkpoint_or_protocol_rejected(self):
        original = copy.deepcopy(self.meta)
        for key, value in (("config", dict(self.cfg, seed=44)), ("checkpoint_update", 2500),
                           ("length_penalty", 0.0), ("inference_precision", "float16")):
            self.meta = dict(original, **{key: value}); self.save()
            with self.assertRaisesRegex(ValueError, "identity/configuration"):
                self.read()

    def test_corrupt_prediction_and_fabricated_score_rejected(self):
        self.meta["bleu"] += 1; self.save()
        with self.assertRaisesRegex(ValueError, "recomputed BLEU"):
            self.read()
        self.meta["bleu"] -= 1; self.save()
        (self.folder / "predictions.txt").write_text("corrupt\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.read()

    def test_tokenizer_fingerprint_and_reference_membership_rejected(self):
        self.meta["data_fingerprints"]["tokenizer_sha256"] = "0" * 64; self.save()
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            self.read()

    def test_completion_marker_cannot_hide_missing_training(self):
        config_path = self.root / "cfg.json"
        write_json(config_path, self.cfg)
        marker = self.root / "completed" / "bpe_42.json"
        write_json(marker, {"run": "bpe_42"})
        plan = {"configs": {"bpe_42": {"file": str(config_path), "seed": 42, "representation": "bpe"}}}
        with self.assertRaisesRegex(ValueError, "without completed training"):
            _read_run(self.root, plan, "bpe_42", {}, self.protocol)


class ReportArtifactTests(unittest.TestCase):
    def test_pending_report_renders_null_without_fabricated_zero_scores(self):
        tables, sets = fixtures()
        per_seed = {str(seed): {"seed": seed, "status": "pending", "controls": {"status": "pending"},
            "representations": {rep: {"status": "pending", "training": None,
                "evaluations": {split + "_" + method: None for split in SPLITS for method in METHODS}}
                for rep in REPRESENTATIONS}} for seed in (42, 43, 44)}
        representation_stats = {split: {rep: {side: {
            "encoded_length_including_bos_eos": {"mean": 8}, "corpus_pieces_per_whitespace_word": 1.5,
            "unk_tokens": 0} for side in ("src", "tgt")} for rep in REPRESENTATIONS}
            for split in ("train", "validation", "test")}
        representation_stats["common_training_set"] = {"cleaned_pairs": 4, "retained_pairs": 4, "excluded_pairs": 0}
        result = {"metrics": {"status": "pending", "per_seed": per_seed, "representation_statistics": representation_stats,
                              "plan_sha256": "0" * 64}, "aggregate": aggregate_pairs(per_seed),
                  "groups": grouped_scores(tables, sets, []), "examples": []}
        with tempfile.TemporaryDirectory() as folder:
            with patch("mt.representation_report.collect", return_value=result):
                write_report("unused-fixture-root", folder)
            out = Path(folder)
            for stem in ("quality", "representation_cost", "validation_curves", "groups"):
                for extension in ("png", "pdf"):
                    self.assertGreater((out / "figures" / (stem + "." + extension)).stat().st_size, 1000)
            saved = json.loads((out / "aggregate.json").read_text())
            self.assertIsNone(saved["conditions"]["test"]["beam"]["unigram"]["bleu"]["mean"])
            self.assertIn("待完成", (out / "report.md").read_text(encoding="utf-8"))
            self.assertEqual((out / "examples.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
