"""Selection fixtures use synthetic predictions and real corpus/journal checks.

The completed-training verifier is mocked only in the end-to-end selector
fixture; its actual checkpoint/RNG/optimizer behavior has dedicated tests.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mt.common import read_json, sha256, write_json
from mt.cv_evaluate import (_digest, _derived_metrics, _output_bytes, _progress,
                            INFERENCE_SOURCES, corpus_metrics)
from mt.cv_prepare import fit_scope, source_digest
from mt.cv_select import (candidate_tag, merge_oof, select, summarize_candidates,
                          validate_candidate_config, validate_pair)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fold_scores(bleu, chrf):
    return [{"fold": i, "samples": i + 1, "bleu": bleu[i], "chrf2": chrf[i]} for i in range(3)]


class CVSelectionStatisticTests(unittest.TestCase):
    def test_mean_is_unweighted_fold_mean_and_sd_is_sample_sd(self):
        result = summarize_candidates({"ls0": fold_scores([10, 20, 30], [20, 30, 40]),
                                       "ls01": fold_scores([15, 20, 24], [80, 80, 80])})
        self.assertEqual(result["selected_label_smoothing"], 0)
        self.assertEqual(result["candidates"]["ls0"]["summary"]["bleu"], {"mean": 20, "sample_std": 10.0, "ddof": 1})

    def test_exact_tie_chrf_then_fixed_candidate_order(self):
        scores = {"ls0": fold_scores([10, 20, 30], [30, 30, 30]),
                  "ls01": fold_scores([20, 20, 20], [30, 31, 30])}
        self.assertEqual(summarize_candidates(scores)["selected_label_smoothing"], .1)
        scores["ls01"] = fold_scores([20, 20, 20], [30, 30, 30])
        self.assertEqual(summarize_candidates(scores)["selected_label_smoothing"], 0)
        scores["ls01"][0]["bleu"] += 1e-10
        self.assertEqual(summarize_candidates(scores)["selected_label_smoothing"], .1)

    def test_missing_duplicate_nonfinite_and_wrong_candidates_rejected(self):
        good = {"ls0": fold_scores([10, 20, 30], [20, 30, 40]), "ls01": fold_scores([15, 20, 24], [30, 30, 30])}
        for mutation in (lambda x: x["ls0"].pop(), lambda x: x["ls0"][0].update(fold=1),
                         lambda x: x["ls0"][0].update(bleu=float("nan")), lambda x: x.pop("ls01")):
            values = json.loads(json.dumps(good))
            mutation(values)
            with self.assertRaises(ValueError):
                summarize_candidates(values)
        with self.assertRaises(ValueError):
            summarize_candidates(good, (.1, 0))

    def test_oof_is_corpus_score_not_mean_fold_and_does_not_choose_candidate(self):
        population = [{"id": 7, "en": "first", "de": "a b c d"},
                      {"id": 2, "en": "second", "de": "e f g h i j k l m n"},
                      {"id": 99, "en": "third", "de": "o p q r s t u v w x"}]
        predictions = ["a b c d", "wrong wrong wrong wrong", "wrong wrong wrong wrong"]
        folds = [[{"id": row["id"], "source": row["en"], "reference": row["de"], "prediction": pred}]
                 for row, pred in zip(population, predictions)]
        examples, result = merge_oof(population, folds)
        mean_fold = sum(corpus_metrics([pred], [row["de"]])["bleu"] for row, pred in zip(population, predictions)) / 3
        self.assertNotAlmostEqual(mean_fold, result["bleu"])
        self.assertEqual([e["id"] for e in examples], [7, 2, 99])
        self.assertEqual(result["bleu"], corpus_metrics(predictions, [r["de"] for r in population])["bleu"])
        # Extra OOF scores cannot displace a candidate's primary fold mean.
        scores = {"ls0": fold_scores([15, 15, 15], [20, 20, 20]), "ls01": fold_scores([14, 14, 14], [30, 30, 30])}
        for row in scores["ls0"]:
            row["oof_bleu"] = 1
        for row in scores["ls01"]:
            row["oof_bleu"] = 100
        self.assertEqual(summarize_candidates(scores)["selected_label_smoothing"], 0)

    def test_oof_rejects_duplicate_missing_wrong_source_and_reference(self):
        population = [{"id": i, "en": f"source {i}", "de": f"reference {i}"} for i in range(3)]
        good = [[{"id": row["id"], "source": row["en"], "reference": row["de"], "prediction": "prediction"}] for row in population]
        for mutation in (lambda x: x[0][0].update(id=1), lambda x: x[0].clear(),
                         lambda x: x[0][0].update(source="wrong"), lambda x: x[0][0].update(reference="wrong")):
            folds = json.loads(json.dumps(good))
            mutation(folds)
            with self.assertRaises(ValueError):
                merge_oof(population, folds)


class CVSelectorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"
        self.source = Path(self.temp.name) / "official"
        self.source.mkdir()
        self.population = [{"id": 7 * i + 1, "en": f"unique english sentence number {i}",
                            "de": f"eine deutsche beispiel zeile nummer {i}"} for i in range(12)]
        jsonl(self.source / "train.jsonl", self.population)
        write_json(self.source / "manifest.json", {"files": {"train": {"rows_saved": 12}}})
        # These sentinels must never be opened by selection.
        for name in ("validation.jsonl", "test.jsonl", "test_metrics.json"):
            (self.source / name).write_text("DO NOT READ", encoding="utf-8")
        write_json(self.root / "design.json", {"cv": {"candidate_order": [0, .1], "updates": 3000, "model_seed": 42, "folds": 3}})
        groups_dir = self.root / "data" / "cv"
        self.assignment = {r["id"]: i % 3 for i, r in enumerate(self.population)}
        jsonl(groups_dir / "groups.jsonl", [{"id": r["id"], "fold": self.assignment[r["id"]],
            "normalized_source_sha256": source_digest(r["en"])} for r in self.population])
        write_json(groups_dir / "grouping_complete.json", {"request": {
            "design_sha256": sha256(self.root / "design.json"),
            "source_train_sha256": sha256(self.source / "train.jsonl"), "source_manifest_sha256": sha256(self.source / "manifest.json")},
            "artifact_sha256": {"groups.jsonl": sha256(groups_dir / "groups.jsonl")}})
        self.configs = []
        for fold in range(3):
            data = groups_dir / f"fold{fold}"
            train = [r for r in self.population if self.assignment[r["id"]] != fold]
            holdout = [r for r in self.population if self.assignment[r["id"]] == fold]
            jsonl(data / "train.jsonl", train)
            jsonl(data / "holdout.jsonl", holdout)
            write_json(data / "tokenizer_fit_ids.json", fit_scope(train, holdout))
            write_json(data / "tokenizer.json", {"synthetic_test_fixture": True})
            (data / "spm.model").write_bytes(b"synthetic fixture: optimizer verifier is mocked")
            write_json(data / "common_ids.json", {"ids": [r["id"] for r in train]})
            jsonl(data / "batch_plan_seed42.jsonl", [{"fixture": True}])
            write_json(data / "preparation_complete.json", {"artifact_sha256": {p.name: sha256(p) for p in data.iterdir() if p.is_file()}})
            for smoothing in (0, .1):
                run = self.root / "cv" / "runs" / f"fold{fold}_{candidate_tag(smoothing)}"
                run.mkdir(parents=True)
                cfg = {"stage": "cv", "fold": fold, "candidate_smoothing": smoothing, "label_smoothing": smoothing,
                    "seed": 42, "max_updates": 3000, "checkpoint_selection": "fixed_last", "representation": "bpe",
                    "data_dir": str(data), "run_dir": str(run), "common_ids": str(data / "common_ids.json"),
                    "batch_plan": str(data / "batch_plan_seed42.jsonl"), "decode_batch_size": 16,
                    "beam_size": 4, "max_generation_tokens": 256, "length_penalty": .6}
                self.configs.append(cfg)
                write_json(self.root / "cv" / "configs" / f"fold{fold}_{candidate_tag(smoothing)}.json", cfg)
                records = [{"update": i, "epoch": i // 10, "batch_cursor": i % 10, "lr_used": .0005,
                    "sentences": 8, "source_tokens": 40, "target_tokens": 40, "padded_tokens": 80,
                    "data_order_sha256": "order", "microbatch_sha256": "boundaries"} for i in range(1, 3001)]
                jsonl(run / "train.jsonl", records)
                write_json(run / "data_link.json", {"initial_model_sha256": "initial", "tokenizer_sha256": sha256(data / "spm.model"),
                    "parameter_counts": {"total": 123, "trainable": 123}})
                write_json(run / "source_hashes.json", {"fixture": True})
                write_json(run / "summary.json", {"training_seconds_total": 1, "exposure": {"sentences": 24000}, "complete": True})
                torch.save({"model_sha256": "fixture-model"}, run / "last.pt")
                out = run / "holdout_greedy"
                out.mkdir()
                identity = {"schema_version": 1, "checkpoint": str((run / "last.pt").resolve()),
                    "checkpoint_sha256": sha256(run / "last.pt"), "checkpoint_update": 3000,
                    "model_sha256": "fixture-model", "config_sha256": _digest(cfg), "split": "holdout", "method": "greedy",
                    "samples": len(holdout), "input_path": str((data / "holdout.jsonl").resolve()),
                    "input_sha256": sha256(data / "holdout.jsonl"), "ids_sha256": _digest([r["id"] for r in holdout]),
                    "tokenizer_sha256": sha256(data / "spm.model"), "protocol": {k: cfg[k] for k in ("beam_size", "length_penalty", "max_generation_tokens")},
                    "batch_size": 16, "precision": "float32", "device": "cpu", "environment": {},
                    "source_hashes": {name: sha256(Path(__file__).parents[1] / "mt" / name) for name in INFERENCE_SOURCES}}
                examples = [{"id": r["id"], "source": r["en"], "reference": r["de"],
                    "prediction": r["de"] if smoothing == .1 else "eine andere falsche zeile nummer null",
                    "generation_limit": False, "unfinished": False} for r in holdout]
                payload = {"index": 0, "offset": 0, "previous_sha256": _digest(identity), "examples": examples,
                    "seconds": 1.0, "peak_cuda_bytes": None}
                records = [{**payload, "sha256": _digest(payload)}]
                write_json(out / "identity.json", identity)
                write_json(out / "batches" / "batch_000000.json", records[0])
                _progress(out, identity, records)
                metrics = _derived_metrics(identity, records)
                for name, content in _output_bytes(records).items():
                    (out / name).write_bytes(content)
                    metrics[name.removesuffix(".txt").removesuffix(".jsonl") + "_sha256"] = sha256(out / name)
                write_json(out / "metrics.json", metrics)

    def run_selector(self):
        def verified_fixture(cfg):
            return read_json(Path(cfg["run_dir"]) / "summary.json")
        original_open = Path.open
        def guard(path, *args, **kwargs):
            if path.name in {"validation.jsonl", "test.jsonl", "test_metrics.json"} or any(
                    part in {"test_greedy", "test_beam", "validation_greedy", "validation_beam"} for part in path.parts):
                raise AssertionError("official validation/test evidence was accessed")
            return original_open(path, *args, **kwargs)
        with patch("mt.cv_select.validate_completed_run", side_effect=verified_fixture) as validation, patch.object(Path, "open", guard):
            result = select(self.root, self.source)
            self.assertEqual(validation.call_count, 6)
            return result

    def test_full_selector_real_journal_rescore_no_test_access_and_readonly_reuse(self):
        result = self.run_selector()
        self.assertEqual(result["selected_label_smoothing"], .1)
        self.assertEqual(result["official_training_samples"], 12)
        self.assertFalse(result["official_validation_or_test_used_for_selection"])
        for tag in ("ls0", "ls01"):
            out = self.root / "cv" / "oof" / tag
            examples = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
            self.assertEqual([r["id"] for r in examples], [r["id"] for r in self.population])
            metrics = read_json(out / "metrics.json")
            self.assertEqual(metrics["bleu"], corpus_metrics([r["prediction"] for r in examples], [r["reference"] for r in examples])["bleu"])
        before = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in (self.root / "cv").rglob("*") if p.is_file()}
        self.assertEqual(self.run_selector(), result)
        after = {str(p): (sha256(p), p.stat().st_mtime_ns) for p in (self.root / "cv").rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_wrong_config_candidate_or_test_input_rejected_before_read(self):
        cfg = self.configs[0]
        with self.assertRaises(ValueError):
            validate_candidate_config(dict(cfg, candidate_smoothing=.1), self.root, 0, 0)
        out = Path(cfg["run_dir"]) / "holdout_greedy"
        identity = read_json(out / "identity.json")
        identity["input_path"] = str(self.source / "test.jsonl")
        write_json(out / "identity.json", identity)
        with self.assertRaisesRegex(ValueError, "evaluation identity"):
            self.run_selector()

    def test_corrupt_metric_rejected_by_actual_journal_rescore(self):
        path = Path(self.configs[0]["run_dir"]) / "holdout_greedy" / "metrics.json"
        metrics = read_json(path)
        write_json(path, {**metrics, "bleu": metrics["bleu"] + 1})
        with self.assertRaisesRegex(ValueError, "independent corpus recomputation"):
            self.run_selector()
        self.assertFalse((self.root / "cv" / "selection.json").exists())

    def test_missing_fold_checkpoint_rejects_incomplete_selection(self):
        (Path(self.configs[-1]["run_dir"]) / "last.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_selector()
        self.assertFalse((self.root / "cv" / "selection.json").exists())

    def test_fit_scope_leak_rejected_even_if_preparation_manifest_is_resigned(self):
        data = self.root / "data" / "cv" / "fold0"
        path = data / "tokenizer_fit_ids.json"
        scope = read_json(path)
        scope["train_ids"].append(scope["excluded_holdout_ids"][0])
        write_json(path, scope)
        marker = data / "preparation_complete.json"
        complete = read_json(marker)
        complete["artifact_sha256"][path.name] = sha256(path)
        write_json(marker, complete)
        with self.assertRaisesRegex(ValueError, "fit scope differs"):
            self.run_selector()

    def test_pair_rejects_seed_initialization_exposure_and_holdout_selection(self):
        left, right = self.configs[:2]
        read_records = lambda cfg: [json.loads(x) for x in (Path(cfg["run_dir"]) / "train.jsonl").read_text().splitlines()]
        a, b = read_records(left), read_records(right)
        la, lb = [read_json(Path(c["run_dir"]) / "data_link.json") for c in (left, right)]
        with self.assertRaises(ValueError):
            validate_pair(left, dict(right, seed=43), a, b, la, lb)
        with self.assertRaises(ValueError):
            validate_pair(left, right, a, b, la, dict(lb, initial_model_sha256="wrong"))
        b[1]["microbatch_sha256"] = "different boundaries"
        with self.assertRaises(ValueError):
            validate_pair(left, right, a, b, la, lb)
        b = read_records(right)
        b[1]["validation_bleu"] = 50
        with self.assertRaises(ValueError):
            validate_pair(left, right, a, b, la, lb)

    def test_existing_different_choice_and_changed_oof_not_overwritten(self):
        result = self.run_selector()
        path = self.root / "cv" / "selection.json"
        write_json(path, dict(result, selected_label_smoothing=0))
        before = path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "different CV selection"):
            self.run_selector()
        self.assertEqual(path.read_bytes(), before)
        write_json(path, result)
        predictions = self.root / "cv" / "oof" / "ls0" / "predictions.txt"
        predictions.write_text("damaged\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_selector()
        self.assertEqual(predictions.read_text(), "damaged\n")


if __name__ == "__main__":
    unittest.main()
