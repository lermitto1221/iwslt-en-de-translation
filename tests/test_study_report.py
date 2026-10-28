"""Statistical, shared-group, pending-state and paired-exposure report checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mt.common import read_json, sha256, write_json
from mt.study_report import (SEEDS, SPLITS, DECODERS, _pending, aggregate_groups, aggregate_runs,
    collect, default_inputs, fixed_samples, group_scores, paired, stats, verify_adaptation_pair,
    stage_costs, gpu_observations, execution_attempts, resource_costs, cv_partition_sensitivity,
    repeated_trigrams, generation_diagnostic_ids, corpus_repetition_diagnostics,
    summarize_reproduction_audit, collect_reproduction_audits, collect_recovery_incidents,
    ScientificInputClosure, verify_input_files, build_input_files, selected_checkpoint_exposure)
from mt.cv_select import summarize_candidates


def examples():
    return [{"id": i, "source": f"source sentence number {i}", "reference": f"eine deutsche zeile nummer {i}",
             "prediction": f"eine deutsche zeile nummer {i}"} for i in (0, 400, 401, 800)]


def membership(rows, split="test"):
    return {"split": split, "rows": [{"id": r["id"],
        "source_sha256": hashlib.sha256(r["source"].encode()).hexdigest(),
        "reference_sha256": hashlib.sha256(r["reference"].encode()).hexdigest(),
        "groups": {"source_words": i % 2, "source_rare_fraction": 0, "anchor_fragmentation": 1,
                   **({"cluster": i % 2} if split == "validation" else {})}}
        for i, r in enumerate(rows)]}


def runs_fixture():
    runs = {method: {str(seed): _pending(f"{method}/{seed}") for seed in SEEDS} for method in ("cv_final", "lora", "full")}
    runs["direct"] = _pending("direct")
    return runs


def mark_complete(run, bleu, chrf2, rows=None):
    run["complete_for_aggregation"] = True
    for split in SPLITS:
        for decoder in DECODERS:
            run["evaluations"][split][decoder] = {"bleu": bleu, "chrf2": chrf2}
            if rows is not None:
                run["examples"][f"{split}/{decoder}"] = rows


class StudyReportTests(unittest.TestCase):
    def test_selected_best_prefix_exposure_is_not_replaced_by_full_budget(self):
        records = [{"update": i, "sentences": 64, "source_tokens": 10 if i <= 1500 else 30,
                    "target_tokens": 20 if i <= 1500 else 40, "ids_sha256": f"batch-{i}"} for i in range(1, 3001)]
        prefix = selected_checkpoint_exposure(records, 1500)
        whole = selected_checkpoint_exposure(records, 3000)
        self.assertEqual((prefix["sentences"], prefix["source_tokens"], prefix["target_tokens"]), (96000, 15000, 30000))
        self.assertEqual((whole["sentences"], whole["source_tokens"], whole["target_tokens"]), (192000, 60000, 90000))
        self.assertNotEqual(prefix["effective_batch_prefix_sha256"], whole["effective_batch_prefix_sha256"])
        modified = copy.deepcopy(records)
        modified[0]["ids_sha256"] = "different-batch-same-token-count"
        self.assertNotEqual(selected_checkpoint_exposure(modified, 1500)["effective_batch_prefix_sha256"], prefix["effective_batch_prefix_sha256"])

    def test_selected_prefix_rejects_partial_log_noncheckpoint_step_and_corrupt_exposure(self):
        records = [{"update": i, "sentences": 64, "source_tokens": 10, "target_tokens": 20,
                    "ids_sha256": f"batch-{i}"} for i in range(1, 3001)]
        for rows, update in ((records[:-1], 1500), (records, 250), (records, 0), (records, 3500)):
            with self.assertRaises(ValueError):
                selected_checkpoint_exposure(rows, update)
        records[0]["target_tokens"] = None
        with self.assertRaises(ValueError):
            selected_checkpoint_exposure(records, 1500)
        self.assertIsNone(_pending("fixture")["selected_checkpoint_exposure"])

    def test_input_closure_rejects_changed_or_missing_required_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = root / "science.json"
            required.write_text('{"samples": 3}\n', encoding="utf8")
            closure = ScientificInputClosure(root)
            closure.add(required, sha256(required))
            self.assertEqual(verify_input_files(closure.files), {"passed": True, "files_verified": 1})
            required.write_text('{"samples": 4}\n', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "report input changed"):
                verify_input_files(closure.files)
            required.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_input_files(closure.files)
            with self.assertRaises(FileNotFoundError):
                ScientificInputClosure(root).add(required)

    def test_input_closure_expected_fingerprints_are_checked_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = root / "model_fixture.bin"
            required.write_bytes(b"small fixture, not a model")
            closure = ScientificInputClosure(root)
            with self.assertRaisesRegex(ValueError, "scientific report input changed"):
                closure.add(required, "0" * 64)
            closure.add(required, sha256(required))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                closure.add(required, "1" * 64)
            with self.assertRaises(FileNotFoundError):
                closure.referenced({str(root / "absent_required_metrics.json"): "0" * 64})

    def test_input_closure_excludes_output_cycles_but_allows_optional_agent_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            report.mkdir()
            closure = ScientificInputClosure(root)
            for name in ("results.json", "samples.json", "report_manifest.json", "review_manifest.json"):
                path = report / name
                path.write_text("{}", encoding="utf8")
                with self.assertRaisesRegex(ValueError, "excluded report input"):
                    closure.add(path)
            review = report / "sample_review.md"
            review.write_text("Synthetic agent-review fixture", encoding="utf8")
            closure.add(review)
            self.assertEqual(set(closure.files), {str(review.resolve())})
            for path in (root / "progress.json", root / "execution.lock", root / "review_manifest.json", root / "new_files_inventory.json"):
                closure.referenced({str(path): "0" * 64})
                self.assertIn(str(path.resolve()), closure.excluded_metadata)
                self.assertNotIn(str(path.resolve()), closure.files)

    def test_completed_evaluation_progress_is_scientific_input_not_live_training_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "cv" / "runs" / "fold0_ls0" / "holdout_greedy" / "progress.json"
            evaluation.parent.mkdir(parents=True)
            evaluation.write_text('{"complete": true}\n', encoding="utf8")
            closure = ScientificInputClosure(root)
            closure.add(evaluation)
            self.assertIn(str(evaluation.resolve()), closure.files)
            with self.assertRaisesRegex(ValueError, "excluded report input"):
                closure.add(evaluation.parent.parent / "progress.json")

    def test_pending_cannot_claim_complete_input_closure_and_empty_or_relative_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("mt.study_report.sha256", side_effect=AssertionError("pending must not hash any models")):
                with self.assertRaisesRegex(ValueError, "complete experimental"):
                    build_input_files(directory, {}, {"complete": False})
            for mapping in ({}, {"relative.json": "0" * 64}):
                with self.assertRaises(ValueError):
                    verify_input_files(mapping)

    @staticmethod
    def recovery_fixture(root):
        plan = root / "plan.json"
        write_json(plan, {"fixture": "no model data"})
        folder = root / "recovery_incidents" / "fixture_loss"
        folder.mkdir(parents=True)
        old_attempt = root / "execution" / "train_fixture" / "attempt_001.json"
        old_attempt.parent.mkdir(parents=True)
        write_json(old_attempt, {"name": "train_fixture", "attempt": 1, "started_at_utc": "2026-10-28T08:25:00+00:00"})
        (folder / "last_committed_before_resume.pt").write_bytes(b"fixture bytes, never deserialized")
        (folder / "train_before_resume.jsonl").write_text('{"fixture": true}\n', encoding="utf8")
        write_json(folder / "controller_progress_before_resume.json", {"log": str(old_attempt.with_suffix(".log")), "stage": "train_fixture"})
        value = {"observed_at_utc": "2026-10-28T08:47:00+00:00", "cause": "unknown; no captured exit or traceback",
            "frozen_study_plan_sha256": sha256(plan), "native_run": str(root / "adaptation" / "runs" / "fixture"),
            "lost_tool_sessions": [1, 2, 3], "checkpoint_update": 1500, "complete_log_updates": 1735,
            "uncommitted_updates_to_replay": 235, "uncommitted_step_seconds_recorded": 94.3,
            "uncommitted_sentences": 15040, "uncommitted_source_tokens": 353359, "uncommitted_target_tokens": 346550,
            "previous_completed_formal_runs": 9,
            "archived_files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in folder.iterdir() if path.is_file()}}
        write_json(folder / "incident.json", value)
        return folder

    def test_known_interrupted_attempt_is_not_reported_as_running_or_given_fake_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.recovery_fixture(root)
            recovery = collect_recovery_incidents(root)
            attempts = execution_attempts(root, recovery)
            self.assertEqual(attempts["interrupted_attempts_without_exit"], 1)
            self.assertEqual(attempts["unfinished_attempts"], 0)
            self.assertEqual(attempts["unclosed_attempt_records"], 1)
            self.assertEqual(attempts["finished_attempts"], 0)
            self.assertEqual(attempts["failed_attempts"], 0)
            self.assertIsNone(attempts["attempts"][0]["exit_code"])
            self.assertIsNone(attempts["attempts"][0]["wall_seconds"])
            self.assertEqual(attempts["attempts"][0]["observed_activity"], "interrupted_without_captured_exit_or_end_time")

    def test_recovery_records_extra_work_without_expanding_net_budget_or_guessing_unlogged_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.recovery_fixture(root)
            result = collect_recovery_incidents(root)
            self.assertEqual(result["formal_net_update_budget_unchanged"], 57000)
            self.assertEqual(result["recorded_discarded_updates_lower_bound"], 235)
            self.assertEqual(result["recorded_discarded_step_seconds_lower_bound"], 94.3)
            self.assertEqual(result["recorded_discarded_sentences"], 15040)
            self.assertEqual(result["unlogged_additional_work"], "unknown, not estimated")
            self.assertEqual(result["incidents"][0]["replay_verification"]["status"], "pending")
            write_json(folder / "replay_verification.json", {"passed": True, "fixture_replayed_updates": 235})
            available = collect_recovery_incidents(root)["incidents"][0]["replay_verification"]
            self.assertEqual(available["status"], "record_available")
            self.assertEqual(available["sha256"], sha256(folder / "replay_verification.json"))

    def test_recovery_rejects_inconsistent_counts_and_changed_archive_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.recovery_fixture(root)
            incident = folder / "incident.json"
            original = read_json(incident)
            write_json(incident, dict(original, uncommitted_updates_to_replay=236))
            with self.assertRaises(ValueError):
                collect_recovery_incidents(root)
            write_json(incident, original)
            (folder / "last_committed_before_resume.pt").write_bytes(b"changed fixture bytes")
            with patch("torch.load", side_effect=AssertionError("report helper must not deserialize checkpoint")):
                with self.assertRaisesRegex(ValueError, "archived recovery evidence"):
                    collect_recovery_incidents(root)

    @staticmethod
    def reproduction_fixture(root):
        return {"schema_version": 1, "seed": 42, "new_study_root": str(root),
            "new_run": str(root / "cv" / "final_runs" / "seed42"),
            "old_run": str(Path(__file__).resolve().parents[1] / "artifacts" / "runs" / "A42"),
            "model": {"all_tensors_bitwise_equal": True, "parameter_tensors_compared": 94,
                "checkpoint_updates": {"old": 3000, "new": 3000}, "tensor_differences": [],
                "model_tensor_sha256": {"old": "same", "new": "same"}},
            "tokenizer": {"all_ID_piece_score_triples_equal": True, "normalizer": {"effective_equal": True},
                "model_file_bytes_equal": False, "trainer_spec": {"effective_equal": False, "serialized_field_differences": []}},
            "independent_execution": {"started_without_resume_argument": True, "new_continuous_optimizer_log_records": 3000,
                "new_training_command": {"exit_code": 0}},
            "evaluations": [{"split": split, "method": decoder, "old_prediction_available": split == "test",
                **({"raw_ID_source_reference_prediction_alignment_equal": True, "bleu_delta": 0} if split == "test" else {})}
                for split in SPLITS for decoder in DECODERS],
            "training_logs": {"old_validation_checkpoints": [{}] * 6, "new_validation_selected_checkpoints": 0},
            "limitations": ["fixture only"], "completed_at_utc": "2026-10-28T08:00:00+00:00"}

    def test_reproduction_summary_does_not_infer_missing_old_predictions_or_other_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = self.reproduction_fixture(root)
            proof = root / "fixture_evidence.txt"
            proof.write_text("synthetic fixture, no training claims", encoding="utf8")
            audit["input_fingerprints"] = {str(proof): sha256(proof)}
            write_json(root / "cv" / "historical_reproduction_seed42.json", audit)
            result = collect_reproduction_audits(root)
            self.assertEqual(result["audited_seeds"], [42])
            self.assertEqual(result["pending_seeds"], [43, 44])
            item = result["seeds"][0]
            self.assertEqual(item["available_historical_evaluation_groups"], 2)
            self.assertEqual(item["aligned_equal_evaluation_groups"], 2)
            self.assertFalse(item["all_four_predictions_compared_and_equal"])
            self.assertTrue(item["both_test_predictions_compared_and_equal"])
            self.assertIn("not pooled into n=6", result["pooling_policy"])
            proof.write_text("changed", encoding="utf8")
            with self.assertRaises(ValueError):
                collect_reproduction_audits(root)

    def test_reproduction_summary_rejects_wrong_seed_and_inconsistent_tensor_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.reproduction_fixture(root)
            for mutate in (lambda x: x.update(seed=43),
                           lambda x: x["model"].update(all_tensors_bitwise_equal="true"),
                           lambda x: x["model"]["model_tensor_sha256"].update(new="different")):
                audit = copy.deepcopy(original)
                mutate(audit)
                with self.assertRaises(ValueError):
                    summarize_reproduction_audit(audit, root, 42)

    def test_repetition_proxy_is_overlapping_case_sensitive_and_punctuation_preserving(self):
        self.assertTrue(repeated_trigrams("a a a a a")["flag"])
        self.assertEqual(repeated_trigrams("a a a a a")["maximum_trigram_occurrences"], 3)
        self.assertFalse(repeated_trigrams("a a a a")["flag"])
        self.assertFalse(repeated_trigrams("A b c a b c A b c")["flag"])
        self.assertFalse(repeated_trigrams("a b c a b c a b c.")["flag"])
        self.assertTrue(repeated_trigrams("a\tb c\n a b  c a b c")["flag"])

    def test_source_or_reference_repetition_is_not_prediction_only_repetition(self):
        repeat = "a b c a b c a b c"
        rows = [
            {"id": 10, "source": repeat, "reference": "a normal reference", "prediction": repeat},
            {"id": 20, "source": "a normal source", "reference": repeat, "prediction": repeat},
            {"id": 30, "source": "another normal source", "reference": "another normal reference", "prediction": repeat},
            {"id": 40, "source": "a last source", "reference": "a last reference", "prediction": "a last prediction"}]
        for row in rows:
            row.update(generation_limit=False, unfinished=False)
        metrics = {"samples": 4, "generation_limit_hits": 0, "unfinished_generations": 0}
        result = corpus_repetition_diagnostics(rows, membership(rows), metrics, generation_diagnostic_ids(rows, metrics))
        self.assertEqual(result["source"]["ids"], [10])
        self.assertEqual(result["reference"]["ids"], [20])
        self.assertEqual(result["prediction"]["ids"], [10, 20, 30])
        self.assertEqual(result["prediction_only"]["ids"], [30])
        self.assertEqual(result["prediction_only"]["count"], 1)
        self.assertEqual(result["generation_limit"]["ids"], [])

    def test_repetition_missing_result_stays_null_not_zero(self):
        result = corpus_repetition_diagnostics(None, None, None)
        self.assertEqual(result["status"], "pending")
        for field in ("source", "reference", "prediction", "prediction_only", "generation_limit", "unfinished"):
            self.assertIsNone(result[field]["count"])
            self.assertIsNone(result[field]["ids"])

    def test_repetition_rejects_wrong_order_population_and_reference_identity(self):
        rows = examples()
        groups = membership(rows)
        metrics = {"samples": 4}
        generation = {"generation_limit": {"count": 0, "ids": []}, "unfinished": {"count": 0, "ids": []}}
        with self.assertRaises(ValueError):
            corpus_repetition_diagnostics(rows[::-1], groups, metrics, generation)
        with self.assertRaises(ValueError):
            corpus_repetition_diagnostics(rows[:-1], groups, metrics, generation)
        changed = copy.deepcopy(rows)
        changed[0]["reference"] = "changed reference"
        with self.assertRaises(ValueError):
            corpus_repetition_diagnostics(changed, groups, metrics, generation)

    def test_native_limits_recover_ids_but_partial_batch_unfinished_ids_are_unknown(self):
        rows = [{"id": 3}, {"id": 8}]
        metric = {"samples": 2, "generation_limit_hits": 1, "unfinished_generations": 1,
                  "context": {"generation": {"max_length": 512}}}
        batches = [{"ids": [3, 8], "generated_lengths": [511, 20], "limit_hits": 1, "unfinished": 1}]
        result = generation_diagnostic_ids(rows, metric, batches)
        self.assertEqual(result["generation_limit"]["ids"], [3])
        self.assertIsNone(result["unfinished"]["ids"])
        self.assertFalse(result["unfinished"]["exact_ids_available"])
        self.assertEqual(result["unfinished"]["unresolved_candidate_batches"], [{"unfinished_count": 1, "candidate_ids": [3, 8]}])
        metric["unfinished_generations"] = 0
        batches[0]["unfinished"] = 0
        result = generation_diagnostic_ids(rows, metric, batches)
        self.assertEqual(result["unfinished"]["ids"], [])
        self.assertTrue(result["unfinished"]["exact_ids_available"])

    def test_generation_diagnostic_counts_and_id_alignment_are_bound_to_saved_evidence(self):
        rows = [{"id": 3, "generation_limit": True, "unfinished": False},
                {"id": 8, "generation_limit": False, "unfinished": False}]
        metrics = {"samples": 2, "generation_limit_hits": 0, "unfinished_generations": 0}
        with self.assertRaises(ValueError):
            generation_diagnostic_ids(rows, metrics)
        metrics.update(generation_limit_hits=1, context={"generation": {"max_length": 512}})
        with self.assertRaises(ValueError):
            generation_diagnostic_ids(rows, metrics, [{"ids": [8, 3], "generated_lengths": [20, 511], "limit_hits": 1, "unfinished": 0}])
        with self.assertRaises(ValueError):
            generation_diagnostic_ids(rows, metrics, [{"ids": [3, 8], "generated_lengths": [20, 511], "limit_hits": 0, "unfinished": 0}])

    @staticmethod
    def cv_selection(zero_bleu, smooth_bleu, zero_chrf=None, smooth_chrf=None):
        zero_chrf = zero_chrf or [40, 40, 40]
        smooth_chrf = smooth_chrf or [41, 41, 41]
        return summarize_candidates({tag: [{"fold": i, "samples": 100 + i, "bleu": bleu[i], "chrf2": chrf[i]}
            for i in range(3)] for tag, bleu, chrf in (("ls0", zero_bleu, zero_chrf), ("ls01", smooth_bleu, smooth_chrf))})

    def test_cv_fold_sensitivity_consistent_preference_with_opposite_chrf_direction(self):
        selection = self.cv_selection([10, 20, 30], [11, 22, 33], [50, 50, 50], [49, 48, 47])
        before = copy.deepcopy(selection)
        value = cv_partition_sensitivity(selection)
        self.assertTrue(value["all_fold_preferences_identical"])
        self.assertTrue(value["all_fold_preferences_match_global_selection"])
        self.assertEqual(value["global_selected_label_smoothing"], .1)
        self.assertEqual(value["delta_ranges"]["bleu"], {"min": 1, "max": 3})
        self.assertEqual(value["delta_ranges"]["chrf2"], {"min": -3, "max": -1})
        self.assertEqual(selection, before)  # Descriptive result never changes frozen selection.

    def test_cv_fold_preferences_do_not_replace_global_mean_with_majority_vote(self):
        selection = self.cv_selection([10, 20, 30], [11, 21, 25])
        value = cv_partition_sensitivity(selection)
        self.assertEqual([r["preferred_label_smoothing"] for r in value["per_fold"]], [.1, .1, 0])
        self.assertFalse(value["all_fold_preferences_identical"])
        self.assertEqual(value["global_selected_label_smoothing"], 0)
        self.assertEqual(value["global_fold_mean_01_minus_0"]["bleu"], -1)
        self.assertEqual(value["delta_ranges"]["bleu"], {"min": -5, "max": 1})

    def test_cv_local_exact_ties_use_chrf_then_predeclared_candidate_order(self):
        selection = self.cv_selection([10, 10, 10], [10, 10, 9], [40, 40, 40], [41, 40, 50])
        value = cv_partition_sensitivity(selection)
        self.assertEqual([r["preferred_label_smoothing"] for r in value["per_fold"]], [.1, 0, 0])
        self.assertEqual(value["per_fold"][0]["preference_reason"], "higher_chrF2_after_exact_BLEU_tie")
        self.assertEqual(value["per_fold"][1]["preference_reason"], "fixed_candidate_order_after_exact_BLEU_and_chrF2_tie")
        self.assertEqual(value["per_fold"][2]["preference_reason"], "higher_BLEU")

    def test_cv_sensitivity_pending_incomplete_wrong_global_and_wrong_population(self):
        self.assertIsNone(cv_partition_sensitivity(None)["all_fold_preferences_identical"])
        self.assertIsNone(cv_partition_sensitivity(None)["delta_ranges"]["bleu"]["min"])
        selection = self.cv_selection([10, 20, 30], [11, 22, 33])
        for mutate in (lambda x: x.update(selected_label_smoothing=0),
                       lambda x: x["candidates"]["ls01"]["folds"].pop(),
                       lambda x: x["candidates"]["ls01"]["folds"][0].update(samples=999)):
            changed = copy.deepcopy(selection)
            mutate(changed)
            with self.assertRaises(ValueError):
                cv_partition_sensitivity(changed)

    def test_missing_is_null_not_zero_and_single_seed_sd_is_undefined(self):
        self.assertEqual(stats({})["mean"], None)
        value = stats({42: 12, 43: None})
        self.assertEqual(value["n"], 1)
        self.assertEqual(value["mean"], 12)
        self.assertIsNone(value["sample_std"])
        self.assertEqual(stats({42: 10, 43: 20, 44: 30})["sample_std"], 10)
        with self.assertRaises(ValueError):
            stats({42: float("nan")})

    def test_pair_uses_seed_intersection_and_sd_of_actual_differences(self):
        result = paired({"42": 10, "43": 30, "44": 99}, {"42": 8, "43": 20})
        self.assertEqual(result["per_seed"], {"42": 2, "43": 10})
        self.assertEqual(result["mean"], 6)
        self.assertAlmostEqual(result["sample_std"], 32**.5)

    def test_only_fully_verified_seeds_enter_aggregates_and_direct_is_single(self):
        runs = runs_fixture()
        mark_complete(runs["direct"], 30, 50)
        mark_complete(runs["lora"]["42"], 32, 51)
        mark_complete(runs["full"]["42"], 31, 53)
        # This score exists, but its seed has incomplete other evaluations.
        runs["lora"]["43"]["evaluations"]["test"]["beam"] = {"bleu": 99, "chrf2": 99}
        result = aggregate_runs(runs)["test"]["beam"]
        self.assertEqual(result["lora"]["included_seeds"], [42])
        self.assertEqual(result["lora"]["bleu"]["mean"], 32)
        self.assertEqual(result["direct"]["fixed_model_count"], 1)
        self.assertIsNone(result["direct"]["sample_std"])
        self.assertEqual(result["lora_minus_direct"]["bleu"]["mean"], 2)
        self.assertEqual(result["paired_lora_minus_full"]["bleu"]["mean"], 1)

    def test_group_identity_order_hash_and_empty_group_are_checked(self):
        rows = examples()
        groups = membership(rows)
        result = group_scores(groups, rows)
        self.assertEqual(result["source_words"][0]["ids"], [0, 401])
        self.assertTrue(result["source_words"][0]["small"])
        self.assertIsNone(result["source_words"][3]["bleu"])
        self.assertEqual(result["source_words"][3]["n"], 0)
        with self.assertRaises(ValueError):
            group_scores(groups, rows[::-1])
        broken = copy.deepcopy(rows)
        broken[0]["reference"] += " wrong"
        with self.assertRaises(ValueError):
            group_scores(groups, broken)

    def test_group_comparisons_retain_exact_same_ids_and_null_missing_methods(self):
        rows = examples()
        runs = runs_fixture()
        mark_complete(runs["direct"], 20, 40, rows)
        mark_complete(runs["lora"]["42"], 21, 41, rows)
        result = aggregate_groups(runs, {split: membership(rows, split) for split in SPLITS})
        group = result["test"]["beam"]["source_words"][0]
        self.assertEqual(group["ids"], [0, 401])
        self.assertEqual(group["methods"]["lora"]["bleu"]["n"], 1)
        self.assertEqual(group["methods"]["full"]["bleu"]["n"], 0)
        self.assertIsNone(group["paired_lora_minus_full"]["bleu"]["mean"])
        self.assertEqual(group["lora_minus_direct"]["bleu"]["mean"], 0)

    def test_fixed_samples_follow_id_rule_and_do_not_claim_human_annotation(self):
        rows = examples()
        runs = runs_fixture()
        mark_complete(runs["direct"], 20, 40, rows)
        result = fixed_samples(runs, {split: membership(rows, split) for split in SPLITS})
        self.assertEqual(result["test"]["planned_ids"], [0, 400, 800])
        self.assertEqual([e["id"] for e in result["test"]["examples"]], [0, 400, 800])
        self.assertIn("not independent human annotation", result["test"]["review_status"])
        self.assertEqual(result["test"]["examples"][0]["predictions"]["direct/beam"]["text"], rows[0]["prediction"])

    def test_pending_collection_does_not_initialize_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "study"
            source = Path(directory) / "source"
            group_root = Path(directory) / "groups"
            source.mkdir()
            write_json(root / "design.json", {"fixture": True})
            write_json(root / "data" / "cv" / "group_audit.json", {"fixture": True})
            rows = examples()
            for split in SPLITS:
                path = source / f"{split}.jsonl"
                path.write_text("".join(json.dumps({"id": r["id"], "en": r["source"], "de": r["reference"]}) + "\n" for r in rows), encoding="utf8")
                write_json(group_root / f"{split}.json", {**membership(rows, split), "source_file_sha256": sha256(path)})
            inputs = default_inputs(root)
            inputs.update(source_data=str(source), groups_root=str(group_root))
            with patch("mt.study_report.collect_direct", return_value=_pending("direct")), patch("mt.study_report.historical_context", return_value={}), patch(
                    "torch.cuda._lazy_init", side_effect=AssertionError("report must not initialize GPU")):
                result = collect(root, inputs)
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "pending")
            self.assertFalse(result["execution_plan_frozen"])
            self.assertIsNone(result["aggregate"]["test"]["beam"]["lora"]["bleu"]["mean"])

    def test_pair_rejects_changed_effective_batches_even_equal_total_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"seed": 42, "model_dir": "same-base", "data_root": "same-data", "updates": 3000,
                "train_precision": "bfloat16", "micro_batch_size": 8, "effective_batch_size": 64,
                "warmup_updates": 300, "validation_every": 500, "eval_batch_size": 16, "generation": {"fixture": True}}
            left, right = {"config": {**common, "lr": 1e-4, "run_dir": str(root / "lora")}}, {"config": {**common, "lr": 1e-5, "run_dir": str(root / "full")}}
            rows = [{"update": i, "ids_sha256": f"ids{i}", "microbatch_sha256": f"micro{i}", "sentences": 64,
                     "source_tokens": 500, "target_tokens": 500} for i in range(1, 3001)]
            for run in (left, right):
                path = Path(run["config"]["run_dir"]) / "train.jsonl"
                path.parent.mkdir()
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
            left["selected_checkpoint_exposure"] = selected_checkpoint_exposure(rows, 500)
            right["selected_checkpoint_exposure"] = selected_checkpoint_exposure(rows, 1500)
            paired_result = verify_adaptation_pair(left, right)
            self.assertTrue(paired_result["verified"])
            self.assertFalse(paired_result["selected_checkpoint_exposure_equal"])
            self.assertEqual(paired_result["selected_checkpoint_updates"], {"lora": 500, "full": 1500})
            rows[0]["microbatch_sha256"] = "changed"
            path = Path(right["config"]["run_dir"]) / "train.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
            with self.assertRaises(ValueError):
                verify_adaptation_pair(left, right)

    def test_all_four_stages_include_tuning_validation_and_full_oof_cost(self):
        runs = runs_fixture()
        tuning = {m: [_pending(f"tune/{m}/{i}") for i in range(2)] for m in ("lora", "full")}
        for method in ("cv_final", "lora", "full"):
            for run in runs[method].values():
                run["training"] = {"updates": 3000, "training_seconds": 10.0, "validation_seconds": 0.0 if method == "cv_final" else 6.0}
                run["validation_curve"] = [] if method == "cv_final" else [{"update": x} for x in range(500, 3001, 500)]
                for split, count in (("validation", 888), ("test", 8079)):
                    for decoder in DECODERS:
                        run["evaluations"][split][decoder] = {"samples": count, "elapsed_seconds": 2.0}
        for candidates in tuning.values():
            for run in candidates:
                run["training"] = {"updates": 3000, "training_seconds": 10.0, "validation_seconds": 6.0}
                run["validation_curve"] = [{"update": x} for x in range(500, 3001, 500)]
        cv_rows = [{"training_complete": True, "updates": 3000, "training_seconds": 10.0,
            "evaluations": [{"samples": count, "elapsed_seconds": 2.0}]} for _ in range(2) for count in (68095, 68094, 68096)]
        result = stage_costs(cv_rows, runs, tuning)
        self.assertEqual(sum(r["completed_training_runs"] for r in result.values()), 19)
        self.assertEqual(sum(r["completed_run_updates"] for r in result.values()), 57000)
        self.assertEqual(result["cv"]["evaluated_sentences"], 408570)
        self.assertEqual(result["adapt_tune"]["completed_full_evaluation_passes"], 24)
        self.assertEqual(result["adapt_final"]["completed_full_evaluation_passes"], 60)
        self.assertEqual(result["adapt_tune"]["evaluation_seconds"], 24)
        self.assertEqual(result["adapt_final"]["evaluation_seconds"], 84)  # 36 selection + 24 final passes * 2s.
        self.assertEqual(sum(r["evaluated_sentences"] for r in result.values()), 623256)

    def test_stage_costs_do_not_fill_missing_future_time_or_partial_updates(self):
        runs = runs_fixture()
        tuning = {m: [_pending("pending") for _ in range(2)] for m in ("lora", "full")}
        result = stage_costs([], runs, tuning)
        for row in result.values():
            self.assertEqual(row["completed_training_runs"], 0)
            self.assertEqual(row["completed_run_updates"], 0)
            self.assertIsNone(row["training_seconds_completed_runs"])
            self.assertIsNone(row["evaluation_seconds"])

    def test_gpu_ranges_use_real_numeric_observations_and_ignore_unfinished_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.jsonl"
            records = [{"observed_at_utc": f"2026-10-28T00:0{i}:00Z", "stage": "fixture", "nvidia_smi_exit_code": 0,
                "values": f"local time, GPU, 90, 1337, {temperature}, {power}, 1800"}
                for i, (temperature, power) in enumerate(((87, 90.5), (84, 75.25)))]
            path.write_bytes(("".join(json.dumps(r) + "\n" for r in records) + '{"partial":').encode())
            value = gpu_observations(path)
            self.assertEqual(value["observations"], 2)
            self.assertEqual(value["temperature_celsius"], {"min": 84, "max": 87})
            self.assertEqual(value["power_watts"], {"min": 75.25, "max": 90.5})
            self.assertEqual(value["unusable_or_partial"], 1)
            self.assertIn("not controlled", value["limitation"])
            path.write_text("bad committed JSON\n", encoding="utf8")
            with self.assertRaises(ValueError):
                gpu_observations(path)

    def test_execution_attempts_count_wallclock_and_distinguish_repeat_without_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "execution" / "train_fixture"
            folder.mkdir(parents=True)
            for attempt, started, ended, content in ((1, "00:00:00", "00:00:20", '{"update": 1, "loss": 2}\n'),
                (2, "00:01:00", "00:01:05", '{"complete": true}\n'), (3, "00:02:00", None, "")):
                path = folder / f"attempt_{attempt:03d}.json"
                log = path.with_suffix(".log")
                log.write_text(content, encoding="utf8")
                record = {"name": "train_fixture", "attempt": attempt, "started_at_utc": f"2026-10-28T{started}+00:00"}
                if ended:
                    record.update(ended_at_utc=f"2026-10-28T{ended}+00:00", exit_code=0, log_sha256=sha256(log))
                write_json(path, record)
            value = execution_attempts(root)
            self.assertEqual(value["summed_finished_command_wall_seconds"], 25)
            self.assertEqual(value["finished_attempts"], 2)
            self.assertEqual(value["unfinished_attempts"], 1)
            self.assertEqual(value["repeat_attempts"], 2)
            self.assertEqual(value["attempts"][0]["observed_activity"], "optimizer_updates_observed")
            self.assertIn("read-only reuse", value["attempts"][1]["observed_activity"])
            self.assertIsNone(value["attempts"][2]["wall_seconds"])

    def test_resource_updates_are_extra_and_checked_against_real_step_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "adaptation" / "resource_fixture.json"
            record = {"methods": {"lora": {"optimizer_updates": 2, "step_results": [{"update": 1}, {"update": 2}],
                "elapsed_seconds": 1, "status": "passed"}, "full": {"status": "failed", "error": "fixture failure before update evidence"}}}
            write_json(path, record)
            write_json(root / "resources" / "cv_pilot" / "pilot.json", {"formal": False, "extra_optimizer_updates": 20,
                "training": {"updates": 20, "training_seconds_total": 2}})
            value = resource_costs(root)
            self.assertEqual(value["optimizer_updates"], 22)
            self.assertFalse(value["formal"])
            record["methods"]["lora"]["optimizer_updates"] = 3
            write_json(path, record)
            with self.assertRaises(ValueError):
                resource_costs(root)


if __name__ == "__main__":
    unittest.main()
