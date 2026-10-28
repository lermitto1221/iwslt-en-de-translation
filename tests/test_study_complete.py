"""Small CPU fixtures for final delivery gates; no real study scan or models."""
import copy
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from mt import study_complete as complete


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def touch(path, value=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"; self.root.mkdir()

    def final_metadata(self):
        for key in ("plan.json", "cv/completion.json", "cv/final_completion.json", "cv/selection.json",
                    "adaptation/selection.json", "regression_final.json", "historical_manifest.json"):
            write(self.root / key, {})
        write(self.root / "progress.json", {"status": "completed", "formal_runs": 19, "formal_updates": 57000})
        write(self.root / "report/results.json", {"status": "complete", "complete": True})
        for path in ("report/report_manifest.json", "report/sample_review_manifest.json", "presentation/delivery_manifest.json"):
            write(self.root / path, {"status": "complete"})
        for spec in complete.expected_runs(self.root):
            stage = spec["stage"]
            cfg = {"run_dir": str(spec["run"]), "seed": spec["seed"]}
            if stage in ("cv", "cv_final"):
                cfg.update(max_updates=3000, checkpoint_selection="fixed_last", stage="cv" if stage == "cv" else "final")
                if stage == "cv":
                    data = self.root / "data" / f"fold{spec['fold']}"
                    write(data / "manifest.json", {"files": {"holdout": {"rows_saved": 3}}})
                    cfg.update(fold=spec["fold"], label_smoothing=spec["smoothing"], data_dir=str(data))
            else:
                cfg.update(method=spec["method"], updates=3000, lr=spec.get("lr", 1e-4))
            write(spec["config"], cfg)
            write(spec["run"] / "summary.json", {"updates": 3000, "complete": True})
            for name in ("last.pt", "train.jsonl", "best.pt"):
                touch(spec["run"] / name)
            for path, expected in complete.metrics_for(spec):
                write(path, {"samples": expected or 3})

    def test_check_only_pending_never_hashes_or_walks_real_evidence(self):
        with patch.object(complete, "sha", side_effect=AssertionError("no hash during pending check")), \
             patch.object(complete, "safe_walk", side_effect=AssertionError("no inventory during check")):
            result = complete.inspect(self.root)
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["complete"])
        self.assertTrue(result["pending"])

    def test_only_19_unique_3000_update_runs_with_all_groups_can_be_ready(self):
        self.final_metadata()
        result = complete.inspect(self.root)
        self.assertEqual(result["status"], "ready_for_deep_validation")
        self.assertEqual(len(result["runs"]), 19)
        self.assertEqual(sum(r["updates"] for r in result["runs"]), 57000)
        self.assertEqual(len({r["run"] for r in result["runs"]}), 19)
        self.assertEqual(sum(len(r["evaluations"]) for r in result["runs"]), 102)

    def test_missing_actual_presentation_cannot_finalize(self):
        self.final_metadata()
        (self.root / "presentation/delivery_manifest.json").unlink()
        with patch.object(complete, "verify_report", side_effect=AssertionError("must stop before deep scan")):
            with self.assertRaisesRegex(ValueError, "pending"):
                complete.finalize(self.root)
        self.assertFalse((self.root / "completion.json").exists())

    def test_pending_report_cannot_turn_completed_training_into_delivery(self):
        self.final_metadata()
        write(self.root / "report/results.json", {"status": "pending", "complete": False})
        self.assertIn("strict final report is pending", complete.inspect(self.root)["pending"])

    def test_wrong_seed_budget_or_reused_directory_rejected(self):
        self.final_metadata()
        spec = list(complete.expected_runs(self.root))[-1]
        original = complete.read(spec["config"])
        for mutation in ({"seed": 7}, {"updates": 6000}, {"run_dir": str(self.root / "other")}):
            write(spec["config"], {**original, **mutation})
            with self.assertRaises(ValueError):
                complete.inspect(self.root)
        write(spec["config"], original)

    def test_missing_group_and_short_group_rejected(self):
        self.final_metadata()
        spec = list(complete.expected_runs(self.root))[-1]
        path, count = list(complete.metrics_for(spec))[-1]
        path.unlink()
        self.assertTrue(any("evaluation pending" in x for x in complete.inspect(self.root)["pending"]))
        write(path, {"samples": count - 1})
        with self.assertRaisesRegex(ValueError, "count"):
            complete.inspect(self.root)

    def regression(self, count=317):
        path = self.root / "regression_final.json"
        log = path.with_suffix(".log")
        log.write_text(f"fixture test output\nRan {count} tests in 1.234s\n\nOK\n")
        write(path, {"full_suite": True, "tests_run": count, "passed": count, "failures": 0,
                     "errors": 0, "skips": 0, "log_sha256": complete.sha(log)})
        return path

    def test_actual_suite_count_comes_from_one_log_not_historical_or_subsets(self):
        result = complete.verify_regression(self.regression(317))
        self.assertEqual(result["tests_run"], 317)
        self.assertEqual(result["suite_seconds_from_log"], 1.234)

    def test_regression_rejects_mismatched_count_failure_and_targeted_scope(self):
        path = self.regression(); original = complete.read(path)
        for mutation in ({"tests_run": 999, "passed": 999}, {"failures": 1}, {"full_suite": False}):
            write(path, {**original, **mutation})
            with self.assertRaises(ValueError):
                complete.verify_regression(path)
        write(path, original)
        path.with_suffix(".log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "log"):
            complete.verify_regression(path)

    def test_regression_allows_actual_extension_shutdown_warning_after_ok(self):
        path = self.regression()
        log = path.with_suffix(".log")
        log.write_text(log.read_text() + "<sys>:0: DeprecationWarning: fixture extension shutdown\n")
        value = complete.read(path); value["log_sha256"] = complete.sha(log); write(path, value)
        self.assertEqual(complete.verify_regression(path)["tests_run"], 317)

    def test_bom_json_and_idempotent_partial_publication_preserve_created_time(self):
        path = self.root / "exit.json"
        path.write_text('{"exit_code":0}', encoding="utf-8-sig")
        self.assertEqual(complete.read(path)["exit_code"], 0)
        stage = self.root / "history_after_final.json"
        value = {"created_at_utc": "2026-10-28T01:02:03+00:00", "passed": True}
        complete.write_once(stage, value)
        before = (complete.sha(stage), stage.stat().st_mtime_ns)
        complete.write_once(stage, {"created_at_utc": complete.stable_created(stage), "passed": True})
        self.assertEqual(before, (complete.sha(stage), stage.stat().st_mtime_ns))

    def test_inventory_omits_live_control_logs_but_keeps_committed_training_logs(self):
        for path in (self.root / "controller_resume.log", self.root / "progress.json", self.root / "gpu_observations.jsonl"):
            self.assertTrue(complete.volatile_inventory_file(path, self.root))
        self.assertFalse(complete.volatile_inventory_file(self.root / "execution/train_seed42/attempt_001.log", self.root))
        self.assertFalse(complete.volatile_inventory_file(self.root / "cv/final_runs/seed42/train.jsonl", self.root))

    def test_source_snapshot_includes_full_suite_not_only_study_tests(self):
        project = Path(self.temp.name) / "project"
        root = project / "artifacts" / "fixture_study"; root.mkdir(parents=True)
        for name in ("mt/study_complete.py", "mt/study_report.py", "README.md",
                     "tests/test_cv_prepare.py", "tests/test_cv_training.py", "tests/test_cv_select.py",
                     "tests/test_adapt_training.py", "tests/test_old_regression.py", "tests/test_study.py"):
            touch(project / name, b"# synthetic source fixture\n")
        write(root / "plan.json", {"source_files": {}, "dependencies": {}})
        with patch.object(complete, "PROJECT", project), patch.object(
                complete.importlib.metadata, "distribution", side_effect=complete.importlib.metadata.PackageNotFoundError):
            result = complete.source_snapshot(root)
        for path in (project / "tests").glob("test_*.py"):
            self.assertIn(str(path), result["source_files"])
            self.assertEqual(Path(result["source_files"][str(path)]["snapshot"]).read_bytes(), path.read_bytes())

    def test_readme_classified_as_modified_existing_using_preserved_copy(self):
        project = Path(self.temp.name) / "project"; project.mkdir()
        readme = project / "README.md"; readme.write_text("updated readme\n")
        backup = self.root / "README.before_study_update.md"; backup.write_text("original readme\n")
        with patch.object(complete, "PROJECT", project):
            value = complete.external_source_inventory_entry(readme, self.root)
            self.assertEqual(value["classification"], "intentionally_modified_existing")
            self.assertEqual(value["before_sha256"], complete.sha(backup))
            self.assertTrue(value["changed_from_saved_copy"])
            backup.unlink()
            with self.assertRaisesRegex(ValueError, "saved pre-update"):
                complete.external_source_inventory_entry(readme, self.root)

    def presentation(self):
        folder = self.root / "presentation"; folder.mkdir(exist_ok=True)
        deck = folder / "final.pptx"
        with zipfile.ZipFile(deck, "w") as archive:
            archive.writestr("[Content_Types].xml", "fixture")
            archive.writestr("ppt/presentation.xml", "fixture")
        pdf = folder / "final.pdf"; pdf.write_bytes(b"%PDF-1.7\nfixture only\n")
        qa = folder / "qa.json"; write(qa, {"passed": True, "synthetic_test_fixture": True})
        path = folder / "delivery_manifest.json"
        value = {"status": "complete", "draft": False, "source_results_sha256": "results",
                 "outputs": {p.name: complete.sha(p) for p in (deck, pdf)},
                 "qa": {"path": qa.name, "sha256": complete.sha(qa), "passed": True}}
        write(path, value); return path

    def test_presentation_requires_actual_deck_pdf_qa_and_final_results(self):
        path = self.presentation()
        complete.verify_delivery_manifest(path, "results")
        with self.assertRaisesRegex(ValueError, "stale results"):
            complete.verify_delivery_manifest(path, "changed")
        data = complete.read(path); data["draft"] = True; write(path, data)
        with self.assertRaisesRegex(ValueError, "draft"):
            complete.verify_delivery_manifest(path, "results")

    def test_missing_or_corrupt_presentation_and_failed_qa_refused(self):
        path = self.presentation()
        pdf = path.parent / "final.pdf"; pdf.unlink()
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            complete.verify_delivery_manifest(path, "results")
        pdf.write_bytes(b"not pdf")
        data = complete.read(path); data["outputs"][pdf.name] = complete.sha(pdf); write(path, data)
        with self.assertRaisesRegex(ValueError, "not a PDF"):
            complete.verify_delivery_manifest(path, "results")

    def test_sample_review_must_be_agent_review_with_bound_actual_text(self):
        path = self.root / "report/sample_review_manifest.json"
        review = path.parent / "sample_review.md"; touch(review, "代理复核：fixture evidence only, not human annotation".encode())
        value = {"status": "complete", "review_type": "agent_review", "source_results_sha256": "result",
                 "source_samples_sha256": "samples", "outputs": {review.name: complete.sha(review)}}
        write(path, value); complete.verify_delivery_manifest(path, "result", "samples")
        for mutation in ({"review_type": "human_annotation"}, {"source_samples_sha256": "old"}):
            write(path, {**value, **mutation})
            with self.assertRaises(ValueError):
                complete.verify_delivery_manifest(path, "result", "samples")

    def test_history_byte_change_is_reported_not_hidden(self):
        project = Path(self.temp.name) / "project"; project.mkdir()
        file = project / "historical.txt"; file.write_text("history")
        manifest = {"files": {file.name: {"sha256": complete.sha(file), "bytes": file.stat().st_size}}}
        write(self.root / "historical_manifest.json", manifest)
        with patch.object(complete, "PROJECT", project), patch.object(complete, "HISTORICAL_FILES", 1):
            self.assertTrue(complete.historical_after(self.root, manifest)["passed"])
            file.write_text("changed")
            checked = complete.historical_after(self.root, manifest)
        self.assertFalse(checked["passed"])
        self.assertEqual(checked["changed"], [file.name])

    def test_inventory_prunes_node_modules_cache_and_windows_junction(self):
        for folder in ("normal", "node_modules", "__pycache__", ".analysis-deps", "junction_runtime"):
            touch(self.root / folder / "file.txt")
        original = getattr(Path, "is_junction", None)
        with patch.object(Path, "is_junction", lambda path: path.name == "junction_runtime", create=original is None):
            found = [p.relative_to(self.root).as_posix() for p in complete.safe_walk(self.root)]
        self.assertEqual(found, ["normal/file.txt"])

    def test_fingerprint_path_escape_and_damaged_content_rejected(self):
        file = self.root / "a.txt"; file.write_text("a")
        with self.assertRaisesRegex(ValueError, "escapes"):
            complete.verify_files({"../outside.txt": "0" * 64}, self.root)
        with self.assertRaisesRegex(ValueError, "changed"):
            complete.verify_files({file.name: "0" * 64}, self.root)

    def test_resource_and_discarded_compute_are_separate_from_formal_net_budget(self):
        snapshot = self.root / "replay.pt"; snapshot.write_bytes(b"fixture snapshot")
        replay = {"incident_sha256": "incident", "replayed_updates_checked": 235,
                  "same_data_lr_token_exposure": True, "all_logged_numerical_values_exactly_equal": True,
                  "identity_differences": [], "numeric_comparison": {
                      "loss": {"exact_equal": True, "different_updates": 0},
                      "gradient_norm": {"exact_equal": True, "different_updates": 0}},
                  "captured_checkpoint_path": str(snapshot), "captured_checkpoint_sha256": complete.sha(snapshot)}
        recovery = {"recorded_discarded_updates_lower_bound": 235, "incidents": [{"sha256": "incident",
                    "recorded_discarded_updates": 235, "replay_verification": {"evidence": replay}}]}
        resources = {"optimizer_updates": 108}
        result = {"resource_costs": resources, "recovery_incidents": recovery}
        with patch("mt.study_report.resource_costs", return_value=resources), \
             patch("mt.study_report.collect_recovery_incidents", return_value=recovery):
            value = complete.verify_recovery(self.root, result)
            self.assertEqual(value["formal_net_updates"], 57000)
            self.assertEqual(value["recorded_optimizer_compute_updates_lower_bound"], 57343)
            replay["all_logged_numerical_values_exactly_equal"] = False
            with self.assertRaisesRegex(ValueError, "verified exact"):
                complete.verify_recovery(self.root, result)

    def report_fixture(self):
        folder = self.root / "report"; folder.mkdir()
        source = self.root / "report_generator.py"; source.write_text("# fixture\n")
        evidence = self.root / "input.txt"; evidence.write_text("source evidence")
        result = {"complete": True, "status": "complete", "generated_at_utc": "fixture-time", "input_sha256": {},
                  "runs": {"direct": {"complete_for_aggregation": True}}, "aggregate": {},
                  "stage_costs": {name: {"completed_training_runs": n, "completed_run_updates": u,
                      "completed_full_evaluation_passes": p, "evaluated_sentences": s}
                      for name, (n,u,p,s) in complete.EXPECTED_STAGES.items()}}
        for method, offset in (("cv_final",0),("lora",2),("full",1)):
            result["runs"][method] = {str(seed): {"evaluations": {split: {decoder: {
                "bleu": seed - 30 + offset, "chrf2": seed - 10 + offset} for decoder in ("greedy","beam")}
                for split in ("validation","test")}} for seed in complete.SEEDS}
        def stat(values):
            return {"n":3,"ddof":1,"per_seed":values,"mean":statistics.mean(values.values()),
                    "sample_std":statistics.stdev(values.values())}
        for split in ("validation","test"):
            result["aggregate"][split] = {}
            for decoder in ("greedy","beam"):
                result["aggregate"][split][decoder] = {method: {metric: stat({str(seed):
                    result["runs"][method][str(seed)]["evaluations"][split][decoder][metric] for seed in complete.SEEDS})
                    for metric in ("bleu","chrf2")} for method in ("cv_final","lora","full")}
                result["aggregate"][split][decoder]["paired_lora_minus_full"] = {
                    metric:stat({str(seed):1 for seed in complete.SEEDS}) for metric in ("bleu","chrf2")}
        write(folder / "results.json", result)
        (folder / "report.md").write_text("fixture final report\n")
        manifest = {"status":"complete","generated_at_utc":"fixture-time","input_sha256":{},
                    "input_files_complete":True,"input_files":{str(evidence):complete.sha(evidence)},
                    "source_generator":{"path":str(source),"sha256":complete.sha(source)},
                    "outputs":{p.name:complete.sha(p) for p in (folder/'results.json',folder/'report.md')}}
        return result,manifest,evidence

    def test_report_requires_complete_real_input_closure_and_untampered_outputs(self):
        result,manifest,evidence=self.report_fixture()
        complete.verify_report(self.root,result,manifest,[])
        evidence.write_text("changed source")
        with self.assertRaises(ValueError):
            complete.verify_report(self.root,result,manifest,[])
        evidence.write_text("source evidence")
        (self.root/'report/report.md').write_text("changed report")
        with self.assertRaisesRegex(ValueError,"changed evidence"):
            complete.verify_report(self.root,result,manifest,[])

    def test_report_fabricated_n6_or_wrong_mean_is_rejected(self):
        result,manifest,_=self.report_fixture()
        result['aggregate']['test']['beam']['lora']['bleu']['n']=6
        # Update the synthetic serialized report to isolate the mathematical gate.
        write(self.root/'report/results.json',result)
        manifest['outputs']['results.json']=complete.sha(self.root/'report/results.json')
        with self.assertRaisesRegex(ValueError,'aggregate'):
            complete.verify_report(self.root,result,manifest,[])
        result['aggregate']['test']['beam']['lora']['bleu']['n']=3
        result['aggregate']['test']['beam']['lora']['bleu']['mean']+=1
        write(self.root/'report/results.json',result)
        manifest['outputs']['results.json']=complete.sha(self.root/'report/results.json')
        with self.assertRaisesRegex(ValueError,'aggregate'):
            complete.verify_report(self.root,result,manifest,[])

    def test_report_incomplete_input_closure_cannot_pass(self):
        result,manifest,_=self.report_fixture()
        manifest['input_files_complete']=False
        with self.assertRaisesRegex(ValueError,'closure'):
            complete.verify_report(self.root,result,manifest,[])


if __name__ == "__main__":
    unittest.main()
