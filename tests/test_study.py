"""CPU-only controller gates: freeze, phase isolation, paths and selection.

No child process, GPU, or formal experiment is started by these tests. Native
checkpoint checks are mocked only where their invocation/order is the contract;
the actual tensor and metric verifiers have dedicated adaptation tests.
"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mt import study
from mt.common import read_json, sha256, write_json


class StudyGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"
        self.root.mkdir()
        self.dependencies = {"fixture": True}
        self.addCleanup(patch.stopall)
        patch("mt.pretrained_mt.dependency_evidence", return_value=self.dependencies).start()

    def plan(self, configs=(), sources=None, inputs=None):
        value = {"root": str(self.root.resolve()), "source_files": sources or {},
                 "inputs": inputs or {}, "initial_configs": study.hashes(configs),
                 "dependencies": self.dependencies}
        write_json(self.root / "plan.json", value)
        return value

    def config(self, name, **values):
        path = self.root / "configs" / (name + ".json")
        write_json(path, {"run_dir": str(self.root / "runs" / name), **values})
        return path

    def stage(self, name, config, decision):
        path = self.root / name / "final_plan.json"
        return study.freeze_stage(self.root, path, [config], decision)

    def test_paths_cannot_escape_or_equal_study_root(self):
        self.assertEqual(study.inside(self.root, self.root / "runs" / "new"),
                         (self.root / "runs" / "new").resolve())
        for path in (self.root, self.root.parent / "historical", self.root / ".." / "outside"):
            with self.assertRaises(ValueError):
                study.inside(self.root, path)

    def test_missing_plan_blocks_child_process_before_execution(self):
        with patch("mt.study.subprocess.run", side_effect=AssertionError("must never launch")) as child:
            with self.assertRaises(FileNotFoundError):
                study.run_command(self.root, "train", "project-python.ps1", ["-m", "mt.cv_train"])
        child.assert_not_called()
        self.assertFalse((self.root / "execution").exists())

    def test_modified_frozen_config_blocks_child_before_execution(self):
        config = self.config("candidate", stage="cv", seed=42, label_smoothing=0)
        self.plan([config])
        write_json(config, dict(read_json(config), label_smoothing=.1))
        with patch("mt.study.subprocess.run", side_effect=AssertionError("must never launch")) as child:
            with self.assertRaisesRegex(ValueError, "Frozen evidence"):
                study.run_command(self.root, "train", "project-python.ps1", [])
        child.assert_not_called()

    def test_modified_input_source_snapshot_and_dependencies_rejected(self):
        project = Path(self.temp.name) / "project"
        project.mkdir()
        code = project / "code.py"; code.write_text("original\n")
        snapshot = self.root / "source_snapshot" / "code.py"
        snapshot.parent.mkdir(); snapshot.write_bytes(code.read_bytes())
        data = self.root / "input.txt"; data.write_text("data\n")
        self.plan(sources={"code.py": sha256(code)}, inputs={str(data): sha256(data)})
        with patch("mt.study.PROJECT", project):
            study.verify_plan(self.root)
            snapshot.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                study.verify_plan(self.root)
            snapshot.write_bytes(code.read_bytes())
            data.write_text("different\n")
            with self.assertRaisesRegex(ValueError, "Frozen evidence"):
                study.verify_plan(self.root)
            data.write_text("data\n")
            with patch("mt.pretrained_mt.dependency_evidence", return_value={"fixture": "changed"}):
                with self.assertRaisesRegex(ValueError, "dependencies"):
                    study.verify_plan(self.root)

    def test_freeze_refuses_started_training_without_last_checkpoint(self):
        started = self.root / "cv/runs/fold0_ls0"
        started.mkdir(parents=True)
        (started / "train.jsonl").write_text('{"update":1}\n')
        with self.assertRaisesRegex(ValueError, "before|Formal|formal|history"):
            study.freeze(self.root)
        self.assertFalse((self.root / "plan.json").exists())

    def test_immutable_json_preserves_complete_evidence(self):
        path = self.root / "evidence.json"
        study.immutable_json(path, {"value": 1})
        before = (sha256(path), path.stat().st_mtime_ns)
        study.immutable_json(path, {"value": 1})
        self.assertEqual(before, (sha256(path), path.stat().st_mtime_ns))
        with self.assertRaises(FileExistsError):
            study.immutable_json(path, {"value": 2})
        self.assertEqual(read_json(path), {"value": 1})

    def test_missing_frozen_input_is_not_silently_omitted(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            study.hashes([self.root / "missing-required-input.json"])

    def test_stage_rejects_changed_input_and_another_root_plan(self):
        cfg = self.config("final", stage="final", seed=42, label_smoothing=.1)
        self.plan()
        self.stage("cv", cfg, {"selected_label_smoothing": .1})
        study.verify_stage(self.root, self.root / "cv/final_plan.json")
        saved = cfg.read_bytes()
        write_json(cfg, dict(read_json(cfg), seed=43))
        with self.assertRaisesRegex(ValueError, "Frozen evidence"):
            study.verify_stage(self.root, self.root / "cv/final_plan.json")
        cfg.write_bytes(saved)
        write_json(self.root / "plan.json", {"replacement": True})
        with self.assertRaisesRegex(ValueError, "another study"):
            study.verify_stage(self.root, self.root / "cv/final_plan.json")

    def test_stage_rechecks_upstream_selection_input_hashes(self):
        self.plan()
        for name in ("cv", "adaptation"):
            config = self.config("final-" + name, seed=42)
            upstream = self.root / name / "runs" / "candidate" / "summary.json"
            write_json(upstream, {"verified_candidate_metric": 20})
            selection = self.root / name / "selection.json"
            decision = {"selected_label_smoothing": .1} if name == "cv" else {
                "selected_lrs": {"lora": 1e-4, "full": 1e-5}}
            write_json(selection, {"complete": True, "input_hashes": {str(upstream): sha256(upstream)}, **decision})
            stage = self.root / name / "final_plan.json"
            study.freeze_stage(self.root, stage, [config, selection], {"selection_sha256": sha256(selection), **decision})
            study.verify_stage(self.root, stage)
            write_json(upstream, {"verified_candidate_metric": 99})
            with self.assertRaisesRegex(ValueError, "Frozen evidence"):
                study.verify_stage(self.root, stage)

    def test_cv_evaluation_gate_rejects_test_beam_unfrozen_or_changed_config(self):
        cfg = self.config("fold0_ls0", stage="cv", seed=42, label_smoothing=0)
        self.plan([cfg])
        self.assertEqual(study.authorize_evaluation(self.root, cfg, "holdout", "greedy")["seed"], 42)
        for split, method in (("test", "greedy"), ("holdout", "beam"), ("validation", "greedy")):
            with self.assertRaises(ValueError):
                study.authorize_evaluation(self.root, cfg, split, method)
        extra = self.config("unfrozen", stage="cv", seed=42)
        with self.assertRaisesRegex(ValueError, "frozen"):
            study.authorize_evaluation(self.root, extra, "holdout", "greedy")
        write_json(cfg, dict(read_json(cfg), label_smoothing=.1))
        with self.assertRaises(ValueError):
            study.authorize_evaluation(self.root, cfg, "holdout", "greedy")

    def test_final_cv_requires_stage_matching_seed_and_selected_smoothing(self):
        cfg = self.config("selected", stage="final", seed=42, label_smoothing=.1)
        self.plan()
        with self.assertRaises(FileNotFoundError):
            study.authorize_evaluation(self.root, cfg, "test", "beam")
        self.stage("cv", cfg, {"selected_label_smoothing": 0})
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            study.authorize_evaluation(self.root, cfg, "test", "beam")

    def test_native_tuning_never_authorized_for_final_test(self):
        cfg = self.config("tune", method="lora", seed=7, lr=1e-4)
        self.plan([cfg])
        with self.assertRaisesRegex(ValueError, "Tuning adaptation"):
            study.authorize_evaluation(self.root, cfg, "test", "beam")

    def test_final_native_lr_must_match_frozen_selection(self):
        cfg = self.config("formal", method="lora", seed=42, lr=3e-4)
        self.plan()
        self.stage("adaptation", cfg, {"selected_lrs": {"lora": 1e-4, "full": 1e-5}})
        with self.assertRaisesRegex(ValueError, "selected learning rate"):
            study.authorize_evaluation(self.root, cfg, "test", "beam")

    def test_cv_controller_only_runs_holdout_and_resumes_own_last(self):
        cfg = self.config("fold0_ls0", stage="cv", seed=42, label_smoothing=0)
        self.plan([cfg])
        run = Path(read_json(cfg)["run_dir"]); run.mkdir(parents=True)
        (run / "last.pt").write_bytes(b"fixture marker")
        with patch("mt.study.run_command") as child:
            study.run_cv(self.root, cfg)
        self.assertEqual(child.call_count, 2)
        training = child.call_args_list[0].args[3]
        self.assertIn("--resume", training)
        self.assertEqual(Path(training[training.index("--resume") + 1]).resolve(),
                         (run / "last.pt").resolve())
        evaluation = child.call_args_list[1].args[3]
        self.assertEqual(evaluation[evaluation.index("--split") + 1], "holdout")
        self.assertEqual(evaluation[evaluation.index("--method") + 1], "greedy")

    def test_tuning_controller_freezes_run_then_trains_without_final_evaluation(self):
        cfg = self.config("tune_lora", method="lora", seed=7, lr=1e-4)
        self.plan([cfg]); calls = []
        with patch("mt.study.ensure_adapt_plan", side_effect=lambda *a: calls.append("freeze_run")), \
             patch("mt.study.run_command", side_effect=lambda *a: calls.append(a[3])):
            study.run_adapt(self.root, cfg, formal=False)
        self.assertEqual(calls[0], "freeze_run")
        self.assertEqual(len(calls), 2)
        self.assertIn("train", calls[1])
        self.assertNotIn("--split", calls[1])

    def test_final_native_controller_runs_four_complete_split_method_pairs(self):
        cfg = self.config("formal", method="full", seed=43, lr=1e-5)
        self.plan()
        self.stage("adaptation", cfg, {"selected_lrs": {"lora": 1e-4, "full": 1e-5}})
        with patch("mt.study.ensure_adapt_plan"), patch("mt.study.run_command") as child:
            study.run_adapt(self.root, cfg, formal=True)
        self.assertEqual(child.call_count, 5)
        pairs = []
        for call in child.call_args_list[1:]:
            arguments = call.args[3]
            pairs.append((arguments[arguments.index("--split") + 1], arguments[arguments.index("--method") + 1]))
        self.assertEqual(pairs, [("validation", "greedy"), ("validation", "beam"), ("test", "greedy"), ("test", "beam")])

    def test_forged_cv_selection_does_not_unlock_final_preparation(self):
        self.plan()
        write_json(self.root / "cv/selection.json", {"selected_label_smoothing": .1, "complete": True})
        def deny_selection(root, name, launcher, arguments):
            self.assertEqual(name, "cv_select")
            raise ValueError("six fold evidence is incomplete")
        with patch("mt.study.run_command", side_effect=deny_selection) as child:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study.prepare_cv_final(self.root)
        self.assertEqual(child.call_count, 1)
        self.assertFalse((self.root / "cv/final_configs").exists())


class AdaptationSelectionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "study"; self.root.mkdir()
        self.configs = {}; self.summaries = {}; self.states = {}
        for method, rates in study.LRS.items():
            for i, lr in enumerate(rates):
                name = study.adapt_name(method, lr)
                cfg_path = self.root / "adaptation/configs" / (name + ".json")
                run = self.root / "adaptation/runs" / name; run.mkdir(parents=True)
                cfg = {"method": method, "seed": 7, "lr": lr, "run_dir": str(run), "data_root": "fixture"}
                write_json(cfg_path, cfg); self.configs[str(cfg_path)] = cfg
                torch.save({"update": 3000}, run / "last.pt")
                # Raw summary deliberately points to the second LR. Only the
                # completed-evidence verifier's result may choose the first.
                raw = {"updates": 3000, "last_sha256": sha256(run / "last.pt"),
                       "best": {"bleu": 999 if i else 1, "chrf2": 20, "update": 1000}}
                write_json(run / "summary.json", raw)
                self.summaries[str(run)] = {"best": {"bleu": 20 if i == 0 else 19, "chrf2": 40, "update": 1500}}
                (run / "best.pt").write_bytes(b"fixture")
                (run / "train.jsonl").write_text("{}\n")
                write_json(run / "plan.json", {"fixture": True})
                write_json(run / "validation/step_01500/metrics.json", {"fixture": True})
                write_json(run / "evaluation/test/greedy/metrics.json", {"bleu": 999})
        write_json(self.root / "plan.json", {"root": str(self.root), "source_files": {},
            "inputs": {}, "initial_configs": study.hashes([Path(path) for path in self.configs]), "dependencies": {}})

    @contextlib.contextmanager
    def selection_mocks(self, completed=None):
        if completed is None:
            completed = lambda cfg, *args: self.summaries[cfg["run_dir"]]
        original_open = Path.open
        def guard(path, *args, **kwargs):
            if "test" in path.parts or path.name == "test.jsonl":
                raise AssertionError("test evidence must not enter LR selection")
            return original_open(path, *args, **kwargs)
        with patch("mt.study.verify_plan", return_value={}), \
             patch("mt.adapt_train.verify_run", side_effect=lambda path: (self.configs[str(path)], {})), \
             patch("mt.adapt_data.load_data", return_value=({}, {}, [])), \
             patch("mt.adapt_train._verify_checkpoint") as checkpoint, \
             patch("mt.adapt_train.verify_completed", side_effect=completed) as final, \
             patch.object(Path, "open", guard):
            yield checkpoint, final

    def test_selection_uses_full_verified_evidence_and_no_test_files(self):
        with self.selection_mocks() as (checkpoint, final):
            result = study.select_adaptation(self.root)
        self.assertEqual(checkpoint.call_count, 4)
        self.assertEqual(final.call_count, 4)
        self.assertEqual(result["selected_lrs"], {"lora": 1e-4, "full": 1e-5})
        self.assertFalse(result["test_used"])
        self.assertFalse(any("test" in Path(p).parts for p in result["input_hashes"]))

    def test_corrupt_complete_evidence_rejected_before_selection_is_written(self):
        def fail(*args):
            raise ValueError("completed validation metric is damaged")
        with self.selection_mocks(fail):
            with self.assertRaisesRegex(ValueError, "damaged"):
                study.select_adaptation(self.root)
        self.assertFalse((self.root / "adaptation/selection.json").exists())

    def test_incomplete_candidate_never_calls_complete_verifier_or_selects(self):
        first = next(iter(self.configs.values()))
        torch.save({"update": 2999}, Path(first["run_dir"]) / "last.pt")
        with self.selection_mocks() as (_, final):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                study.select_adaptation(self.root)
        final.assert_not_called()
        self.assertFalse((self.root / "adaptation/selection.json").exists())

    def test_completed_selection_reuses_readonly_and_changed_inputs_are_rejected(self):
        with self.selection_mocks():
            result = study.select_adaptation(self.root)
        output = self.root / "adaptation/selection.json"
        before = (sha256(output), output.stat().st_mtime_ns)
        with self.selection_mocks():
            self.assertEqual(study.select_adaptation(self.root), result)
        self.assertEqual(before, (sha256(output), output.stat().st_mtime_ns))
        config = Path(next(iter(self.configs)))
        config.write_text(config.read_text() + "\n")
        with self.selection_mocks():
            with self.assertRaises(FileExistsError):
                study.select_adaptation(self.root)
        self.assertEqual(before, (sha256(output), output.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
