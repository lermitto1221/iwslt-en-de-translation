"""Predeclaration tests: do not mint plans for started or changed experiments."""
import datetime
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sacrebleu

from mt.common import read_json, sha256, write_json
from mt.plan_quality import create_plan, prepare_quality_config, quality_paths
from mt.summary import EXPECTED, FINGERPRINT_KEYS


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class QualityPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline, self.quality = self.root / "A42", self.root / "quality"
        self.config, self.protocol, self.output = (self.root / name for name in ("quality.json", "protocol.json", "plan.json"))
        base = {"seed": 42, "label_smoothing": .1, "max_updates": 2, "warmup_updates": 1,
                "validate_every": 1, "run_dir": str(self.baseline)}
        self.cfg = dict(base, max_updates=4, run_dir=str(self.quality))
        write_json(self.config, self.cfg)
        write_json(self.baseline / "config.json", base)
        write_json(self.baseline / "summary.json", {"complete": True, "smoke": False, "pilot": False, "updates": 2})
        write_json(self.baseline / "data_link.json", {"initial_model_sha256": "b" * 64})
        (self.baseline / "best.pt").write_bytes(b"a saved baseline checkpoint fixture")
        jsonl(self.baseline / "train.jsonl", [{"update": update, "data_order_sha256": f"{update:064x}",
                    "target_tokens": 20, "sentences": 2} for update in (1, 2)])
        rows = [{"id": 0, "en": "The house is next to the river.", "de": "Das Haus steht neben dem Fluss."},
                {"id": 1, "en": "The birds fly over the small village.", "de": "Die Vögel fliegen über das kleine Dorf."}]
        source = self.root / "test.jsonl"
        jsonl(source, rows)
        score = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
        references = [row["de"] for row in rows]
        bleu = score.corpus_score(references, [references]).score
        fingerprints = {key: "a" * 64 for key in FINGERPRINT_KEYS}
        fingerprints["test_sha256"] = sha256(source)
        self.fingerprints = {"manifest_sha256": fingerprints["manifest_sha256"],
                             "tokenizer_sha256": fingerprints["tokenizer_sha256"],
                             "splits": {split: fingerprints[split + "_sha256"]
                                        for split in ("train", "validation", "test")}}
        write_json(self.baseline / "data_link.json", {**self.fingerprints, "initial_model_sha256": "b" * 64})
        patcher = patch("mt.plan_quality.check_inputs", return_value=self.fingerprints)
        patcher.start()
        self.addCleanup(patcher.stop)
        for method in ("greedy", "beam"):
            out = self.baseline / method
            jsonl(out / "examples.jsonl", [{"id": row["id"], "source": row["en"],
                "reference": row["de"], "prediction": row["de"]} for row in rows])
            for name in ("predictions.txt", "references.txt"):
                (out / name).write_text("\n".join(references) + "\n", encoding="utf-8")
            write_json(out / "metrics.json", {"split": "test", "smoke": False, "pilot": False,
                "samples": 2, "expected_samples": 2, "method": method, "beam_size": 1 if method == "greedy" else 4,
                "length_penalty": .6, "max_generation_tokens": 256, "config": base,
                "bleu": bleu, "signature": str(score.get_signature()), "sacrebleu_version": sacrebleu.__version__,
                "checkpoint": str(self.baseline / "best.pt"), "checkpoint_sha256": sha256(self.baseline / "best.pt"),
                "initial_model_sha256": "b" * 64, "inference_source_sha256": "c" * 64,
                "data_fingerprints": fingerprints, "reference_source": str(source), "reference_sha256": sha256(source),
                "predictions_sha256": sha256(out / "predictions.txt"), "inference_precision": "float32",
                "inference_batch_size": 16, "environment": {"gpu": "fixture only"}})
        write_json(self.protocol, {"selection_split": "validation", "beam_size": 4,
                                  "length_penalty": .6, "max_generation_tokens": 256})
        patcher = patch.dict(EXPECTED, test=2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def create(self):
        return create_plan(self.config, self.protocol, self.output, self.baseline)

    def test_plan_uses_real_creation_time_and_binds_complete_baseline(self):
        before = datetime.datetime.now(datetime.timezone.utc)
        plan = self.create()
        after = datetime.datetime.now(datetime.timezone.utc)
        created = datetime.datetime.fromisoformat(plan["created_at_utc"])
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, after)
        self.assertEqual(plan["created_date"], created.date().isoformat())
        self.assertEqual(plan["quality_extension"]["config_sha256"], sha256(self.config))
        self.assertEqual(plan["protocol_sha256"], sha256(self.protocol))
        self.assertNotIn("core_additional_seeds", plan)
        self.assertTrue(plan["baseline"]["test_results_already_available"])
        self.assertEqual(len(plan["baseline"]["files"]), 13)
        self.assertEqual(len(plan["core_source_hashes"]), 7)
        self.assertEqual(plan["data_fingerprints"], self.fingerprints)
        for path, fingerprint in plan["baseline"]["files"].items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(sha256(path), fingerprint)
        self.assertEqual(read_json(self.output), plan)

    def test_existing_plan_is_preserved_without_force_override(self):
        self.output.write_text("original declaration", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
            self.create()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "original declaration")

    def test_started_runs_cannot_receive_retroactive_plans(self):
        for filename in ("last.pt", "train.jsonl", "greedy/metrics.json", "summary.json", "source_snapshot/manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                marker = run / filename
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("", encoding="utf-8")
                write_json(self.config, dict(self.cfg, run_dir=str(run)))
                with self.assertRaisesRegex(ValueError, "post-hoc predeclared plan"):
                    self.create()
                self.assertFalse(self.output.exists())

    def test_prepared_config_only_is_allowed_but_a_different_config_is_not(self):
        write_json(self.quality / "config.json", dict(self.cfg, max_updates=9))
        with self.assertRaisesRegex(ValueError, "config.json differs"):
            self.create()
        write_json(self.quality / "config.json", self.cfg)
        self.assertEqual(self.create()["quality_extension"]["updates"], 4)

    def test_budget_and_other_training_controls_cannot_change(self):
        for change, message in (({"max_updates": 2}, "budget"), ({"max_updates": 4.0}, "budget"),
                                ({"warmup_updates": 2}, "only in"), ({"seed": 43}, "seed=42"),
                                ({"run_dir": str(self.baseline)}, "differ from baseline")):
            with self.subTest(change=change):
                write_json(self.config, dict(self.cfg, **change))
                with self.assertRaisesRegex(ValueError, message):
                    self.create()

    def test_baseline_must_be_complete_and_both_decoders_must_be_consistent(self):
        path = self.baseline / "summary.json"
        summary = read_json(path)
        write_json(path, dict(summary, complete=False))
        with self.assertRaisesRegex(ValueError, "completed formal training"):
            self.create()
        write_json(path, summary)
        path = self.baseline / "beam" / "metrics.json"
        metric = read_json(path)
        write_json(path, dict(metric, initial_model_sha256="d" * 64))
        with self.assertRaisesRegex(ValueError, "initialization fingerprints differ"):
            self.create()

    def test_test_selected_or_mismatched_decoding_protocol_is_rejected(self):
        frozen = read_json(self.protocol)
        for changes, message in (({"selection_split": "test"}, "validation-selected"),
                                 ({"length_penalty": 0.0}, "differs from frozen protocol")):
            with self.subTest(changes=changes):
                write_json(self.protocol, dict(frozen, **changes))
                with self.assertRaisesRegex(ValueError, message):
                    self.create()

    def test_baseline_output_corruption_prevents_a_new_declaration(self):
        (self.baseline / "greedy" / "predictions.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_execution_rejects_baseline_changed_after_plan_creation(self):
        from mt.run_quality import verify_baseline_evidence

        plan = self.create()
        self.assertEqual(verify_baseline_evidence(plan), self.baseline.resolve())
        (self.baseline / "best.pt").write_bytes(b"a replacement checkpoint after the plan was frozen")
        with self.assertRaisesRegex(ValueError, "baseline evidence changed.*best.pt"):
            verify_baseline_evidence(plan)

    def test_execution_keeps_original_plan_without_baseline_evidence_compatible(self):
        from mt.run_quality import verify_baseline_evidence

        historical = {"created_date": "2026-10-10", "status": "predeclared_before_new_training",
                      "quality_extension": {"updates": 12000, "seed": 42}}
        self.assertEqual(verify_baseline_evidence(historical), Path("artifacts/runs/A42"))
        self.assertNotIn("baseline", historical)

    def retag(self, seed):
        self.cfg["seed"] = seed
        write_json(self.config, self.cfg)
        cfg = read_json(self.baseline / "config.json")
        cfg["seed"] = seed
        write_json(self.baseline / "config.json", cfg)
        for method in ("greedy", "beam"):
            path = self.baseline / method / "metrics.json"
            metrics = read_json(path)
            metrics["config"] = cfg
            write_json(path, metrics)

    def test_additional_seeds_match_their_own_baseline_and_acknowledge_prior_result(self):
        for seed in (43, 44):
            with self.subTest(seed=seed):
                self.retag(seed)
                output = self.root / f"plan{seed}.json"
                plan = create_plan(self.config, self.protocol, output, self.baseline, seed=seed)
                self.assertEqual(plan["quality_extension"]["seed"], seed)
                self.assertEqual(plan["baseline"]["seed"], seed)
                self.assertTrue(plan["prior_knowledge"]["seed42_long_result_already_known"])
                self.assertTrue(plan["prior_knowledge"]["additional_replication_after_seed42"])

    def test_seed_label_and_baseline_mismatch_are_rejected(self):
        self.retag(43)
        with self.assertRaisesRegex(ValueError, "seed=44"):
            create_plan(self.config, self.protocol, self.output, self.baseline, seed=44)
        base = read_json(self.baseline / "config.json")
        write_json(self.baseline / "config.json", dict(base, seed=42))
        with self.assertRaisesRegex(ValueError, "seed=43"):
            create_plan(self.config, self.protocol, self.output, self.baseline, seed=43)

    def test_data_mismatch_prevents_plan_creation(self):
        write_json(self.baseline / "data_link.json", {"initial_model_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "data/tokenizer fingerprints differ"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_default_outputs_are_disjoint_and_seed42_paths_are_preserved(self):
        values = [quality_paths(seed) for seed in (42, 43, 44)]
        for key in ("config", "baseline", "run", "plan", "progress", "summary"):
            self.assertEqual(len({value[key] for value in values}), 3)
        self.assertEqual(values[0]["progress"], Path("artifacts/quality_progress.json"))
        self.assertEqual(values[1]["plan"], Path("artifacts/quality_seed43/plan.json"))

    def test_prepared_config_changes_only_seed_run_and_never_overwrites(self):
        template = self.root / "template.json"
        out = self.root / "newconfig.json"
        cfg = dict(self.cfg, seed=42, max_updates=12000, validate_every=500)
        write_json(template, cfg)
        actual = prepare_quality_config(43, out, template, self.quality)
        self.assertEqual(actual, dict(cfg, seed=43, run_dir=self.quality.as_posix()))
        before = out.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
            prepare_quality_config(44, out, template, self.root / "other")
        self.assertEqual(out.read_bytes(), before)

    def test_new_seed_cannot_use_legacy_plan_without_source_freeze(self):
        from mt.run_quality import verify_baseline_evidence, verify_frozen_inputs

        with self.assertRaisesRegex(ValueError, "matching baseline evidence"):
            verify_baseline_evidence({"quality_extension": {"seed": 43}})
        with patch("mt.run_quality.check_inputs", return_value=self.fingerprints):
            with self.assertRaisesRegex(ValueError, "frozen sources and data"):
                verify_frozen_inputs({"schema_version": 1}, self.cfg, 43)

    def test_execution_rejects_source_and_data_changes_before_training(self):
        from mt.run_quality import verify_frozen_inputs

        plan = self.create()
        with patch("mt.run_quality.check_inputs", return_value=self.fingerprints):
            self.assertEqual(verify_frozen_inputs(plan, self.cfg, 42), self.fingerprints)
            plan["core_source_text"]["model.py"] += "\n# changed after declaration\n"
            with self.assertRaisesRegex(ValueError, "source changed.*model.py"):
                verify_frozen_inputs(plan, self.cfg, 42)
        plan = read_json(self.output)
        with patch("mt.run_quality.check_inputs", return_value={**self.fingerprints, "tokenizer_sha256": "e" * 64}):
            with self.assertRaisesRegex(ValueError, "changed after plan freeze"):
                verify_frozen_inputs(plan, self.cfg, 42)

    def completed_quality_fixture(self):
        """Small real serialized checkpoints/logs; no model or CUDA execution."""
        import torch
        from mt.plan_quality import freeze_core_sources

        sources, content = freeze_core_sources()
        write_json(self.quality / "config.json", self.cfg)
        write_json(self.quality / "summary.json", {"updates": 4, "complete": True, "smoke": False,
                                                   "pilot": False, "best_validation_bleu": 12.0})
        write_json(self.quality / "data_link.json", read_json(self.baseline / "data_link.json"))
        write_json(self.quality / "source_hashes.json", sources)
        write_json(self.baseline / "source_hashes.json", sources)
        write_json(self.quality / "source_snapshot" / "manifest.json", {"core_source_hashes": sources})
        for name, source in content.items():
            (self.quality / "source_snapshot" / name).write_bytes(source.encode("utf-8"))
        rows = [{"update": update, "data_order_sha256": f"{update:064x}", "target_tokens": 20,
                 "sentences": 2, "lr_used": .001, "loss": 1 / update,
                 "validation_bleu_greedy": score, "validation_samples": 2}
                for update, score in enumerate((11.0, 12.0, 12.0, 11.5), 1)]
        jsonl(self.baseline / "train.jsonl", rows[:2])
        jsonl(self.quality / "train.jsonl", rows)
        lines = (self.quality / "train.jsonl").read_bytes().splitlines(keepends=True)
        for name, update in (("last.pt", 4), ("best.pt", 2)):
            torch.save({"config": self.cfg, "fingerprints": self.fingerprints, "smoke": False,
                        "pilot": False, "update": update, "initial_model_sha256": "b" * 64,
                        "source_hashes": sources, "best_bleu": 12.0,
                        "log_sha256": hashlib.sha256(b"".join(lines[:update])).hexdigest()}, self.quality / name)
        patcher = patch.dict(EXPECTED, validation=2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_completed_reuse_checks_checkpoint_selection_and_entire_paired_prefix(self):
        from mt.run_quality import verify_completed_training

        self.completed_quality_fixture()
        result = verify_completed_training(self.quality, self.cfg, self.baseline, self.fingerprints, {})
        self.assertEqual(result["best_validation"]["update"], 2)
        self.assertEqual(result["prefix_updates"], 2)
        self.assertEqual(result["numeric_difference_count"], 0)

    def test_completed_reuse_rejects_later_tied_best_checkpoint(self):
        import torch
        from mt.run_quality import verify_completed_training

        self.completed_quality_fixture()
        state = torch.load(self.quality / "best.pt", weights_only=False)
        state["update"] = 3
        torch.save(state, self.quality / "best.pt")
        with self.assertRaisesRegex(ValueError, "validation-best selection"):
            verify_completed_training(self.quality, self.cfg, self.baseline, self.fingerprints, {})

    def test_completed_reuse_rejects_checkpoint_log_damage(self):
        import torch
        from mt.run_quality import verify_completed_training

        self.completed_quality_fixture()
        state = torch.load(self.quality / "last.pt", weights_only=False)
        state["log_sha256"] = "0" * 64
        torch.save(state, self.quality / "last.pt")
        with self.assertRaisesRegex(ValueError, "log SHA-256"):
            verify_completed_training(self.quality, self.cfg, self.baseline, self.fingerprints, {})

    def test_completed_reuse_rejects_prefix_control_changes_and_records_numeric_changes(self):
        from mt.run_quality import verify_completed_training

        self.completed_quality_fixture()
        rows = [json.loads(line) for line in (self.baseline / "train.jsonl").read_text().splitlines()]
        rows[0]["loss"] = 9.0
        jsonl(self.baseline / "train.jsonl", rows)
        result = verify_completed_training(self.quality, self.cfg, self.baseline, self.fingerprints, {})
        self.assertEqual(result["numeric_difference_count"], 1)
        self.assertEqual(result["numeric_differences"][0]["differences"]["loss"]["baseline"], 9.0)
        rows[0]["lr_used"] = .5
        jsonl(self.baseline / "train.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "prefix differs.*lr_used"):
            verify_completed_training(self.quality, self.cfg, self.baseline, self.fingerprints, {})

    def test_runner_refuses_wrong_seed_and_damaged_plan_before_training_or_state_writes(self):
        from mt.run_quality import run_quality

        self.retag(43)
        plan = create_plan(self.config, self.protocol, self.output, self.baseline, seed=43)
        paths = {**quality_paths(43), "progress": self.root / "progress.json", "summary": self.root / "result.json"}
        with patch("mt.run_quality.train") as trainer, patch("mt.run_quality.quality_paths", return_value=paths):
            with self.assertRaisesRegex(ValueError, "plan seed differs"):
                run_quality(44, self.config, self.protocol, self.output, self.baseline)
            plan["core_source_hashes"]["train.py"] = "0" * 64
            write_json(self.output, plan)
            with patch("mt.run_quality.check_inputs", return_value=self.fingerprints):
                with self.assertRaisesRegex(ValueError, "source changed.*train.py"):
                    run_quality(43, self.config, self.protocol, self.output, self.baseline)
            trainer.assert_not_called()
        self.assertFalse(paths["progress"].exists())
        self.assertFalse(paths["summary"].exists())

    def test_runner_refuses_other_seed_state_output_and_never_starts_training(self):
        from mt.run_quality import run_quality

        self.retag(43)
        create_plan(self.config, self.protocol, self.output, self.baseline, seed=43)
        progress = self.root / "progress.json"
        write_json(progress, {"seed": 44, "stage": "complete"})
        original = progress.read_bytes()
        paths = {**quality_paths(43), "progress": progress, "summary": self.root / "result.json"}
        with patch("mt.run_quality.train") as trainer, patch("mt.run_quality.quality_paths", return_value=paths), \
                patch("mt.run_quality.check_inputs", return_value=self.fingerprints), \
                patch("mt.run_quality.verify_initialization"):
            with self.assertRaisesRegex(ValueError, "belongs to a different frozen experiment"):
                run_quality(43, self.config, self.protocol, self.output, self.baseline)
            trainer.assert_not_called()
        self.assertEqual(progress.read_bytes(), original)

    def test_complete_legacy_seed42_reuse_preserves_global_state_and_artifacts(self):
        import shutil
        from mt.run_quality import run_quality

        self.completed_quality_fixture()
        results = {}
        for method in ("greedy", "beam"):
            directory = self.quality / method
            directory.mkdir()
            for name in ("predictions.txt", "references.txt", "examples.jsonl"):
                shutil.copyfile(self.baseline / method / name, directory / name)
            metrics = read_json(self.baseline / method / "metrics.json")
            metrics.update(config=self.cfg, checkpoint=str(self.quality / "best.pt"),
                           checkpoint_update=2, checkpoint_sha256=sha256(self.quality / "best.pt"),
                           inference_source_sha256=sha256(Path(__file__).resolve().parents[1] / "mt" / "decoding.py"))
            write_json(directory / "metrics.json", metrics)
            results[method] = dict(metrics, bleu_difference_from_A42=0.0)
        write_json(self.output, {"protocol_sha256": sha256(self.protocol),
                   "quality_extension": {"seed": 42, "config_sha256": sha256(self.config),
                                         "updates": 4, "smoothing": .1, "fresh_initialization": True}})
        progress, summary = self.root / "legacy_progress.json", self.root / "legacy_summary.json"
        write_json(progress, {"stage": "complete", "updated_at_utc": "historical timestamp"})
        write_json(summary, {"config": self.cfg, "training": read_json(self.quality / "summary.json"),
                             "results": results, "plan": "original relative plan path"})
        protected = {path: path.read_bytes() for path in (progress, summary, self.quality / "best.pt",
                                                        self.quality / "last.pt", self.quality / "train.jsonl")}
        paths = {**quality_paths(42), "progress": progress, "summary": summary}
        with patch("mt.run_quality.train") as trainer, patch("mt.run_quality.quality_paths", return_value=paths), \
                patch("mt.run_quality.check_inputs", return_value=self.fingerprints), \
                patch("mt.run_quality.verify_baseline_evidence", return_value=self.baseline), \
                patch("mt.run_quality.verify_initialization"), patch("builtins.print"):
            result = run_quality(42, self.config, self.protocol, self.output, self.baseline)
            trainer.assert_not_called()
        self.assertEqual(result["controls"]["status"], "verified")
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        corrupted = read_json(summary)
        corrupted["results"]["greedy"]["bleu_difference_from_A42"] = 99.0
        write_json(summary, corrupted)
        before = summary.read_bytes()
        with patch("mt.run_quality.train") as trainer, patch("mt.run_quality.quality_paths", return_value=paths), \
                patch("mt.run_quality.check_inputs", return_value=self.fingerprints), \
                patch("mt.run_quality.verify_baseline_evidence", return_value=self.baseline), \
                patch("mt.run_quality.verify_initialization"), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "inconsistent derived greedy difference"):
                run_quality(42, self.config, self.protocol, self.output, self.baseline)
            trainer.assert_not_called()
        self.assertEqual(summary.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
