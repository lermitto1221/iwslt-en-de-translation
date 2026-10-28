"""Portable recipe checks: temporary inputs only, no GPU/model/downloads."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_cv_adaptation_recipe", PROJECT / "scripts/prepare_cv_adaptation.py")
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class PublicCVAdaptationRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / "recipes").mkdir()
        for name in ("cv_adaptation_design.json", "decode_protocol_seed42.json"):
            shutil.copyfile(PROJECT / "recipes" / name, self.project / "recipes" / name)
        self.root = self.project / "outputs/study"

    def initialize(self):
        recipe.initialize(self.project, "outputs/study", "input/data")
        return recipe.checked_design(self.root)

    def fixture_fold_files(self):
        # These are preparation metadata fixtures, never fed to a model.
        for fold in range(3):
            data = self.root / "data/cv" / f"fold{fold}"
            data.mkdir(parents=True)
            for name in ("manifest.json", "tokenizer.json", "common_ids.json", "batch_plan_seed42.jsonl"):
                (data / name).write_text(json.dumps({"fixture_fold": fold}), encoding="utf-8")
            (data / "spm.model").write_bytes(f"fold-local-tokenizer-{fold}".encode())
            recipe.exclusive(data / "preparation_complete.json", {"artifact_sha256": {
                p.name: recipe.sha(p) for p in data.iterdir() if p.is_file()}})

    def test_relative_paths_resolve_against_new_checkout(self):
        design = self.initialize()
        self.assertEqual(Path(design["output_root"]), self.root.resolve())
        self.assertEqual(Path(design["source_data"]), (self.project / "input/data").resolve())
        self.assertFalse((self.project / "input/data").exists())
        self.assertEqual(design["budgets"]["total_runs"], 19)
        self.assertEqual(design["budgets"]["total_updates"], 57000)
        self.assertFalse((self.root / "historical_manifest.json").exists())

    def test_init_refuses_existing_nonempty_directory_without_modification(self):
        self.initialize()
        before = (self.root / "design.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(before, (self.root / "design.json").read_bytes())

    def test_protocol_serialization_does_not_depend_on_checkout_line_endings(self):
        path = self.project / "recipes/decode_protocol_seed42.json"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.initialize()
        recipe.checked_design(self.root)

    def test_frozen_design_or_protocol_change_is_rejected(self):
        self.initialize()
        with (self.root / "design.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Design changed"):
            recipe.checked_design(self.root)

    def test_cv_candidates_share_fold_inputs_and_only_change_smoothing_and_output(self):
        self.initialize()
        self.fixture_fold_files()
        paths = recipe.cv_configs(self.root)
        self.assertEqual(len(paths), 7)
        for fold in range(3):
            left = recipe.load(self.root / "cv/configs" / f"fold{fold}_ls0.json")
            right = recipe.load(self.root / "cv/configs" / f"fold{fold}_ls01.json")
            allowed = {"label_smoothing", "candidate_smoothing", "run_dir"}
            self.assertEqual({k: v for k, v in left.items() if k not in allowed},
                             {k: v for k, v in right.items() if k not in allowed})
            self.assertEqual(left["checkpoint_selection"], "fixed_last")
            self.assertEqual(left["max_updates"], 3000)
            self.assertEqual(left["common_ids_sha256"], recipe.sha(left["common_ids"]))
            self.assertIn(f"fold{fold}", left["data_dir"])
        before = {p: p.read_bytes() for p in paths}
        with self.assertRaises(FileExistsError):
            recipe.cv_configs(self.root)
        self.assertEqual(before, {p: p.read_bytes() for p in paths})

    def test_missing_preparation_never_publishes_partial_configs(self):
        self.initialize()
        with self.assertRaises(FileNotFoundError):
            recipe.cv_configs(self.root)
        self.assertFalse((self.root / "cv/configs").exists())

    def test_stage_inputs_are_bound_to_current_files(self):
        self.initialize()
        self.fixture_fold_files()
        recipe.cv_configs(self.root)
        mapping = recipe.load(self.root / "cv/configs/stage_plan.json")["input_hashes"]
        recipe.verify_hashes(mapping)
        path = self.root / "data/cv/fold0/spm.model"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            recipe.verify_hashes(mapping)

    def native_fixture(self):
        design = self.initialize()
        evidence, sources = {"weights": {"sha256": "base-id", "bytes": 1}}, {"source.py": "source-id"}
        manifest = {"model_files": evidence, "updates": 3000, "effective_batch_size": 64,
                    "micro_batch_size": 4, "max_train_tokens": 256, "data_dir": design["source_data"]}
        recipe.exclusive(self.root / "adaptation/data/manifest.json", manifest)
        native = {"model_id": design["adapt"]["base"], "revision": design["adapt"]["revision"],
                  "data_dir": design["source_data"], "batch_size": 16,
                  "generation": {"greedy": {"num_beams": 1}, "beam": {"num_beams": 4}}}
        download = {"revision": design["adapt"]["revision"], "files": evidence}
        probe = {"passed": True, "model_files": evidence, "source_files": sources,
                 "data_manifest_sha256": recipe.sha(self.root / "adaptation/data/manifest.json"),
                 "micro_batch_size": 4, "precision": "float16"}
        return design, native, download, probe, manifest, evidence, sources, self.root / "base/model"

    def test_native_template_carries_actual_common_resource_settings(self):
        values = self.native_fixture()
        result = recipe.native_template(self.root, *values)
        self.assertEqual(result["micro_batch_size"], 4)
        self.assertEqual(result["effective_batch_size"], 64)
        self.assertEqual(result["train_precision"], "float16")
        self.assertEqual(result["generation"], values[1]["generation"])
        # The future file may not exist; normalize Windows 8.3 aliases first.
        self.assertEqual(Path(result["study_plan"]).resolve(), (self.root / "plan.json").resolve())
        self.assertFalse((self.root / "plan.json").exists())

    def test_native_rejects_failed_or_stale_resource_identity(self):
        values = self.native_fixture()
        values[3]["passed"] = False
        with self.assertRaisesRegex(ValueError, "resource evidence"):
            recipe.native_template(self.root, *values)
        values[3]["passed"] = True
        values[3]["source_files"] = {"source.py": "changed-source"}
        with self.assertRaisesRegex(ValueError, "resource evidence"):
            recipe.native_template(self.root, *values)

    def test_native_rejects_model_revision_and_input_source_mismatches(self):
        values = self.native_fixture()
        values[2]["revision"] = "another-revision"
        with self.assertRaisesRegex(ValueError, "pinned downloaded"):
            recipe.native_template(self.root, *values)
        values[2]["revision"] = values[0]["adapt"]["revision"]
        values[1]["data_dir"] = str(self.project / "different-input")
        with self.assertRaisesRegex(ValueError, "settings differ"):
            recipe.native_template(self.root, *values)


if __name__ == "__main__":
    unittest.main()
