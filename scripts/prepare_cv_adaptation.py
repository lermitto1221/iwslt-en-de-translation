"""Create local CV/adaptation recipes without the author's historical artifacts.

This script never starts training or downloads a dataset/model. Run the stages
in docs/REPRODUCING_CV_ADAPTATION.md from the repository root.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(project, path):
    path = Path(path)
    return (path if path.is_absolute() else Path(project) / path).resolve()


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def publish_files(files):
    # Check the complete stage before publishing any configuration.
    if any(Path(path).exists() for path in files):
        raise FileExistsError("Stage outputs already exist; preserve them and resume the existing runs.")
    for path, value in files.items():
        exclusive(path, value)


def initialize(project=PROJECT, root="artifacts/cv_adaptation", source="artifacts/data"):
    project, root = Path(project).resolve(), local(project, root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Choose a new empty study root; no historical run is overwritten.")
    template = project / "recipes/cv_adaptation_design.json"
    protocol = project / "recipes/decode_protocol_seed42.json"
    design = load(template)
    design.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  output_root=str(root), source_data=str(local(project, source)),
                  status="new_reproduction_before_preparation")
    design["cv"]["decode_protocol"] = str(root / "decode_protocol.json")
    protocol_value = load(protocol)
    publish_files({root / "design.json": design, root / "decode_protocol.json": protocol_value,
                   root / "recipe_identity.json": {"design_sha256": hashlib.sha256(
                       (json.dumps(design, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest(),
                       "template_sha256": sha(template), "protocol_template_sha256": sha(protocol),
                       "protocol_sha256": hashlib.sha256((json.dumps(protocol_value, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()}})
    return root / "design.json"


def checked_design(root):
    root = Path(root).resolve()
    identity = load(root / "recipe_identity.json")
    if sha(root / "design.json") != identity["design_sha256"]:
        raise ValueError("Design changed after initialization; start a new study instead.")
    if sha(root / "decode_protocol.json") != identity["protocol_sha256"]:
        raise ValueError("Fixed decode protocol changed.")
    design = load(root / "design.json")
    if Path(design["output_root"]).resolve() != root:
        raise ValueError("Root moved after local initialization; create a fresh local recipe.")
    return design


def hashes(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError("Frozen input changed: " + path)


def make_cv_config(root, design, seed, smoothing, fold=None):
    root = Path(root).resolve()
    stage = "cv" if fold is not None else "final"
    data = root / "data/cv" / (f"fold{fold}" if fold is not None else "final")
    name = f"fold{fold}_{'ls0' if smoothing == 0 else 'ls01'}" if fold is not None else f"seed{seed}"
    run = root / "cv" / ("runs" if stage == "cv" else "final_runs") / name
    cfg = dict(design["cv_training"])
    cfg.update(seed=seed, label_smoothing=smoothing, candidate_smoothing=smoothing,
               stage=stage, data_dir=str(data), run_dir=str(run),
               run_role="formal_cv_candidate" if stage == "cv" else "formal_selected_cv_retrain")
    if fold is not None:
        cfg["fold"] = fold
    for key, filename in (("common_ids", "common_ids.json"), ("batch_plan", f"batch_plan_seed{seed}.jsonl")):
        cfg[key] = str(data / filename)
        cfg[key + "_sha256"] = sha(data / filename)
    return cfg


def cv_configs(root, final=False):
    root = Path(root).resolve()
    design = checked_design(root)
    inputs = [root / "design.json", root / "decode_protocol.json"]
    if final:
        # Revalidate all six completed runs/OOF evidence before unlocking finals.
        from mt.cv_select import select
        selection = select(root, design["source_data"])
        verify_hashes(selection["input_hashes"])
        smoothing = selection["selected_label_smoothing"]
        configurations = [(f"seed{seed}", make_cv_config(root, design, seed, smoothing)) for seed in (42, 43, 44)]
        output = root / "cv/final_configs"
        inputs.append(root / "cv/selection.json")
    else:
        configurations = [(f"fold{fold}_{'ls0' if smoothing == 0 else 'ls01'}",
                           make_cv_config(root, design, 42, smoothing, fold))
                          for fold in range(3) for smoothing in (0.0, 0.1)]
        output = root / "cv/configs"
    for _, cfg in configurations:
        data = Path(cfg["data_dir"])
        complete = load(data / "preparation_complete.json")
        verify_hashes({str(data / name): digest for name, digest in complete["artifact_sha256"].items()})
        if final and (complete["provenance"]["selected_label_smoothing"] != smoothing
                      or complete["provenance"]["selection_sha256"] != sha(root / "cv/selection.json")):
            raise ValueError("Full-training refit is not bound to the validated CV selection.")
        inputs.extend(Path(cfg["data_dir"]) / name for name in ("preparation_complete.json", "manifest.json", "tokenizer.json", "spm.model", "common_ids.json"))
        inputs.append(Path(cfg["batch_plan"]))
    files = {output / (name + ".json"): cfg for name, cfg in configurations}
    files[output / "stage_plan.json"] = {"scope": "new-machine fixed-3000-step CV reproduction",
        "input_hashes": hashes(inputs), "configurations": configurations,
        "checkpoint_selection": "fixed_last", "test_used_for_selection": False}
    publish_files(files)
    return list(files)


def native_template(root, design, native, download, probe, manifest, evidence, source_files, model_dir):
    """Validate machine-local probe identities before creating any run config."""
    root = Path(root).resolve()
    data_root = root / "adaptation/data"
    if (native["model_id"] != design["adapt"]["base"] or native["revision"] != design["adapt"]["revision"]
            or download["revision"] != design["adapt"]["revision"] or download["files"] != evidence):
        raise ValueError("Native model is not the pinned downloaded OPUS revision.")
    if (probe.get("passed") is not True or probe["model_files"] != evidence
            or probe["source_files"] != source_files or probe["data_manifest_sha256"] != sha(data_root / "manifest.json")):
        raise ValueError("Successful current-model/current-source resource evidence is required.")
    if (manifest["model_files"] != evidence or manifest["updates"] != 3000
            or manifest["effective_batch_size"] != 64 or manifest["max_train_tokens"] != 256
            or probe["micro_batch_size"] != manifest["micro_batch_size"]
            or Path(manifest["data_dir"]).resolve() != Path(design["source_data"]).resolve()
            or Path(native["data_dir"]).resolve() != Path(design["source_data"]).resolve()):
        raise ValueError("Shared native data/schedule/resource settings differ.")
    return {"model_dir": str(Path(model_dir).resolve()), "data_root": str(data_root), "study_plan": str(root / "plan.json"),
        "updates": 3000, "warmup_updates": 300, "train_precision": probe["precision"],
        "micro_batch_size": manifest["micro_batch_size"], "effective_batch_size": 64,
        "validation_every": 500, "save_every": 250, "eval_batch_size": native["batch_size"],
        "cpu_threads": 4, "generation": native["generation"]}


def adapt_configs(root, native_root, model_dir, smoke):
    root = Path(root).resolve()
    design = checked_design(root)
    native_root, model_dir, smoke = map(lambda value: local(PROJECT, value), (native_root, model_dir, smoke))
    native, download, probe = load(native_root / "config.json"), load(native_root / "download.json"), load(smoke)
    from mt.adapt_train import sources, validate_config
    from mt.pretrained_mt import model_evidence
    data_root, plan = root / "adaptation/data", root / "plan.json"
    template = native_template(root, design, native, download, probe, load(data_root / "manifest.json"),
                               model_evidence(model_dir), sources(), model_dir)
    configs = {}
    for method, rates in design["adapt"]["lr_candidates"].items():
        for lr in rates:
            name = f"tune_{method}_lr{lr:g}_seed7"
            cfg = dict(template, method=method, seed=7, lr=lr,
                       run_dir=str(root / "adaptation/runs" / name))
            validate_config(cfg)
            configs[root / "adaptation/configs" / (name + ".json")] = cfg
    inputs = [root / "design.json", data_root / "manifest.json", smoke,
              native_root / "config.json", native_root / "download.json"]
    configs[plan] = {"schema_version": 1, "scope": "portable low-level adaptation recipe; not the historical mt.study plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_hashes": hashes(inputs),
        "source_files": sources(), "budgets": design["budgets"], "history_known": True,
        "selection_rule": design["adapt"]["selection"], "template": template}
    publish_files(configs)
    return list(configs)


def adapt_final_configs(root):
    root = Path(root).resolve()
    checked_design(root)
    plan = load(root / "plan.json")
    verify_hashes(plan["input_hashes"])
    # This selector is independent of study.freeze/run and its historical inputs.
    # It validates frozen run identity, complete training, validation, and hashes.
    from mt.study import select_adaptation
    selection = select_adaptation(root)
    verify_hashes(selection["input_hashes"])
    files = {}
    for seed in (42, 43, 44):
        for method in ("lora", "full"):
            name = f"final_{method}_seed{seed}"
            files[root / "adaptation/configs" / (name + ".json")] = dict(
                plan["template"], method=method, seed=seed, lr=selection["selected_lrs"][method],
                run_dir=str(root / "adaptation/runs" / name))
    files[root / "adaptation/final_plan.json"] = {"selection_sha256": sha(root / "adaptation/selection.json"),
        "study_plan_sha256": sha(root / "plan.json"), "selected_lrs": selection["selected_lrs"],
        "configurations": {str(path): value for path, value in files.items()},
        "fresh_base_for_every_seed": True}
    publish_files(files)
    return list(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "cv-configs", "cv-final-configs", "adapt-configs", "adapt-final-configs"))
    parser.add_argument("--root", default="artifacts/cv_adaptation")
    parser.add_argument("--source", default="artifacts/data")
    parser.add_argument("--native-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--smoke")
    args = parser.parse_args()
    root = local(PROJECT, args.root)
    if args.action == "init":
        result = initialize(root=root, source=args.source)
    elif args.action == "cv-configs":
        result = cv_configs(root)
    elif args.action == "cv-final-configs":
        result = cv_configs(root, final=True)
    elif args.action == "adapt-final-configs":
        result = adapt_final_configs(root)
    else:
        if not all((args.native_root, args.model_dir, args.smoke)):
            parser.error("adapt-configs requires --native-root, --model-dir and --smoke")
        result = adapt_configs(root, args.native_root, args.model_dir, args.smoke)
    print(json.dumps([str(path) for path in result] if isinstance(result, list) else str(result), indent=2))


if __name__ == "__main__":
    main()
