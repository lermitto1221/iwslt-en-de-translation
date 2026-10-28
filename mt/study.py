"""Frozen sequential CV selection and same-base adaptation study.

Use project-pretrained-python.ps1 for this controller. Child processes use the
original project entry points; every child finishes before the next starts.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .common import read_json, sha256

PROJECT = Path(__file__).resolve().parent.parent
CANDIDATES = ((0.0, "ls0"), (0.1, "ls01"))
LRS = {"lora": [1e-4, 3e-4], "full": [1e-5, 3e-5]}
SEEDS = (42, 43, 44)
SOURCES = (
    "mt/study.py", "mt/cv_prepare.py", "mt/cv_train.py", "mt/cv_evaluate.py", "mt/cv_select.py",
    "mt/adapt_model.py", "mt/adapt_data.py", "mt/adapt_train.py", "mt/pretrained_mt.py",
    "mt/common.py", "mt/data.py", "mt/model.py", "mt/train.py", "mt/batching.py",
    "mt/tokenizer.py", "mt/decoding.py", "mt/inference.py", "mt/evaluate.py",
    "mt/representation_batches.py", "mt/representation_prepare.py", "mt/export.py",
    "project-python.ps1", "project-pretrained-python.ps1",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Refusing to replace frozen evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication, without replacing an existing published file.
    temporary = path.with_name(path.name + f".{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    temporary.rename(path)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Output must stay inside the new study directory: {path}")
    return path


def hashes(paths):
    result = {}
    for path in sorted(set(map(Path, paths))):
        if not path.exists():
            raise FileNotFoundError(f"Missing required freeze input: {path}")
        if path.is_file():
            result[str(path.resolve())] = sha256(path)
    return result


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise ValueError(f"Frozen evidence changed or missing: {name}")


def cv_config(root, data, run, seed, smoothing, stage, fold=None):
    cfg = read_json(root / "resources/cv_pilot/config.json")
    cfg.pop("run_role", None); cfg.pop("validate_every", None)
    cfg.update(data_dir=str(data), run_dir=str(run), seed=seed, label_smoothing=smoothing,
               candidate_smoothing=smoothing, stage=stage,
               run_role="formal_cv_candidate" if stage == "cv" else "formal_selected_cv_retrain")
    if fold is not None:
        cfg["fold"] = fold
    for key, filename in (("common_ids", "common_ids.json"), ("batch_plan", f"batch_plan_seed{seed}.jsonl")):
        cfg[key] = str(data / filename); cfg[key + "_sha256"] = sha256(cfg[key])
    inside(root, run)
    return cfg


def adapt_name(method, lr):
    return f"tune_{method}_lr{lr:g}_seed7"


def adapt_config(root, name, method, seed, lr):
    cfg = read_json(root / "adaptation/config_template.json")
    cfg.update(method=method, seed=seed, lr=lr, run_dir=str(root / "adaptation/runs" / name),
               study_plan=str(root / "plan.json"))
    return cfg


def estimate_cost(root):
    cv = read_json(root / "resources/cv_pilot/pilot.json")
    native = read_json(root / "adaptation/resource_smoke_micro8_v2.json")
    train = cv["steady_mean_update_seconds"]
    parts = {
        "cv_training_18000_updates": 18000 * train,
        "cv_complete_holdout_408570_sentences": 408570 / cv["decode_sentences_per_second"],
        "cv_final_training_9000_updates": 9000 * train,
        "cv_final_full_evaluations_proxy": 3 * 8967 * (1 / cv["decode_sentences_per_second"] + 342.0 / 8079),
        "lora_training_15000_updates": 15000 * native["methods"]["lora"]["steady_mean_step_seconds"],
        "full_training_15000_updates": 15000 * native["methods"]["full"]["steady_mean_step_seconds"],
        "adaptation_60_validation_evaluations_proxy": 60 * 20.154,
        "adaptation_24_formal_evaluations_proxy": 6 * (20.154 + 46.89 + 163.374 + 422.05),
    }
    return {"seconds_by_component": parts, "total_hours_before_overheads": sum(parts.values()) / 3600,
        "planning_range_hours": [8, 12], "extra_resource_optimizer_updates": 108,
        "basis": "Actual CV train-only 20-update/512-sentence timing; actual native training probes. Final decode proxies from historical bpe42/native timings; no quality used in estimate.",
        "limits": "Not a guarantee. Startup, tokenization, checkpoint I/O, OOF scoring, thermal/power changes and generated lengths add uncertainty. Complete CV holdouts explicitly included."}


def freeze(root):
    root = Path(root).resolve()
    if (root / "plan.json").exists():
        return verify_plan(root)
    # No formal computation may precede the full frozen root plan.
    for subtree in (root / "cv/runs", root / "cv/final_runs", root / "adaptation/runs"):
        if subtree.exists() and any(p.is_file() for p in subtree.rglob("*")):
            raise ValueError("Formal run evidence exists before the study freeze")
    from .adapt_train import sources
    from .pretrained_mt import dependency_evidence
    resource = read_json(root / "adaptation/resource_ready.json")
    if not resource["passed"] or resource["source_files"] != sources():
        raise ValueError("Final resource audit does not match current adaptation implementation")
    if not read_json(root / "baseline_reuse_audit.json")["passed"]:
        raise ValueError("Direct-baseline reuse audit did not pass")
    configs = []
    for fold in range(3):
        for smoothing, tag in CANDIDATES:
            name = f"fold{fold}_{tag}"
            path = root / "cv/configs" / (name + ".json")
            immutable_json(path, cv_config(root, root / f"data/cv/fold{fold}", root / "cv/runs" / name,
                                           42, smoothing, "cv", fold))
            configs.append(path)
    for method, rates in LRS.items():
        for lr in rates:
            name = adapt_name(method, lr)
            path = root / "adaptation/configs" / (name + ".json")
            immutable_json(path, adapt_config(root, name, method, 7, lr)); configs.append(path)
    source_hashes = {name: sha256(PROJECT / name) for name in SOURCES}
    for name, checksum in source_hashes.items():
        out = root / "source_snapshot" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if sha256(out) != checksum:
                raise ValueError("Partial source snapshot differs: " + name)
        else:
            with out.open("xb") as f:
                f.write((PROJECT / name).read_bytes())
    inputs = [root / name for name in ("design.json", "historical_manifest.json", "baseline_reuse_audit.json",
        "resources/cv_pilot/pilot.json", "resources/cv_pilot/config.json", "adaptation/resource_ready.json",
        "adaptation/resource_smoke_micro8_v2.json", "adaptation/config_template.json", "adaptation/implementation_ready.json")]
    for directory in (root / "data/cv", root / "adaptation/data"):
        inputs += [p for p in directory.rglob("*") if p.is_file() and p.suffix != ".log"]
    inputs += [PROJECT / "artifacts/decode_protocol_seed42.json"]
    inputs += [PROJECT / "artifacts/data" / (s + ".jsonl") for s in ("train", "validation", "test")]
    inputs += list((PROJECT / "artifacts/pretrained_mt_2026_10_27/model").glob("*"))
    plan = {"schema_version": 1, "frozen_at_utc": now(), "root": str(root),
        "history_known": True, "status": "frozen_before_all_19_formal_trainings_and_new_test_results",
        "design_sha256": sha256(root / "design.json"), "source_files": source_hashes,
        "inputs": hashes(inputs), "initial_configs": hashes(configs), "dependencies": dependency_evidence(),
        "budgets": read_json(root / "design.json")["budgets"],
        "resolved_conditions": read_json(root / "adaptation/implementation_ready.json")["common_conditions"],
        "execution_order": "CV folds0,1,2 each ls0 then ls01; CV selection; new full-train tokenizer; CV final seeds42,43,44; tune LoRA LR1e-4,3e-4 then full LR1e-5,3e-5; LR selection; formal seeds42,43,44 each LoRA then full",
        "dynamic_stage_rule": "After data-only CV selection / validation-only LR selection, freeze final configs and selection hashes in separate immutable stage plans before any corresponding training or test. Root plan remains unchanged.",
        "selection_ties": "CV mean fold BLEU, mean fold chrF2, candidate order[0,0.1]. Adaptation within run: validation BLEU, chrF2, earliest update; across learning rates: selected checkpoint BLEU, chrF2, declared LR order. Exact ties only.",
        "parameters": read_json(root / "adaptation/implementation_ready.json")["parameters"],
        "cost_estimate": estimate_cost(root),
        "interpretation_limits": ["CV is budget3000-only parameter selection, not independent final confirmation.",
            "Same update counts do not imply same compute or sentence exposures across scratch/native families.",
            "All native trainable parameters are tuned in full; two native fixed sinusoidal position tables stay fixed.",
            "Prior OPUS pretraining may overlap IWSLT/TED. n3 does not establish statistical significance."]}
    immutable_json(root / "plan.json", plan)
    return verify_plan(root)


def verify_plan(root):
    root = Path(root).resolve(); plan = read_json(root / "plan.json")
    if plan["root"] != str(root):
        raise ValueError("Study root identity differs")
    for name, checksum in plan["source_files"].items():
        if sha256(PROJECT / name) != checksum or sha256(root / "source_snapshot" / name) != checksum:
            raise ValueError("Frozen execution source changed: " + name)
    verify_hashes(plan["inputs"]); verify_hashes(plan["initial_configs"])
    from .pretrained_mt import dependency_evidence
    if dependency_evidence() != plan["dependencies"]:
        raise ValueError("Frozen dependencies changed")
    return plan


def freeze_stage(root, path, paths, decision):
    path = inside(root, path)
    value = {"created_at_utc": read_json(path)["created_at_utc"] if path.exists() else now(),
             "study_plan_sha256": sha256(root / "plan.json"), "decision": decision, "files": hashes(paths)}
    immutable_json(path, value)
    verify_hashes(value["files"])
    return value


def verify_stage(root, path):
    stage = read_json(inside(root, path))
    if stage["study_plan_sha256"] != sha256(root / "plan.json"):
        raise ValueError("Stage belongs to another study plan")
    verify_hashes(stage["files"])
    for name in stage["files"]:
        if Path(name).name == "selection.json":
            decision = read_json(name)
            if decision.get("complete") is not True:
                raise ValueError("Stage selection is incomplete")
            verify_hashes(decision["input_hashes"])
            key = "selected_lrs" if "selected_lrs" in decision else "selected_label_smoothing"
            if stage["decision"][key] != decision[key]:
                raise ValueError("Stage decision differs from its selection evidence")
    return stage


def authorize_evaluation(root, config, split, method):
    """Reject tuning test use and require selection-bound final configurations."""
    config = inside(root, config); cfg = read_json(config)
    inside(root, cfg["run_dir"])
    if cfg.get("stage") == "cv":
        if (split, method) != ("holdout", "greedy"):
            raise ValueError("CV candidates may only evaluate complete heldout Greedy")
        frozen = read_json(root / "plan.json")["initial_configs"]
        if frozen.get(str(config)) != sha256(config):
            raise ValueError("CV candidate is not frozen")
    elif cfg.get("stage") == "final":
        stage = verify_stage(root, root / "cv/final_plan.json")
        if str(config) not in stage["files"] or cfg["seed"] not in SEEDS or cfg["label_smoothing"] != stage["decision"]["selected_label_smoothing"]:
            raise ValueError("Final CV config does not match frozen selection")
        if split not in ("validation", "test") or method not in ("greedy", "beam"):
            raise ValueError("Invalid final evaluation")
    else:
        if cfg.get("seed") not in SEEDS:
            raise ValueError("Tuning adaptation models cannot enter final evaluation")
        stage = verify_stage(root, root / "adaptation/final_plan.json")
        if str(config) not in stage["files"] or cfg["lr"] != stage["decision"]["selected_lrs"][cfg["method"]]:
            raise ValueError("Final adaptation config differs from selected learning rate")
        if split not in ("validation", "test") or method not in ("greedy", "beam"):
            raise ValueError("Invalid final native evaluation")
    return cfg


@contextmanager
def study_lock(root):
    import msvcrt
    path = root / "execution.lock"
    with path.open("a+b") as handle:
        if not path.stat().st_size:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("This study already has an active sequential controller") from exc
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def run_command(root, name, launcher, arguments):
    verify_plan(root)
    out = root / "execution" / name
    out.mkdir(parents=True, exist_ok=True)
    attempt = len(list(out.glob("attempt_*.json"))) + 1
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(PROJECT / launcher), *map(str, arguments)]
    start = now()
    record = {"name": name, "attempt": attempt, "started_at_utc": start, "command": command,
              "study_plan_sha256": sha256(root / "plan.json")}
    record_path = out / f"attempt_{attempt:03d}.json"
    atomic_json(record_path, record)
    atomic_json(root / "progress.json", {"stage": name, "status": "running", "updated_at_utc": start,
        "command": command, "log": str(out / f"attempt_{attempt:03d}.log"),
        "recovery_command": f'.\\project-pretrained-python.ps1 -m mt.study run --root "{root}"'})
    print(json.dumps({"stage": name, "started_at": start}), flush=True)
    with (out / f"attempt_{attempt:03d}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
    record.update(ended_at_utc=now(), exit_code=result.returncode,
                  log_sha256=sha256(out / f"attempt_{attempt:03d}.log"))
    atomic_json(record_path, record)
    if result.returncode:
        raise RuntimeError(f"Stage {name} failed (exit {result.returncode}); see {out}")
    print(json.dumps({"stage": name, "finished_at": record["ended_at_utc"]}), flush=True)


def run_cv(root, config):
    cfg = read_json(config); run = inside(root, cfg["run_dir"])
    if cfg["stage"] == "final":
        authorize_evaluation(root, config, "validation", "greedy")
    else:
        authorize_evaluation(root, config, "holdout", "greedy")
    args = ["-m", "mt.cv_train", "--config", config]
    if (run / "last.pt").exists():
        args += ["--resume", run / "last.pt"]
    run_command(root, "train_" + run.name, "project-python.ps1", args)
    splits = ("holdout",) if cfg["stage"] == "cv" else ("validation", "test")
    methods = ("greedy",) if cfg["stage"] == "cv" else ("greedy", "beam")
    for split in splits:
        for method in methods:
            authorize_evaluation(root, config, split, method)
            run_command(root, f"eval_{run.name}_{split}_{method}", "project-python.ps1",
                ["-m", "mt.cv_evaluate", "--checkpoint", run / "last.pt", "--split", split, "--method", method,
                 "--out", run / f"{split}_{method}", "--protocol", PROJECT / "artifacts/decode_protocol_seed42.json",
                 "--decode-batch-size", "16", "--device", "cuda"])


def prepare_cv_final(root):
    # This selector validates all six completed runs/holdouts; final preprocessing
    # cannot be unlocked by dropping a hand-written selection label into place.
    run_command(root, "cv_select", "project-python.ps1", ["-m", "mt.cv_select", "--root", root])
    decision = read_json(root / "cv/selection.json")
    run_command(root, "cv_final_prepare", "project-python.ps1",
                ["-m", "mt.cv_prepare", "--root", root, "--stage", "final"])
    data = root / "data/cv/final"; paths = []
    for seed in SEEDS:
        path = root / "cv/final_configs" / f"seed{seed}.json"
        immutable_json(path, cv_config(root, data, root / "cv/final_runs" / f"seed{seed}", seed,
                                      decision["selected_label_smoothing"], "final"))
        paths.append(path)
    freeze_stage(root, root / "cv/final_plan.json", paths + list(data.rglob("*")) + [root / "cv/selection.json"],
                 {"selected_label_smoothing": decision["selected_label_smoothing"],
                  "selection_sha256": sha256(root / "cv/selection.json"), "checkpoint": "fixed last3000"})
    return paths


def ensure_adapt_plan(root, config):
    from .adapt_train import freeze_run, verify_run, sources
    cfg = read_json(config); run = inside(root, cfg["run_dir"])
    if (run / "plan.json").exists():
        verify_run(config); return
    snapshot = run / "source_snapshot"
    if snapshot.exists():
        if (run / "last.pt").exists() or (run / "train.jsonl").exists():
            raise ValueError("Training history exists without a frozen run plan")
        expected = sources()
        for p in snapshot.rglob("*"):
            if p.is_file() and expected.get(p.relative_to(snapshot).as_posix()) != sha256(p):
                raise ValueError("Partial adaptation snapshot is corrupt")
        archived = run / ("source_snapshot.before-plan-recovery." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        snapshot.rename(archived)
    freeze_run(config)


def run_adapt(root, config, formal):
    verify_plan(root)
    cfg = read_json(config); run = inside(root, cfg["run_dir"])
    if formal:
        authorize_evaluation(root, config, "validation", "greedy")
    elif cfg["seed"] != 7 or str(Path(config).resolve()) not in read_json(root / "plan.json")["initial_configs"]:
        raise ValueError("Tuning configuration is not the frozen seed7 candidate")
    ensure_adapt_plan(root, config)
    run_command(root, "train_" + run.name, "project-pretrained-python.ps1",
                ["-m", "mt.adapt_train", "train", "--config", config])
    if formal:
        for split in ("validation", "test"):
            for method in ("greedy", "beam"):
                authorize_evaluation(root, config, split, method)
                run_command(root, f"eval_{run.name}_{split}_{method}", "project-pretrained-python.ps1",
                    ["-m", "mt.adapt_train", "evaluate", "--config", config, "--split", split, "--method", method])


def select_adaptation(root):
    import torch
    from .adapt_data import load_data
    from .adapt_train import select_learning_rate, verify_run, _verify_checkpoint, verify_completed
    records, selected, inputs = {}, {}, []
    for method, rates in LRS.items():
        records[method] = []
        for lr in rates:
            name = adapt_name(method, lr); config = root / "adaptation/configs" / (name + ".json")
            cfg, plan = verify_run(config); run = Path(cfg["run_dir"])
            manifest, indexed, schedule = load_data(cfg["data_root"], cfg["seed"])
            state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            _verify_checkpoint(state, cfg, plan, schedule)
            if state["update"] != 3000:
                raise ValueError("Learning-rate candidate is incomplete or changed")
            summary = verify_completed(cfg, plan, state, manifest, "cuda")
            del state, indexed, schedule
            best = summary["best"]
            records[method].append({"method": method, "lr": lr, "seed": 7, "updates": 3000, "split": "validation",
                "bleu": best["bleu"], "chrf2": best["chrf2"], "checkpoint_update": best["update"], "run": str(run)})
            inputs += [config, run / "summary.json", run / "last.pt", run / "best.pt", run / "train.jsonl", run / "plan.json"]
            inputs += [p for p in (run / "validation").rglob("*") if p.is_file()]
        selected[method] = select_learning_rate(records[method], rates)["lr"]
    path = root / "adaptation/selection.json"
    value = {"created_at_utc": read_json(path)["created_at_utc"] if path.exists() else now(), "complete": True,
        "selected_lrs": selected, "candidates": records, "input_hashes": hashes(inputs),
        "test_used": False, "rule": "complete seed7 validation BLEU, then chrF2, then fixed LR order"}
    immutable_json(path, value)
    return value


def prepare_adapt_final(root):
    selection = select_adaptation(root); paths = []
    for seed in SEEDS:
        for method in LRS:
            name = f"final_{method}_seed{seed}"; path = root / "adaptation/configs" / (name + ".json")
            immutable_json(path, adapt_config(root, name, method, seed, selection["selected_lrs"][method])); paths.append(path)
    freeze_stage(root, root / "adaptation/final_plan.json", paths + [root / "adaptation/selection.json"],
                 {"selected_lrs": selection["selected_lrs"], "fresh_base_for_every_seed": True,
                  "selection_sha256": sha256(root / "adaptation/selection.json")})
    return paths


def history_check(root, output):
    manifest = read_json(root / "historical_manifest.json")
    changed = []
    for name, value in manifest["files"].items():
        path = PROJECT / name
        if not path.is_file() or path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
            changed.append(name)
    result = {"checked_at_utc": now(), "files_checked": len(manifest["files"]), "changed": changed, "passed": not changed,
              "manifest_sha256": sha256(root / "historical_manifest.json")}
    immutable_json(output, result)
    if changed:
        raise ValueError("Historical files changed: " + repr(changed))
    return result


def run(root):
    root = Path(root).resolve()
    with study_lock(root):
        verify_plan(root)
        try:
            if not (root / "history_before_formal.json").exists():
                history_check(root, root / "history_before_formal.json")
            for fold in range(3):
                for _, tag in CANDIDATES:
                    run_cv(root, root / "cv/configs" / f"fold{fold}_{tag}.json")
            for config in prepare_cv_final(root):
                run_cv(root, config)
            for method, rates in LRS.items():
                for lr in rates:
                    run_adapt(root, root / "adaptation/configs" / (adapt_name(method, lr) + ".json"), formal=False)
            for config in prepare_adapt_final(root):
                run_adapt(root, config, formal=True)
            atomic_json(root / "progress.json", {"stage": "experiments_complete_reporting_pending", "status": "completed",
                "updated_at_utc": now(), "formal_runs": 19, "formal_updates": 57000,
                "next": "CPU-only aggregate verification, report/presentation and historical fingerprint check"})
        except BaseException as exc:
            old = read_json(root / "progress.json") if (root / "progress.json").exists() else {}
            atomic_json(root / "progress.json", {**old, "status": "failed", "updated_at_utc": now(),
                "error": repr(exc), "traceback": traceback.format_exc(),
                "recovery_command": f'.\\project-pretrained-python.ps1 -m mt.study run --root "{root}"'})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "history"))
    parser.add_argument("--root")
    args = parser.parse_args()
    root = Path(args.root or read_json(PROJECT / "artifacts/cv_adapt_active.json")["root"]).resolve()
    if args.action == "run":
        run(root)
    else:
        result = {"freeze": freeze, "verify": verify_plan}.get(args.action)
        result = result(root) if result else history_check(root, root / ("history_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
