"""Freeze a seed-specific extension before training; keep seed-42 defaults."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

from .common import sha256
from .summary import (artifact_path, read_object, require, valid_hash,
                      validate_comparability, validate_result, validate_training_log)
from .train import CORE_TRAINING_SOURCES, check_inputs


def quality_paths(seed=42):
    require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    folder = Path(f"artifacts/quality_seed{seed}")
    return {"config": Path(f"experiments/quality_A{seed}_12000.json"),
            "baseline": Path(f"artifacts/runs/A{seed}"),
            "run": Path(f"artifacts/runs/quality_A{seed}_12000"),
            "plan": Path("artifacts/quality_plan.json") if seed == 42 else folder / "plan.json",
            "progress": Path("artifacts/quality_progress.json") if seed == 42 else folder / "progress.json",
            "summary": Path("artifacts/quality_summary.json") if seed == 42 else folder / "summary.json"}


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_quality_config(seed, out=None, template="experiments/quality_A42_12000.json", run_dir=None):
    """Copy the fixed template, changing only seed/run_dir; never overwrite."""
    paths = quality_paths(seed)
    target = Path(out or paths["config"])
    if target.exists():
        raise FileExistsError(f"quality config already exists and will not be overwritten: {target}")
    cfg = read_object(template, "quality template")
    require(cfg.get("max_updates") == 12000 and cfg.get("label_smoothing") == .1
            and cfg.get("validate_every") == 500, "quality template does not match the fixed 12000-update experiment")
    cfg = dict(cfg, seed=seed, run_dir=str(run_dir or paths["run"]).replace("\\", "/"))
    require_unstarted(Path(cfg["run_dir"]), cfg)
    exclusive_json(target, cfg)
    return cfg


def read_hashed_object(path, context):
    path = artifact_path(path, context).resolve()
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{context}: expected a JSON object: {path}")
    return path, value, hashlib.sha256(content).hexdigest()


def require_unstarted(run, config):
    """Only an empty directory or its matching prepared config is permitted."""
    if not run.exists():
        return
    require(run.is_dir(), f"quality run path is not a directory: {run}")
    evidence = [path for path in run.iterdir() if path.name != "config.json"]
    require(not evidence, f"quality run already contains artifacts ({', '.join(path.name for path in evidence[:5])}); "
            "refusing to create a post-hoc predeclared plan; use a new run directory before training")
    prepared = run / "config.json"
    if prepared.exists():
        require(read_object(prepared, "prepared quality config") == config,
                "existing quality run config.json differs from the planned configuration")


def validate_extension_config(cfg, base_cfg, seed):
    require(type(cfg.get("seed")) is int and cfg["seed"] == seed and base_cfg.get("seed") == seed,
            f"quality and baseline configs must both match seed={seed}")
    require(cfg.get("label_smoothing") == .1, "quality extension requires label_smoothing=0.1")
    require(type(cfg.get("validate_every")) is int and cfg["validate_every"] > 0,
            "quality config requires a positive integer validation interval")
    require(type(cfg.get("max_updates")) is int and type(base_cfg.get("max_updates")) is int
            and cfg["max_updates"] > base_cfg["max_updates"] > 0,
            "quality extension budget must be an integer larger than the completed baseline budget")
    ignored = {"run_dir", "max_updates"}
    require({key: value for key, value in cfg.items() if key not in ignored}
            == {key: value for key, value in base_cfg.items() if key not in ignored},
            f"quality config may differ from A{seed} only in run_dir and max_updates")


def freeze_core_sources():
    # Preserve exact source bytes as JSON text, including line endings.
    content = {name: Path(__file__).with_name(name).read_bytes() for name in CORE_TRAINING_SOURCES}
    return ({name: hashlib.sha256(value).hexdigest() for name, value in content.items()},
            {name: value.decode("utf-8") for name, value in content.items()})


def create_plan(config=None, protocol="artifacts/decode_protocol_seed42.json",
                out=None, baseline=None, seed=42):
    paths = quality_paths(seed)
    output, baseline = Path(out or paths["plan"]).resolve(), Path(baseline or paths["baseline"]).resolve()
    if output.exists():
        raise FileExistsError(f"plan already exists and will not be overwritten: {output}")
    config_path, cfg, config_hash = read_hashed_object(config or paths["config"], "quality config")
    require(isinstance(cfg.get("run_dir"), str) and cfg["run_dir"].strip(), "quality config has no run_dir")
    run = Path(cfg["run_dir"]).resolve()
    require(not run.is_relative_to(baseline) and not baseline.is_relative_to(run),
            f"quality run directory must differ from baseline A{seed} and not contain it")
    require(not output.is_relative_to(run), "save the quality plan outside the reserved quality run directory")
    require_unstarted(run, cfg)
    protocol_path, frozen, protocol_hash = read_hashed_object(protocol, "frozen decoding protocol")
    require(frozen.get("selection_split") == "validation" and type(frozen.get("beam_size")) is int
            and frozen["beam_size"] == 4, "quality extension requires a validation-selected beam=4 protocol")
    require(type(frozen.get("max_generation_tokens")) is int and 1 <= frozen["max_generation_tokens"] <= 4095
            and type(frozen.get("length_penalty")) in (float, int) and math.isfinite(frozen["length_penalty"]),
            "invalid length penalty or generation limit in frozen protocol")
    base_cfg = read_object(baseline / "config.json", f"baseline A{seed} config")
    summary = read_object(baseline / "summary.json", f"baseline A{seed} summary")
    require(summary.get("complete") is True and summary.get("smoke") is False
            and summary.get("pilot", False) is False, f"baseline A{seed} must have completed formal training")
    validate_extension_config(cfg, base_cfg, seed)
    validate_training_log(baseline, base_cfg, f"baseline A{seed}")
    records = {}
    for method, group in (("greedy", f"A1_seed{seed}"), ("beam", f"A2_seed{seed}")):
        metrics = read_object(baseline / method / "metrics.json", f"baseline A{seed} {method}")
        validate_result(metrics, baseline, "A", seed, method)
        require(artifact_path(metrics.get("checkpoint"), group).resolve() == baseline / "best.pt",
                f"{group}: baseline evaluation does not use its best.pt")
        for field in ("length_penalty", "max_generation_tokens"):
            require(metrics.get(field) == frozen[field], f"baseline {method}: {field} differs from frozen protocol")
        records[group] = metrics
    validate_comparability(records)
    initial = records[f"A1_seed{seed}"]["initial_model_sha256"]
    require(initial == records[f"A2_seed{seed}"]["initial_model_sha256"],
            "baseline greedy/beam initialization fingerprints differ")
    linked = read_object(baseline / "data_link.json", "baseline data link")
    require(valid_hash(initial) and linked.get("initial_model_sha256") == initial,
            "baseline data link initialization differs from verified evaluation")
    fingerprints = check_inputs(cfg)
    require(all(linked.get(key) == value for key, value in fingerprints.items()),
            "quality data/tokenizer fingerprints differ from baseline")
    expected_metric_data = {"manifest_sha256": fingerprints["manifest_sha256"],
                           "tokenizer_sha256": fingerprints["tokenizer_sha256"],
                           **{split + "_sha256": value for split, value in fingerprints["splits"].items()}}
    require(records[f"A1_seed{seed}"]["data_fingerprints"] == expected_metric_data,
            "baseline metrics data fingerprints differ from verified inputs")
    evidence_names = ["config.json", "summary.json", "train.jsonl", "best.pt", "data_link.json",
                      "greedy/metrics.json", "beam/metrics.json"]
    evidence_names += [f"{method}/{name}" for method in ("greedy", "beam")
                       for name in ("predictions.txt", "references.txt", "examples.jsonl")]
    evidence_names += [name for name in ("source_hashes.json", "source_snapshot/manifest.json")
                       if (baseline / name).is_file()]
    evidence = {artifact_path(baseline / name, "baseline evidence").resolve().as_posix():
                sha256(baseline / name) for name in evidence_names}
    sources, source_text = freeze_core_sources()
    require_unstarted(run, cfg)
    require(sha256(config_path) == config_hash and sha256(protocol_path) == protocol_hash,
            "quality config or frozen protocol changed while the plan was being validated")
    now = datetime.datetime.now(datetime.timezone.utc)
    plan = {
        "schema_version": 2, "created_at_utc": now.isoformat(), "created_date": now.date().isoformat(),
        "created_date_timezone": "UTC", "status": "predeclared_before_quality_training",
        "path_convention": "All recorded paths are absolute; baseline.files maps absolute paths to SHA-256.",
        "shared_decoding_protocol": protocol_path.as_posix(), "protocol_sha256": protocol_hash,
        "core_source_hashes": sources, "core_source_text": source_text, "data_fingerprints": fingerprints,
        "baseline": {"run_dir": baseline.as_posix(), "seed": seed, "test_results_already_available": True,
                     "initial_model_sha256": initial, "files": evidence},
        "quality_extension": {
            "config": config_path.as_posix(), "config_sha256": config_hash,
            "run_dir": run.as_posix(), "updates": cfg["max_updates"], "seed": seed, "smoothing": .1,
            "fresh_initialization": True, "only_training_changes_from_baseline": ["run_dir", "max_updates"],
            "selection": f"full validation greedy BLEU every {cfg['validate_every']} updates; earliest best checkpoint; no test tuning",
            "decode": "reuse frozen seed42 validation protocol for greedy/beam test; report separately from the core matrix",
        },
        "prior_knowledge": {"seed42_long_result_already_known": seed != 42,
                            "additional_replication_after_seed42": seed != 42},
        "limitations": "Baseline test results already exist. For additional seeds the seed-42 long-training result "
                       "is already known. This plan freezes only this new seed's extension before it starts; it is not "
                       "a predeclaration of all historical experiments. Test results cannot select checkpoints, "
                       "change budgets or tune decoding. Few seeds do not establish general or statistically significant superiority.",
    }
    if seed == 42:
        plan["quality_extension"]["only_training_changes_from_A42"] = ["run_dir", "max_updates"]
    exclusive_json(output, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config")
    parser.add_argument("--baseline")
    parser.add_argument("--protocol", default="artifacts/decode_protocol_seed42.json")
    parser.add_argument("--out")
    parser.add_argument("--prepare-config", action="store_true", help="exclusively copy the seed-42 template to a new config")
    args = parser.parse_args()
    if args.prepare_config:
        prepare_quality_config(args.seed, args.config)
    create_plan(args.config, args.protocol, args.out, args.baseline, args.seed)
    print(f"Frozen quality-extension plan: {Path(args.out or quality_paths(args.seed)['plan']).resolve()}")


if __name__ == "__main__":
    main()

