"""Run the predeclared paired experiment, validation-only tuning, and full test."""
import argparse
import datetime
from pathlib import Path

import torch

from .common import read_json, sha256, write_json
from .evaluate import evaluate_checkpoint
from .inference import validate_protocol
from .summary import collect, validate_saved_outputs
from .train import train


def evaluate_or_reuse(checkpoint, split, method, out, beam_size=None, alpha=None, max_tokens=None):
    """Reuse only a complete, verified result for this exact request.

    Incomplete output directories can be retried. A completed but inconsistent
    result is an error: changing an experiment must use a separate directory.
    """
    checkpoint, out = Path(checkpoint), Path(out)
    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        return evaluate_checkpoint(checkpoint, split, method, out, beam_size=beam_size,
                                   alpha=alpha, max_tokens=max_tokens)
    cfg = read_json(checkpoint.parent / "config.json")
    metrics = read_json(metrics_path)
    expected = {"split": split, "method": method, "config": cfg,
                "checkpoint_sha256": sha256(checkpoint),
                "inference_source_sha256": sha256(Path(__file__).with_name("decoding.py")),
                "beam_size": 1 if method == "greedy" else (cfg["beam_size"] if beam_size is None else beam_size),
                "length_penalty": cfg["length_penalty"] if alpha is None else alpha,
                "max_generation_tokens": cfg["max_generation_tokens"] if max_tokens is None else max_tokens,
                "inference_batch_size": cfg.get("decode_batch_size", 16), "inference_precision": "float32"}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f"{out}: saved {key} differs from this evaluation; use a new output directory")
    validate_saved_outputs(metrics, out, context=f"cached {split}/{method}")
    print({"stage": "reuse_verified_evaluation", "directory": str(out)}, flush=True)
    return metrics


def frozen_protocol(path):
    value = read_json(path)
    validate_protocol(value)
    if value.get("selection_split") != "validation" or value["beam_size"] != 4:
        raise ValueError("core experiments require a validation-selected beam=4 protocol")
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol", help="reuse an already frozen common validation protocol; required for extra seeds")
    args = p.parse_args()
    if args.seed != 42 and not args.protocol:
        raise ValueError("additional seeds must reuse --protocol artifacts/decode_protocol_seed42.json")
    configs = {name: read_json(f"experiments/{name}{args.seed}.json") for name in ("A", "B")}
    compare = [{k: v for k, v in cfg.items() if k not in ("run_dir", "label_smoothing")}
               for cfg in configs.values()]
    if compare[0] != compare[1]:
        raise ValueError("A/B configurations differ beyond smoothing and output directory")
    for name, cfg in configs.items():
        if cfg["seed"] != args.seed or cfg["label_smoothing"] != (0.1 if name == "A" else 0.0):
            raise ValueError("experiment labels must match seed and smoothing configuration")
    protocol_path = Path(f"artifacts/decode_protocol_seed{args.seed}.json")
    # Freeze once. Re-entering an interrupted pipeline does not tune on a new
    # protocol after test outputs have become available.
    protocol = frozen_protocol(args.protocol) if args.protocol else None
    if protocol_path.exists():
        saved_protocol = frozen_protocol(protocol_path)
        if protocol is not None and saved_protocol != protocol:
            raise ValueError("saved decoding protocol differs from --protocol; use a separate experiment")
        protocol = saved_protocol
    if protocol is None and any((Path(cfg["run_dir"]) / method / "metrics.json").exists()
                                for cfg in configs.values() for method in ("greedy", "beam")):
        raise ValueError("test results exist but the frozen protocol is missing; restore the original protocol")
    status_path = Path(f"artifacts/pipeline_seed{args.seed}.json")

    def status(stage, **extra):
        record = {"stage": stage, "seed": args.seed,
                  "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
        write_json(status_path, record)
        print(record, flush=True)

    for name, cfg in configs.items():
        run = Path(cfg["run_dir"])
        done = run / "summary.json"
        if (done.exists() and read_json(done).get("complete") and not read_json(done).get("pilot")
                and not read_json(done).get("smoke")):
            if read_json(run / "config.json") != cfg:
                raise ValueError("completed run configuration differs")
            if read_json(done).get("updates") != cfg["max_updates"] or not (run / "best.pt").is_file():
                raise ValueError("completed run is missing its fixed budget or best checkpoint")
            continue
        status("training", condition=name, run_dir=str(run))
        last = run / "last.pt"
        train(cfg, resume=last if last.exists() else None)
        torch.cuda.empty_cache()
    if protocol is None:
        protocol = tune_validation(configs, args.seed, status)
    elif not protocol_path.exists():
        write_json(protocol_path, protocol)
    alpha, max_tokens = protocol["length_penalty"], protocol["max_generation_tokens"]
    for name, cfg in configs.items():
        for method in ("greedy", "beam"):
            status("test", condition=name, method=method)
            evaluate_or_reuse(Path(cfg["run_dir"]) / "best.pt", "test", method,
                Path(cfg["run_dir"]) / method, beam_size=4, alpha=alpha, max_tokens=max_tokens)
    write_json("artifacts/experiment_summary.json", collect("artifacts/runs"))
    status("complete")


def tune_validation(configs, seed, status):
    # This grid and averaging rule are declared before reading any test output.
    validation = {}
    max_tokens = configs["A"]["max_generation_tokens"]
    for name, cfg in configs.items():
        status("validation_greedy", condition=name)
        validation[name] = {}
        validation[name]["greedy"] = evaluate_or_reuse(Path(cfg["run_dir"]) / "best.pt", "validation", "greedy",
            Path(cfg["run_dir"]) / "validation_greedy", max_tokens=max_tokens)
    if max(v["greedy"]["generation_limit_fraction"] for v in validation.values()) > 0.01:
        max_tokens *= 2
    candidate_scores = {}
    for alpha in (0.0, 0.6):
        scores = []
        for name, cfg in configs.items():
            status("validation_beam", condition=name, length_penalty=alpha, max_tokens=max_tokens)
            result = evaluate_or_reuse(Path(cfg["run_dir"]) / "best.pt", "validation", "beam",
                Path(cfg["run_dir"]) / f"validation_beam_alpha{alpha}", beam_size=4, alpha=alpha,
                max_tokens=max_tokens)
            validation[name][f"beam_alpha{alpha}"] = result
            scores.append(result["bleu"])
        candidate_scores[alpha] = sum(scores) / len(scores)
    alpha = max(candidate_scores, key=candidate_scores.get)
    protocol = {"selection_split": "validation", "beam_size": 4, "length_penalty": alpha,
        "max_generation_tokens": max_tokens, "candidate_mean_validation_bleu": candidate_scores,
        "rule": "select shared alpha by mean A/B validation BLEU; tie selects 0.0; double generation cap once if validation greedy cap rate >1%",
        "validation_results": validation}
    write_json(f"artifacts/decode_protocol_seed{seed}.json", protocol)
    return protocol


if __name__ == "__main__":
    main()
