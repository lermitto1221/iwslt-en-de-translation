"""Run one frozen quality extension, with independent state for each seed."""
import argparse
import datetime
import hashlib
from pathlib import Path

import torch

from .common import read_json, sha256, write_json
from .plan_quality import (quality_paths, require_unstarted, validate_extension_config)
from .run_experiments import evaluate_or_reuse, frozen_protocol
from .summary import read_jsonl, require, validate_result, validate_saved_outputs
from .train import (CORE_TRAINING_SOURCES, _validate_resume_sources, check_inputs,
                    model_digest, train)


def verify_baseline_evidence(plan):
    """The original dated seed-42 plan predates explicit baseline evidence."""
    baseline = plan.get("baseline")
    if baseline is None:
        require(plan.get("quality_extension", {}).get("seed") == 42,
                "new seed requires a plan with matching baseline evidence")
        return Path("artifacts/runs/A42")
    files = baseline.get("files")
    require(isinstance(files, dict) and bool(files), "quality plan baseline evidence is missing")
    for filename, expected in files.items():
        path = Path(filename)
        require(path.is_absolute() and path.is_file() and sha256(path) == expected,
                f"baseline evidence changed after the quality plan was frozen: {filename}")
    return Path(baseline["run_dir"])


def verify_frozen_inputs(plan, cfg, seed):
    """Verify the recorded source and actual data before any training or writes."""
    fingerprints = check_inputs(cfg)
    if plan.get("schema_version", 0) >= 2:
        sources, text = plan.get("core_source_hashes"), plan.get("core_source_text")
        require(isinstance(sources, dict) and set(sources) == set(CORE_TRAINING_SOURCES)
                and isinstance(text, dict) and set(text) == set(CORE_TRAINING_SOURCES),
                "quality plan must freeze all seven core sources before training")
        for name in CORE_TRAINING_SOURCES:
            require(isinstance(text[name], str)
                    and hashlib.sha256(text[name].encode("utf-8")).hexdigest() == sources[name]
                    and sha256(Path(__file__).with_name(name)) == sources[name],
                    f"core source changed after plan freeze: {name}")
        require(fingerprints == plan.get("data_fingerprints"),
                "data/tokenizer fingerprints changed after plan freeze")
    else:
        require(seed == 42, "additional seed requires a new plan with frozen sources and data")
    return fingerprints


def verify_initialization(cfg, initial):
    """Check the seed-specific fresh initialization on CPU before allocating CUDA."""
    from .model import TranslationTransformer
    from .tokenizer import load

    torch.set_num_threads(cfg.get("cpu_threads", 4))
    tok = load(cfg["data_dir"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        model = TranslationTransformer(tok.vocab_size(), cfg)
        actual = model_digest(model)
    require(actual == initial, "fresh model initialization differs from the same-seed baseline")
    return actual


def verify_completed_training(run, cfg, baseline_run, fingerprints, plan):
    """Check saved completion, best selection, checkpoints and paired prefix."""
    from .quality_report import read_training

    saved = read_json(run / "config.json")
    require(saved == cfg, "completed quality training does not match the fixed configuration")
    training = read_training(run)
    require(training["status"] == "complete" and training["summary"].get("smoke") is False
            and training["summary"].get("pilot", False) is False,
            "quality training must complete the formal fixed budget")
    linked, baseline_link = read_json(run / "data_link.json"), read_json(baseline_run / "data_link.json")
    require(linked == baseline_link, "long-budget data/initialization controls differ from same-seed baseline")
    current = {name: sha256(Path(__file__).with_name(name)) for name in CORE_TRAINING_SOURCES}
    log_bytes = (run / "train.jsonl").read_bytes().splitlines(keepends=True)
    for name, expected_update in (("last.pt", cfg["max_updates"]),
                                  ("best.pt", training["best_validation"]["update"])):
        state = torch.load(run / name, map_location="cpu", weights_only=False)
        require(state.get("config") == cfg and state.get("fingerprints") == fingerprints
                and state.get("smoke") is False and state.get("pilot", False) is False
                and state.get("update") == expected_update
                and state.get("initial_model_sha256") == linked.get("initial_model_sha256"),
                f"{name}: completed checkpoint differs from frozen run or validation-best selection")
        _validate_resume_sources(state, current, run)
        prefix = b"".join(log_bytes[:expected_update])
        require(state.get("log_sha256") == hashlib.sha256(prefix).hexdigest(),
                f"{name}: training log SHA-256 differs from saved checkpoint")
        require(state.get("best_bleu") == training["summary"]["best_validation_bleu"],
                f"{name}: best validation score differs from completed summary")
        del state
    short = read_jsonl(baseline_run / "train.jsonl", "baseline training prefix")
    long = read_jsonl(run / "train.jsonl", "long training prefix")
    require(len(long) >= len(short), "long training prefix is shorter than baseline")
    numeric = []
    for left, right in zip(short, long):
        for field in ("update", "data_order_sha256", "lr_used", "target_tokens", "sentences"):
            require(field in left and left[field] == right.get(field),
                    f"paired training prefix differs at update {left['update']}: {field}")
        changes = {field: {"baseline": left.get(field), "long": right.get(field)}
                   for field in ("epoch", "loss", "validation_bleu_greedy",
                                 "validation_signature", "validation_samples", "validation_generation_limit_fraction")
                   if left.get(field) != right.get(field)}
        if changes:
            numeric.append({"update": left["update"], "differences": changes})
    old_sources = read_json(baseline_run / "source_hashes.json")
    source_changes = {name: {"baseline": old_sources.get(name), "long": current[name]}
                      for name in CORE_TRAINING_SOURCES if old_sources.get(name) != current[name]}
    require(not (set(source_changes) - {"train.py", "decoding.py"}),
            "baseline has unexpected historical core source differences")
    return {"status": "verified", "prefix_updates": len(short),
            "data_initialization_equal": True, "data_order_lr_token_sentence_equal": True,
            "numeric_difference_count": len(numeric), "numeric_differences": numeric,
            "historical_core_source_differences": source_changes,
            "best_validation": training["best_validation"]}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def run_quality(seed=42, config=None, protocol="artifacts/decode_protocol_seed42.json",
                plan=None, baseline=None):
    paths = quality_paths(seed)
    config = Path(config or paths["config"])
    plan_path = Path(plan or ("artifacts/improvement_plan_2026_10_10.json" if seed == 42 else paths["plan"]))
    declaration = read_json(plan_path)
    extension = declaration["quality_extension"]
    require(extension.get("seed") == seed, "quality plan seed differs from requested seed")
    require(sha256(config) == extension["config_sha256"],
            "quality configuration changed after its budget was fixed")
    require(sha256(protocol) == declaration["protocol_sha256"],
            "decoding protocol changed after the quality experiment was declared")
    baseline_run = verify_baseline_evidence(declaration)
    if baseline is not None:
        require(Path(baseline).resolve() == baseline_run.resolve(), "baseline differs from frozen plan")
    cfg, base_cfg = read_json(config), read_json(baseline_run / "config.json")
    validate_extension_config(cfg, base_cfg, seed)
    run = Path(cfg["run_dir"])
    require(not run.resolve().is_relative_to(baseline_run.resolve())
            and not baseline_run.resolve().is_relative_to(run.resolve()),
            "quality run must not overlap the baseline")
    require(extension.get("updates") == cfg["max_updates"] and extension.get("smoothing") == cfg["label_smoothing"]
            and extension.get("fresh_initialization") is True, "quality plan metadata differs from configuration")
    if extension.get("run_dir") is not None:
        require(Path(extension["run_dir"]).resolve() == run.resolve(), "quality plan run directory differs from configuration")
    fingerprints = verify_frozen_inputs(declaration, cfg, seed)
    linked = read_json(baseline_run / "data_link.json")
    require(all(linked.get(key) == value for key, value in fingerprints.items()),
            "baseline data/tokenizer fingerprints differ from frozen quality inputs")
    for method in ("greedy", "beam"):
        validate_result(read_json(baseline_run / method / "metrics.json"), baseline_run, "A", seed, method)
    verify_initialization(cfg, linked["initial_model_sha256"])
    frozen = frozen_protocol(protocol)
    identity = {"seed": seed, "run_dir": str(run.resolve()), "plan": str(plan_path.resolve()),
                "plan_sha256": sha256(plan_path)}
    # State paths are derived from the explicit seed. Refuse an unrelated prior
    # record instead of silently replacing another experiment's status.
    for key in ("progress", "summary"):
        if paths[key].exists():
            existing = read_json(paths[key])
            if seed != 42 or "seed" in existing:
                require(all(existing.get(name) == value for name, value in identity.items()),
                        f"{key} output belongs to a different frozen experiment")
    # A fully saved run is verified read-only. In particular, the legacy
    # seed-42 global files keep their original timestamps and byte contents.
    preserve_outputs = (paths["summary"].is_file() and paths["progress"].is_file()
                        and read_json(paths["progress"]).get("stage") == "complete"
                        and all((run / method / "metrics.json").is_file() for method in ("greedy", "beam")))

    def status(stage):
        record = {**identity, "stage": stage,
                  "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if not preserve_outputs:
            atomic_json(paths["progress"], record)
        print(record, flush=True)

    completed = run / "summary.json"
    if not (completed.exists() and read_json(completed).get("complete") is True):
        last = run / "last.pt"
        if not last.exists():
            require_unstarted(run, cfg)
        status("training")
        train(cfg, resume=last if last.exists() else None)
        torch.cuda.empty_cache()
    verify_frozen_inputs(declaration, cfg, seed)
    controls = verify_completed_training(run, cfg, baseline_run, fingerprints, declaration)
    results = {}
    for method in ("greedy", "beam"):
        status("test_" + method)
        metrics = evaluate_or_reuse(run / "best.pt", "test", method, run / method,
            beam_size=frozen["beam_size"], alpha=frozen["length_penalty"],
            max_tokens=frozen["max_generation_tokens"])
        validate_result(metrics, run, "A", seed, method)
        require(metrics.get("checkpoint_update") == controls["best_validation"]["update"],
                "test result does not use the earliest best validation checkpoint")
        old = read_json(baseline_run / method / "metrics.json")
        validate_saved_outputs(old, baseline_run / method, context=f"quality baseline A{seed} {method}")
        results[method] = dict(metrics, bleu_difference_from_baseline=metrics["bleu"] - old["bleu"])
        if seed == 42:
            results[method]["bleu_difference_from_A42"] = results[method]["bleu_difference_from_baseline"]
    verify_frozen_inputs(declaration, cfg, seed)
    summary = {**identity, "training": read_json(completed), "results": results,
               "baseline": str(baseline_run.resolve()), "config": cfg, "controls": controls,
               "note": "This seed's frozen longer budget is separate from the 3000-update A/B matrix. "
                       "Full validation selects checkpoints; test never changes training or decoding."}
    if preserve_outputs:
        previous = read_json(paths["summary"])
        require(previous.get("config") == cfg and previous.get("training") == summary["training"],
                "saved quality summary differs from verified training")
        for method in ("greedy", "beam"):
            # New derived comparison keys were not present in the historical
            # summary; all original evaluation fields must still agree.
            evaluation = read_json(run / method / "metrics.json")
            require(all(previous.get("results", {}).get(method, {}).get(key) == value
                        for key, value in evaluation.items()),
                    f"saved quality summary differs from verified {method} evaluation")
            for key in ("bleu_difference_from_baseline", "bleu_difference_from_A42"):
                prior_result = previous.get("results", {}).get(method, {})
                if key in prior_result:
                    require(prior_result[key] == results[method]["bleu_difference_from_baseline"],
                            f"saved quality summary has an inconsistent derived {method} difference")
        if "controls" in previous:
            require(previous["controls"] == controls, "saved quality summary has inconsistent control checks")
        if "baseline" in previous:
            require(Path(previous["baseline"]).resolve() == baseline_run.resolve(),
                    "saved quality summary has an inconsistent baseline")
    else:
        atomic_json(paths["summary"], summary)
    status("complete")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config")
    parser.add_argument("--baseline")
    parser.add_argument("--protocol", default="artifacts/decode_protocol_seed42.json")
    parser.add_argument("--plan")
    args = parser.parse_args()
    run_quality(args.seed, args.config, args.protocol, args.plan, args.baseline)


if __name__ == "__main__":
    main()

