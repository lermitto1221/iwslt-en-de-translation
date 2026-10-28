"""CPU-only evidence/reporting for grouped CV and same-base MT adaptation.

This module never calls training, model loading for generation, or generation.
Incomplete runs remain pending/null. Saved complete results must verify before
they can enter seed aggregates. It writes only the new study's report folder.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from .common import read_json, sha256, write_json
from .cv_evaluate import INFERENCE_SOURCES, _digest, corpus_metrics, verify_results
from .data import read_pairs

SEEDS = (42, 43, 44)
SPLITS = ("validation", "test")
DECODERS = ("greedy", "beam")
GROUP_LABELS = {"source_words": ["≤10词", "11–20词", "21–40词", ">40词"],
    "source_rare_fraction": ["0", "(0,5%]", "(5%,15%]", ">15%"],
    "anchor_fragmentation": ["≤1.25", "(1.25,1.5]", "(1.5,2]", ">2"],
    "cluster": [f"簇{i}" for i in range(5)]}
TITLES = {"cv_final": "CV选参后重训", "direct": "原始OPUS-MT", "lora": "LoRA", "full": "全参数微调"}
STAGE_BUDGETS = {"cv": ("CV候选", 6, 18000, 6, 408570),
    "cv_final": ("CV选择后全训练重训", 3, 9000, 12, 53802),
    "adapt_tune": ("适配学习率调参", 4, 12000, 24, 21312),
    "adapt_final": ("适配正式重复", 6, 18000, 60, 139572)}


def stats(values):
    """Missing is excluded, never zero; n=1 has no sample standard deviation."""
    present = {str(key): float(value) for key, value in values.items() if value is not None}
    if any(not math.isfinite(value) for value in present.values()):
        raise ValueError("nonfinite values cannot enter report statistics")
    array = list(present.values())
    return {"n": len(array), "included": list(present), "per_seed": present,
            "mean": statistics.mean(array) if array else None,
            "sample_std": statistics.stdev(array) if len(array) >= 2 else None, "ddof": 1}


def paired(left, right):
    return stats({str(seed): left[str(seed)] - right[str(seed)] for seed in SEEDS
                  if left.get(str(seed)) is not None and right.get(str(seed)) is not None})


def cv_partition_sensitivity(selection):
    """Describe per-fold preferences without changing the global CV rule."""
    if selection is None:
        return {"schema_version": 1, "status": "pending", "per_fold": [],
            "all_fold_preferences_identical": None, "global_selected_label_smoothing": None,
            "delta_ranges": {metric: {"min": None, "max": None} for metric in ("bleu", "chrf2")}}
    from .cv_select import summarize_candidates
    scores = {tag: item["folds"] for tag, item in selection["candidates"].items()}
    checked = summarize_candidates(scores)
    selected = checked["selected_label_smoothing"]
    if selected != selection["selected_label_smoothing"]:
        raise ValueError("global CV selection does not match its saved fold-score evidence")
    indexed = {tag: {row["fold"]: row for row in rows} for tag, rows in scores.items()}
    records = []
    for fold in range(3):
        zero, smooth = indexed["ls0"][fold], indexed["ls01"][fold]
        if zero["samples"] != smooth["samples"]:
            raise ValueError("fold sensitivity candidates must score the same number of heldout sentences")
        preference = max((0.0, .1), key=lambda value: (
            indexed["ls0" if value == 0 else "ls01"][fold]["bleu"],
            indexed["ls0" if value == 0 else "ls01"][fold]["chrf2"], -value))
        reason = ("higher_BLEU" if zero["bleu"] != smooth["bleu"] else
                  "higher_chrF2_after_exact_BLEU_tie" if zero["chrf2"] != smooth["chrf2"] else
                  "fixed_candidate_order_after_exact_BLEU_and_chrF2_tie")
        records.append({"fold": fold,
            "delta_01_minus_0": {metric: smooth[metric] - zero[metric] for metric in ("bleu", "chrf2")},
            "preferred_label_smoothing": preference, "preference_reason": reason,
            "matches_global_selection": preference == selected})
    return {"schema_version": 1, "status": "complete", "per_fold": records,
        "all_fold_preferences_identical": len({r["preferred_label_smoothing"] for r in records}) == 1,
        "all_fold_preferences_match_global_selection": all(r["matches_global_selection"] for r in records),
        "global_selected_label_smoothing": selected,
        "global_fold_mean_01_minus_0": {metric: checked["candidates"]["ls01"]["summary"][metric]["mean"] -
            checked["candidates"]["ls0"]["summary"][metric]["mean"] for metric in ("bleu", "chrf2")},
        "delta_ranges": {metric: {"min": min(r["delta_01_minus_0"][metric] for r in records),
                                   "max": max(r["delta_01_minus_0"][metric] for r in records)} for metric in ("bleu", "chrf2")},
        "fold_preference_rule": "fold corpus BLEU, then corpus chrF2 on exact BLEU tie, then fixed order [0,0.1] on exact double tie",
        "role": "descriptive partition-sensitivity analysis only; the global selection remains arithmetic three-fold mean BLEU, mean chrF2, fixed order",
        "limits": "three dependent folds at one model seed and3000 updates; small differences do not establish statistical significance or general insensitivity"}


def fmt(value, digits=3):
    return "待完成" if value is None else f"{value:.{digits}f}"


def fmt_stat(value):
    if not value["n"]:
        return "待完成（n=0）"
    if value["n"] == 1:
        return f"{fmt(value['mean'])}（n=1，SD未定义）"
    return f"{fmt(value['mean'])} ± {fmt(value['sample_std'])}（n={value['n']}）"


def default_inputs(root):
    root = Path(root).resolve()
    return {"schema_version": 1, "root": str(root),
        "source_data": str(Path("artifacts/data").resolve()),
        "groups_root": str(Path("artifacts/representation_2026_10_27/groups").resolve()),
        "direct_root": str(Path("artifacts/pretrained_mt_2026_10_27").resolve()),
        "direct_reuse_audit": str(root / "baseline_reuse_audit.json"),
        "direct_backend_addendum": str(root / "baseline_backend_environment_addendum.json"),
        "repetition_analysis_plan": str(root / "repetition_analysis_plan.json"),
        "cv_selection": str(root / "cv" / "selection.json"),
        "cv_final": {str(seed): str(root / "cv" / "final_configs" / f"seed{seed}.json") for seed in SEEDS},
        "adaptation_final": {method: {str(seed): str(root / "adaptation" / "configs" / f"final_{method}_seed{seed}.json")
            for seed in SEEDS} for method in ("lora", "full")},
        "adaptation_tuning": {method: [str(root / "adaptation" / "configs" / f"tune_{method}_lr{lr:g}_seed7.json")
            for lr in ([1e-4, 3e-4] if method == "lora" else [1e-5, 3e-5])] for method in ("lora", "full")},
        "adaptation_selection": str(root / "adaptation" / "selection.json")}


def _pending(config_path, reason="configuration not yet created"):
    return {"status": "pending", "reason": reason, "config_path": str(config_path),
            "training": None, "evaluations": {split: {decoder: None for decoder in DECODERS} for split in SPLITS},
            "validation_curve": [], "selected_checkpoint_exposure": None,
            "examples": {}, "complete_for_aggregation": False}


def selected_checkpoint_exposure(records, selected_update):
    """Exposure of the independently selected weights, not all3000-step cost."""
    if (len(records) != 3000 or [r.get("update") for r in records] != list(range(1, 3001))
            or type(selected_update) is not int or selected_update not in range(500, 3001, 500)):
        raise ValueError("selected checkpoint exposure requires the full3000 log and a declared checkpoint update")
    prefix = records[:selected_update]
    fields = ("sentences", "source_tokens", "target_tokens")
    if any(type(row.get(field)) is not int or row[field] < 0 for row in prefix for field in fields):
        raise ValueError("selected checkpoint prefix exposure is missing or invalid")
    if any(not isinstance(row.get("ids_sha256"), str) for row in prefix):
        raise ValueError("selected checkpoint prefix lacks its effective batch-ID evidence")
    return {"update": selected_update, **{field: sum(row[field] for row in prefix) for field in fields},
        "effective_batch_prefix_sha256": _digest([r["ids_sha256"] for r in prefix]),
        "scope": "committed updates1 through the selected best update, inclusive; sentence exposures are occurrences, not unique sentence coverage",
        "full_training_budget_updates": 3000}


def _training_progress(run):
    path = Path(run) / "progress.json"
    return read_json(path) if path.exists() else None


def collect_cv_final(config_path, seed):
    path = Path(config_path)
    result = _pending(path)
    if not path.exists():
        return result
    cfg = read_json(path)
    if cfg.get("stage") != "final" or cfg.get("seed") != seed or cfg.get("max_updates") != 3000:
        raise ValueError("wrong CV final configuration label/budget")
    run = Path(cfg["run_dir"])
    result.update(config=cfg, progress=_training_progress(run))
    if not (run / "summary.json").exists() or not read_json(run / "summary.json").get("complete"):
        result["reason"] = "training incomplete"
        return result
    from .cv_train import validate_completed_run
    summary = validate_completed_run(cfg)  # CPU-only checkpoint/tensor/log checks.
    result["training_summary_sha256"] = sha256(run / "summary.json")
    result["training"] = {"updates": summary["updates"], "selected_update": 3000,
        "training_seconds": summary["training_seconds_total"], "validation_seconds": 0.0,
        "peak_training_cuda_bytes": summary["peak_cuda_bytes"], "parameter_evidence": {
            "total_parameters": summary["parameters"]["total"], "trainable_parameters": summary["parameters"]["trainable"],
            "trainable_fraction": summary["parameters"]["trainable"] / summary["parameters"]["total"]},
        "sentence_exposures": summary["exposure"]["sentences"], "source_token_exposures": summary["exposure"]["source_tokens"],
        "target_token_exposures": summary["exposure"]["target_tokens"], "best_export_bytes": (run / "last.pt").stat().st_size,
        "export_size_definition": "complete last.pt resume checkpoint including optimizer; not an inference-only model export"}
    for split in SPLITS:
        rows = read_pairs(Path(cfg["data_dir"]) / f"{split}.jsonl")
        for decoder in DECODERS:
            out = run / f"{split}_{decoder}"
            if not (out / "metrics.json").exists():
                continue
            identity = read_json(out / "identity.json")
            if (identity["split"] != split or identity["method"] != decoder
                    or Path(identity["checkpoint"]).resolve() != (run / "last.pt").resolve()
                    or identity["checkpoint_sha256"] != sha256(run / "last.pt")
                    or identity["checkpoint_update"] != 3000 or identity["config_sha256"] != _digest(cfg)
                    or identity["precision"] != "float32" or identity["batch_size"] != cfg["decode_batch_size"]
                    or identity["protocol"] != {key: cfg[key] for key in ("beam_size", "length_penalty", "max_generation_tokens")}
                    or identity["source_hashes"] != {name: sha256(Path(__file__).with_name(name)) for name in INFERENCE_SOURCES}):
                raise ValueError("CV final evaluation belongs to a different checkpoint/split")
            result["evaluations"][split][decoder] = verify_results(out, rows)
            result["examples"][f"{split}/{decoder}"] = read_pairs(out / "examples.jsonl")
    result["complete_for_aggregation"] = all(result["evaluations"][s][d] is not None for s in SPLITS for d in DECODERS)
    result["status"] = "complete" if result["complete_for_aggregation"] else "pending_evaluation"
    result["reason"] = None if result["complete_for_aggregation"] else "one or more complete official evaluation passes pending"
    return result


def collect_adaptation(config_path, method, seed, tuning=False):
    path = Path(config_path)
    result = _pending(path)
    if not path.exists():
        return result
    cfg = read_json(path)
    if cfg.get("method") != method or cfg.get("seed") != seed or cfg.get("updates") != 3000:
        raise ValueError("adaptation run has a wrong method/seed/budget label")
    run = Path(cfg["run_dir"])
    result.update(config=cfg, progress=_training_progress(run))
    if not (run / "summary.json").exists():
        result["reason"] = "training incomplete"
        return result
    # These helpers inspect CPU state only. Do not call train/evaluate_export,
    # which can initialize a model or resume GPU work when evidence is absent.
    import torch
    from .adapt_train import verify_run, _verify_checkpoint, verify_completed, evaluation_context
    from .adapt_data import load_data
    from .pretrained_mt import verify_result
    cfg, plan = verify_run(path)
    manifest, indexed, schedule = load_data(cfg["data_root"], seed)
    del indexed
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    _verify_checkpoint(state, cfg, plan, schedule)
    if state["update"] != 3000:
        raise ValueError("completed adaptation summary has incomplete checkpoint")
    # Context uses the recorded evaluation device category; this is not a GPU call.
    first = run / "validation" / "step_00500" / "metrics.json"
    device = read_json(first)["context"]["device"]
    summary = verify_completed(cfg, plan, state, manifest, device)
    del state, schedule
    result["training"] = summary
    records = read_pairs(run / "train.jsonl")
    result["selected_checkpoint_exposure"] = selected_checkpoint_exposure(records, summary["best"]["update"])
    result["validation_curve"] = [r["validation"] for r in records if "validation" in r]
    if tuning:
        result.update(status="complete", reason=None, complete_for_aggregation=False)
        return result
    for split in SPLITS:
        rows = read_pairs(Path(manifest["data_dir"]) / f"{split}.jsonl")
        for decoder in DECODERS:
            out = run / "evaluation" / split / decoder
            if not (out / "metrics.json").exists():
                continue
            saved = read_json(out / "metrics.json")
            context = evaluation_context(cfg, run, split, decoder, summary["best"]["state_sha256"], saved["context"]["device"])
            result["evaluations"][split][decoder] = verify_result(out, rows, context)
            result["examples"][f"{split}/{decoder}"] = read_pairs(out / "examples.jsonl")
    result["complete_for_aggregation"] = all(result["evaluations"][s][d] is not None for s in SPLITS for d in DECODERS)
    result["status"] = "complete" if result["complete_for_aggregation"] else "pending_evaluation"
    result["reason"] = None if result["complete_for_aggregation"] else "one or more complete official evaluation passes pending"
    return result


def collect_direct(inputs):
    from .pretrained_mt import verify_plan, verify_result
    out = Path(inputs["direct_root"])
    audit_path = Path(inputs["direct_reuse_audit"])
    result = _pending(out, "direct-baseline reuse audit not complete")
    if not audit_path.exists():
        return result
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        result["reason"] = "baseline reuse audit rejected; require a separately rerun fixed baseline"
        return result
    verify_plan(out)
    config = read_json(out / "config.json")
    if sha256(out / "config.json") != audit["config_file_sha256"]:
        raise ValueError("direct baseline changed since this study's reuse audit")
    result["config"] = config
    result["training"] = {"updates": 0, "training_seconds": 0.0, "trainable_parameters": 0,
        "project_adaptation_cost_only": True, "pretraining_cost": "not disclosed; not zero",
        "total_parameters": config["parameters"],
        "model_file_bytes": (out / "model" / "pytorch_model.bin").stat().st_size}
    for split in SPLITS:
        rows = read_pairs(Path(config["data_dir"]) / f"{split}.jsonl")
        for decoder in DECODERS:
            directory = out / split / decoder
            if not (directory / "metrics.json").exists():
                continue
            saved = read_json(directory / "metrics.json")
            context = {"plan_sha256": sha256(out / "plan.json"), "config_sha256": sha256(out / "config.json"),
                "split": split, "method": decoder, "batch_size": config["batch_size"],
                "input": read_json(out / "inputs.json")["splits"][split], "generation": config["generation"][decoder],
                "precision": config["precision"], "device": saved["context"]["device"]}
            result["evaluations"][split][decoder] = verify_result(directory, rows, context)
            result["examples"][f"{split}/{decoder}"] = read_pairs(directory / "examples.jsonl")
    result["complete_for_aggregation"] = all(result["evaluations"][s][d] is not None for s in SPLITS for d in DECODERS)
    result.update(status="complete" if result["complete_for_aggregation"] else "pending_evaluation", reason=None,
                  reuse_audit_sha256=sha256(audit_path), fixed_model_count=1)
    return result


def group_scores(membership, examples):
    members = membership["rows"]
    if [e["id"] for e in examples] != [m["id"] for m in members]:
        raise ValueError("grouping and prediction ID order differ")
    for example, member in zip(examples, members):
        for field, key in (("source", "source_sha256"), ("reference", "reference_sha256")):
            if hashlib.sha256(example[field].encode("utf-8")).hexdigest() != member[key]:
                raise ValueError("fixed grouping source/reference hash mismatch")
    output = {}
    for category in members[0]["groups"]:
        output[category] = []
        for label in range(len(GROUP_LABELS[category])):
            chosen = [e for e, m in zip(examples, members) if m["groups"][category] == label]
            scores = corpus_metrics([e["prediction"] for e in chosen], [e["reference"] for e in chosen]) if chosen else {"bleu": None, "chrf2": None}
            output[category].append({"group": label, "label": GROUP_LABELS[category][label], "n": len(chosen),
                "ids": [e["id"] for e in chosen], "small": 0 < len(chosen) < 30, **scores})
    return output


def aggregate_runs(runs):
    output = {}
    for split in SPLITS:
        output[split] = {}
        for decoder in DECODERS:
            output[split][decoder] = {}
            for method in ("cv_final", "lora", "full"):
                eligible = {seed: run for seed, run in runs[method].items() if run["complete_for_aggregation"]}
                output[split][decoder][method] = {"included_seeds": [int(s) for s in eligible], "n": len(eligible),
                    "excluded_pending_seeds": [s for s in SEEDS if str(s) not in eligible],
                    **{metric: stats({seed: run["evaluations"][split][decoder][metric] for seed, run in eligible.items()})
                       for metric in ("bleu", "chrf2")}}
            output[split][decoder]["paired_lora_minus_full"] = {metric: paired(
                output[split][decoder]["lora"][metric]["per_seed"], output[split][decoder]["full"][metric]["per_seed"])
                for metric in ("bleu", "chrf2")}
            direct = runs["direct"]["evaluations"][split][decoder]
            output[split][decoder]["direct"] = {"fixed_model_count": 1 if direct else 0,
                "bleu": direct["bleu"] if direct else None, "chrf2": direct["chrf2"] if direct else None,
                "sample_std": None, "note": "one fixed model, not three training seeds"}
            for method in ("lora", "full"):
                output[split][decoder][method + "_minus_direct"] = {metric: stats({
                    seed: value - direct[metric] for seed, value in output[split][decoder][method][metric]["per_seed"].items()}
                    if direct else {}) for metric in ("bleu", "chrf2")}
    return output


def aggregate_groups(runs, memberships):
    result = {}
    for split in SPLITS:
        result[split] = {}
        for decoder in DECODERS:
            computed = {}
            if runs["direct"]["evaluations"][split][decoder] is not None:
                computed["direct"] = group_scores(memberships[split], runs["direct"]["examples"][f"{split}/{decoder}"])
            for method in ("cv_final", "lora", "full"):
                for seed, run in runs[method].items():
                    if run["complete_for_aggregation"]:
                        computed[f"{method}/{seed}"] = group_scores(memberships[split], run["examples"][f"{split}/{decoder}"])
            categories = {}
            for category, labels in GROUP_LABELS.items():
                if category not in memberships[split]["rows"][0]["groups"]:
                    continue
                categories[category] = []
                for label, text in enumerate(labels):
                    ids = [row["id"] for row in memberships[split]["rows"] if row["groups"][category] == label]
                    entry = {"group": label, "label": text, "n": len(ids), "ids": ids, "small": 0 < len(ids) < 30,
                        "methods": {}, "paired_lora_minus_full": {}}
                    for method in ("cv_final", "lora", "full"):
                        entry["methods"][method] = {metric: stats({str(seed): computed[f"{method}/{seed}"][category][label][metric]
                            for seed in SEEDS if f"{method}/{seed}" in computed}) for metric in ("bleu", "chrf2")}
                    direct = computed.get("direct", {}).get(category)
                    entry["methods"]["direct"] = {metric: direct[label][metric] if direct else None for metric in ("bleu", "chrf2")}
                    entry["paired_lora_minus_full"] = {metric: paired(entry["methods"]["lora"][metric]["per_seed"],
                        entry["methods"]["full"][metric]["per_seed"]) for metric in ("bleu", "chrf2")}
                    for method in ("lora", "full"):
                        entry[method + "_minus_direct"] = {metric: stats({s: v - entry["methods"]["direct"][metric]
                            for s, v in entry["methods"][method][metric]["per_seed"].items()})
                            if entry["methods"]["direct"][metric] is not None else stats({}) for metric in ("bleu", "chrf2")}
                    categories[category].append(entry)
            result[split][decoder] = categories
    return result


def fixed_samples(runs, memberships):
    import sacrebleu
    metric = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2)
    result = {}
    for split in SPLITS:
        members = memberships[split]["rows"]
        ids = ([row["id"] for row in members if row["id"] % 400 == 0] if split == "test" else sorted(
            i for cluster in range(5) for i in sorted(r["id"] for r in members if r["groups"]["cluster"] == cluster)[:3]))
        examples_by_condition = {}
        for decoder in DECODERS:
            key = f"{split}/{decoder}"
            if key in runs["direct"]["examples"]:
                examples_by_condition[f"direct/{decoder}"] = {e["id"]: e for e in runs["direct"]["examples"][key]}
            for method in ("cv_final", "lora", "full"):
                for seed, run in runs[method].items():
                    if key in run["examples"]:
                        examples_by_condition[f"{method}/{seed}/{decoder}"] = {e["id"]: e for e in run["examples"][key]}
        chosen = []
        for identifier in ids:
            actual = next((values[identifier] for values in examples_by_condition.values()), None)
            if actual is None:
                continue
            predictions = {}
            for condition, values in examples_by_condition.items():
                example = values[identifier]
                if (example["source"], example["reference"]) != (actual["source"], actual["reference"]):
                    raise ValueError("fixed example conditions do not share source/reference")
                words = example["prediction"].split()
                triples = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
                predictions[condition] = {"text": example["prediction"],
                    "sentence_chrf2_proxy": metric.sentence_score(example["prediction"], [example["reference"]]).score,
                    "maximum_repeated_trigram_count": max(triples.values(), default=0),
                    "source_digit_strings": re.findall(r"\d+(?:[.,]\d+)*", example["source"]),
                    "prediction_digit_strings": re.findall(r"\d+(?:[.,]\d+)*", example["prediction"])}
            chosen.append({"id": identifier, "source": actual["source"], "reference": actual["reference"], "predictions": predictions})
        result[split] = {"planned_ids": ids, "available_ids": [e["id"] for e in chosen], "examples": chosen,
            "selection_rule": "raw ID % 400 == 0" if split == "test" else "lowest three IDs per frozen old validation cluster",
            "review_status": "automatic proxy diagnostics; semantic success/failure requires explicitly labeled agent review, not independent human annotation"}
    return result


def repeated_trigrams(text):
    """Frozen descriptive proxy: case-sensitive whitespace, punctuation kept."""
    if not isinstance(text, str):
        raise ValueError("repetition analysis requires actual saved text")
    words = text.split()
    counts = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    maximum = max(counts.values(), default=0)
    return {"flag": maximum >= 3, "maximum_trigram_occurrences": maximum}


def generation_diagnostic_ids(examples, metrics, native_batches=None):
    """Recover IDs only where per-sentence or complete-batch evidence permits."""
    ids = [e["id"] for e in examples]
    if len(ids) != len(set(ids)) or len(ids) != metrics["samples"]:
        raise ValueError("generation diagnostic sample IDs are incomplete or duplicated")
    if native_batches is None:
        if any(type(e.get("generation_limit")) is not bool or type(e.get("unfinished")) is not bool for e in examples):
            raise ValueError("CV generation diagnostics require saved per-example booleans")
        limits = [e["id"] for e in examples if e["generation_limit"]]
        unfinished = [e["id"] for e in examples if e["unfinished"]]
        unknown, confirmed_unfinished = [], unfinished
        definition = "saved CV generation_limit/unfinished per-example flags; generated-token budget excludes BOS"
    else:
        if [i for batch in native_batches for i in batch["ids"]] != ids:
            raise ValueError("native generation journal IDs differ from verified examples")
        maximum = metrics["context"]["generation"]["max_length"]
        limits, confirmed_unfinished, unknown = [], [], []
        for batch in native_batches:
            batch_ids, lengths, count = batch["ids"], batch["generated_lengths"], batch["unfinished"]
            if (len(lengths) != len(batch_ids) or any(type(length) is not int or length < 1 for length in lengths)
                    or type(count) is not int or not 0 <= count <= len(batch_ids)):
                raise ValueError("native generation-length/unfinished evidence is invalid")
            hits = [identifier for identifier, length in zip(batch_ids, lengths) if length + 1 >= maximum]
            if len(hits) != batch["limit_hits"]:
                raise ValueError("native limit-hit count does not match saved per-ID generated lengths")
            limits.extend(hits)
            if count == len(batch_ids):
                confirmed_unfinished.extend(batch_ids)
            elif count:
                unknown.append({"unfinished_count": count, "candidate_ids": batch_ids})
        unfinished = None if unknown else confirmed_unfinished
        definition = "native generated_length+1 >= max_length; +1 is decoder-start token; unfinished exact IDs cannot be inferred from a nonzero partial-batch count alone"
    unfinished_count = len(confirmed_unfinished) + sum(batch["unfinished_count"] for batch in unknown)
    if len(limits) != metrics["generation_limit_hits"] or unfinished_count != metrics["unfinished_generations"]:
        raise ValueError("generation diagnostic counts differ from strictly verified corpus metrics")
    return {"generation_limit": {"count": len(limits), "ids": limits},
        "unfinished": {"count": unfinished_count, "ids": unfinished,
            "confirmed_ids": confirmed_unfinished, "unresolved_candidate_batches": unknown,
            "exact_ids_available": not unknown}, "definition": definition}


def corpus_repetition_diagnostics(examples, membership, metrics, generation=None):
    if metrics is None:
        return {"status": "pending", "samples": None,
            **{name: {"count": None, "ids": None} for name in ("source", "reference", "prediction", "prediction_only")},
            "generation_limit": {"count": None, "ids": None}, "unfinished": {"count": None, "ids": None}}
    members = membership["rows"]
    ids = [row["id"] for row in members]
    if (metrics["samples"] != len(members) or [e["id"] for e in examples] != ids
            or len(ids) != len(set(ids)) or not ids):
        raise ValueError("repetition diagnostics require complete ordered original IDs")
    flags = {name: [] for name in ("source", "reference", "prediction", "prediction_only")}
    maxima = {name: 0 for name in ("source", "reference", "prediction")}
    for example, member in zip(examples, members):
        for field, key in (("source", "source_sha256"), ("reference", "reference_sha256")):
            if hashlib.sha256(example[field].encode("utf-8")).hexdigest() != member[key]:
                raise ValueError("repetition source/reference identity differs from frozen full-split membership")
        evidence = {field: repeated_trigrams(example[field]) for field in maxima}
        for field, item in evidence.items():
            maxima[field] = max(maxima[field], item["maximum_trigram_occurrences"])
            if item["flag"]:
                flags[field].append(example["id"])
        if evidence["prediction"]["flag"] and not evidence["source"]["flag"] and not evidence["reference"]["flag"]:
            flags["prediction_only"].append(example["id"])
    if generation is None:
        raise ValueError("complete diagnostics require verified generation-limit/unfinished evidence")
    return {"status": "complete", "samples": len(ids), "ids_sha256": _digest(ids),
        **{field: {"count": len(values), "ids": values,
            **({"maximum_trigram_occurrences": maxima[field]} if field in maxima else {})} for field, values in flags.items()},
        **generation, "interpretation": "descriptive proxy, not a semantic-error label or causal effect; source/reference repetition is retained as a control"}


def collect_generation_diagnostics(root, runs, memberships, inputs):
    path = Path(inputs.get("repetition_analysis_plan", Path(root) / "repetition_analysis_plan.json"))
    if not path.exists():
        return {"status": "plan_pending", "plan": None, "evaluations": []}
    plan = read_json(path)
    if (plan["original_plan_sha256"] != sha256(Path(root) / "plan.json")
            or plan["ngram_order"] != 3 or plan["overlapping"] is not True or plan["flag_threshold_occurrences"] != 3
            or plan["token_definition"] != "Python str.split() whitespace tokens; case-sensitive; punctuation preserved; no lowercasing, stemming or token rewriting"):
        raise ValueError("repetition diagnostic plan differs from the independently frozen specification")
    for name, digest in plan["inputs"].items():
        if sha256(name) != digest:
            raise ValueError("repetition analysis input fingerprint changed")
    evaluations = []
    conditions = [("direct", None, runs["direct"])] + [(method, int(seed), run) for method in ("cv_final", "lora", "full") for seed, run in runs[method].items()]
    for method, seed, run in conditions:
        for split in SPLITS:
            for decoder in DECODERS:
                metric = run["evaluations"][split][decoder]
                if metric is None:
                    result = corpus_repetition_diagnostics(None, None, None)
                else:
                    examples = run["examples"][f"{split}/{decoder}"]
                    if method == "direct":
                        directory = Path(inputs["direct_root"]) / split / decoder
                    elif method == "cv_final":
                        directory = Path(run["config"]["run_dir"]) / f"{split}_{decoder}"
                    else:
                        directory = Path(run["config"]["run_dir"]) / "evaluation" / split / decoder
                    if method == "cv_final":
                        generation = generation_diagnostic_ids(examples, metric)
                    else:
                        journal = directory / "batches.jsonl"
                        if sha256(journal) != metric["file_sha256"]["batches.jsonl"]:
                            raise ValueError("native repetition analysis journal changed after strict verification")
                        generation = generation_diagnostic_ids(examples, metric, read_pairs(journal))
                    result = corpus_repetition_diagnostics(examples, memberships[split], metric, generation)
                    result["evidence"] = {"evaluation_directory": str(directory.resolve()),
                        "files": {name: sha256(directory / name) for name in ("metrics.json", "examples.jsonl", "predictions.txt", "references.txt")},
                        "fixed_membership_sha256": sha256(Path(inputs["groups_root"]) / f"{split}.json")}
                    if method != "cv_final":
                        result["evidence"]["files"]["batches.jsonl"] = sha256(directory / "batches.jsonl")
                evaluations.append({"condition": method, "seed": seed, "split": split, "decoder": decoder, **result})
    done = sum(item["status"] == "complete" for item in evaluations)
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": sha256(__file__), "status": "complete" if done == len(evaluations) else "pending",
        "plan": {"path": str(path.resolve()), "sha256": sha256(path), "created_at_utc": plan["created_at_utc"]},
        "planned_complete_evaluations": len(evaluations), "available_complete_evaluations": done,
        "evaluations": evaluations, "definition": plan["repetition_flag"], "token_definition": plan["token_definition"],
        "selection_role": "supplementary description only, no effect on rank/LR/checkpoint/smoothing/budget/generation settings"}


def verify_adaptation_pair(left, right):
    """Both methods must consume the same data; their selected LRs may differ."""
    fields = ("seed", "model_dir", "data_root", "updates", "train_precision", "micro_batch_size",
              "effective_batch_size", "warmup_updates", "validation_every", "eval_batch_size", "generation")
    if any(k not in left["config"] or k not in right["config"] or left["config"][k] != right["config"][k] for k in fields):
        raise ValueError("same-seed adaptation conditions differ in a shared controlled setting")
    records = [read_pairs(Path(run["config"]["run_dir"]) / "train.jsonl") for run in (left, right)]
    if any(len(rows) != 3000 for rows in records):
        raise ValueError("same-seed adaptation pair lacks 3000 complete updates")
    columns = ("update", "ids_sha256", "microbatch_sha256", "sentences", "source_tokens", "target_tokens")
    if any(any(a.get(k) != b.get(k) or k not in a or k not in b for k in columns) for a, b in zip(*records)):
        raise ValueError("LoRA/full consumed different effective batches or token exposures")
    return {"verified": True, "updates": 3000, "same_base": True, "same_precision": True,
            "same_effective_ids_microbatch_boundaries_and_tokens": True,
            "comparison_scope": "the complete3000-step trajectories; independently selected best checkpoints may have different prefix exposures",
            "selected_checkpoint_updates": {"lora": left.get("selected_checkpoint_exposure", {}).get("update") if left.get("selected_checkpoint_exposure") else None,
                "full": right.get("selected_checkpoint_exposure", {}).get("update") if right.get("selected_checkpoint_exposure") else None},
            "selected_checkpoint_exposure_equal": (left["selected_checkpoint_exposure"] == right["selected_checkpoint_exposure"])
                if left.get("selected_checkpoint_exposure") and right.get("selected_checkpoint_exposure") else None,
            "learning_rate": "method-specific value chosen from equal-sized validation searches"}


def verify_adaptation_selection(selection, tuning):
    if selection.get("complete") is not True or selection.get("test_used") is not False:
        raise ValueError("adaptation selection is incomplete or used test evidence")
    from .adapt_train import select_learning_rate
    for path, digest in selection["input_hashes"].items():
        if sha256(path) != digest:
            raise ValueError("adaptation selection input hash differs")
    selected = {}
    for method, rates in (("lora", [1e-4, 3e-4]), ("full", [1e-5, 3e-5])):
        if any(run["status"] != "complete" for run in tuning[method]):
            raise ValueError("adaptation LR selection lacks complete candidate runs")
        candidates = []
        for run in tuning[method]:
            cfg, best = run["config"], run["training"]["best"]
            candidates.append({"method": method, "lr": cfg["lr"], "seed": 7, "updates": 3000, "split": "validation",
                "bleu": best["bleu"], "chrf2": best["chrf2"], "checkpoint_update": best["update"], "run": cfg["run_dir"]})
        if candidates != selection["candidates"][method]:
            raise ValueError("saved LR candidate scores differ from verified training summaries")
        selected[method] = select_learning_rate(candidates, rates)["lr"]
    if selected != selection["selected_lrs"]:
        raise ValueError("selected learning rates differ from predeclared validation rule")
    return selected


def cv_costs(root, selection):
    """Count completed CV work even before all six runs permit selection."""
    if selection:
        return [{"name": f"fold{row['fold']}_{tag}", "training_complete": True, "updates": 3000,
                 "training_seconds": row["training_seconds"], "evaluations": [{"samples": row["samples"],
                    "elapsed_seconds": row["heldout_decode_seconds"]}]} for tag, item in selection["candidates"].items() for row in item["folds"]]
    costs = []
    from .cv_train import validate_completed_run
    from .cv_select import validate_candidate_config, validate_evaluation_identity
    for fold in range(3):
        for value, tag in ((0.0, "ls0"), (.1, "ls01")):
            path = Path(root) / "cv" / "configs" / f"fold{fold}_{tag}.json"
            row = {"name": path.stem, "training_complete": False, "updates": None,
                   "training_seconds": None, "evaluations": []}
            if path.exists():
                cfg = read_json(path)
                data, run = validate_candidate_config(cfg, root, fold, value)
                if (run / "summary.json").exists() and read_json(run / "summary.json").get("complete"):
                    summary = validate_completed_run(cfg)
                    row.update(training_complete=True, updates=summary["updates"], training_seconds=summary["training_seconds_total"])
                    out = run / "holdout_greedy"
                    if (out / "metrics.json").exists():
                        validate_evaluation_identity(read_json(out / "identity.json"), cfg, run, data)
                        metric = verify_results(out, read_pairs(data / "holdout.jsonl"))
                        row["evaluations"].append({"samples": metric["samples"], "elapsed_seconds": metric["elapsed_seconds"]})
            costs.append(row)
    return costs


def stage_costs(cv_rows, runs, tuning):
    """Completed saved work only, excluding partial runs and projected times."""
    normalized = {"cv": cv_rows, "cv_final": [], "adapt_tune": [], "adapt_final": []}
    for stage, entries in (("cv_final", list(runs["cv_final"].values())),
        ("adapt_tune", [run for values in tuning.values() for run in values]),
        ("adapt_final", [run for method in ("lora", "full") for run in runs[method].values()])):
        for run in entries:
            training = run["training"]
            evaluations = [{"samples": metric["samples"], "elapsed_seconds": metric["elapsed_seconds"]}
                for split in SPLITS for decoder in DECODERS if (metric := run["evaluations"][split][decoder]) is not None]
            if training and stage in ("adapt_tune", "adapt_final"):
                for point in run["validation_curve"]:
                    # Verified complete run requires exactly six full888 passes.
                    evaluations.append({"samples": 888, "elapsed_seconds": None, "checkpoint_selection": True})
            normalized[stage].append({"training_complete": training is not None,
                "updates": training["updates"] if training else None,
                "training_seconds": training["training_seconds"] if training else None,
                "checkpoint_selection_seconds": training.get("validation_seconds", 0.0) if training else None,
                "evaluations": evaluations})
    result = {}
    for stage, (title, budget_runs, budget_updates, budget_passes, budget_sentences) in STAGE_BUDGETS.items():
        rows = normalized[stage]
        done = [row for row in rows if row["training_complete"]]
        evals = [e for row in rows for e in row["evaluations"]]
        measured = [e["elapsed_seconds"] for e in evals if e["elapsed_seconds"] is not None]
        selection_times = [row.get("checkpoint_selection_seconds", 0.0) for row in done]
        result[stage] = {"title": title, "planned_runs": budget_runs, "planned_optimizer_updates": budget_updates,
            "completed_training_runs": len(done), "completed_run_updates": sum(row["updates"] for row in done),
            "training_seconds_completed_runs": sum(row["training_seconds"] for row in done) if done else None,
            "planned_full_evaluation_passes": budget_passes, "planned_evaluation_sentences": budget_sentences,
            "completed_full_evaluation_passes": len(evals), "evaluated_sentences": sum(e["samples"] for e in evals),
            "evaluation_seconds": (sum(measured) + sum(selection_times)) if evals else None,
            "checkpoint_selection_seconds": sum(selection_times) if any(e.get("checkpoint_selection") for e in evals) else None,
            "cost_scope": "only completed verified runs and completed full evaluation passes; ongoing partial work not estimated or replaced by zero"}
    return result


def gpu_observations(path):
    path = Path(path)
    if not path.exists():
        return {"observations": 0, "temperature_celsius": None, "power_watts": None}
    values, skipped = [], 0
    raw = path.read_bytes()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                skipped += 1
                continue
            raise ValueError("GPU observation has corrupt committed JSON") from None
        if record.get("nvidia_smi_exit_code") != 0:
            skipped += 1
            continue
        parts = next(csv.reader([record["values"]], skipinitialspace=True))
        if len(parts) != 7:
            skipped += 1
            continue
        try:
            utilization, memory, temperature, power, clocks = map(float, parts[2:])
        except ValueError:
            skipped += 1
            continue
        if not all(math.isfinite(v) for v in (utilization, memory, temperature, power, clocks)):
            skipped += 1
            continue
        values.append({"observed_at_utc": record["observed_at_utc"], "stage": record.get("stage"),
            "temperature": temperature, "power": power, "utilization": utilization, "memory": memory, "clocks": clocks})
    def extent(key):
        return {"min": min(row[key] for row in values), "max": max(row[key] for row in values)} if values else None
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "observations": len(values), "unusable_or_partial": skipped,
        "first_observed_at_utc": values[0]["observed_at_utc"] if values else None,
        "last_observed_at_utc": values[-1]["observed_at_utc"] if values else None,
        "temperature_celsius": extent("temperature"), "power_watts": extent("power"),
        "utilization_percent": extent("utilization"), "memory_used_mib": extent("memory"), "sm_clock_mhz": extent("clocks"),
        "limitation": "sparse passive observations, not controlled or continuously sampled conditions; not a causal explanation of speed differences"}


def collect_recovery_incidents(root):
    """Metadata/file-hash checks only; never deserialize or rebuild a model."""
    root = Path(root).resolve()
    incidents = []
    for path in sorted((root / "recovery_incidents").glob("*/incident.json")):
        value = read_json(path)
        if value["frozen_study_plan_sha256"] != sha256(root / "plan.json"):
            raise ValueError("recovery incident belongs to a different frozen study plan")
        committed, logged, replay = (value[k] for k in ("checkpoint_update", "complete_log_updates", "uncommitted_updates_to_replay"))
        if (any(type(v) is not int or v < 0 for v in (committed, logged, replay))
                or logged - committed != replay):
            raise ValueError("recovery committed/logged/replay update counts disagree")
        for key in ("uncommitted_sentences", "uncommitted_source_tokens", "uncommitted_target_tokens"):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError("recovery exposure count is invalid")
        seconds = value["uncommitted_step_seconds_recorded"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("recovery discarded-step timing is invalid")
        for name, item in value["archived_files"].items():
            archive = path.parent / name
            if archive.stat().st_size != item["bytes"] or sha256(archive) != item["sha256"]:
                raise ValueError("archived recovery evidence changed: " + name)
        controller = read_json(path.parent / "controller_progress_before_resume.json")
        old_attempt = Path(controller["log"]).with_suffix(".json").resolve()
        if not old_attempt.is_relative_to(root / "execution"):
            raise ValueError("recovery incident points outside this study's execution directory")
        replay_path = path.parent / "replay_verification.json"
        replay_verification = {"status": "pending", "path": str(replay_path), "sha256": None, "evidence": None}
        if replay_path.exists():
            replay_verification = {"status": "record_available", "path": str(replay_path), "sha256": sha256(replay_path),
                                   "evidence": read_json(replay_path)}
        incidents.append({"path": str(path.resolve()), "sha256": sha256(path),
            "observed_at_utc": value["observed_at_utc"], "reported_cause": value["cause"],
            "lost_tool_sessions": value.get("lost_tool_sessions", []), "native_run": value["native_run"],
            "previous_completed_formal_runs": value.get("previous_completed_formal_runs"),
            "affected_attempt": str(old_attempt), "checkpoint_update": committed, "complete_log_updates": logged,
            "recorded_discarded_updates": replay, "recorded_discarded_step_seconds": seconds,
            "recorded_discarded_sentences": value["uncommitted_sentences"],
            "recorded_discarded_source_tokens": value["uncommitted_source_tokens"],
            "recorded_discarded_target_tokens": value["uncommitted_target_tokens"],
            "captured_exit_code": None, "exact_process_end_time": None,
            "checkpoint_validation_basis": "incident's saved CPU verification and archived-file fingerprints; reporting does not load model checkpoints",
            "replay_verification": replay_verification})
    return {"incidents": incidents, "incident_count": len(incidents),
        "recorded_discarded_updates_lower_bound": sum(i["recorded_discarded_updates"] for i in incidents),
        "recorded_discarded_step_seconds_lower_bound": sum(i["recorded_discarded_step_seconds"] for i in incidents),
        "recorded_discarded_sentences": sum(i["recorded_discarded_sentences"] for i in incidents),
        "recorded_discarded_source_tokens": sum(i["recorded_discarded_source_tokens"] for i in incidents),
        "recorded_discarded_target_tokens": sum(i["recorded_discarded_target_tokens"] for i in incidents),
        "unlogged_additional_work": "unknown, not estimated", "formal_net_update_budget_unchanged": 57000,
        "interpretation": "discarded pre-interruption work is additional computation outside the final net committed57000 updates; replay is not a new candidate, seed or training-budget expansion; missing exit/end records do not prove a process is still running"}


def execution_attempts(root, recovery=None):
    """Actual command intervals, kept separate from overlapping kernel timings."""
    root = Path(root)
    recovery = collect_recovery_incidents(root) if recovery is None else recovery
    known_interrupted = {str(Path(item["affected_attempt"]).resolve()): item for item in recovery["incidents"]}
    records = []
    for path in sorted((root / "execution").glob("*/attempt_*.json")):
        record = read_json(path)
        started = datetime.fromisoformat(record["started_at_utc"].replace("Z", "+00:00"))
        ended = datetime.fromisoformat(record["ended_at_utc"].replace("Z", "+00:00")) if record.get("ended_at_utc") else None
        seconds = (ended - started).total_seconds() if ended else None
        if seconds is not None and seconds < 0:
            raise ValueError("execution attempt ended before it began")
        log = path.with_suffix(".log")
        incident = known_interrupted.get(str(path.resolve())) if ended is None else None
        activity = "interrupted_without_captured_exit_or_end_time" if incident else "open_record_process_status_not_determined"
        if ended:
            if not log.exists() or sha256(log) != record["log_sha256"]:
                raise ValueError("completed execution attempt log is missing or changed")
            events = []
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            if any("update" in event and ("loss" in event or "run" in event) for event in events):
                activity = "optimizer_updates_observed"
            elif any("decode_progress" in event or ("evaluation" in event and "completed" in event) for event in events):
                activity = "generation_batches_observed"
            elif record.get("exit_code") == 0 and record["attempt"] > 1:
                activity = "repeat_command_without_update_or_generation_progress; may be strict read-only reuse/verification"
            else:
                activity = "control_or_verification_without_logged_optimizer_or_generation_progress"
        records.append({"path": str(path.resolve()), "sha256": sha256(path), "name": record["name"],
            "attempt": record["attempt"], "started_at_utc": record["started_at_utc"], "ended_at_utc": record.get("ended_at_utc"),
            "wall_seconds": seconds, "exit_code": record.get("exit_code"), "observed_activity": activity,
            "interruption_incident": {"path": incident["path"], "sha256": incident["sha256"]} if incident else None})
    done = [record for record in records if record["wall_seconds"] is not None]
    interrupted = [record for record in records if record["interruption_incident"] is not None]
    return {"attempts": records, "finished_attempts": len(done), "unfinished_attempts": len(records) - len(done) - len(interrupted),
        "interrupted_attempts_without_exit": len(interrupted), "unclosed_attempt_records": len(records) - len(done),
        "failed_attempts": sum(record["exit_code"] != 0 for record in done),
        "summed_finished_command_wall_seconds": sum(record["wall_seconds"] for record in done) if done else None,
        "repeat_attempts": sum(record["attempt"] > 1 for record in records),
        "time_definition": "sum of finished command end-start intervals; includes startup, checks, I/O and any repeated read-only verification. Known interrupted attempts with no captured end/exit remain untimed, not marked running. Other unclosed records do not establish live process state. Not pure optimizer/generation time; do not add to stage compute times or treat repeat attempts as new experimental runs."}


def resource_costs(root):
    root = Path(root)
    rows = []
    for path in sorted((root / "adaptation").glob("resource*.json")):
        value = read_json(path)
        for method, entry in value.get("methods", {}).items():
            if "optimizer_updates" not in entry:
                continue
            if entry["optimizer_updates"] != len(entry.get("step_results", [])):
                raise ValueError("resource optimizer-update count differs from saved step records")
            rows.append({"file": str(path.resolve()), "method": method, "optimizer_updates": entry["optimizer_updates"],
                "seconds": entry.get("elapsed_seconds"), "status": entry.get("status"), "sha256": sha256(path)})
    pilot_path = root / "resources" / "cv_pilot" / "pilot.json"
    if pilot_path.exists():
        value = read_json(pilot_path)
        if value.get("formal") is not False or value["extra_optimizer_updates"] != value["training"]["updates"]:
            raise ValueError("CV resource pilot is mislabeled")
        rows.append({"file": str(pilot_path.resolve()), "method": "cv_pilot", "optimizer_updates": value["extra_optimizer_updates"],
            "seconds": value["training"]["training_seconds_total"], "status": "completed_resource_probe", "sha256": sha256(pilot_path)})
    return {"optimizer_updates": sum(r["optimizer_updates"] for r in rows), "attempts": rows,
        "formal": False, "cost_scope": "measured resource optimizer updates only; excluded from 19 formal runs/57000 updates; failed attempts with no published update count are retained separately without invented counts"}


def historical_context(root):
    root = Path(root)
    paths = {"long_aggregate": Path("artifacts/quality_multiseed_report/aggregate.json"),
             "public_seed42_beam": Path("artifacts/runs/quality_A42_12000/beam/metrics.json"),
             "representation_report": Path("artifacts/representation_2026_10_27/report/report.md"),
             "historical_course_report": Path("artifacts/course_report_2026_10_27/report.md")}
    manifest = read_json(root / "historical_manifest.json")
    for path in paths.values():
        expected = manifest["files"].get(path.as_posix())
        if expected is None or sha256(path) != expected["sha256"]:
            raise ValueError("historical report reference changed from pre-execution fingerprint")
    aggregate = read_json(paths["long_aggregate"])
    public = read_json(paths["public_seed42_beam"])
    return {"long_budget_updates": 12000, "included_seeds": aggregate["included_seeds"],
        "long_beam": aggregate["groups"]["long_beam"], "public_seed42_beam_bleu": public["bleu"],
        "sources": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()},
        "scope": "previously completed historical references; not new fixed3000 runs and not used to choose this study's settings"}


def summarize_reproduction_audit(audit, root, seed):
    root = Path(root).resolve()
    project = Path(__file__).resolve().parents[1]
    if (audit.get("schema_version") != 1 or audit.get("seed") != seed
            or Path(audit.get("new_study_root", "")).resolve() != root
            or Path(audit.get("new_run", "")).resolve() != root / "cv" / "final_runs" / f"seed{seed}"
            or Path(audit.get("old_run", "")).resolve() != project / "artifacts" / "runs" / f"A{seed}"):
        raise ValueError("historical reproduction audit belongs to a different seed/run/study")
    model, tokenizer, execution = audit["model"], audit["tokenizer"], audit["independent_execution"]
    if (type(model.get("all_tensors_bitwise_equal")) is not bool
            or type(model.get("parameter_tensors_compared")) is not int or model["parameter_tensors_compared"] < 1
            or model.get("checkpoint_updates") != {"old": 3000, "new": 3000}
            or type(tokenizer.get("all_ID_piece_score_triples_equal")) is not bool
            or type(tokenizer["normalizer"].get("effective_equal")) is not bool):
        raise ValueError("historical reproduction tensor/tokenizer evidence is incomplete")
    if model["all_tensors_bitwise_equal"] and (model["tensor_differences"]
            or model["model_tensor_sha256"]["old"] != model["model_tensor_sha256"]["new"]):
        raise ValueError("historical reproduction tensor-equality evidence is inconsistent")
    evaluations = audit["evaluations"]
    if (len(evaluations) != 4 or {(r["split"], r["method"]) for r in evaluations} != {
            (split, decoder) for split in SPLITS for decoder in DECODERS}
            or any(type(r.get("old_prediction_available")) is not bool for r in evaluations)):
        raise ValueError("historical reproduction must distinguish all four evaluation availability states")
    available = [r for r in evaluations if r["old_prediction_available"]]
    if any(type(r.get("raw_ID_source_reference_prediction_alignment_equal")) is not bool for r in available):
        raise ValueError("historical prediction equality is missing or mislabeled")
    equal = [r for r in available if r["raw_ID_source_reference_prediction_alignment_equal"]]
    test_equal = [r for r in equal if r["split"] == "test"]
    fresh = (execution.get("started_without_resume_argument") is True
             and execution.get("new_continuous_optimizer_log_records") == 3000
             and execution.get("new_training_command", {}).get("exit_code") == 0)
    return {"seed": seed, "status": "audited", "model_tensors_compared": model["parameter_tensors_compared"],
        "all_tensors_bitwise_equal": model["all_tensors_bitwise_equal"],
        "tensor_differences": model["tensor_differences"],
        "tokenizer_ID_piece_score_equal": tokenizer["all_ID_piece_score_triples_equal"],
        "tokenizer_effective_normalizer_equal": tokenizer["normalizer"]["effective_equal"],
        "tokenizer_raw_model_bytes_equal": tokenizer["model_file_bytes_equal"],
        "trainer_spec_effective_equal": tokenizer["trainer_spec"]["effective_equal"],
        "trainer_spec_differences": tokenizer["trainer_spec"]["serialized_field_differences"],
        "available_historical_evaluation_groups": len(available), "aligned_equal_evaluation_groups": len(equal),
        "all_four_predictions_compared_and_equal": len(equal) == 4,
        "both_test_predictions_compared_and_equal": len(test_equal) == 2,
        "evaluations": [{key: row[key] for key in ("split", "method", "old_prediction_available", "samples", "bleu_delta",
            "raw_ID_source_reference_prediction_alignment_equal", "note") if key in row} for row in evaluations],
        "fresh_executed_training_evidence": fresh,
        "old_validation_checkpoints": len(audit["training_logs"].get("old_validation_checkpoints", [])),
        "new_validation_selected_checkpoints": audit["training_logs"].get("new_validation_selected_checkpoints"),
        "same_endpoint_and_test_predictions": model["all_tensors_bitwise_equal"] and len(test_equal) == 2,
        "limitations": audit["limitations"], "completed_at_utc": audit["completed_at_utc"]}


def collect_reproduction_audits(root):
    root = Path(root)
    rows = []
    for seed in SEEDS:
        path = root / "cv" / f"historical_reproduction_seed{seed}.json"
        if not path.exists():
            rows.append({"seed": seed, "status": "pending", "audit_path": str(path.resolve()), "audit_sha256": None})
            continue
        audit = read_json(path)
        item = summarize_reproduction_audit(audit, root, seed)
        if not audit.get("input_fingerprints"):
            raise ValueError("historical reproduction audit has no input evidence")
        for name, digest in audit["input_fingerprints"].items():
            if sha256(name) != digest:
                raise ValueError("historical reproduction audit input changed: " + name)
        rows.append({**item, "audit_path": str(path.resolve()), "audit_sha256": sha256(path),
            "input_file_count": len(audit["input_fingerprints"])})
    return {"schema_version": 1, "seeds": rows, "audited_seeds": [r["seed"] for r in rows if r["status"] == "audited"],
        "pending_seeds": [r["seed"] for r in rows if r["status"] == "pending"],
        "pooling_policy": "historical and new runs with the same seeds are not pooled into n=6; endpoint/prediction identity is reproduction evidence, not additional quality improvement",
        "scope": "only existing per-seed CPU audit evidence is summarized; no other seed is inferred from equal scores"}


def collect(root, inputs):
    root = Path(root)
    design = read_json(root / "design.json")
    group_audit = read_json(root / "data" / "cv" / "group_audit.json")
    selection_path = Path(inputs["cv_selection"])
    cv_selection = None
    if selection_path.exists():
        cv_selection = read_json(selection_path)
        for path, digest in cv_selection["input_hashes"].items():
            if sha256(path) != digest:
                raise ValueError("CV selection input evidence changed")
        from .cv_select import summarize_candidates, merge_oof
        recalculated = summarize_candidates({tag: candidate["folds"] for tag, candidate in cv_selection["candidates"].items()})
        if recalculated["selected_label_smoothing"] != cv_selection["selected_label_smoothing"]:
            raise ValueError("saved CV selection differs from its declared fold-score rule")
        # Saved OOF text must independently reproduce its metrics and original IDs.
        population = read_pairs(Path(inputs["source_data"]) / "train.jsonl")
        for tag in ("ls0", "ls01"):
            out = root / "cv" / "oof" / tag
            examples = read_pairs(out / "examples.jsonl")
            merged, metrics = merge_oof(population, [[e for e in examples if e["fold"] == fold] for fold in range(3)])
            if merged != examples:
                raise ValueError("OOF saved order differs")
            saved = read_json(out / "metrics.json")
            if any(saved[k] != v for k, v in metrics.items()) or any(sha256(out / name) != digest for name, digest in saved["artifacts"].items()):
                raise ValueError("OOF saved scores/text fingerprints differ")
    runs = {"cv_final": {str(seed): collect_cv_final(inputs["cv_final"][str(seed)], seed) for seed in SEEDS},
            "direct": collect_direct(inputs)}
    for method in ("lora", "full"):
        runs[method] = {str(seed): collect_adaptation(inputs["adaptation_final"][method][str(seed)], method, seed) for seed in SEEDS}
    tuning = {method: [collect_adaptation(path, method, 7, tuning=True) for path in inputs["adaptation_tuning"][method]]
              for method in ("lora", "full")}
    selection = Path(inputs["adaptation_selection"])
    adapt_selection = read_json(selection) if selection.exists() else None
    if adapt_selection:
        selected_lrs = verify_adaptation_selection(adapt_selection, tuning)
        for method in ("lora", "full"):
            for run in runs[method].values():
                if run.get("config") and run["config"]["lr"] != selected_lrs[method]:
                    raise ValueError("formal adaptation did not use its validation-selected learning rate")
    elif any(run["training"] for method in ("lora", "full") for run in runs[method].values()):
        raise ValueError("formal adaptation training exists without a completed LR selection")
    for run in runs["cv_final"].values():
        if run["training"] and (cv_selection is None or run["config"]["label_smoothing"] != cv_selection["selected_label_smoothing"]):
            raise ValueError("CV final training lacks its corresponding completed smoothing selection")
    paired_controls = {}
    for seed in map(str, SEEDS):
        left, right = runs["lora"][seed], runs["full"][seed]
        paired_controls[seed] = verify_adaptation_pair(left, right) if left["training"] and right["training"] else {"verified": False, "status": "pending"}
    # Check only the explicitly bound generation dictionary, precision, batch,
    # and device category. This does not assert complete backend equivalence.
    for method in ("lora", "full"):
        for run in runs[method].values():
            for split in SPLITS:
                for decoder in DECODERS:
                    actual, direct = run["evaluations"][split][decoder], runs["direct"]["evaluations"][split][decoder]
                    if actual and direct and any(actual["context"].get(k) != direct["context"].get(k)
                            for k in ("generation", "precision", "batch_size", "device")):
                        raise ValueError("adapted/direct formal generation conditions differ")
    memberships = {split: read_json(Path(inputs["groups_root"]) / f"{split}.json") for split in SPLITS}
    for split, membership in memberships.items():
        if sha256(Path(inputs["source_data"]) / f"{split}.jsonl") != membership["source_file_sha256"]:
            raise ValueError("official data differs from frozen historical group membership")
    aggregates = aggregate_runs(runs)
    groups = aggregate_groups(runs, memberships)
    samples = fixed_samples(runs, memberships)
    generation_diagnostics = collect_generation_diagnostics(root, runs, memberships, inputs)
    resource_records = []
    for path in sorted((root / "adaptation").glob("resource*.json")):
        value = read_json(path)
        if "methods" in value:
            resource_records.append({"path": str(path.resolve()), "sha256": sha256(path), "passed": value.get("passed"),
                "methods": {method: {k: v for k, v in item.items() if k in ("status", "error", "error_type", "peak_cuda_bytes",
                    "steady_mean_step_seconds", "learned_export_reload_logits_bitwise_equal", "zero_adapter_equivalence")}
                    for method, item in value["methods"].items()}})
    complete = cv_selection is not None and adapt_selection is not None and all(run["complete_for_aggregation"] for method in ("cv_final", "lora", "full") for run in runs[method].values()) and runs["direct"]["complete_for_aggregation"] and all(r["status"] == "complete" for rows in tuning.values() for r in rows)
    costs = stage_costs(cv_costs(root, cv_selection), runs, tuning)
    history = historical_context(root)
    reproduction = collect_reproduction_audits(root)
    recovery = collect_recovery_incidents(root)
    regression_records = []
    for path in root.glob("regression*.json"):
        item = read_json(path)
        log = path.with_suffix(".log")
        if "tests_run" in item and log.exists() and item.get("log_sha256") == sha256(log):
            regression_records.append({"path": str(path.resolve()), "sha256": sha256(path), **item})
    backend_path = Path(inputs.get("direct_backend_addendum", root / "baseline_backend_environment_addendum.json"))
    backend = None
    if backend_path.exists():
        addendum = read_json(backend_path)
        if (addendum["parent_audit_sha256"] != sha256(inputs["direct_reuse_audit"])
                or addendum["frozen_plan_sha256"] != sha256(root / "plan.json")):
            raise ValueError("baseline backend addendum belongs to a different reuse audit or frozen plan")
        backend = {"path": str(backend_path.resolve()), "sha256": sha256(backend_path),
            "historical_CUBLAS_WORKSPACE_CONFIG": addendum["historical_CUBLAS_WORKSPACE_CONFIG"],
            "not_established": addendum["not_established"],
            "demonstrated_backend_value_mismatch": addendum["demonstrated_backend_value_mismatch"],
            "report_text_zh": addendum["recommended_report_text_zh"],
            "reuse_decision": addendum["decision"]}
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
        "status": "complete" if complete else "pending", "design": design, "cv_group_audit": group_audit,
        "cv_selection": cv_selection, "cv_partition_sensitivity": cv_partition_sensitivity(cv_selection),
        "adaptation_selection": adapt_selection, "adaptation_paired_controls": paired_controls,
        "execution_plan_frozen": (root / "plan.json").is_file(),
        "execution_plan_frozen_at_utc": read_json(root / "plan.json").get("frozen_at_utc") if (root / "plan.json").is_file() else None,
        "runs": {method: ({k: v for k, v in values.items() if k != "examples"} if method == "direct" else
            {seed: {k: v for k, v in run.items() if k != "examples"} for seed, run in values.items()}) for method, values in runs.items()},
        "adaptation_tuning": tuning, "aggregate": aggregates, "groups": groups, "samples": samples,
        "generation_diagnostics": generation_diagnostics, "generation_diagnostics_file": "generation_diagnostics.json",
        "resource_attempts": resource_records, "inputs": inputs,
        "stage_costs": costs, "resource_costs": resource_costs(root), "gpu_observations": gpu_observations(root / "gpu_observations.jsonl"),
        "execution_attempts": execution_attempts(root, recovery), "historical_context": history,
        "recovery_incidents": recovery,
        "historical_reproduction": reproduction,
        "executed_regression_records": regression_records,
        "baseline_backend_environment": backend,
        "input_sha256": {"design": sha256(root / "design.json"), "group_audit": sha256(root / "data" / "cv" / "group_audit.json"),
            **({"baseline_backend_environment_addendum": backend["sha256"]} if backend else {}),
            **({"repetition_analysis_plan": generation_diagnostics["plan"]["sha256"]} if generation_diagnostics["plan"] else {}),
            **{f"historical_reproduction_seed{item['seed']}": item["audit_sha256"] for item in reproduction["seeds"] if item["status"] == "audited"},
            **{f"recovery_incident_{index}": item["sha256"] for index, item in enumerate(recovery["incidents"])},
            **{f"replay_verification_{index}": item["replay_verification"]["sha256"] for index, item in enumerate(recovery["incidents"]) if item["replay_verification"]["sha256"] is not None},
            **{f"groups_{split}": sha256(Path(inputs["groups_root"]) / f"{split}.json") for split in SPLITS}},
        "aggregation_policy": "only a completed verified training seed with all four complete official evaluations enters seed statistics; absent values remain null; direct is one fixed model"}
    return result


def _figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "pdf.fonttype": 42})
    paths = []

    def finish(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = out / (name + "." + suffix)
            fig.savefig(path, dpi=160, bbox_inches="tight")
            paths.append(path.name)
        plt.close(fig)

    audit = result["cv_group_audit"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    folds = audit["folds"]
    axes[0].bar([f"折{r['fold']}" for r in folds], [r["holdout_pairs"] for r in folds], color="#4472a3")
    axes[0].set_ylabel("完整保留折句对数")
    axes[0].set_title("分组CV：句对数接近平衡")
    for i, row in enumerate(folds):
        axes[0].text(i, row["holdout_pairs"], f"{row['holdout_pairs']:,}", ha="center", va="bottom")
    axes[0].set_ylim(0, max(r["holdout_pairs"] for r in folds) * 1.15)
    axes[1].bar([f"折{r['fold']}" for r in folds], [r["components"] for r in folds], color="#ca8b39")
    axes[1].set_ylabel("不可拆分连通分量数")
    axes[1].set_title("句数平衡不代表群体组成相同")
    finish(fig, "cv_group_balance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("bleu", "chrf2")):
        selection = result["cv_selection"]
        if selection is None:
            axis.text(.5, .5, "CV正式结果待完成", transform=axis.transAxes, ha="center")
            axis.set_axis_off()
            continue
        for tag, candidate in selection["candidates"].items():
            axis.plot([r["fold"] for r in candidate["folds"]], [r[metric] for r in candidate["folds"]], marker="o", label=f"平滑={candidate['label_smoothing']}")
        axis.set_xticks([0, 1, 2], ["折0", "折1", "折2"])
        axis.set_ylabel(metric)
        axis.legend()
        axis.set_title("CV保留折Greedy，固定3000更新")
    finish(fig, "cv_candidate_scores")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for axis, metric in zip(axes, ("bleu", "chrf2")):
        condition = result["aggregate"]["test"]["beam"]
        for i, method in enumerate(("cv_final", "direct", "lora", "full")):
            if method == "direct":
                mean, std = condition[method][metric], None
                points = []
            else:
                item = condition[method][metric]
                mean, std = item["mean"], item["sample_std"]
                points = list(item["per_seed"].values())
            if mean is None:
                axis.text(i, .05, "待完成", transform=axis.get_xaxis_transform(), ha="center", color="#888888")
                continue
            axis.scatter([i] * len(points), points, s=24, color="#6291ad", zorder=3)
            axis.errorbar([i], [mean], yerr=[std] if std is not None else None, fmt="D", color="#173d5e", capsize=5)
        axis.set_xticks(range(4), ["CV重训", "直接推理\n固定模型", "LoRA", "全参"])
        axis.set_xlim(-.5, 3.5)
        axis.set_ylabel(metric)
        axis.set_title("完整测试 Beam；误差条为种子样本SD")
        axis.grid(axis="y", alpha=.2)
    finish(fig, "test_quality")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, method in zip(axes, ("lora", "full")):
        found = False
        for run in result["adaptation_tuning"][method]:
            if run["validation_curve"]:
                curve = run["validation_curve"]
                axis.plot([p["update"] for p in curve], [p["bleu"] for p in curve], "--o", markersize=3,
                    label=f"调参7 lr={run['config']['lr']:g}")
                found = True
        for seed, run in result["runs"][method].items():
            curve = run["validation_curve"]
            if curve:
                axis.plot([p["update"] for p in curve], [p["bleu"] for p in curve], "-o", markersize=3, label=f"正式seed{seed}")
                found = True
        axis.set_title(TITLES[method] + "完整验证Greedy")
        axis.set_xlabel("优化器更新")
        axis.set_ylabel("corpus BLEU")
        if found:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "待完成", ha="center", transform=axis.transAxes)
    finish(fig, "validation_curves")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cost_fields = (("training_seconds", 60, "训练经过时间 / 分钟"), ("peak_training_cuda_bytes", 2**20, "峰值已分配训练显存 / MiB"))
    for axis, (field, scale, label) in zip(axes, cost_fields):
        for i, method in enumerate(("cv_final", "lora", "full")):
            values = [run["training"][field] / scale for run in result["runs"][method].values()
                      if run["training"] and run["training"].get(field) is not None]
            if values:
                axis.scatter([i] * len(values), values, color=["#4472a3", "#d09537", "#5b8b67"][i])
                axis.plot([i - .14, i + .14], [statistics.mean(values)] * 2, color="black")
            else:
                axis.text(i, .05, "待完成", transform=axis.get_xaxis_transform(), ha="center", color="#888888")
        axis.set_xticks(range(3), ["CV重训", "LoRA", "全参"])
        axis.set_xlim(-.5, 2.5)
        axis.set_ylabel(label)
    finish(fig, "training_cost")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, category in zip(axes, ("source_words", "source_rare_fraction", "anchor_fragmentation")):
        rows = result["groups"]["test"]["beam"][category]
        any_values = False
        for method, color in (("lora", "#d09537"), ("full", "#5b8b67")):
            indices, means, errors = [], [], []
            for i, row in enumerate(rows):
                item = row[method + "_minus_direct"]["bleu"]
                if item["mean"] is not None:
                    indices.append(i); means.append(item["mean"]); errors.append(item["sample_std"])
            if means:
                axis.plot(indices, means, "o-", label=TITLES[method], color=color)
                for i, mean, error in zip(indices, means, errors):
                    if error is not None:
                        axis.errorbar(i, mean, yerr=error, capsize=3, color=color)
                any_values = True
        if not any_values:
            axis.text(.5, .5, "适配分组结果待完成", transform=axis.transAxes, ha="center")
        axis.axhline(0, color="#999999", linewidth=.7)
        axis.set_xticks(range(len(rows)), [row["label"] + f"\nn={row['n']}" for row in rows], fontsize=8)
        axis.set_title(category)
        axis.set_ylabel("Beam BLEU Δ（适配−直接）")
        if any_values:
            axis.legend(fontsize=8)
    finish(fig, "fixed_group_deltas")
    return paths


def _write_examples(result, out):
    lines = ["# 固定句子集合的实际译文与代理诊断", "",
        "全部示例遵守预先固定ID规则；不因新模型成绩替换句子。以下句级chrF2只用作定位参考字面重合变化的代理，不平均后充当corpus指标，也不等同语义准确度。重复三元组和数字集合差异只提示复核，不能自动证明漏译、实体错误或否定反转。", "",
        "本文件由程序汇集实际输出。若另附sample_review.md，其中的语义判断应明确标为代理复核，不能写成独立人工标注。", ""]
    for split in SPLITS:
        item = result["samples"][split]
        lines += [f"## {split}", "", f"规则：{item['selection_rule']}；计划{len(item['planned_ids'])}句，当前有输出{len(item['available_ids'])}句。", ""]
        if not item["examples"]:
            lines += ["待完成。", ""]
        for example in item["examples"]:
            lines += [f"### ID {example['id']}", "", f"源句：{example['source']}", "", f"参考：{example['reference']}", ""]
            for condition, pred in example["predictions"].items():
                lines += [f"- **{condition}**：{pred['text']}  ",
                    f"  句级chrF2代理={pred['sentence_chrf2_proxy']:.2f}；最大重复三元组次数={pred['maximum_repeated_trigram_count']}。"]
            lines += [""]
    (out / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(result, out):
    a = result["cv_group_audit"]
    status = "完成" if result["complete"] else "进行中：缺失结果标为待完成，不计作零分"
    frozen = f"正式执行计划于{result['execution_plan_frozen_at_utc']}（UTC）冻结" if result["execution_plan_frozen"] else "候选设计及分组方案已保存，正式执行计划待冻结"
    lines = ["# 分组交叉验证调参与同底座适配研究", "", f"报告状态：**{status}**。生成时间（UTC）：{result['generated_at_utc']}。", "",
        f"本轮在已知历史测试结果后规划。{frozen}；规则要求在查看本轮新测试成绩前冻结设置与选择规则。CV和官方验证用于选择，完整官方测试用于报告。历史15次从零训练、表示对照和预训练直接推理已完成，未将它们重新列为待执行。", "",
        "## 1. 问题：标签平滑选择是否对训练划分敏感？", "",
        "### 方法", "",
        "仅在官方204,285句清洗训练集内部做三折分组CV，固定模型seed42与3,000更新，候选0、0.1。每折独立拟合BPE8k；同折候选共享分词器、随机初始化、训练ID及每次更新微批边界。完整保留折只用于Greedy评分；固定取第3,000步，不根据它选择checkpoint。", "",
        f"通过原始带标签archive与Parquet逐行对齐，核验{a['raw_pairs_verified']:,}句并恢复{a['talks']:,}个演讲。分组策略：`{a['strategy']}`；跨折同规范化源句={a['cross_fold_normalized_source_keys']}，跨折演讲={a['cross_fold_talk_ids']}。", "",
        f"最大连通分量{a['selected_grouping']['largest_component']:,}句，占{a['selected_grouping']['largest_component_fraction']:.2%}。规范化仅作用于分组键（NFKC、casefold、合并空白、保留标点），翻译原文不改。", "",
        "| 折 | 折训练数（过滤前） | 完整保留数 | 组件数 | 演讲数 | 源空格词均值 |", "|---|---:|---:|---:|---:|---:|"]
    for row in a["folds"]:
        lines.append(f"| {row['fold']} | {row['train_pairs_before_length_filter']} | {row['holdout_pairs']} | {row['components']} | {row['talks_present']} | {row['mean_source_whitespace_words']:.2f} |")
    lines += ["", "![CV分组平衡](cv_group_balance.png)", "",
        "句对数平衡不代表组件组成或语言分布相同。演讲与重复源句之间的连通关系会形成大组件；严格保持组件完整，因此折间差异同时包含不同演讲和内容组成。", "", "### 结果与选择", ""]
    selection = result["cv_selection"]
    if selection is None:
        lines += ["六组CV训练及完整保留折推理尚未全部形成已验证选择记录；标签平滑选择待完成。", ""]
    else:
        lines += ["| 平滑 | 三折Greedy BLEU均值±折间SD | 三折chrF2均值±折间SD | OOF corpus BLEU | OOF corpus chrF2 |", "|---|---:|---:|---:|---:|"]
        for candidate in selection["candidates"].values():
            mean = candidate["summary"]
            lines.append(f"| {candidate['label_smoothing']} | {mean['bleu']['mean']:.3f} ± {mean['bleu']['sample_std']:.3f} | {mean['chrf2']['mean']:.3f} ± {mean['chrf2']['sample_std']:.3f} | {candidate['oof']['bleu']:.3f} | {candidate['oof']['chrf2']:.3f} |")
        sensitivity = result["cv_partition_sensitivity"]
        wins = [str(row["preferred_label_smoothing"]) for row in sensitivity["per_fold"]]
        lines += ["", f"全局三折均值规则选择标签平滑 **{selection['selected_label_smoothing']}**。各折描述性偏好依次为：{', '.join(wins)}。", "",
            "主指标为三折corpus BLEU的算术平均；仅完全同分时按平均chrF2、预先候选顺序决胜。OOF按完整原始训练ID顺序拼接后重算，不用于改变主选择。这里SD是折间差异，三折共享部分训练数据，不是三个随机种子的SD。", ""]
        lines += ["| 折 | ΔBLEU（0.1−0） | ΔchrF2（0.1−0） | 局部偏好 | 决胜依据 |", "|---|---:|---:|---:|---|"]
        for row in sensitivity["per_fold"]:
            reason = {"higher_BLEU": "BLEU更高", "higher_chrF2_after_exact_BLEU_tie": "BLEU完全同分后chrF2更高",
                      "fixed_candidate_order_after_exact_BLEU_and_chrF2_tie": "两指标完全同分后固定顺序"}[row["preference_reason"]]
            lines.append(f"| {row['fold']} | {row['delta_01_minus_0']['bleu']:+.6f} | {row['delta_01_minus_0']['chrf2']:+.6f} | {row['preferred_label_smoothing']} | {reason} |")
        ranges = sensitivity["delta_ranges"]
        agreement = "三折局部偏好一致" if sensitivity["all_fold_preferences_identical"] else "三折局部偏好不一致，存在划分相关变化"
        lines += ["", f"对研究问题A的直接回答：在本次固定模型seed42与3,000更新预算下，**{agreement}**。0.1−0的折内BLEU差值范围为[{ranges['bleu']['min']:+.6f}, {ranges['bleu']['max']:+.6f}]，chrF2差值范围为[{ranges['chrf2']['min']:+.6f}, {ranges['chrf2']['max']:+.6f}]；全局仍按三折均值选择{selection['selected_label_smoothing']}。", "",
            "各折偏好也依次使用BLEU→chrF2→固定候选顺序，仅用于描述敏感性，不替代全局选择。即使三折偏好一致，也只能说明在这三个具体划分上观测到方向一致；小差值不能证明统计显著、普遍不敏感，或其他预算下同样成立。", ""]
    lines += ["![CV候选结果](cv_candidate_scores.png)", "", "### 解释与局限", "",
        "CV得到的是3,000更新下的选参证据；不能据此宣称12,000更新的最优参数，也不能将用于选择的CV成绩包装成独立最终确认。三个正式重训种子与三折分工不同：前者测量固定官方划分下的初始化/顺序变化，后者检验训练划分敏感性。", "",
        "## 2. 问题：同一底座上，LoRA与全参适配的收益和代价如何？", "",
        "### 方法", "",
        "底座固定为Helsinki-NLP/opus-mt-en-de，revision `6183067f769a302e3861815543b9f312c71b0ca4`。原始参数74,410,496；保留原生分词器、PAD58100、decoder start58100、EOS0及完整生成设置。LoRA为r8/alpha16/dropout0/bias none，覆盖36个q_proj/v_proj；全参更新全部可学习底座权重，保留原生无梯度正弦位置表。实际可训练数量以各运行证据为准。LoRA正式评测使用未合并的adapter，包含额外A/B投影算子；较少的可训练参数不保证更快推理。", "",
        "先以训练样本检查前后向、更新、显存及零adapter输出一致性；两方法在完整3,000步训练轨迹上使用共同有效批次64、相同原始ID顺序和微批边界，完整预算为每次192,000句次。句次不是唯一句子覆盖数。调参seed7：LoRA学习率1e-4/3e-4，全参1e-5/3e-5，每候选3,000更新，每500更新完整888句Greedy验证。运行内按BLEU、chrF2、最早checkpoint取最佳；学习率之间按最佳BLEU、chrF2、预先候选顺序决胜。正式42/43/44每次重新加载原始底座，各3,000更新。", "",
        "被评测的best权重只经历截至其所选步数的训练前缀；若LoRA和全参的最佳步不同，两个所选模型累计见到的句次/token也不同。“共同暴露”只约束完整3,000步轨迹，不能误述为两个最佳模型看到了相同的全部训练句子。这是预先声明的独立验证选点方案的一部分，不是因显存差异改变有效批次。", "",
        "正式解码统一FP32、原生Greedy/Beam4、batch16、max_length512（含decoder起始token）、length_penalty1、forced_eos0。强制EOS结束不代表没有触限或重复。直接推理是一个固定模型；经过本轮输入、模型与原生分词器、生成/加载源码、依赖、FP32、batch16和完整GenerationConfig审计后，只读复用历史四组推理及原计时。", "",
        "历史直接推理进程未记录 `CUBLAS_WORKSPACE_CONFIG`，其值未知；未记录不能推断为当时未设置，也不能推断为与适配进程不同。适配入口的seed_all仅在变量不存在时默认设为`:4096:8`，继承值则保留。因此不声称全部后端环境相同，也未证明历史输出可以逐位重现。已核验身份仍支持本轮只读复用，这项缺失属于比较与计时的残余限制，不是已观察到的配置差异。见[后端环境补充审计](../baseline_backend_environment_addendum.json)。", "",
        "### 调参记录", "", "| 方法 | 学习率 | 状态 | 最佳验证步 | 最佳Greedy BLEU | chrF2 |", "|---|---:|---|---:|---:|---:|"]
    for method, candidates in result["adaptation_tuning"].items():
        for run in candidates:
            cfg = run.get("config", {})
            best = run["training"]["best"] if run["training"] else {}
            lines.append(f"| {TITLES[method]} | {cfg.get('lr', '待完成')} | {run['status']} | {best.get('update', '待完成')} | {fmt(best.get('bleu'))} | {fmt(best.get('chrf2'))} |")
    if result["adaptation_selection"] is not None:
        chosen = result["adaptation_selection"]["selected_lrs"]
        lines += ["", f"按已冻结验证规则选定学习率：LoRA **{chosen['lora']:g}**，全参 **{chosen['full']:g}**；正式重复从原始底座重新开始，不继续训练调参模型。", ""]
    lines += ["", "### 正式重复的所选checkpoint及其实际训练前缀", "",
        "| 方法/seed | best更新 | best验证Greedy BLEU | best验证chrF2 | 所选前缀句次 | 所选前缀源/目标token | 完整3000步句次 |", "|---|---:|---:|---:|---:|---|---:|"]
    for method in ("lora", "full"):
        for seed, run in result["runs"][method].items():
            prefix = run["selected_checkpoint_exposure"]
            if prefix is None:
                lines.append(f"| {TITLES[method]}/{seed} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
                continue
            best = run["training"]["best"]
            lines.append(f"| {TITLES[method]}/{seed} | {prefix['update']} | {best['bleu']:.3f} | {best['chrf2']:.3f} | {prefix['sentences']:,} | {prefix['source_tokens']:,}/{prefix['target_tokens']:,} | {run['training']['sentence_exposures']:,} |")
    lines += ["", "前缀暴露从完整已核验日志的第1步累加至best.update（含该步），同时保存前缀批次ID摘要；不按最后一步192,000句次替代所选模型实际暴露。学习率调参运行也在机器结果中保存同样字段。训练成本仍计实际完成的全部3,000步及其验证工作，选择较早best不会抹去后续已经发生的计算。", ""]
    lines += ["", "![完整验证学习曲线](validation_curves.png)", "", "## 3. 完整官方评测与种子重复", "",
        "只有训练及验证/测试×Greedy/Beam四组评测均完整、核验通过的seed进入均值。缺失值为null，不作零分。BLEU为大小写敏感13a，chrF2为字符6阶、word0、beta2；逐组保存完整签名。", "",
        "| 划分/解码 | 方法 | BLEU | chrF2 | 纳入seed／固定模型数 |", "|---|---|---:|---:|---|"]
    for split in SPLITS:
        for decoder in DECODERS:
            condition = result["aggregate"][split][decoder]
            for method in ("cv_final", "direct", "lora", "full"):
                item = condition[method]
                if method == "direct":
                    lines.append(f"| {split}/{decoder} | {TITLES[method]} | {fmt(item['bleu'])} | {fmt(item['chrf2'])} | {item['fixed_model_count']}个固定模型 |")
                else:
                    lines.append(f"| {split}/{decoder} | {TITLES[method]} | {fmt_stat(item['bleu'])} | {fmt_stat(item['chrf2'])} | {item['included_seeds']} |")
    lines += ["", "![测试质量](test_quality.png)", "", "### 同seed LoRA减全参的配对差", "",
        "| 划分/解码 | ΔBLEU均值±配对SD | ΔchrF2均值±配对SD |", "|---|---:|---:|"]
    for split in SPLITS:
        for decoder in DECODERS:
            delta = result["aggregate"][split][decoder]["paired_lora_minus_full"]
            lines.append(f"| {split}/{decoder} | {fmt_stat(delta['bleu'])} | {fmt_stat(delta['chrf2'])} |")
    history = result["historical_context"]
    historical_beam = history["long_beam"]["bleu"]
    lines += ["", "先计算同seed差，再求均值和ddof=1样本SD，不相减两个SD；n=3不能直接作为统计显著。官方验证曾用于选择适配学习率和checkpoint，其分数用于诊断，不是独立确认。CV重训与历史12,000更新成绩分开呈现，不能混合成同预算排名。", "",
        "### 已完成历史实验的独立参照", "",
        f"历史12,000更新的三种子Beam BLEU为 **{historical_beam['mean']:.3f} ± {historical_beam['sample_std']:.3f}**；公开seed42模型为 **{history['public_seed42_beam_bleu']:.3f}**。这些模型采用历史验证选checkpoint规则，与本轮CV选择后固定第3,000步的预算和选择方式不同，仅列为历史参照。", "",
        "- [三种子12,000更新报告](../../quality_multiseed_report/report.md)。",
        "- [已完成BPE8k/unigram8k共同批次表示对照](../../representation_2026_10_27/report/report.md)。",
        "- [已完成表示与独立预训练方法综合报告](../../course_report_2026_10_27/report.md)。", ""]
    reproduction = result["historical_reproduction"]
    lines += ["### 本轮CV重训与历史同seed、同3000更新模型的复现核验", "",
        f"只收录已有逐种子审计，当前已审计seed={reproduction['audited_seeds']}，尚待核验seed={reproduction['pending_seeds']}。同样分数本身不足以证明模型或预测逐位相同，未完成审计的种子不从其他种子推断。", "",
        "| seed | 状态 | 终点张量逐位相同 | 已比预测组/4 | 对齐预测相同组 | 词表piece/score与有效normalizer相同 | 新训练执行证据 |", "|---|---|---|---:|---:|---|---|"]
    for item in reproduction["seeds"]:
        if item["status"] != "audited":
            lines.append(f"| {item['seed']} | 待核验 | 未判断 | 未判断 | 未判断 | 未判断 | 未判断 |")
            continue
        lines.append(f"| {item['seed']} | 已审计 | {item['all_tensors_bitwise_equal']}（{item['model_tensors_compared']}张量） | {item['available_historical_evaluation_groups']} | {item['aligned_equal_evaluation_groups']} | {item['tokenizer_ID_piece_score_equal'] and item['tokenizer_effective_normalizer_equal']} | {item['fresh_executed_training_evidence']} |")
    matching = [item["seed"] for item in reproduction["seeds"] if item["status"] == "audited" and item["same_endpoint_and_test_predictions"]]
    if matching:
        lines += ["", f"已审计seed={matching}的终点模型及两组测试预测精确复现历史同seed模型，已比较部分没有新增翻译质量收益。CV选参增加的是训练集内严格分组、折内拟合分词器与固定预算选择的规范证据。", ""]
    fresh = [item["seed"] for item in reproduction["seeds"] if item["status"] == "audited" and item["fresh_executed_training_evidence"]]
    if fresh:
        lines += [f"seed={fresh}的新训练命令、完整独立日志和计时证明执行了新的从头训练，不能写成直接复用了旧checkpoint。", ""]
    lines += ["新拟合分词器的原始model哈希与旧模型不同；学得的piece/score、特殊ID和有效normalizer是否相同以审计为准。TrainerSpec中的路径、shuffle/hard_vocab_limit等训练设置及显式默认值差异均保留，不把文件差异说成仅有路径变化。预测文本可以逐行相同而原文件SHA不同（例如旧CRLF、新LF）；不把原始字节哈希不同误判成译文变化。", "",
        "历史运行在训练中每500更新做验证并选择best；本轮CV重训固定第3000步，不做训练中验证选checkpoint。旧best恰好位于3000步时，两者终点可以相同，但验证、保存等工作量和跨日期硬件状态不同，计时不能直接解释为算法加速。历史日志缺失的新微批哈希等字段不冒充已经逐项核验。", "",
        "历史三种子与本轮相同三种子的重复结果不合并成n=6；本轮正式统计仍为实际纳入的42/43/44三个种子。", ""]
    for item in reproduction["seeds"]:
        if item["status"] == "audited":
            lines.append(f"- [seed{item['seed']}完整复现审计](../cv/historical_reproduction_seed{item['seed']}.json)，SHA-256 `{item['audit_sha256']}`；旧训练内验证{item['old_validation_checkpoints']}次，新训练内验证选checkpoint {item['new_validation_selected_checkpoints']}次。")
    lines += ["", "## 4. 全研究预算与实际资源成本", "",
        "正式预算共 **19次训练、57,000次优化器更新**：CV六次、选参重训三次、学习率调参四次、适配正式重复六次。下表只累计已完成并核验的运行及完整评测；进行中的部分更新、时间及未完成推理不估算为未来成本。完成数为0是实际已完成数量，不代表缺失质量得零分。", "",
        "| 阶段 | 计划运行/更新 | 已完成训练/更新 | 已完成训练秒 | 完整评测遍数（完成/计划） | 完整评测句次（完成/计划） | 已完成评测秒 |",
        "|---|---|---|---:|---|---|---:|"]
    for cost in result["stage_costs"].values():
        lines.append(f"| {cost['title']} | {cost['planned_runs']}/{cost['planned_optimizer_updates']:,} | {cost['completed_training_runs']}/{cost['completed_run_updates']:,} | {fmt(cost['training_seconds_completed_runs'], 1)} | {cost['completed_full_evaluation_passes']}/{cost['planned_full_evaluation_passes']} | {cost['evaluated_sentences']:,}/{cost['planned_evaluation_sentences']:,} | {fmt(cost['evaluation_seconds'], 1)} |")
    actual_runs = sum(c["completed_training_runs"] for c in result["stage_costs"].values())
    actual_updates = sum(c["completed_run_updates"] for c in result["stage_costs"].values())
    lines += ["", f"当前已验证完整训练共{actual_runs}/19次、其中累计{actual_updates:,}/57,000更新。CV完整保留折需两候选各覆盖全部204,285句，共 **408,570句次**；其推理成本单独计入。原生适配检查点选择共有调参24遍+正式36遍，即 **60×888=53,280句次**；正式验证/测试两种解码另计24遍。不能仅按更新数忽略这些成本。", "",
        f"资源探测另有实际记录 **{result['resource_costs']['optimizer_updates']}次优化器更新**（本轮冻结时记录108次），不混入19次正式训练。原始直接底座项目适配更新为0；既有外部预训练成本未知，不等于0。资源探测和命令执行明细见results.json。", "",
        "### 正式重复与四组学习率调参的逐运行成本", "",
        "| 方法/seed/学习率 | 训练秒 | 训练中验证秒（单列） | 峰值MiB | 可训练/总参数 | 暴露句数 | 源/目标token | 保存文件MiB |", "|---|---:|---:|---:|---|---:|---|---:|"]
    training_entries = [(f"{TITLES[method]}/{seed}", run) for method in ("cv_final", "lora", "full") for seed, run in result["runs"][method].items()]
    training_entries += [(f"调参{TITLES[method]}/7/lr={run.get('config', {}).get('lr', '待创建')}", run)
        for method in ("lora", "full") for run in result["adaptation_tuning"][method]]
    for label, run in training_entries:
        training = run["training"]
        if not training:
            lines.append(f"| {label} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
            continue
        p = training["parameter_evidence"]
        memory = training.get("peak_training_cuda_bytes")
        lines.append(f"| {label} | {training['training_seconds']:.1f} | {training.get('validation_seconds', 0):.1f} | {fmt(memory / 2**20 if memory is not None else None, 1)} | {p['trainable_parameters']:,}/{p['total_parameters']:,} ({p['trainable_fraction']:.3%}) | {training['sentence_exposures']:,} | {training['source_token_exposures']:,}/{training['target_token_exposures']:,} | {training['best_export_bytes'] / 2**20:.2f} |")
    lines += ["", "CV保存尺寸在当前实现中是含优化器的完整last.pt；适配为所选best.pt学习权重导出，LoRA还需要原始底座。二者存储定义不同，不直接用大小推断压缩率。适配训练秒为优化器步耗时累计，不含验证，验证秒另列；CV训练总时长包括其训练/保存开销。最终推理计时另列。同更新数不等于同算力。GPU任务顺序执行，但温度、电源、执行顺序、生成长度和允许并行的CPU测试/审计/报告负载未完全控制，不能从时间差断言算法速度因果。", "",
        "Token暴露均排除批次补齐的PAD。CV源端计入BOS和EOS，目标端排除BOS、计入EOS；Marian源端和目标端计入原生EOS，不另加BOS。两类分词器与句子暴露不同，跨模型token数不能直接解释为相同信息量或计算量。", ""]
    if selection:
        lines += ["### CV完整保留折推理成本", "", "| 平滑/折 | 训练秒 | 完整保留折推理秒 | 句数 |", "|---|---:|---:|---:|"]
        for candidate in selection["candidates"].values():
            for row in candidate["folds"]:
                lines.append(f"| {candidate['label_smoothing']}/{row['fold']} | {row['training_seconds']:.1f} | {row['heldout_decode_seconds']:.1f} | {row['samples']} |")
        lines += [""]
    lines += ["### 最终测试Beam推理成本", "", "| 方法/seed | 秒 | 句/秒 | 摊销每句ms | 批服务ms | 峰值MiB | 触限 |", "|---|---:|---:|---:|---:|---:|---:|"]
    entries = [("direct/固定模型", result["runs"]["direct"])] + [(f"{method}/{seed}", run) for method in ("cv_final", "lora", "full") for seed, run in result["runs"][method].items()]
    for name, run in entries:
        metric = run["evaluations"]["test"]["beam"]
        if metric:
            memory = metric.get("peak_cuda_bytes")
            lines.append(f"| {name} | {metric['elapsed_seconds']:.1f} | {metric['sentences_per_second']:.2f} | {metric['mean_latency_ms']:.2f} | {metric['mean_batch_latency_ms']:.2f} | {fmt(memory / 2**20 if memory is not None else None, 1)} | {metric['generation_limit_hits']} |")
        else:
            lines.append(f"| {name} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
    lines += ["", "推理秒为同步批服务时间之和，包含分词、生成和目标还原，排除模型加载、每次调用两批预热、打分、写盘与中断等待。摊销每句延迟=总服务秒/句数，批服务延迟=总服务秒/批数；二者均不等于独占单请求延迟。直接模型保留历史计时，不伪称本日重新测速。", "",
        "![训练资源](training_cost.png)", "", "资源试验及失败记录："]
    for attempt in result["resource_attempts"]:
        lines.append(f"- `{Path(attempt['path']).name}`：passed={attempt['passed']}；" + "；".join(f"{m}: {v.get('status')} {v.get('error', '')}" for m, v in attempt["methods"].items()))
    observations = result["gpu_observations"]
    commands = result["execution_attempts"]
    lines += ["", "### 硬件观察与命令实际耗时", ""]
    if observations["observations"]:
        temp, power = observations["temperature_celsius"], observations["power_watts"]
        lines += [f"GPU日志当前有{observations['observations']}条可用被动观测：温度 **{temp['min']:.0f}–{temp['max']:.0f}℃**、功耗 **{power['min']:.2f}–{power['max']:.2f}W**。时间覆盖{observations['first_observed_at_utc']}至{observations['last_observed_at_utc']}（UTC）。这是稀疏观察，不是连续受控测量；不能把记录到温度/功耗等同于控制了它们，也不能据此单独解释性能差异。", ""]
    else:
        lines += ["尚无可用GPU观察记录，不虚构温度或功耗范围。", ""]
    lines += [f"执行目录记录已结束命令{commands['finished_attempts']}次、已由事故记录确认中断但缺少退出码/结束时间{commands['interrupted_attempts_without_exit']}次、其他尚未封口记录{commands['unfinished_attempts']}次、已捕获失败退出{commands['failed_attempts']}次、同名命令追加尝试{commands['repeat_attempts']}次；已结束命令start→end实际经过时间总和为{fmt(commands['summed_finished_command_wall_seconds'], 1)}秒。未封口记录本身不证明进程仍在运行，已确认中断的旧attempt不列为运行中。", "",
        "命令时间包含启动、严格校验、读写、训练/推理以及可能的重复只读复用；不等于纯GPU计算时间，不与上表训练/评测秒相加。重复尝试只统计命令调用，不算新增正式实验。机器结果逐条保存观测到的更新/生成进度；没有进度的重复调用标为可能只读验证，不臆测为额外训练。", ""]
    recovery = result["recovery_incidents"]
    if recovery["incident_count"]:
        extra = recovery["recorded_discarded_updates_lower_bound"]
        resource = result["resource_costs"]["optimizer_updates"]
        lines += ["### 实际中断、恢复与额外计算", "",
            "曾发生工具会话及控制器/训练/报告/观察进程同时消失的实际中断。原因未知，未捕获退出码或traceback；不能据此归因于CPU报告任务、内存或其他未经证实原因。事故确认时间不同于精确进程退出时间，缺失的结束时间不补造。", "",
            "| 事故确认时间UTC | 已提交checkpoint步 | 已完整落盘日志步 | 未提交需回放步 | 已记录额外步秒 | 额外句次 | 源/目标token |", "|---|---:|---:|---:|---:|---:|---|"]
        for incident in recovery["incidents"]:
            lines.append(f"| {incident['observed_at_utc']} | {incident['checkpoint_update']} | {incident['complete_log_updates']} | {incident['recorded_discarded_updates']} | {incident['recorded_discarded_step_seconds']:.6f} | {incident['recorded_discarded_sentences']} | {incident['recorded_discarded_source_tokens']}/{incident['recorded_discarded_target_tokens']} |")
        lines += ["", f"恢复继续同一冻结方案，从匹配的完整last.pt恢复；已经完成的阶段只读严格重验。净提交预算仍为19次正式运行、57,000更新，不把回放当新增候选、种子或扩展预算。另列资源探测{resource}更新，以及中断前已经计算、但未提交且需重做的至少 **{extra}更新／{recovery['recorded_discarded_step_seconds_lower_bound']:.6f}秒**；可能还有未落盘工作，数量和耗时未知。", "",
            f"若全部正式净预算完成，则已记录计算次数的下界为57,000+{resource}+{extra}=**{57000 + resource + extra:,}次更新**，不是新的净训练预算。正式summary计时覆盖保留的提交轨迹；丢弃分支的已记录耗时在此单列，不隐藏额外计算，也不把未完成原attempt的未知wall time算成0。", ""]
        for incident in recovery["incidents"]:
            folder = Path(incident["path"]).parent.name
            lines.append(f"- [事故与存档证据](../recovery_incidents/{folder}/incident.json)，SHA-256 `{incident['sha256']}`。")
            replay = incident["replay_verification"]
            if replay["sha256"] is not None:
                lines.append(f"- [实际回放核验记录](../recovery_incidents/{folder}/replay_verification.json)，SHA-256 `{replay['sha256']}`；具体核验结论以该文件实际字段为准。")
            else:
                lines.append("- 实际回放核验文件尚未保存，当前不声称已验证完整回放一致性。")
        lines += [""]
    lines += ["", "## 5. 相同句子群体与失败分析", "",
        "复用旧表示研究的原始ID分组：源空格词长、仅训练频率定义的稀有比例、历史锚定BPE碎片化；验证另复用固定旧聚类。任何方法均不重新分桶。组内直接计算corpus BLEU/chrF2，空组为null，n<30显式标小组；不能按组分数加权还原总体BLEU。", ""]
    diagnostics = result["generation_diagnostics"]
    lines += ["### 完整语料生成诊断", "",
        "此项为[独立冻结的补充描述性分析](../repetition_analysis_plan.json)，冻结时本轮新官方测试结果文件尚未产生，原训练/生成/选择计划保持不变。使用case-sensitive Python str.split空格token，保留标点；同一句中任何重叠词3gram出现至少3次即标记。分别统计源句、参考和预测；“仅预测”要求预测触发而源句、参考均未触发。", "",
        "这些标记仅提示代理复核，不证明语义错误、因果影响或解码退化；合法演讲原文和参考也可能重复。未触发不能证明没有重复、漏译或错误。不根据此分析选参或增加no_repeat生成规则。", "",
        "| 条件/seed | 划分/解码 | 完整n | 源重复 | 参考重复 | 预测重复 | 仅预测重复 | 触限 | 未结束 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in diagnostics["evaluations"]:
        label = TITLES[item["condition"]] + ("/固定模型" if item["seed"] is None else f"/{item['seed']}")
        values = [fmt(item["samples"], 0)] + [fmt(item[key]["count"], 0) for key in
            ("source", "reference", "prediction", "prediction_only", "generation_limit", "unfinished")]
        lines.append(f"| {label} | {item['split']}/{item['decoder']} | " + " | ".join(values) + " |")
    lines += ["", "每项只纳入严格核验的完整评测，缺失仍为待完成。触限与未结束的定义沿用各自冻结解码实现；原生forced EOS可能使触限句仍显示已结束。机器结果保存标记的原始ID及输入/评测指纹。若原生journal只留下非零的批级未结束计数而无法唯一确定句子，精确ID明确为null并保留候选批ID，不凭触限或文本猜测。见[完整语料生成诊断JSON](generation_diagnostics.json)。", ""]
    for category in ("source_words", "source_rare_fraction", "anchor_fragmentation"):
        lines += [f"### 测试Beam：{category}", "", "| 固定组 | n | 直接BLEU | LoRA BLEU | 全参BLEU | LoRA−全参 |", "|---|---:|---:|---:|---:|---:|"]
        for row in result["groups"]["test"]["beam"][category]:
            methods = row["methods"]
            lines.append(f"| {row['label']} | {row['n']} | {fmt(methods['direct']['bleu'])} | {fmt_stat(methods['lora']['bleu'])} | {fmt_stat(methods['full']['bleu'])} | {fmt_stat(row['paired_lora_minus_full']['bleu'])} |")
    lines += ["", "![固定群体增益](fixed_group_deltas.png)", "",
        "固定样例保存测试原ID%400==0的全部21句、验证每个旧簇最低3个ID，所有可用条件展示同一句的实际译文。见[全部译文与代理诊断](examples.md)和[机器样例](samples.json)。句级chrF2、重复三元组与数字差异只是提示复核，不能自动认定语义改善或失败。", ""]
    if (out / "sample_review.md").exists():
        lines += ["另见[代理复核：改善与失败](sample_review.md)。该文件由代理阅读实际译文后编写，报告生成器保留其原文；不冒充独立人工标注。", ""]
    else:
        lines += ["实际译文的语义代理复核仍待结果齐备后完成；当前不填写未执行的人工或语言学结论。", ""]
    condition = result["aggregate"]["test"]["beam"]
    lines += ["## 6. 结果支持什么、不能支持什么", ""]
    for method in ("lora", "full"):
        difference = condition[method + "_minus_direct"]["bleu"]
        if difference["n"]:
            values = list(difference["per_seed"].values())
            lines.append(f"- {TITLES[method]}相对同一个未适配底座的测试Beam ΔBLEU为{fmt_stat(difference)}；纳入seed中改善{sum(v > 0 for v in values)}个、下降{sum(v < 0 for v in values)}个、持平{sum(v == 0 for v in values)}个。收益或退化均按固定计划呈现。")
        else:
            lines.append(f"- {TITLES[method]}正式重复尚未形成可纳入的完整结果，当前不能判断其相对底座收益。")
    lines += ["- LoRA与全参比较采用同一底座、相同完整3,000步句次轨迹与评测协议，能描述这两种适配方案的预算内差异；各自独立验证选择best，故最佳步不同时被评测权重的前缀暴露并不相同。各自学习率由等数量候选验证选择，结论针对该有限搜索范围内的设置。", "- 预训练语料可能含IWSLT/TED，缺少完整逐句清单，无法排除预训练重叠；这不因后续适配而消失。", "- 测试集在历史多轮研究中已经观察过，不是秘密未见数据；本轮冻结能约束本轮选择，不抹去历史知识。", "- 分组关系不证明因果，参考措辞、长度、领域与稀有度相互混杂；少量种子、固定预算和有限学习率候选限制推广。", "",
        "## 7. 复现、恢复与课程证据", "", "```powershell",
        f".\\project-pretrained-python.ps1 -m mt.study_report --root {result['inputs']['root']}", "```", "",
        "本命令只做CPU校验、统计与图表生成，不启动训练或推理。运行恢复使用各阶段progress记录及原配置，从对应last.pt继续；推理只复用完整身份和校验一致的journal。", "",
        "- [机器结果](results.json)、[完整分组](groups.json)、[固定样例](samples.json)、[本报告输入映射](report_inputs.json)。",
        "- [课程对应表](course_alignment.md)。本轮测试与历史完整性以根目录最终completion记录为准，不把历史194项通过记录冒充本轮执行。", ""]
    for record in result["executed_regression_records"]:
        lines.append(f"- 已执行检查记录 `{Path(record['path']).name}`：运行{record['tests_run']}项，通过{record.get('passed')}项，失败{record.get('failures')}、错误{record.get('errors')}；日志SHA已核验。记录时间{record.get('observed_at_utc', '见原记录')}。")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    alignment = ["# 课程关注点与实际证据", "", "课程认定仍需教师确认，不承诺得分。已完成旧表示对照和预训练直接推理不再列为待执行。", "",
        "| 关注点 | 当前证据 | 状态／解释边界 |", "|---|---|---|",
        "| 数据探索、降维、聚类 | 历史探索报告与固定ID群体 | 已完成；聚类强长度关联，不解释为主题真值 |",
        "| 不同特征表示 | 历史BPE8k/unigram8k共同批次三种子对照 | 已完成；本轮复用其固定句子群体 |",
        f"| CV用于选择超参数 | TED/重复源连通分组、折内BPE、固定第3000步、OOF | {'选择记录已验证' if result['cv_selection'] else '正式选参结果待完成'}；三种子不是CV |",
        "| 同底座新方法比较 | 原始模型、LoRA、全参，共同有效批次 | 逐方法状态见报告；不以外部预训练差异解释LoRA独立作用 |",
        "| 实验设置 | 冻结方案、源码/输入/环境、共同句子暴露、恢复检查 | 已知历史测试之后规划，未追溯称历史事前声明 |",
        "| 结果解释 | seed均值/SD、配对差、成本、完整固定分组 | n=3不直接宣称显著；CV fold SD与seed SD明确区分 |",
        "| 图表与报告 | 中文报告、JSON、PNG/PDF、固定实际译文 | pending不填零；语义代理复核与独立人工标注区分 |",
        "| 展示 | 基于最终实际结果更新PPT | 演示文稿状态以本轮完成记录为准 |",
        "| 待教师确认 | MT是否按字面要求k-fold、新方法/新表示认定范围 | 不阻塞已授权本地研究，不声称教师已认可 |", ""]
    (out / "course_alignment.md").write_text("\n".join(alignment), encoding="utf-8")


class ScientificInputClosure:
    """An explicit file registry, never a recursive scan of the whole study."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}
        self.excluded_metadata = {}

    def exclusion_reason(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.root / "report") and path != self.root / "report" / "sample_review.md":
            return "report output or report-derived metadata would create an input/output cycle"
        if path.is_relative_to(self.root / "presentation"):
            return "presentation is downstream of the final report"
        if path.is_relative_to(self.root):
            if path.name in {"execution.lock", "review_manifest.json", "report_manifest.json", "completion.json", "new_files.json", "new_files_manifest.json"}:
                return "mutable downstream, completion, review, inventory or lock metadata"
            if path.name.startswith("new_files") or "inventory" in path.name:
                return "new-file inventory is downstream mutable metadata"
            if path.name == "progress.json" and "batches" not in path.parts:
                # Complete CV evaluation progress is required by verify_results;
                # root/run training progress is live controller metadata.
                parent = path.parent.name
                if not (parent == "holdout_greedy" or parent in {f"{s}_{d}" for s in SPLITS for d in DECODERS}):
                    return "live root/training progress, not immutable scientific evidence"
        return None

    def add(self, path, expected=None):
        path = Path(path).resolve()
        reason = self.exclusion_reason(path)
        if reason:
            raise ValueError(f"excluded report input: {path}: {reason}")
        key = str(path)
        if expected is not None and (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("expected scientific input SHA-256 is invalid: " + key)
        if key in self.files:
            if expected is not None and self.files[key] != expected:
                raise ValueError("conflicting scientific input fingerprints: " + key)
            return self.files[key]
        if not path.is_file():
            raise FileNotFoundError("required scientific report input is missing: " + key)
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError("scientific report input changed: " + key)
        self.files[key] = actual
        return actual

    def referenced(self, mapping, base=None):
        for name, item in mapping.items():
            path = Path(name)
            if not path.is_absolute():
                if base is None:
                    raise ValueError("relative evidence path needs an explicit base")
                path = Path(base) / path
            reason = self.exclusion_reason(path)
            if reason:
                self.excluded_metadata[str(path.resolve())] = reason
                continue
            self.add(path, item["sha256"] if isinstance(item, dict) else item)

    def json(self, path, expected=None):
        self.add(path, expected)
        return read_json(path)


def verify_input_files(input_files):
    """Finalizer contract: reject any missing/changed input, without models."""
    if not isinstance(input_files, dict) or not input_files:
        raise ValueError("final report requires a nonempty complete scientific input closure")
    for name, expected in input_files.items():
        path = Path(name)
        if not path.is_absolute() or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("report input closure must contain absolute file paths and SHA-256 digests")
        if not path.is_file():
            raise FileNotFoundError("report input is missing: " + name)
        if sha256(path) != expected:
            raise ValueError("report input changed: " + name)
    return {"passed": True, "files_verified": len(input_files)}


def build_input_files(root, inputs, result, inputs_path=None):
    """Build the final scientific evidence closure after all GPU work ends.

    Role-specific files are required explicitly: CV has last.pt and a source
    manifest; native adaptation has last/best and all six checkpoint exports.
    Pending reports do not call this function or claim a finalizable closure.
    """
    if result.get("complete") is not True:
        raise ValueError("complete input closure requires the complete experimental result set")
    root = Path(root).resolve()
    project = Path(__file__).resolve().parents[1]
    evidence = ScientificInputClosure(root)
    expected_metrics = {}
    for split in SPLITS:
        for decoder in DECODERS:
            expected_metrics[str((Path(inputs["direct_root"]) / split / decoder).resolve())] = result["runs"]["direct"]["evaluations"][split][decoder]
    for method in ("cv_final", "lora", "full"):
        for run in result["runs"][method].values():
            folder = Path(run["config"]["run_dir"])
            for split in SPLITS:
                for decoder in DECODERS:
                    path = folder / f"{split}_{decoder}" if method == "cv_final" else folder / "evaluation" / split / decoder
                    expected_metrics[str(path.resolve())] = run["evaluations"][split][decoder]

    def sources(mapping, snapshot=None, current_base=project):
        evidence.referenced(mapping, current_base)
        if snapshot is not None:
            evidence.referenced(mapping, snapshot)

    def prepared(folder, marker="preparation_complete.json", key="artifact_sha256"):
        complete = evidence.json(Path(folder) / marker)
        evidence.referenced(complete[key], folder)
        return complete

    def native_evaluation(directory):
        directory = Path(directory)
        metric = evidence.json(directory / "metrics.json")
        if str(directory.resolve()) in expected_metrics and metric != expected_metrics[str(directory.resolve())]:
            raise ValueError("native metric changed after strict report collection")
        for name in ("predictions.txt", "references.txt", "examples.jsonl", "batches.jsonl"):
            evidence.add(directory / name, metric["file_sha256"][name])
        evidence.add(directory / "started.json")
        return metric

    def cv_evaluation(directory):
        directory = Path(directory)
        metric = evidence.json(directory / "metrics.json")
        if str(directory.resolve()) in expected_metrics and metric != expected_metrics[str(directory.resolve())]:
            raise ValueError("CV final metric changed after strict report collection")
        identity = evidence.json(directory / "identity.json")
        evidence.add(directory / "progress.json")
        for name in ("predictions.txt", "references.txt", "examples.jsonl"):
            key = name.removesuffix(".txt").removesuffix(".jsonl") + "_sha256"
            evidence.add(directory / name, metric[key])
        total = (identity["samples"] + identity["batch_size"] - 1) // identity["batch_size"]
        for index in range(total):
            evidence.add(directory / "batches" / f"batch_{index:06d}.json")
        evidence.add(identity["input_path"], identity["input_sha256"])
        evidence.add(identity["checkpoint"], identity["checkpoint_sha256"])
        sources(identity["source_hashes"], current_base=project / "mt")
        return metric

    def cv_run(config_path, splits, verified=None):
        cfg = evidence.json(config_path)
        run = Path(cfg["run_dir"])
        evidence.add(run / "config.json")
        evidence.add(run / "summary.json", verified["training_summary_sha256"] if verified else None)
        evidence.add(run / "train.jsonl")
        evidence.add(run / "last.pt")
        evidence.add(run / "data_link.json")
        evidence.add(run / "environment.json")
        hashes = evidence.json(run / "source_hashes.json")
        sources(hashes, run / "source_snapshot", project / "mt")
        evidence.add(run / "source_snapshot" / "manifest.json")
        prepared(cfg["data_dir"])
        for split in splits:
            for decoder in (("greedy",) if split == "holdout" else DECODERS):
                cv_evaluation(run / f"{split}_{decoder}")

    def native_run(config_path, formal, verified):
        cfg = evidence.json(config_path)
        run = Path(cfg["run_dir"])
        plan = evidence.json(run / "plan.json")
        evidence.add(run / "config.json", plan["config_sha256"])
        sources(plan["source_files"], run / "source_snapshot")
        evidence.referenced(plan["model_files"], cfg["model_dir"])
        summary = evidence.json(run / "summary.json")
        if summary != verified["training"] or cfg != verified["config"]:
            raise ValueError("native training summary/config changed after strict collection")
        evidence.add(run / "last.pt", summary["last_sha256"])
        evidence.add(run / "best.pt", summary["best_sha256"])
        evidence.add(run / "train.jsonl", summary["train_log_sha256"])
        evidence.add(run / "plan.json", summary["plan_sha256"])
        for update in range(500, 3001, 500):
            point = next(p for p in verified["validation_curve"] if p["update"] == update)
            evidence.add(run / f"checkpoint_{update:05d}.pt", point["export_sha256"])
            evidence.add(run / "validation" / f"step_{update:05d}" / "metrics.json", point["metrics_sha256"])
            native_evaluation(run / "validation" / f"step_{update:05d}")
        for extra in run.glob("checkpoint_*.pt"):
            evidence.add(extra)
        if formal:
            for split in SPLITS:
                for decoder in DECODERS:
                    native_evaluation(run / "evaluation" / split / decoder)

    plan = evidence.json(root / "plan.json")
    evidence.add(root / "design.json", plan["design_sha256"])
    historical = evidence.json(root / "historical_manifest.json")
    if "gptprompt.txt" in historical["files"]:
        evidence.add(project / "gptprompt.txt", historical["files"]["gptprompt.txt"]["sha256"])
    evidence.referenced(plan["inputs"])
    evidence.referenced(plan["initial_configs"])
    sources(plan["source_files"], root / "source_snapshot")
    prepared(root / "data" / "cv", "grouping_complete.json")
    for stage in ("cv", "adaptation"):
        stage_plan = evidence.json(root / stage / "final_plan.json")
        evidence.referenced(stage_plan["files"])
        selection = evidence.json(root / stage / "selection.json")
        evidence.referenced(selection["input_hashes"])
    for tag in ("ls0", "ls01"):
        folder = root / "cv" / "oof" / tag
        metric = evidence.json(folder / "metrics.json")
        evidence.referenced(metric["artifacts"], folder)
    for fold in range(3):
        for tag in ("ls0", "ls01"):
            cv_run(root / "cv" / "configs" / f"fold{fold}_{tag}.json", ("holdout",))
    for seed in SEEDS:
        cv_run(inputs["cv_final"][str(seed)], SPLITS, result["runs"]["cv_final"][str(seed)])
    for method in ("lora", "full"):
        for path, verified in zip(inputs["adaptation_tuning"][method], result["adaptation_tuning"][method]):
            native_run(path, formal=False, verified=verified)
        for seed in SEEDS:
            native_run(inputs["adaptation_final"][method][str(seed)], formal=True, verified=result["runs"][method][str(seed)])

    native_data = root / "adaptation" / "data"
    data_manifest = evidence.json(native_data / "manifest.json")
    for split, item in data_manifest["splits"].items():
        evidence.add(native_data / f"{split}.jsonl", item["cache_sha256"])
        evidence.add(Path(data_manifest["data_dir"]) / f"{split}.jsonl", item["input_sha256"])
    for item in data_manifest["schedules"].values():
        evidence.add(native_data / item["path"], item["sha256"])
    for name, key in (("retained_train_ids.json", "retained_train_ids_sha256"), ("excluded_train.json", "excluded_train_sha256")):
        evidence.add(native_data / name, data_manifest[key])

    direct = Path(inputs["direct_root"])
    direct_plan = evidence.json(direct / "plan.json")
    for name, key in (("config", "config_sha256"), ("inputs", "input_sha256"), ("environment", "environment_sha256"),
                      ("download", "download_sha256"), ("smoke", "smoke_sha256"), ("length_audit", "length_audit_sha256")):
        evidence.add(direct / f"{name}.json", direct_plan[key])
    sources(direct_plan["source_files"])
    evidence.referenced(direct_plan["model_files"], direct / "model")
    direct_cfg = read_json(direct / "config.json")
    direct_inputs = read_json(direct / "inputs.json")
    evidence.add(Path(direct_cfg["data_dir"]) / "manifest.json", direct_inputs["manifest_sha256"])
    for split, item in direct_inputs["splits"].items():
        evidence.add(Path(direct_cfg["data_dir"]) / f"{split}.jsonl", item["sha256"])
    for split in SPLITS:
        for decoder in DECODERS:
            native_evaluation(direct / split / decoder)
    audit = evidence.json(inputs["direct_reuse_audit"])
    evidence.add(audit["audit_source"]["path"], audit["audit_source"]["sha256"])
    evidence.add(audit["adaptation_template"], audit["adaptation_template_sha256"])
    evidence.referenced(audit["result_files_before"], direct)
    sources(audit["source_files"])
    sources(audit["adaptation_source_files"])
    evidence.add(audit["native_generator"]["file"], audit["native_generator"]["file_sha256"])
    backend = evidence.json(inputs.get("direct_backend_addendum", root / "baseline_backend_environment_addendum.json"),
                            result["baseline_backend_environment"]["sha256"])
    evidence.referenced(backend["historical_search"]["file_sha256"], backend["historical_search"]["directory"])
    sources(backend["source_files"])

    # Bind the dependency RECORD files and specifically audited implementation
    # modules read by dependency_evidence; this is not a scan of all site-packages.
    import importlib.metadata
    environment = plan["dependencies"]
    for package, metadata in environment["packages"].items():
        dist = importlib.metadata.distribution(package)
        record = next((p for p in (dist.files or []) if p.name == "RECORD"), None)
        if metadata["record_sha256"] is not None:
            if record is None:
                raise FileNotFoundError("required dependency RECORD is missing: " + package)
            evidence.add(dist.locate_file(record), metadata["record_sha256"])
    for key in ("torch_module", "numpy_module", "transformers_module"):
        evidence.add(environment[key])
    evidence.referenced(environment["transformers_implementation"], Path(environment["transformers_module"]).parent)

    for split in SPLITS:
        membership = evidence.json(Path(inputs["groups_root"]) / f"{split}.json")
        evidence.add(Path(inputs["source_data"]) / f"{split}.jsonl", membership["source_file_sha256"])
    repetition = evidence.json(inputs.get("repetition_analysis_plan", root / "repetition_analysis_plan.json"),
                               result["generation_diagnostics"]["plan"]["sha256"])
    evidence.referenced(repetition["inputs"])
    for item in result["historical_context"]["sources"].values():
        evidence.add(item["path"], item["sha256"])
    for item in result["historical_reproduction"]["seeds"]:
        if item["status"] == "audited":
            reproduction = evidence.json(item["audit_path"], item["audit_sha256"])
            evidence.referenced(reproduction["input_fingerprints"])
            descriptor = reproduction["tokenizer"]["descriptor_source"]
            evidence.add(descriptor["path"], descriptor["sha256"])
    for item in result["resource_costs"]["attempts"]:
        evidence.add(item["file"], item["sha256"])
    for item in result["resource_attempts"]:
        evidence.add(item["path"], item["sha256"])
    for item in result["recovery_incidents"]["incidents"]:
        incident_path = Path(item["path"])
        incident = evidence.json(incident_path, item["sha256"])
        evidence.referenced(incident["archived_files"], incident_path.parent)
        evidence.add(Path(incident["native_run"]) / "plan.json", incident["run_plan_sha256"])
        replay_info = item["replay_verification"]
        if replay_info["sha256"] is not None:
            replay = evidence.json(replay_info["path"], replay_info["sha256"])
            if "captured_checkpoint_path" in replay:
                evidence.add(replay["captured_checkpoint_path"], replay["captured_checkpoint_sha256"])
    for item in result["execution_attempts"]["attempts"]:
        if item["wall_seconds"] is not None or item["interruption_incident"] is not None:
            record_path = Path(item["path"])
            record = evidence.json(record_path, item["sha256"])
            evidence.add(record_path.with_suffix(".log"), record.get("log_sha256"))
    telemetry = result["gpu_observations"]
    if telemetry.get("path") is not None:
        evidence.add(telemetry["path"], telemetry["sha256"])
    for item in result["executed_regression_records"]:
        record_path = Path(item["path"])
        evidence.add(record_path, item["sha256"])
        evidence.add(record_path.with_suffix(".log"), item["log_sha256"])
    review = root / "report" / "sample_review.md"
    if review.exists():
        evidence.add(review)
    if inputs_path is not None:
        reason = evidence.exclusion_reason(inputs_path)
        if reason:
            evidence.excluded_metadata[str(Path(inputs_path).resolve())] = reason
        else:
            evidence.add(inputs_path)
    return {"input_files": dict(sorted(evidence.files.items())), "excluded_metadata": dict(sorted(evidence.excluded_metadata.items()))}


def generate(root, inputs_path=None):
    root = Path(root).resolve()
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    inputs = read_json(inputs_path) if inputs_path else default_inputs(root)
    if Path(inputs["root"]).resolve() != root:
        raise ValueError("report input mapping belongs to a different study")
    result = collect(root, inputs)
    closure = build_input_files(root, inputs, result, inputs_path) if result["complete"] else {"input_files": {}, "excluded_metadata": {}}
    write_json(out / "report_inputs.json", inputs)
    write_json(out / "results.json", {key: value for key, value in result.items() if key not in ("groups", "samples", "generation_diagnostics")})
    write_json(out / "groups.json", result["groups"])
    write_json(out / "samples.json", result["samples"])
    write_json(out / "generation_diagnostics.json", result["generation_diagnostics"])
    figures = _figures(result, out)
    _write_examples(result, out)
    _write_report(result, out)
    write_json(out / "report_manifest.json", {"generated_at_utc": result["generated_at_utc"], "status": result["status"],
        "generator_sha256": sha256(__file__), "input_sha256": result["input_sha256"],
        "source_generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "input_files_complete": result["complete"], "input_files": closure["input_files"],
        "excluded_mutable_or_downstream_metadata": closure["excluded_metadata"],
        "input_files_scope": "complete scientific evidence closure" if result["complete"] else "pending experimental set; empty closure is explicitly not finalizable",
        "outputs": {name: sha256(out / name) for name in ["report.md", "results.json", "groups.json", "samples.json", "generation_diagnostics.json",
            "examples.md", "course_alignment.md", "report_inputs.json", *figures]}})
    return {"report": str(out / "report.md"), "status": result["status"], "figures": figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--inputs")
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.inputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
