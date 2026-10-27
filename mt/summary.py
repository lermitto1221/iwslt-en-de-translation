import argparse
import json
import math
import re
import statistics
from pathlib import Path

from .common import read_json, sha256, write_json
from .data import EXPECTED


FINGERPRINT_KEYS = ("manifest_sha256", "tokenizer_sha256", "train_sha256",
                    "validation_sha256", "test_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def artifact_path(value, context):
    require(isinstance(value, (str, Path)) and bool(str(value)), f"{context}: file path is missing")
    path = Path(str(value).replace("\\", "/"))
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[1] / path
    require(path.is_file(), f"{context}: required file is missing: {path}")
    return path


def verify_hash(path, expected, context):
    require(valid_hash(expected), f"{context}: recorded SHA-256 is missing or invalid")
    require(sha256(path) == expected, f"{context}: SHA-256 mismatch: {path}")


def read_object(path, context):
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON from {path}: {exc}") from exc
    require(isinstance(value, dict), f"{context}: expected a JSON object in {path}")
    return value


def read_jsonl(path, context):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{context}: invalid JSON at {path}:{line_number}: {exc.msg}") from exc
                require(isinstance(row, dict), f"{context}: expected an object at {path}:{line_number}")
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{context}: cannot read {path}: {exc}") from exc
    return rows


def validate_saved_outputs(metrics, output_dir, context="evaluation"):
    """Verify saved evaluation artifacts without loading a model or running CUDA.

    Works for validation and test. Formal dataset cardinality and experimental
    controls are checked separately by validate_result. Historical results do
    not have to match the current decoding source code.
    """
    import sacrebleu

    require(isinstance(metrics, dict), f"{context}: metrics must be an object")
    expected = metrics.get("expected_samples")
    require(type(expected) is int and expected > 0 and metrics.get("samples") == expected,
            f"{context}: invalid or inconsistent evaluation sample count")
    split = metrics.get("split")
    require(split in ("validation", "test"), f"{context}: expected validation or test results")
    directory = Path(output_dir)
    paths = {name: artifact_path(directory / name, context) for name in
             ("predictions.txt", "references.txt", "examples.jsonl")}
    verify_hash(paths["predictions.txt"], metrics.get("predictions_sha256"), f"{context}: predictions")
    # Older metrics did not hash references.txt. Exact comparison against the
    # hashed source references below still verifies that file's full contents.
    if "references_sha256" in metrics:
        verify_hash(paths["references.txt"], metrics["references_sha256"], f"{context}: references")
    source_path = artifact_path(metrics.get("reference_source"), context)
    verify_hash(source_path, metrics.get("reference_sha256"), f"{context}: reference source")
    fingerprints = metrics.get("data_fingerprints") or {}
    if split + "_sha256" in fingerprints:
        require(fingerprints[split + "_sha256"] == metrics["reference_sha256"],
                f"{context}: reference source differs from recorded {split} fingerprint")
    if "checkpoint_sha256" in metrics:
        checkpoint = artifact_path(metrics.get("checkpoint"), context)
        verify_hash(checkpoint, metrics["checkpoint_sha256"], f"{context}: checkpoint")
    try:
        predictions = paths["predictions.txt"].read_text(encoding="utf-8").splitlines()
        references = paths["references.txt"].read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{context}: cannot read prediction/reference text: {exc}") from exc
    examples = read_jsonl(paths["examples.jsonl"], context)
    source_rows = read_jsonl(source_path, context)
    for name, rows in (("predictions", predictions), ("references", references),
                       ("examples", examples), ("reference source", source_rows)):
        require(len(rows) == expected, f"{context}: {name} has {len(rows)} rows; expected {expected}")
    seen_source, seen_examples = set(), set()
    for index, (source, example, prediction, reference) in enumerate(
            zip(source_rows, examples, predictions, references), 1):
        for label, row, seen in (("reference source", source, seen_source), ("examples", example, seen_examples)):
            identity = row.get("id")
            require(type(identity) is int and identity >= 0,
                    f"{context}: {label} row {index} has an invalid id")
            require(identity not in seen, f"{context}: duplicate id {identity} in {label} at row {index}")
            seen.add(identity)
        require(isinstance(source.get("en"), str) and isinstance(source.get("de"), str)
                and source["en"] and source["de"], f"{context}: source row {index} is not a nonempty aligned pair")
        require(example.get("id") == source["id"], f"{context}: example/source id order mismatch at row {index}")
        require(example.get("source") == source["en"], f"{context}: example source mismatch at row {index}")
        require(example.get("reference") == reference == source["de"],
                f"{context}: reference mismatch at row {index}")
        require(example.get("prediction") == prediction, f"{context}: example/prediction mismatch at row {index}")
    metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    actual_bleu = metric.corpus_score(predictions, [references]).score
    score = metrics.get("bleu")
    require(type(score) in (int, float) and math.isfinite(score)
            and math.isclose(actual_bleu, score, rel_tol=0, abs_tol=1e-8),
            f"{context}: saved BLEU {score!r} does not match recomputed BLEU {actual_bleu}")
    require(str(metric.get_signature()) == metrics.get("signature")
            and sacrebleu.__version__ == metrics.get("sacrebleu_version"),
            f"{context}: SacreBLEU signature/version differs; use the recorded dependency versions")
    return examples


def validate_training_log(run, cfg, context):
    path = artifact_path(Path(run) / "train.jsonl", context)
    rows = read_jsonl(path, context)
    budget = cfg.get("max_updates")
    require(type(budget) is int and budget > 0, f"{context}: invalid update budget")
    require(len(rows) == budget, f"{context}: training log has {len(rows)} updates; expected {budget}")
    orders = []
    for expected, row in enumerate(rows, 1):
        require(type(row.get("update")) is int and row["update"] == expected,
                f"{context}: training log update sequence must be 1..{budget}; row {expected} "
                f"contains {row.get('update')!r} (missing, duplicate, or reordered update)")
        require(valid_hash(row.get("data_order_sha256")),
                f"{context}: invalid data order SHA-256 at update {expected}")
        require(type(row.get("target_tokens")) is int and row["target_tokens"] > 0
                and type(row.get("sentences")) is int and row["sentences"] > 0,
                f"{context}: invalid target token or sentence count at update {expected}")
        orders.append((row["data_order_sha256"], row["target_tokens"], row["sentences"]))
    return orders


def validate_result(metrics, run, condition, seed, method):
    """Reject diagnostic, partial, or untraceable results before reporting BLEU."""
    context = f"{condition}{seed}/{method}"
    require(isinstance(metrics, dict), f"{context}: metrics must be an object")
    require(metrics.get("split") == "test", f"{context}: only test results can enter the experiment table")
    require(metrics.get("smoke") is False, f"{context}: missing smoke flag or diagnostic checkpoint")
    require(metrics.get("pilot", False) is False, f"{context}: pilot checkpoint cannot enter formal results")
    require(metrics.get("samples") == metrics.get("expected_samples") == EXPECTED["test"],
            f"{context}: the complete {EXPECTED['test']}-sentence test set is required")
    require(metrics.get("method") == method, f"{context}: decoding method differs from output directory")
    require(metrics.get("beam_size") == (1 if method == "greedy" else 4),
            f"{context}: core experiment requires greedy or beam=4")
    score = metrics.get("bleu")
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 100 + 1e-8,
            f"{context}: invalid corpus BLEU")
    signature_text = metrics.get("signature")
    require(isinstance(signature_text, str), f"{context}: SacreBLEU signature is missing")
    signature = dict(item.split(":", 1) for item in signature_text.split("|") if ":" in item)
    require(signature.get("nrefs") == "1" and signature.get("case") == "mixed"
            and signature.get("tok") == "13a" and signature.get("version") == metrics.get("sacrebleu_version"),
            f"{context}: expected a complete case-sensitive SacreBLEU 13a signature")
    require(valid_hash(metrics.get("checkpoint_sha256")),
            f"{context}: missing checkpoint SHA-256; rerun evaluation with current code")
    require(valid_hash(metrics.get("inference_source_sha256")),
            f"{context}: missing inference source SHA-256")
    require(valid_hash(metrics.get("initial_model_sha256")),
            f"{context}: missing initialization fingerprint; paired initialization cannot be verified")
    fingerprints = metrics.get("data_fingerprints", {})
    require(isinstance(fingerprints, dict) and all(valid_hash(fingerprints.get(key)) for key in FINGERPRINT_KEYS),
            f"{context}: incomplete data/tokenizer fingerprints")
    require(metrics.get("reference_sha256") == fingerprints["test_sha256"],
            f"{context}: test reference differs from the checkpoint's verified data")
    cfg = metrics.get("config", {})
    require(isinstance(cfg, dict) and cfg.get("seed") == seed
            and cfg.get("label_smoothing") == (0.1 if condition == "A" else 0.0),
            f"{context}: seed or label smoothing differs from experiment label")
    summary_path = run / "summary.json"
    require(summary_path.exists(), f"{context}: training summary is missing; training budget not verified")
    training = read_object(summary_path, context)
    require(training.get("smoke") is False and training.get("pilot", False) is False
            and training.get("updates") == cfg.get("max_updates")
            and isinstance(cfg.get("max_updates"), int) and cfg["max_updates"] > 0,
            f"{context}: training did not complete its fixed update budget")
    config_path = run / "config.json"
    require(config_path.exists() and read_object(config_path, context) == cfg,
            f"{context}: evaluation checkpoint config differs from completed training")
    require(metrics.get("inference_precision") and isinstance(metrics.get("inference_batch_size"), int)
            and metrics["inference_batch_size"] > 0 and isinstance(metrics.get("environment"), dict)
            and bool(metrics["environment"]), f"{context}: inference hardware/precision/batch not recorded")
    require(isinstance(metrics.get("length_penalty"), (int, float))
            and math.isfinite(metrics["length_penalty"])
            and isinstance(metrics.get("max_generation_tokens"), int)
            and metrics["max_generation_tokens"] > 0, f"{context}: invalid decoding settings")
    return validate_saved_outputs(metrics, run / method, context)


def validate_comparability(records):
    # All seeds use one dataset, tokenizer, decoding rule, precision and speed-test hardware.
    common_fields = ("data_fingerprints", "inference_source_sha256", "length_penalty", "max_generation_tokens", "signature",
                     "inference_precision", "inference_batch_size", "environment")
    anchor = next(iter(records.values()), None)
    for name, metrics in records.items():
        for field in common_fields:
            require(metrics[field] == anchor[field], f"{name}: {field} differs across core experiments")
    for seed in (42, 43, 44):
        conditions = {}
        for condition in ("A", "B"):
            greedy, beam = (records.get(f"{condition}{i}_seed{seed}") for i in (1, 2))
            if greedy and beam:
                require(greedy["checkpoint_sha256"] == beam["checkpoint_sha256"],
                        f"{condition}{seed}: greedy and beam must use the same checkpoint")
                require(greedy["config"] == beam["config"], f"{condition}{seed}: checkpoint configs differ")
            present = greedy or beam
            if present:
                conditions[condition] = present
        if len(conditions) == 2:
            require(conditions["A"]["initial_model_sha256"] == conditions["B"]["initial_model_sha256"],
                    f"seed {seed}: A/B model initialization fingerprints differ")
            normalized = [{k: v for k, v in conditions[c]["config"].items() if k not in ("label_smoothing", "run_dir")}
                          for c in ("A", "B")]
            require(normalized[0] == normalized[1],
                    f"seed {seed}: A/B training config differs beyond label_smoothing and run_dir")
            if "training_source_hashes" in conditions["A"] or "training_source_hashes" in conditions["B"]:
                require(conditions["A"].get("training_source_hashes") == conditions["B"].get("training_source_hashes"),
                        f"seed {seed}: training source code changed between A and B")


def collect(root):
    root = Path(root)
    result, records, training_orders = {}, {}, {}
    for seed in (42, 43, 44):
        for condition in ("A", "B"):
            run = root / f"{condition}{seed}"
            for method in ("greedy", "beam"):
                name = f"{condition}{'1' if method == 'greedy' else '2'}_seed{seed}"
                path = run / method / "metrics.json"
                if path.exists():
                    metrics = read_object(path, name)
                    validate_result(metrics, run, condition, seed, method)
                    if (condition, seed) not in training_orders:
                        training_orders[condition, seed] = validate_training_log(run, metrics["config"], f"{condition}{seed}")
                    records[name] = metrics
                    result[name] = metrics
                else:
                    result[name] = "待运行"
    validate_comparability(records)
    for seed in (42, 43, 44):
        if all((c, seed) in training_orders for c in ("A", "B")):
            require(training_orders["A", seed] == training_orders["B", seed],
                    f"seed {seed}: training example order or token/sentence counts differ between A/B")
    aggregates = {}
    for condition in ("A1", "A2", "B1", "B2"):
        scores = {seed: records[f"{condition}_seed{seed}"]["bleu"] for seed in (42, 43, 44)
                  if f"{condition}_seed{seed}" in records}
        values = list(scores.values())
        aggregates[condition] = ({"seeds_completed": len(values), "seeds": list(scores),
            "bleu_mean": statistics.mean(values),
            "bleu_sample_std": statistics.stdev(values) if len(values) > 1 else None}
            if values else "待运行")
    result["aggregate"] = aggregates
    result["paired_seeds_completed"] = [seed for seed in (42, 43, 44)
                                         if all(f"{c}_seed{seed}" in records for c in ("A1", "A2", "B1", "B2"))]
    result["comparison_note"] = ("仅已完成全部四组的种子可支持成对实验比较；单种子不支持稳定统计结论。"
                                 "缺少的实验保持待运行。")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="artifacts/runs")
    p.add_argument("--out", default="artifacts/experiment_summary.json")
    args = p.parse_args()
    result = collect(args.runs)
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
