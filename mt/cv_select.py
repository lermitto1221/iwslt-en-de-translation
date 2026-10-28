"""Select smoothing using only complete grouped CV holdouts; rescore full OOF.

The primary selection statistic is an unweighted arithmetic mean of three
fold corpus scores. OOF corpus scores are separate descriptive statistics.
No official validation/test prediction, text, or metric is read here.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

import torch

from .common import read_json, sha256
from .cv_evaluate import INFERENCE_SOURCES, _digest, corpus_metrics, verify_results
from .cv_prepare import _verify_files, fit_scope, partition_rows, source_digest
from .cv_train import validate_completed_run
from .data import read_pairs
from .inference import validate_protocol
from .representation_prepare import _immutable_bytes, _immutable_json, _jsonl_bytes

CANDIDATES = (0.0, 0.1)
PAIRED_LOG_FIELDS = ("update", "epoch", "batch_cursor", "lr_used", "sentences",
                     "source_tokens", "target_tokens", "padded_tokens",
                     "data_order_sha256", "microbatch_sha256")
SELECTION_RULE = {
    "primary": "maximum arithmetic mean of three complete holdout Greedy corpus BLEU scores",
    "exact_tie_1": "maximum arithmetic mean of the same three holdout corpus chrF2 scores",
    "exact_tie_2": "first candidate in fixed order [0.0, 0.1]",
    "oof_role": "descriptive corpus rescore; never used to choose or reorder candidates",
    "fold_standard_deviation": "sample standard deviation ddof=1 across folds, not across random seeds",
    "checkpoint": "last.pt at exactly update 3000; no heldout checkpoint choice",
    "scope": "hyperparameter selection at 3000 updates; not independent final confirmation or evidence for 12000-update optimality",
}


def candidate_tag(value):
    if type(value) not in (int, float) or value not in CANDIDATES:
        raise ValueError("unknown label-smoothing candidate")
    return "ls0" if value == 0 else "ls01"


def summarize_candidates(fold_scores, candidate_order=CANDIDATES):
    """Choose only from fold statistics; deliberately no OOF argument."""
    if tuple(candidate_order) != CANDIDATES or set(fold_scores) != {"ls0", "ls01"}:
        raise ValueError("exactly the two frozen smoothing candidates are required")
    candidates = {}
    for smoothing in candidate_order:
        tag = candidate_tag(smoothing)
        rows = sorted(fold_scores[tag], key=lambda row: row["fold"])
        if [row.get("fold") for row in rows] != [0, 1, 2]:
            raise ValueError("each candidate requires all three unique folds; missing scores are not zero")
        for row in rows:
            if type(row.get("samples")) is not int or row["samples"] < 1:
                raise ValueError("fold scores require positive sample counts")
            for metric in ("bleu", "chrf2"):
                value = row.get(metric)
                # SacreBLEU can return 100.00000000000004 for perfect corpora.
                if type(value) not in (int, float) or not math.isfinite(value) or not -1e-9 <= value <= 100 + 1e-9:
                    raise ValueError("fold score is missing or not a finite corpus metric")
        candidates[tag] = {"label_smoothing": smoothing, "folds": rows, "n_folds": 3,
            "summary": {metric: {"mean": statistics.mean(row[metric] for row in rows),
                "sample_std": statistics.stdev(row[metric] for row in rows), "ddof": 1}
                for metric in ("bleu", "chrf2")}}
    selected = max(candidate_order, key=lambda value: (
        candidates[candidate_tag(value)]["summary"]["bleu"]["mean"],
        candidates[candidate_tag(value)]["summary"]["chrf2"]["mean"],
        -candidate_order.index(value)))
    return {"selected_label_smoothing": selected, "candidates": candidates,
            "selection_rule": SELECTION_RULE, "candidate_order": list(candidate_order)}


def merge_oof(population, folds):
    """Restore original full-training row order, refusing any missing/extra ID."""
    originals = {row["id"]: row for row in population}
    if len(originals) != len(population) or not originals or len(folds) != 3:
        raise ValueError("OOF requires unique original IDs and exactly three complete folds")
    merged = {}
    for fold, examples in enumerate(folds):
        if not examples:
            raise ValueError("OOF cannot contain an empty fold")
        for example in examples:
            identifier = example.get("id")
            if type(identifier) is not int or identifier in merged or identifier not in originals:
                raise ValueError("OOF has duplicate, unknown, or invalid original IDs")
            row = originals[identifier]
            if (example.get("source") != row["en"] or example.get("reference") != row["de"]
                    or not isinstance(example.get("prediction"), str)
                    or "\n" in example["prediction"] or "\r" in example["prediction"]):
                raise ValueError("OOF source/reference/prediction alignment differs")
            merged[identifier] = {**example, "fold": fold}
    if set(merged) != set(originals):
        raise ValueError("OOF is incomplete; every cleaned official training row must appear exactly once")
    examples = [merged[row["id"]] for row in population]
    metrics = corpus_metrics([e["prediction"] for e in examples], [e["reference"] for e in examples])
    return examples, {**metrics, "samples": len(examples), "n_folds": 3,
        "order": "original cleaned full official training row order, identified by original raw IDs",
        "statistic": "one corpus rescore of all OOF predictions; not a mean of sentence or fold scores"}


def validate_candidate_config(cfg, root, fold, smoothing):
    root = Path(root)
    tag = candidate_tag(smoothing)
    if (cfg.get("stage") != "cv" or type(cfg.get("fold")) is not int or cfg["fold"] != fold
            or cfg.get("candidate_smoothing") != smoothing or cfg.get("label_smoothing") != smoothing
            or type(cfg.get("candidate_smoothing")) not in (int, float)
            or type(cfg.get("label_smoothing")) not in (int, float)
            or cfg.get("seed") != 42 or cfg.get("max_updates") != 3000
            or cfg.get("checkpoint_selection") != "fixed_last" or cfg.get("representation") != "bpe"):
        raise ValueError("candidate config has wrong fold/smoothing/seed/stage or checkpoint budget")
    data = root / "data" / "cv" / f"fold{fold}"
    expected = {"data_dir": data, "run_dir": root / "cv" / "runs" / f"fold{fold}_{tag}",
                "common_ids": data / "common_ids.json", "batch_plan": data / "batch_plan_seed42.jsonl"}
    for key, path in expected.items():
        if Path(cfg[key]).resolve() != path.resolve():
            raise ValueError(f"candidate {key} does not point to this fold's canonical evidence")
    return data, expected["run_dir"]


def validate_pair(left, right, left_records, right_records, left_link, right_link):
    allowed = {"label_smoothing", "candidate_smoothing", "run_dir"}
    if {k: v for k, v in left.items() if k not in allowed} != {k: v for k, v in right.items() if k not in allowed}:
        raise ValueError("same-fold candidates differ in settings beyond smoothing/output directory")
    for key in ("initial_model_sha256", "tokenizer_sha256", "parameter_counts"):
        if key not in left_link or left_link.get(key) != right_link.get(key):
            raise ValueError("same-fold candidates do not share initialization/tokenizer/model size")
    if (len(left_records) != 3000 or len(right_records) != 3000
            or [r.get("update") for r in left_records] != list(range(1, 3001))
            or [r.get("update") for r in right_records] != list(range(1, 3001))):
        raise ValueError("both candidates must have all 3000 continuous updates")
    for a, b in zip(left_records, right_records):
        if any(key not in a or key not in b or a[key] != b[key] for key in PAIRED_LOG_FIELDS):
            raise ValueError("paired training IDs/boundaries/learning rates/token exposure differ")
        if any(key.startswith(("validation", "holdout")) for key in set(a) | set(b)):
            raise ValueError("holdout evaluation must not select training checkpoint")
    return {"same_initialization": True, "same_tokenizer": True, "same_config_except_smoothing_and_output": True,
        "all_3000_update_ids_boundaries_lr_exposure_match": True,
        "initial_model_sha256": left_link["initial_model_sha256"],
        "tokenizer_sha256": left_link["tokenizer_sha256"]}


def validate_evaluation_identity(identity, cfg, run, data):
    """Check canonical paths before verify_results can open identity.input_path."""
    checkpoint = Path(run) / "last.pt"
    source = Path(data) / "holdout.jsonl"
    if (identity.get("split") != "holdout" or identity.get("method") != "greedy"
            or Path(identity.get("input_path", "")).resolve() != source.resolve()
            or Path(identity.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or identity.get("checkpoint_update") != 3000 or identity.get("precision") != "float32"
            or identity.get("batch_size") != cfg["decode_batch_size"]
            or identity.get("protocol") != validate_protocol(cfg)
            or identity.get("config_sha256") != _digest(cfg)):
        raise ValueError("evaluation identity is not the frozen CV holdout/checkpoint/protocol")
    if identity.get("checkpoint_sha256") != sha256(checkpoint) or identity.get("input_sha256") != sha256(source):
        raise ValueError("evaluation checkpoint/input hashes differ from frozen evidence")
    if identity.get("tokenizer_sha256") != sha256(Path(data) / "spm.model"):
        raise ValueError("evaluation tokenizer differs from fold tokenizer")
    current = {name: sha256(Path(__file__).with_name(name)) for name in INFERENCE_SOURCES}
    if identity.get("source_hashes") != current:
        raise ValueError("evaluation implementation differs from current frozen source")


def _population(root, source):
    group_dir = root / "data" / "cv"
    complete = read_json(group_dir / "grouping_complete.json")
    _verify_files(group_dir, complete)
    request = complete["request"]
    if (sha256(root / "design.json") != request["design_sha256"]
            or sha256(source / "train.jsonl") != request["source_train_sha256"]
            or sha256(source / "manifest.json") != request["source_manifest_sha256"]):
        raise ValueError("design or official training input differs from frozen grouping")
    rows = read_pairs(source / "train.jsonl")
    original = read_json(source / "manifest.json")
    if len(rows) != original["files"]["train"]["rows_saved"]:
        raise ValueError("official training population is incomplete")
    groups = read_pairs(group_dir / "groups.jsonl")
    if [r["id"] for r in groups] != [r["id"] for r in rows] or len({r["id"] for r in groups}) != len(rows):
        raise ValueError("group map does not cover original training IDs exactly in order")
    if any(record["normalized_source_sha256"] != source_digest(row["en"]) for row, record in zip(rows, groups)):
        raise ValueError("normalized grouping sources differ from original training text")
    assignment = {r["id"]: r["fold"] for r in groups}
    return rows, assignment


def select(root, source="artifacts/data"):
    root, source = Path(root), Path(source)
    design = read_json(root / "design.json")
    if (design["cv"]["candidate_order"] != list(CANDIDATES) or design["cv"]["updates"] != 3000
            or design["cv"]["model_seed"] != 42 or design["cv"]["folds"] != 3):
        raise ValueError("selection requires the frozen three-fold/two-candidate design")
    population, assignment = _population(root, source)
    input_hashes = {}

    def remember(path):
        input_hashes[str(Path(path).resolve())] = sha256(path)

    for path in (root / "design.json", root / "data" / "cv" / "grouping_complete.json",
                 root / "data" / "cv" / "groups.jsonl", source / "train.jsonl", source / "manifest.json", Path(__file__)):
        remember(path)
    scores = {candidate_tag(value): [] for value in CANDIDATES}
    all_examples = {candidate_tag(value): [] for value in CANDIDATES}
    controls, protocol_identities = [], []
    for fold in range(3):
        expected_train, holdout = partition_rows(population, assignment, fold)
        data = root / "data" / "cv" / f"fold{fold}"
        _verify_files(data, read_json(data / "preparation_complete.json"))
        if (read_pairs(data / "train.jsonl") != expected_train or read_pairs(data / "holdout.jsonl") != holdout
                or read_json(data / "tokenizer_fit_ids.json") != fit_scope(expected_train, holdout)):
            raise ValueError("fold training/holdout/tokenizer fit scope differs from frozen population partition")
        for filename in ("preparation_complete.json", "train.jsonl", "holdout.jsonl", "tokenizer_fit_ids.json",
                         "tokenizer.json", "spm.model", "common_ids.json", "batch_plan_seed42.jsonl"):
            remember(data / filename)
        pair = []
        for smoothing in CANDIDATES:
            tag = candidate_tag(smoothing)
            config_path = root / "cv" / "configs" / f"fold{fold}_{tag}.json"
            cfg = read_json(config_path)
            _, run = validate_candidate_config(cfg, root, fold, smoothing)
            if any((run / name).exists() for name in ("best.pt", "validation.jsonl", "validation_log.jsonl")):
                raise ValueError("CV run must not contain a best/validation-selected checkpoint history")
            summary = validate_completed_run(cfg)
            records, link = read_pairs(run / "train.jsonl"), read_json(run / "data_link.json")
            out = run / "holdout_greedy"
            identity = read_json(out / "identity.json")
            validate_evaluation_identity(identity, cfg, run, data)
            state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            if identity.get("model_sha256") != state.get("model_sha256"):
                raise ValueError("evaluated tensor hash differs from verified last checkpoint")
            del state
            metrics = verify_results(out, holdout)
            examples = read_pairs(out / "examples.jsonl")
            if (metrics["checkpoint_update"] != 3000 or metrics["split"] != "holdout"
                    or metrics["method"] != "greedy" or metrics["samples"] != len(holdout)):
                raise ValueError("fold metric is not the complete fixed-update Greedy holdout")
            protocol_identities.append({key: identity[key] for key in ("protocol", "precision", "batch_size", "source_hashes")})
            scores[tag].append({"fold": fold, "samples": len(holdout), "bleu": metrics["bleu"], "chrf2": metrics["chrf2"],
                "signature": metrics["signature"], "chrf2_signature": metrics["chrf2_signature"],
                "checkpoint_sha256": metrics["checkpoint_sha256"], "training_seconds": summary["training_seconds_total"],
                "heldout_decode_seconds": metrics["elapsed_seconds"], "generation_limit_hits": metrics["generation_limit_hits"],
                "exposure": summary["exposure"]})
            all_examples[tag].append(examples)
            pair.append((cfg, records, link))
            for path in (config_path, run / "last.pt", run / "train.jsonl", run / "data_link.json", run / "summary.json",
                         run / "source_hashes.json", out / "identity.json", out / "progress.json", out / "metrics.json",
                         out / "predictions.txt", out / "references.txt", out / "examples.jsonl"):
                remember(path)
        controls.append({"fold": fold, **validate_pair(pair[0][0], pair[1][0], pair[0][1], pair[1][1], pair[0][2], pair[1][2])})
    if any(value != protocol_identities[0] for value in protocol_identities[1:]):
        raise ValueError("CV candidates/folds do not share the same formal inference protocol")
    signatures = {(row["signature"], row["chrf2_signature"]) for rows in scores.values() for row in rows}
    if len(signatures) != 1:
        raise ValueError("fold metric signatures differ")
    selected = summarize_candidates(scores)
    prepared_oof = {}
    for tag, folds in all_examples.items():
        examples, metrics = merge_oof(population, folds)
        prepared_oof[tag] = (examples, metrics)
        selected["candidates"][tag]["oof"] = {**metrics,
            "output_directory": str((root / "cv" / "oof" / tag).resolve())}
    output = root / "cv" / "selection.json"
    # Complete result identity is computed before publishing anything. Existing
    # evidence is reused only after a full read-only validation of its inputs.
    created = read_json(output)["created_at_utc"] if output.exists() else datetime.now(timezone.utc).isoformat()
    result = {"schema_version": 1, "created_at_utc": created, "complete": True,
        **selected, "paired_controls": controls, "input_hashes": input_hashes,
        "official_training_samples": len(population), "official_validation_or_test_used_for_selection": False,
        "history_known": True, "grouping_complete_sha256": sha256(root / "data" / "cv" / "grouping_complete.json")}
    if output.exists() and read_json(output) != result:
        raise FileExistsError("refusing to overwrite a different CV selection or its evidence")
    for tag, (examples, metrics) in prepared_oof.items():
        out = root / "cv" / "oof" / tag
        _immutable_bytes(out / "predictions.txt", ("\n".join(e["prediction"] for e in examples) + "\n").encode("utf-8"))
        _immutable_bytes(out / "references.txt", ("\n".join(e["reference"] for e in examples) + "\n").encode("utf-8"))
        _immutable_bytes(out / "examples.jsonl", _jsonl_bytes(examples))
        _immutable_json(out / "metrics.json", {**metrics,
            "artifacts": {name: sha256(out / name) for name in ("predictions.txt", "references.txt", "examples.jsonl")},
            "input_evidence_sha256": _digest(input_hashes), "selection_role": "not used to select the candidate"})
    _immutable_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", default="artifacts/data")
    args = parser.parse_args()
    import json
    print(json.dumps(select(args.root, args.source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
