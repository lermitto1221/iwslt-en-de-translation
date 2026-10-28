"""Check public files, relative links and published numerical evidence.

This checks the curated source package, not every possible secret or the
unpublished raw predictions. Git internals and Python caches are never scanned.
"""
import json
import math
import os
import re
import statistics
from pathlib import Path
from urllib.parse import unquote, urlparse


root = Path(__file__).resolve().parents[1]
ignored_dirs = {".git", "__pycache__"}
forbidden_dirs = {".venv", "venv", ".analysis-deps", "node_modules", "artifacts"}
forbidden_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".parquet", ".model", ".vocab", ".jsonl", ".pem", ".key"}
text_suffixes = {".md", ".json", ".txt", ".yml", ".yaml", ".toml", ".py", ".ps1", ".sh", ".cfg", ".ini"}
local_path = re.compile(r"(?i)(?:\b[a-z]:[\\/]+(?:users|develop)[\\/]+|/" + r"users/|/" + r"home/[^/\s]+/)")
secret_patterns = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
    "API token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
}
issues = []
files = []
for directory, dirs, names in os.walk(root, followlinks=False):
    # Prune before descending, so a local checkout/cache does not affect review.
    dirs[:] = sorted(d for d in dirs if d not in ignored_dirs)
    for name in list(dirs):
        path = Path(directory) / name
        if path.is_symlink():
            issues.append(f"Symlink directory requires manual review: {path.relative_to(root)}")
            dirs.remove(name)
    for name in sorted(names):
        path = Path(directory) / name
        if path.is_symlink():
            issues.append(f"Symlink file requires manual review: {path.relative_to(root)}")
            continue
        files.append(path)
for path in files:
    relative = path.relative_to(root)
    if set(relative.parts) & forbidden_dirs or path.suffix.lower() in forbidden_suffixes:
        issues.append(f"Unexpected generated/private file: {relative}")
    if path.name == ".env" or path.name.startswith(".env."):
        issues.append(f"Environment file requires manual review: {relative}")
    if path.stat().st_size >= 25 * 1024 * 1024:
        issues.append(f"File exceeds browser upload size: {relative}")
    if path.suffix.lower() not in text_suffixes:
        continue
    content = path.read_text(encoding="utf-8")
    if local_path.search(content):
        issues.append(f"Personal local absolute path in public text: {relative}")
    for label, pattern in secret_patterns.items():
        if pattern.search(content):
            issues.append(f"Possible {label} in public text: {relative}")
    if path.suffix.lower() == ".md":
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            href = match.group(1).strip().strip("<>")
            if urlparse(href).scheme or href.startswith("#"):
                continue
            target = unquote(href.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                issues.append(f"Broken link: {relative} -> {target}")


def check(condition, message):
    if not condition:
        issues.append(message)


def close(actual, expected):
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10))


def check_stats(value, values, label, expected_n=None):
    check(all(isinstance(x, (int, float)) and math.isfinite(x) for x in values), f"Non-finite values: {label}")
    check(len(values) > 1, f"Insufficient values for sample SD: {label}")
    if len(values) <= 1:
        return
    check(close(value["mean"], statistics.mean(values)), f"Mean arithmetic mismatch: {label}")
    check(close(value["sample_std"], statistics.stdev(values)), f"Sample SD arithmetic mismatch: {label}")
    check(value.get("ddof") == 1, f"Sample SD must use ddof=1: {label}")
    if expected_n is not None:
        check(value.get("n") == expected_n == len(values), f"Count mismatch: {label}")


def visit_seed_stats(value, location="cv_adaptation"):
    if isinstance(value, dict):
        if {"per_seed", "mean", "sample_std"} <= value.keys():
            by_seed = value["per_seed"]
            check(set(by_seed) == {"42", "43", "44"}, f"Unexpected seed keys: {location}")
            check_stats(value, list(by_seed.values()), location, 3)
            if "included" in value:
                check(value["included"] == ["42", "43", "44"], f"Unexpected included seeds: {location}")
        for key, item in value.items():
            visit_seed_stats(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            visit_seed_stats(item, f"{location}[{index}]")


def check_difference(output, left, right, label):
    for metric in ("bleu", "chrf2"):
        for seed in ("42", "43", "44"):
            right_value = right[metric]["per_seed"][seed] if isinstance(right[metric], dict) else right[metric]
            expected = left[metric]["per_seed"][seed] - right_value
            check(close(output[metric]["per_seed"][seed], expected), f"Paired difference mismatch: {label}.{metric}.{seed}")


# Retain the historical result check; the new study does not replace that work.
summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
budget = summary["training_budget_comparison"]
check(budget["n"] == 3, "Historical budget comparison count differs")
actual = budget["groups"]["long_beam"]["bleu"]
check(close(actual["mean"], 23.13723375311204), "Historical long Beam BLEU mean differs")
check(close(actual["sample_std"], 0.32055083606193596), "Historical long Beam BLEU SD differs")

study = json.loads((root / "results/cv_adaptation.json").read_text(encoding="utf-8"))
check(study["schema_version"] == 1 and study["status"] == "complete", "Unexpected CV/adaptation schema or status")
check(study["seeds"] == [42, 43, 44], "Unexpected study seeds")
check(study["test_samples"] == study["dataset"]["test_pairs"] == 8079, "Test sample counts differ")
visit_seed_stats(study)

cv = study["cv"]
check(cv["folds"] == 3 and cv["updates_each"] == 3000, "Unexpected CV fold count or budget")
candidates = list(cv["candidates"].values())
check([candidate["label_smoothing"] for candidate in candidates] == [0.0, 0.1], "CV candidate order differs")
for candidate in candidates:
    folds = candidate["folds"]
    check([fold["fold"] for fold in folds] == [0, 1, 2], "CV fold IDs differ")
    check(sum(fold["samples"] for fold in folds) == study["dataset"]["cleaned_train_pairs"], "CV holdout coverage differs")
    for metric in ("bleu", "chrf2"):
        check_stats(candidate["summary"][metric], [fold[metric] for fold in folds], f"CV {candidate['label_smoothing']} {metric}")
    check(candidate["oof"]["samples"] == study["dataset"]["cleaned_train_pairs"] and candidate["oof"]["n_folds"] == 3, "OOF coverage differs")
winner = max(enumerate(candidates), key=lambda item: (item[1]["summary"]["bleu"]["mean"], item[1]["summary"]["chrf2"]["mean"], -item[0]))[1]
check(cv["selected_label_smoothing"] == winner["label_smoothing"], "CV selection does not follow frozen mean BLEU/chrF2/order rule")
audit = cv["group_audit"]
check(audit["cross_fold_normalized_source_keys"] == audit["cross_fold_talk_ids"] == 0, "Reported grouping leakage is nonzero")
check(audit["talk_isolation_guaranteed"] is True, "Talk isolation guarantee differs")
check(sum(fold["holdout_pairs"] for fold in audit["folds"]) == audit["cleaned_pairs"], "Group audit coverage differs")

for method in ("lora", "full"):
    choices = study["adaptation_selection"]["candidates"][method]
    check(len(choices) == 2 and all(item["split"] == "validation" and item["seed"] == 7 for item in choices), f"Adaptation candidate setup differs: {method}")
    selected = max(enumerate(choices), key=lambda item: (item[1]["bleu"], item[1]["chrf2"], -item[0]))[1]
    check(study["adaptation_selection"]["selected_lrs"][method] == selected["lr"], f"Adaptation LR selection differs: {method}")
    resource = study["resources"][method]
    check(0 < resource["trainable_parameters"] <= resource["total_parameters"], f"Parameter counts differ: {method}")
    check(close(resource["trainable_fraction"], resource["trainable_parameters"] / resource["total_parameters"]), f"Trainable fraction differs: {method}")

for decoder in ("greedy", "beam"):
    direct = study["methods"]["direct"][decoder]
    check(direct["fixed_model_count"] == 1 and direct["sample_std"] is None, "Direct inference is one fixed model")
    for method in ("cv_final", "lora", "full"):
        result = study["methods"][method][decoder]
        check(result["n"] == 3 and result["included_seeds"] == [42, 43, 44] and not result["excluded_pending_seeds"], f"Incomplete formal method: {method}/{decoder}")
    check_difference(study["paired_lora_minus_full"][decoder], study["methods"]["lora"][decoder], study["methods"]["full"][decoder], f"test/{decoder}/lora-full")
    for method in ("lora", "full"):
        check_difference(study["gains_vs_direct"][decoder][method], study["methods"][method][decoder], direct, f"test/{decoder}/{method}-direct")
    validation = study["validation"][decoder]
    check_difference(validation["paired_lora_minus_full"], validation["lora"], validation["full"], f"validation/{decoder}/lora-full")
    for method in ("lora", "full"):
        check_difference(validation[f"{method}_minus_direct"], validation[method], validation["direct"], f"validation/{decoder}/{method}-direct")

counts = study["study"]
stages = list(study["stage_costs"].values())
for stage in stages:
    for planned, completed in (("planned_runs", "completed_training_runs"), ("planned_optimizer_updates", "completed_run_updates"), ("planned_full_evaluation_passes", "completed_full_evaluation_passes"), ("planned_evaluation_sentences", "evaluated_sentences")):
        check(stage[planned] == stage[completed], f"Incomplete stage count: {stage['title']}/{completed}")
check(sum(stage["completed_training_runs"] for stage in stages) == counts["formal_runs"] == 19, "Formal run totals differ")
check(sum(stage["completed_run_updates"] for stage in stages) == counts["net_updates"] == 57000, "Formal update totals differ")
check(sum(stage["completed_full_evaluation_passes"] for stage in stages) == counts["eval_groups"] == 102, "Evaluation group totals differ")
check(sum(stage["evaluated_sentences"] for stage in stages) == counts["evaluation_sentence_exposures"], "Evaluation sentence totals differ")
tests = counts["original_research_tests"]
check(tests["tests_run"] == tests["passed"] == 349 and tests["failures"] == tests["errors"] == tests["skips"] == 0, "Original research test record differs")
check(counts["historical_files_checked"] == 1393 and counts["historical_files_changed"] == 0, "Historical fingerprint record differs")
if issues:
    raise SystemExit("\n".join(issues))
print(f"PASS: {len(files)} public files; links, privacy patterns, generated-file exclusions, historical results and CV/adaptation arithmetic verified.")
