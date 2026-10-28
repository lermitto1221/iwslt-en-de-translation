"""Final delivery gate; default checks are lightweight and never use a model.

Usage:
  project-pretrained-python.ps1 -m mt.study_complete --root R --check-only
  project-pretrained-python.ps1 -m mt.study_complete --root R --finalize

Required final-only manifests (relative to R, overridable on the CLI):
* regression_final.json: full_suite=true, tests_run/passed/failures/errors/skips,
  log_sha256; log defaults to the same stem with .log. Counts must match real
  unittest output. It is one full-suite run, never a sum of targeted subsets.
* presentation/delivery_manifest.json: status=complete, draft=false,
  source_results_sha256, outputs {relative path: sha256} with PPTX and PDF,
  qa {path, sha256, passed:true}; the QA JSON must itself say passed:true.
* report/sample_review_manifest.json: status=complete, review_type=agent_review,
  source_results_sha256, source_samples_sha256, outputs including sample_review.md.

Only --finalize hashes the complete scientific evidence, 1393 protected old
files, and new-file inventory. Run it after GPU work and observers finish.
It does not regenerate reports, deserialize checkpoints, or run inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
HISTORICAL_FILES = 1393
EXPECTED_STAGES = {"cv": (6, 18000, 6, 408570), "cv_final": (3, 9000, 12, 53802),
                   "adapt_tune": (4, 12000, 24, 21312), "adapt_final": (6, 18000, 60, 139572)}
SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".venv", ".analysis-deps", ".cache", "cache", "caches", ".pytest_cache", ".mypy_cache"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    # Windows PowerShell 5 may publish UTF-8 JSON with a BOM.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path != root and path.is_relative_to(root), "path escapes intended directory: " + str(path))
    return path


def write_once(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "refusing to overwrite existing delivery evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    temporary.rename(path)


def stable_created(path):
    return read(path)["created_at_utc"] if Path(path).exists() else now()


def verify_files(mapping, base=None):
    require(isinstance(mapping, dict) and mapping, "empty file-fingerprint mapping")
    resolved = {}
    for name, expected in mapping.items():
        path = Path(name)
        path = inside(base, base / path) if base is not None else path.resolve()
        require(path.is_file(), "missing evidence file: " + str(path))
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "invalid SHA-256")
        require(sha(path) == expected, "changed evidence file: " + str(path))
        resolved[str(path)] = expected
    return resolved


def expected_runs(root):
    for fold in range(3):
        for smoothing, tag in ((0.0, "ls0"), (.1, "ls01")):
            name = f"fold{fold}_{tag}"
            yield {"stage": "cv", "name": name, "seed": 42, "smoothing": smoothing, "fold": fold,
                   "config": root / "cv/configs" / (name + ".json"), "run": root / "cv/runs" / name}
    for seed in SEEDS:
        yield {"stage": "cv_final", "name": f"seed{seed}", "seed": seed,
               "config": root / "cv/final_configs" / f"seed{seed}.json", "run": root / "cv/final_runs" / f"seed{seed}"}
    for method, rates in (("lora", (1e-4, 3e-4)), ("full", (1e-5, 3e-5))):
        for lr in rates:
            name = f"tune_{method}_lr{lr:g}_seed7"
            yield {"stage": "adapt_tune", "name": name, "seed": 7, "method": method, "lr": lr,
                   "config": root / "adaptation/configs" / (name + ".json"), "run": root / "adaptation/runs" / name}
    for seed in SEEDS:
        for method in ("lora", "full"):
            name = f"final_{method}_seed{seed}"
            yield {"stage": "adapt_final", "name": name, "seed": seed, "method": method,
                   "config": root / "adaptation/configs" / (name + ".json"), "run": root / "adaptation/runs" / name}


def metrics_for(spec):
    run, stage = spec["run"], spec["stage"]
    if stage == "cv":
        yield run / "holdout_greedy/metrics.json", None
    if stage in ("adapt_tune", "adapt_final"):
        for update in range(500, 3001, 500):
            yield run / "validation" / f"step_{update:05d}" / "metrics.json", 888
    if stage in ("cv_final", "adapt_final"):
        for split, count in (("validation", 888), ("test", 8079)):
            for method in ("greedy", "beam"):
                path = run / f"{split}_{method}" if stage == "cv_final" else run / "evaluation" / split / method
                yield path / "metrics.json", count


def collect_run_metadata(root, pending):
    records = []
    specs = list(expected_runs(root))
    require(len(specs) == 19 and len({str(s["run"]) for s in specs}) == 19, "formal runs are not 19 unique directories")
    for spec in specs:
        cfg_path, run = spec["config"], spec["run"]
        if not cfg_path.is_file():
            pending.append("missing configuration: " + str(cfg_path)); continue
        cfg = read(cfg_path)
        require(inside(root, cfg["run_dir"]) == run, "formal run path/label mismatch")
        require(cfg["seed"] == spec["seed"], "wrong formal seed")
        if spec["stage"] in ("cv", "cv_final"):
            require(cfg["max_updates"] == 3000 and cfg["checkpoint_selection"] == "fixed_last", "wrong CV training budget/selection")
            require(cfg["stage"] == ("cv" if spec["stage"] == "cv" else "final"), "wrong CV stage")
            if spec["stage"] == "cv":
                require(cfg["fold"] == spec["fold"] and cfg["label_smoothing"] == spec["smoothing"], "wrong CV candidate")
        else:
            require(cfg["updates"] == 3000 and cfg["method"] == spec["method"], "wrong native method/budget")
            if spec["stage"] == "adapt_tune":
                require(cfg["lr"] == spec["lr"], "wrong tuning learning rate")
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            pending.append("training pending: " + str(run)); continue
        summary = read(summary_path)
        require(summary["updates"] == 3000, "summary claims incomplete or expanded formal budget")
        if spec["stage"] in ("cv", "cv_final"):
            require(summary.get("complete") is True, "CV summary is not complete")
        for name in ("last.pt", "train.jsonl") + (("best.pt",) if spec["stage"].startswith("adapt") else ()):
            require((run / name).is_file(), "completed run lacks " + name)
        evaluations = []
        for path, expected in metrics_for(spec):
            if not path.is_file():
                pending.append("evaluation pending: " + str(path)); continue
            metric = read(path)
            if expected is None:
                expected = read(Path(cfg["data_dir"]) / "manifest.json")["files"]["holdout"]["rows_saved"]
            require(metric["samples"] == expected, "incomplete evaluation count")
            evaluations.append({"path": str(path), "samples": expected})
        records.append({**{k: v for k, v in spec.items() if k not in ("config", "run")},
                        "config": str(cfg_path), "run": str(run), "updates": 3000,
                        "summary": str(summary_path), "evaluations": evaluations})
    return records


def inspect(root, regression_json=None, presentation_manifest=None, sample_review_manifest=None):
    root = Path(root).resolve(); pending = []
    paths = {"progress": root / "progress.json", "plan": root / "plan.json",
             "cv_completion": root / "cv/completion.json", "cv_final_completion": root / "cv/final_completion.json",
             "cv_selection": root / "cv/selection.json", "adaptation_selection": root / "adaptation/selection.json",
             "results": root / "report/results.json", "report_manifest": root / "report/report_manifest.json",
             "regression": Path(regression_json) if regression_json else root / "regression_final.json",
             "presentation": Path(presentation_manifest) if presentation_manifest else root / "presentation/delivery_manifest.json",
             "sample_review": Path(sample_review_manifest) if sample_review_manifest else root / "report/sample_review_manifest.json",
             "historical_manifest": root / "historical_manifest.json"}
    for key, path in paths.items():
        paths[key] = inside(root, path)
        if not path.is_file():
            pending.append("missing " + key + ": " + str(path))
    if paths["progress"].exists():
        progress = read(paths["progress"])
        if not (progress.get("status") == "completed" and progress.get("formal_runs") == 19 and progress.get("formal_updates") == 57000):
            pending.append("controller has not marked all19 formal runs complete")
    records = collect_run_metadata(root, pending)
    if paths["results"].exists():
        results = read(paths["results"])
        if results.get("complete") is not True or results.get("status") != "complete":
            pending.append("strict final report is pending")
    if paths["report_manifest"].exists() and read(paths["report_manifest"]).get("status") != "complete":
        pending.append("final report manifest is pending")
    for key in ("presentation", "sample_review"):
        if paths[key].exists() and read(paths[key]).get("status") != "complete":
            pending.append(key + " manifest is pending")
    if len(records) != 19:
        pending.append(f"only {len(records)}/19 complete-training summaries present")
    return {"status": "pending" if pending else "ready_for_deep_validation", "complete": False,
            "check_scope": "metadata only; no checkpoint deserialization, full report recomputation, protected-history scan or file inventory",
            "pending": pending, "formal_training_summaries": len(records), "runs": records,
            "paths": {k: str(v) for k, v in paths.items()}}


def verify_regression(path):
    path = Path(path); record = read(path)
    require(record.get("full_suite") is True, "final regression record must identify one full suite")
    count, passed, failed, errors, skipped = (record.get(k) for k in ("tests_run", "passed", "failures", "errors", "skips"))
    require(all(type(v) is int and v >= 0 for v in (count, passed, failed, errors, skipped)) and count > 0, "invalid regression counts")
    require(failed == errors == 0 and passed + skipped == count, "final regression did not pass")
    log = Path(record.get("log", path.with_suffix(".log")))
    require(log.is_file() and sha(log) == record.get("log_sha256"), "final regression log missing or changed")
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", text)
    require(len(matches) == 1 and int(matches[0][0]) == count, "regression count differs from actual single-suite log")
    success_lines = re.findall(r"(?m)^OK(?: \(skipped=(\d+)\))?[ \t]*$", text)
    # CPython extension shutdown warnings may follow the unittest OK line.
    require(len(success_lines) == 1 and not re.search(r"(?m)^FAILED(?:\s|$)", text), "regression log lacks successful single-suite status")
    require(int(success_lines[0] or 0) == skipped, "regression skip count differs")
    require(record.get("exit_code", 0) == 0, "final regression process exited unsuccessfully")
    if record.get("source_files"):
        verify_files(record["source_files"])
    return {"path": str(path), "sha256": sha(path), **record, "suite_seconds_from_log": float(matches[0][1]),
            "counting_rule": "one actually executed full suite; targeted subsets and historical runs are not added"}


def verify_delivery_manifest(path, result_hash, sample_hash=None):
    path = Path(path); value = read(path)
    require(value.get("status") == "complete", "delivery manifest is incomplete")
    require(value.get("source_results_sha256") == result_hash, "delivery belongs to stale results")
    outputs = verify_files(value.get("outputs"), path.parent)
    if sample_hash is not None:
        require(value.get("review_type") == "agent_review", "sample review must be labeled agent review")
        require(value.get("source_samples_sha256") == sample_hash, "sample review belongs to stale samples")
        review = [Path(p) for p in outputs if Path(p).name == "sample_review.md"]
        require(len(review) == 1 and review[0].stat().st_size > 20, "actual sample review Markdown missing")
    else:
        require(value.get("draft") is False, "draft presentation cannot complete delivery")
        decks = [Path(p) for p in outputs if Path(p).suffix.lower() == ".pptx"]
        pdfs = [Path(p) for p in outputs if Path(p).suffix.lower() == ".pdf"]
        require(decks and pdfs, "both editable PPTX and PDF are required")
        for deck in decks:
            require(zipfile.is_zipfile(deck), "presentation is not a real PPTX package")
            with zipfile.ZipFile(deck) as archive:
                require({"[Content_Types].xml", "ppt/presentation.xml"}.issubset(archive.namelist()), "PPTX package lacks presentation content")
        for pdf in pdfs:
            with pdf.open("rb") as handle:
                require(handle.read(5) == b"%PDF-", "presentation preview is not a PDF")
        qa = value.get("qa", {})
        qa_path = inside(path.parent, path.parent / qa.get("path", ""))
        require(qa.get("passed") is True and qa_path.is_file() and sha(qa_path) == qa.get("sha256"), "presentation QA evidence missing or changed")
        require(read(qa_path).get("passed") is True, "presentation QA did not pass")
    return {"path": str(path), "sha256": sha(path), "manifest": value, "verified_outputs": outputs}


def verify_report(root, result, manifest, records):
    require(result.get("complete") is True and result.get("status") == "complete" and manifest.get("status") == "complete", "report is incomplete")
    require(manifest["generated_at_utc"] == result["generated_at_utc"], "report result/manifest generation mismatch")
    verify_files(manifest["outputs"], root / "report")
    require(manifest.get("input_files_complete") is True, "report input closure is not marked complete")
    from .study_report import verify_input_files
    verify_input_files(manifest["input_files"])
    generator = manifest.get("source_generator", {"path": str(PROJECT / "mt/study_report.py"), "sha256": manifest.get("generator_sha256")})
    require(sha(generator["path"]) == generator["sha256"], "report generator changed")
    require(manifest["input_sha256"] == result["input_sha256"], "report symbolic input identities differ")
    for name, (runs, updates, passes, sentences) in EXPECTED_STAGES.items():
        cost = result["stage_costs"][name]
        require((cost["completed_training_runs"], cost["completed_run_updates"], cost["completed_full_evaluation_passes"], cost["evaluated_sentences"]) ==
                (runs, updates, passes, sentences), "report stage is incomplete: " + name)
    for record in records:
        cfg = read(record["config"]); stage = record["stage"]
        if stage == "cv":
            tag = "ls0" if record["smoothing"] == 0 else "ls01"
            fold = result["cv_selection"]["candidates"][tag]["folds"][record["fold"]]
            metric = read(record["evaluations"][0]["path"])
            require(fold["fold"] == record["fold"], "CV report fold label differs")
            require(all(fold[key] == metric[key] for key in ("samples", "bleu", "chrf2")), "reported CV fold differs from actual metrics")
        else:
            if stage == "cv_final":
                reported = result["runs"]["cv_final"][str(record["seed"])]
            elif stage == "adapt_tune":
                matches = [v for v in result["adaptation_tuning"][record["method"]] if v["config"]["lr"] == record["lr"]]
                require(len(matches) == 1, "duplicate/missing tuning report entry")
                reported = matches[0]
            else:
                reported = result["runs"][record["method"]][str(record["seed"])]
            require(reported["status"] == "complete" and reported["config"] == cfg and reported["training"]["updates"] == 3000, "report run identity/completion differs")
            if stage != "adapt_tune":
                require(reported["complete_for_aggregation"] is True, "incomplete seed enters report")
                for split in ("validation", "test"):
                    for decoder in ("greedy", "beam"):
                        directory = Path(record["run"]) / (f"{split}_{decoder}" if stage == "cv_final" else f"evaluation/{split}/{decoder}")
                        require(reported["evaluations"][split][decoder] == read(directory / "metrics.json"), "report evaluation differs from saved metric")
        bound = [record["config"], record["summary"], str(Path(record["run"]) / "last.pt"), str(Path(record["run"]) / "train.jsonl")]
        bound += [metric["path"] for metric in record["evaluations"]]
        if record["stage"].startswith("adapt"):
            bound += [str(Path(record["run"]) / "best.pt")]
            bound += [str(Path(record["run"]) / f"checkpoint_{update:05d}.pt") for update in range(500, 3001, 500)]
        for path in bound:
            require(str(Path(path).resolve()) in manifest["input_files"], "report input closure omits formal run evidence: " + path)
    require(result["runs"]["direct"]["complete_for_aggregation"] is True, "direct baseline not verified")
    for split in ("validation", "test"):
        for decoder in ("greedy", "beam"):
            for method in ("cv_final", "lora", "full"):
                for metric in ("bleu", "chrf2"):
                    values = {str(seed): result["runs"][method][str(seed)]["evaluations"][split][decoder][metric] for seed in SEEDS}
                    saved = result["aggregate"][split][decoder][method][metric]
                    require(saved["n"] == 3 and saved["ddof"] == 1 and saved["per_seed"] == values and
                            saved["mean"] == statistics.mean(values.values()) and saved["sample_std"] == statistics.stdev(values.values()),
                            "final seed aggregate differs from actual per-seed results")
            for metric in ("bleu", "chrf2"):
                values = {str(seed): result["runs"]["lora"][str(seed)]["evaluations"][split][decoder][metric] -
                          result["runs"]["full"][str(seed)]["evaluations"][split][decoder][metric] for seed in SEEDS}
                saved = result["aggregate"][split][decoder]["paired_lora_minus_full"][metric]
                require(saved["n"] == 3 and saved["per_seed"] == values and saved["mean"] == statistics.mean(values.values()) and
                        saved["sample_std"] == statistics.stdev(values.values()), "final paired aggregate differs")


def verify_recovery(root, result):
    from .study_report import collect_recovery_incidents, resource_costs
    resources = resource_costs(root)
    require(resources["optimizer_updates"] == 108, "resource probe updates differ from actual frozen108 evidence")
    recovery = collect_recovery_incidents(root)
    require(recovery["recorded_discarded_updates_lower_bound"] == 235, "recorded recovery extra computation differs from235")
    for incident in recovery["incidents"]:
        replay = incident["replay_verification"]["evidence"]
        require(replay is not None and replay["incident_sha256"] == incident["sha256"], "replay verification missing or belongs to another incident")
        require(replay["replayed_updates_checked"] == incident["recorded_discarded_updates"] and
                replay["same_data_lr_token_exposure"] is True and replay["all_logged_numerical_values_exactly_equal"] is True and
                replay["identity_differences"] == [], "replayed data/numerics were not verified exact")
        require(all(v["exact_equal"] is True and v["different_updates"] == 0 for v in replay["numeric_comparison"].values()), "replay numeric differences exist")
        require(sha(replay["captured_checkpoint_path"]) == replay["captured_checkpoint_sha256"], "replay checkpoint snapshot changed")
    require(result["resource_costs"] == resources and result["recovery_incidents"] == recovery, "report resource/recovery costs differ from source evidence")
    return {"formal_net_updates": 57000, "resource_probe_updates": 108, "recorded_discarded_updates_lower_bound": 235,
            "recorded_optimizer_compute_updates_lower_bound": 57343,
            "resources": resources, "recovery": recovery,
            "limitations": "Interruption cause, original process exit code and precise end time remain unknown; unlogged extra work is unknown.235 replay does not create a new formal run or increase net57000, but its discarded branch has real additional cost."}


def historical_after(root, manifest):
    require(len(manifest["files"]) == HISTORICAL_FILES, "protected history file count differs")
    changed = []
    for name, old in manifest["files"].items():
        path = inside(PROJECT, PROJECT / name)
        if not path.is_file() or path.stat().st_size != old["bytes"] or sha(path) != old["sha256"]:
            changed.append(name)
    return {"created_at_utc": stable_created(root / "history_after_final.json"), "files_checked": len(manifest["files"]),
            "changed": changed, "passed": not changed, "historical_manifest_sha256": sha(root / "historical_manifest.json")}


def safe_walk(root):
    """Never traverse symlinks, Windows junctions/reparse points or runtimes."""
    root = Path(root)
    def linked(path):
        return path.is_symlink() or getattr(path, "is_junction", lambda: False)()
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=fail):
        base = Path(directory)
        dirs[:] = sorted(name for name in dirs if name.lower() not in SKIP_DIRS and not linked(base / name))
        for name in sorted(files):
            path = base / name
            if not linked(path) and path.suffix.lower() not in (".pyc", ".lock", ".tmp", ".pending"):
                yield path


def source_snapshot(root):
    sources = {PROJECT / "mt/study_complete.py", PROJECT / "mt/study_report.py", PROJECT / "README.md"}
    for path in safe_walk(root):
        if "delivery_source_snapshot" not in path.parts and path.suffix.lower() in (".py", ".ps1", ".mjs", ".js"):
            sources.add(path)
    sources.update(PROJECT / name for name in read(root / "plan.json")["source_files"])
    # Preserve the source of the actually executed full suite, including CV,
    # adaptation, historical regression tests and final delivery checks.
    sources.update(path for path in (PROJECT / "tests").glob("test_*.py"))
    evidence = {}
    for source in sorted(sources):
        require(source.is_file(), "missing final source: " + str(source))
        relative = source.relative_to(PROJECT)
        target = root / "delivery_source_snapshot" / relative
        expected = sha(source); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == expected, "final source snapshot differs")
        else:
            with source.open("rb") as original, target.open("xb") as copy:
                shutil.copyfileobj(original, copy)
        evidence[str(source)] = {"sha256": expected, "snapshot": str(target)}
    packages = {}
    for name in ("torch", "numpy", "sentencepiece", "sacrebleu", "transformers", "matplotlib", "scikit-learn", "scipy", "portalocker"):
        try:
            dist = importlib.metadata.distribution(name)
            record = next((p for p in dist.files or [] if p.name == "RECORD"), None)
            packages[name] = {"version": dist.version, "record_sha256": sha(dist.locate_file(record)) if record else None}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"available_in_finalizer_entrypoint": False}
    value = {"created_at_utc": stable_created(root / "delivery_source_manifest.json"), "source_files": evidence,
             "dependencies_observed_without_importing_models": packages,
             "frozen_training_dependencies": read(root / "plan.json")["dependencies"]}
    write_once(root / "delivery_source_manifest.json", value)
    return value


def volatile_inventory_file(path, root):
    """Exclude control-plane files that can still be updated around delivery."""
    if path.name == "progress.json":
        return True
    if path.parent == root:
        if path.name == "gpu_observations.jsonl":
            return True
        if path.suffix == ".log" and path.name.startswith(("controller", "observer", "finalize", "study_complete")):
            return True
    return False


def external_source_inventory_entry(path, root):
    """Absence from a protection manifest does not imply a newly created file."""
    path = Path(path)
    value = {"sha256": sha(path), "bytes": path.stat().st_size,
             "classification": "source_absent_from_pre_execution_protection_manifest"}
    if path.resolve() == (PROJECT / "README.md").resolve():
        backup = Path(root) / "README.before_study_update.md"
        require(backup.is_file(), "README update classification requires its saved pre-update copy")
        value.update(classification="intentionally_modified_existing", before_file=str(backup),
                     before_sha256=sha(backup), changed_from_saved_copy=sha(backup) != value["sha256"])
    return value


def finalize(root, **options):
    root = Path(root).resolve()
    require(not (root / "completion.json").exists(), "overall completion already exists; preserve it")
    state = inspect(root, **options)
    require(not state["pending"], "delivery is still pending: " + "; ".join(state["pending"][:8]))
    from .study import verify_plan, verify_stage
    verify_plan(root)
    verify_stage(root, root / "cv/final_plan.json"); verify_stage(root, root / "adaptation/final_plan.json")
    paths = {key: Path(value) for key, value in state["paths"].items()}
    results, report_manifest = read(paths["results"]), read(paths["report_manifest"])
    verify_report(root, results, report_manifest, state["runs"])
    require(read(paths["cv_final_completion"])["complete"] is True and read(paths["cv_final_completion"])["optimizer_updates"] == 9000, "CV final stage is incomplete")
    regression = verify_regression(paths["regression"])
    review = verify_delivery_manifest(paths["sample_review"], sha(paths["results"]), sha(root / "report/samples.json"))
    presentation = verify_delivery_manifest(paths["presentation"], sha(paths["results"]))
    accounting = verify_recovery(root, results)
    history = historical_after(root, read(paths["historical_manifest"]))
    write_once(root / "history_after_final.json", history)
    require(history["passed"], "historical protected files changed")
    snapshot = source_snapshot(root)
    excluded = {root / "completion.json", root / "new_files.json"}
    files = {str(path.relative_to(PROJECT)).replace("\\", "/"): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in safe_walk(root) if path not in excluded and not volatile_inventory_file(path, root)}
    old_files = read(paths["historical_manifest"])["files"]
    for source in snapshot["source_files"]:
        path = Path(source); relative = path.relative_to(PROJECT).as_posix()
        if relative not in old_files and path.is_relative_to(PROJECT) and not path.is_relative_to(root):
            files[relative] = external_source_inventory_entry(path, root)
    inventory = {"created_at_utc": stable_created(root / "new_files.json"), "files": files, "file_count": len(files),
        "scope": "all new study-directory artifacts plus final source files absent from historical protection manifest; absence from that manifest is not by itself a filesystem creation-time claim",
        "excluded": "self/overall completion avoid circular hashes; live progress, locks, temporary/cache/runtime directories, symlinks and junctions are not inventoried"}
    write_once(root / "new_files.json", inventory)
    completion = {"schema_version": 1, "created_at_utc": now(), "status": "complete", "complete": True,
        "whole_study_complete": True, "formal_runs": 19, "formal_net_optimizer_updates": 57000,
        "unique_run_directories": [record["run"] for record in state["runs"]], "runs": state["runs"],
        "aggregate": results["aggregate"], "stage_costs": results["stage_costs"], "computation_accounting": accounting,
        "final_full_regression": regression, "sample_review": review, "presentation": presentation,
        "historical_verification": history, "source_manifest_sha256": sha(root / "delivery_source_manifest.json"),
        "new_files_sha256": sha(root / "new_files.json"),
        "delivery_input_sha256": {str(path): sha(path) for path in paths.values()},
        "failed_or_interrupted_work": accounting["recovery"],
        "resource_attempts": results.get("resource_attempts", []),
        "execution_attempts": results.get("execution_attempts"),
        "unexecuted_authorized_formal_experiments": [],
        "limitations": ["Historical backend workspace unknown; reuse is bounded by recorded identities", "n3 is not statistical significance", "CV optimality applies only to3000updates", "course recognition remains for the teacher", "OPUS overlap cannot be excluded"],
        "stage_completion_records_preserved": True, "no_training_or_inference_in_finalizer": True}
    write_once(root / "completion.json", completion)
    return completion


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--regression-json"); parser.add_argument("--presentation-manifest"); parser.add_argument("--sample-review-manifest")
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in ("regression_json", "presentation_manifest", "sample_review_manifest")}
    result = finalize(args.root, **options) if args.finalize else inspect(args.root, **options)
    print(json.dumps(result if not args.finalize else {"status": result["status"], "formal_runs": 19,
        "formal_net_optimizer_updates": 57000, "full_regression_tests": result["final_full_regression"]["tests_run"],
        "completion": str(Path(args.root).resolve() / "completion.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
