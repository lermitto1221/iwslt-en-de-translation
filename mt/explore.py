"""Freeze and execute independent train/validation exploration, without training."""
import argparse
from contextlib import contextmanager
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from .common import read_json, sha256, write_json

DEFAULT_CONFIG = "experiments/exploration_2026_10_27.json"
ANALYSIS_MODULES = ("explore.py", "explore_data.py", "explore_vectors.py", "explore_quality.py")
CORE_MODULES = ("model.py", "train.py", "batching.py", "tokenizer.py", "decoding.py",
                "common.py", "data.py", "inference.py", "evaluate.py", "summary.py", "run_experiments.py")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def hash_files(files):
    return {str(p.resolve()): sha256(p) for p in sorted(set(files)) if p.is_file()}


def check_hashes(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("file fingerprint manifest must be nonempty")
    for filename, digest in values.items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen file missing or changed: {filename}")


def create_plan(config_path):
    cfg = read_json(config_path)
    out = Path(cfg["output_dir"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("analysis output already exists; keep its original plan and resume")
    data = Path(cfg["data_dir"])
    inputs = [Path(config_path), data / "manifest.json", data / "tokenizer.json", data / "spm.model",
              data / "spm.vocab", data / "train.jsonl", data / "validation.jsonl",
              data / "raw/iwslt2017-train.parquet", data / "raw/iwslt2017-validation.parquet",
              Path(cfg["checkpoint"]), Path(cfg["baseline_checkpoint"]), Path(cfg["protocol"])]
    for ckpt in (cfg["checkpoint"], cfg["baseline_checkpoint"]):
        inputs += [Path(ckpt).parent / name for name in ("config.json", "summary.json", "data_link.json", "source_hashes.json")]
    for folder in cfg["validation_reuse"].values():
        inputs += [Path(folder) / name for name in ("metrics.json", "examples.jsonl", "predictions.txt", "references.txt")]
    inputs += [Path(__file__).with_name(name) for name in CORE_MODULES]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Protect all previous artifacts, including model weights and timing records.
    historical = [p for p in Path("artifacts").rglob("*") if p.is_file() and out not in p.resolve().parents]
    historical += [p for p in Path("experiments").glob("*") if p.is_file() and p.resolve() != Path(config_path).resolve()]
    historical += [Path("gptprompt.txt"), Path("requirements.txt"), Path("requirements.lock.txt")]
    historical += [Path(__file__).with_name(name) for name in CORE_MODULES]
    protected = hash_files(historical)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "matplotlib", "torch", "sentencepiece", "pyarrow", "sacrebleu")}
    record = {
        "created_at_utc": utc_now(), "local_timezone": "Asia/Shanghai", "config": cfg,
        "config_sha256": sha256(config_path), "inputs_sha256": hash_files(inputs),
        "design": "Exploratory analysis added after existing training and test results were known; frozen before this analysis, not retrospective training preregistration.",
        "questions": ["How do sentence length, lexical rarity and BPE fragmentation differ?",
                      "Are validation translation scores associated with these features?",
                      "Do encoder clusters reflect length, vocabulary or interpretable semantic groups?",
                      "Which validation groups improve under the longer training budget?"],
        "word_definition": "str.split() then casefold(); punctuation remains attached or standalone; no extra normalization. Separate EN/DE counts fit on full cleaned training data before BPE length filtering. Rare count <=5; unseen count=0 included in validation rarity.",
        "sampling_rule": "random.Random(20261027).sample 10000 complete cleaned training rows, then sort original IDs; do not restrict to length-eligible rows; all 888 validation rows. Save selection and verify row alignment.",
        "representation": "Pre-fixed seed42 12000-update validation-best checkpoint; final encoder layer float32 eval/no_grad; mean over all source tokens except PAD/BOS/EOS, retain UNK; no truncation, empty effective sequence is an error.",
        "fit_rule": "Training sample only: per-dimension mean/std ddof=0 (constant scale=1), full SVD PCA50, unwhitened scores, kmeans++ K5 n_init10 max_iter300 tol1e-4 seed20261027. Validation transform/predict only. No K sweep.",
        "diagnostics": "Euclidean silhouette on a fixed random 2000 training subset in PCA50 space; Calinski-Harabasz/Davies-Bouldin on full training PCA50. PC Pearson correlations and cluster eta squared for simple features are descriptive, not causal. Report KMeans convergence and empty clusters.",
        "lexical_profile": "Training sample only, per-cluster document prevalence enrichment (add-one smoothed rate / global rate), alphabetic whitespace-casefold tokens excluding fixed stopwords; within-cluster support >=10, top8 with deterministic lexical ties. Lexical hints are not topic labels.",
        "quality_rule": "Same seed42 baseline/long best checkpoints; shared historical protocol. Beam and greedy on all888 validation; corpus SacreBLEU mixed-case13a and chrF2 per cluster and pre-fixed feature bins; n<30 flagged, empty scores=null. No average sentence BLEU, no weighted reconstruction of corpus BLEU. Validation already used for model/protocol selection, so this is diagnostic and not independent held-out confirmation.",
        "example_rule": "Per cluster choose lowest3 original validation IDs; all four translations retained; agent review, not independent human annotation. No score-based selection.",
        "history_policy": "No historical artifact writes; new results isolated. No new model training or test-driven tuning.",
        "python": platform.python_version(), "dependencies": versions,
        "implementation": "Existing packages only; NumPy SVD and explicitly implemented kmeans++/Lloyd; no dependency upgrade.",
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "historical_manifest.json", {"created_at_utc": record["created_at_utc"], "files_sha256": protected})
    with (out / "plan.json").open("x", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    atomic_json(out / "environment.json", {"python": record["python"], "dependencies": versions, "entrypoint": "project-python.ps1"})
    print({"plan": str(out / "plan.json"), "protected_files": len(protected), "created_at_utc": record["created_at_utc"]}, flush=True)
    return record


def validate_plan(config_path):
    cfg = read_json(config_path)
    out = Path(cfg["output_dir"])
    plan = read_json(out / "plan.json")
    if plan["config_sha256"] != sha256(config_path) or plan["config"] != cfg:
        raise ValueError("configuration changed after analysis plan was frozen")
    check_hashes(plan["inputs_sha256"])
    return cfg, out


@contextmanager
def execution_lock(path):
    import msvcrt
    with Path(path).open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0"); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def run(config_path, stage="all", device="cuda"):
    cfg, out = validate_plan(config_path)
    os.environ.setdefault("MPLCONFIGDIR", str((out / "matplotlib").resolve()))
    sources = hash_files([Path(__file__).with_name(name) for name in ANALYSIS_MODULES])
    if len(sources) != len(ANALYSIS_MODULES):
        raise ValueError("analysis implementation is incomplete")
    frozen = out / "analysis_sources.json"
    if frozen.exists():
        if read_json(frozen)["files_sha256"] != sources:
            raise ValueError("analysis source changed; do not silently mix partial results")
    else:
        atomic_json(frozen, {"frozen_at_utc": utc_now(), "files_sha256": sources})
    from .explore_data import run_data_analysis
    from .explore_vectors import run_vectors
    from .explore_quality import run_quality_analysis
    stages = {"data": lambda: run_data_analysis(cfg, out),
              "vectors": lambda: run_vectors(cfg, out, device=device),
              "quality": lambda: run_quality_analysis(cfg, out)}
    with execution_lock(out / "execution.lock"):
        for name, function in stages.items():
            if stage not in ("all", name):
                continue
            done = out / "stages" / f"{name}.json"
            if done.exists():
                saved = read_json(done)["files_sha256"]
                if str((out / name / "summary.json").resolve()) not in saved:
                    raise ValueError(f"{name}: completion marker lacks required summary")
                check_hashes(saved)
                print({"stage": name, "status": "verified_reuse"}, flush=True)
                continue
            for prerequisite in ({"vectors": ["data"], "quality": ["data", "vectors"]}.get(name, [])):
                check_hashes(read_json(out / "stages" / f"{prerequisite}.json")["files_sha256"])
            atomic_json(out / "progress.json", {"stage": name, "status": "running", "at_utc": utc_now()})
            print({"stage": name, "status": "running"}, flush=True)
            function()
            if not (out / name / "summary.json").is_file():
                raise ValueError(f"{name}: required summary was not produced")
            paths = list((out / name).rglob("*"))
            figure_patterns = {"data": ["data_*"], "vectors": ["pca_*", "cluster_*"], "quality": ["quality_*"]}
            for pattern in figure_patterns[name]:
                paths += list((out / "figures").glob(pattern))
            atomic_json(done, {"completed_at_utc": utc_now(), "files_sha256": hash_files(paths)})
            print({"stage": name, "status": "complete"}, flush=True)
        atomic_json(out / "progress.json", {"stage": stage, "status": "complete", "at_utc": utc_now()})


def verify_history(out):
    out = Path(out)
    values = read_json(out / "historical_manifest.json")["files_sha256"]
    check_hashes(values)
    result = {"verified_at_utc": utc_now(), "status": "verified", "unchanged_files": len(values)}
    atomic_json(out / "historical_verification.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "verify-history"))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--stage", choices=("all", "data", "vectors", "quality"), default="all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    if a.command == "plan":
        create_plan(a.config)
    elif a.command == "run":
        run(a.config, a.stage, a.device)
    else:
        print(verify_history(read_json(a.config)["output_dir"]))


if __name__ == "__main__":
    main()
