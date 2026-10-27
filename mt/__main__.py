"""Show saved project progress without loading models or taking GPU memory."""
import argparse
import json
from pathlib import Path
import sys

from .common import read_json


def latest_record(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        offset = max(0, size - 65536)
        stream.seek(offset)
        if offset:
            stream.readline()
        lines = stream.read().split(b"\n")
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue  # A currently written tail is not a committed status.
        if isinstance(value, dict) and "update" in value:
            return value
    return None


def project_status(root):
    root = Path(root)
    runs = []
    for name in ("A42", "B42", "A43", "B43", "A44", "B44",
                 "quality_A42_12000", "quality_A43_12000", "quality_A44_12000"):
        run = root / "artifacts" / "runs" / name
        config_path = run / "config.json"
        if not config_path.exists():
            config_path = root / "experiments" / f"{name}.json"
        cfg = read_json(config_path) if config_path.exists() else {}
        summary_path = run / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        latest = latest_record(run / "train.jsonl") or {}
        metrics = {}
        for method in ("greedy", "beam"):
            path = run / method / "metrics.json"
            if path.exists():
                value = read_json(path)
                metrics[method] = {"bleu": value.get("bleu"), "samples": value.get("samples")}
        runs.append({"run": name, "updates_recorded": latest.get("update", summary.get("updates", 0)),
                     "update_budget": cfg.get("max_updates"), "training_complete": summary.get("complete", False),
                     "best_validation_bleu": summary.get("best_validation_bleu"), "tests": metrics})
    stages = [read_json(path) for path in sorted((root / "artifacts").glob("pipeline_seed*.json"))]
    quality = root / "artifacts" / "quality_progress.json"
    if quality.exists():
        stages.append(read_json(quality))
    for path in sorted((root / "artifacts").glob("quality_seed*/progress.json")):
        stages.append(read_json(path))
    return {"project": str(root.resolve()), "runs": runs, "last_saved_pipeline_stages": stages,
            "note": "Saved records, not a process liveness check. Use mt.summary for full artifact validation."}


def main():
    parser = argparse.ArgumentParser(description="查看本地实验记录，无需加载模型")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    value = project_status(args.project)
    if args.json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    print("英→德实验记录（不代表进程当前仍在运行）")
    for run in value["runs"]:
        state = "训练完成" if run["training_complete"] else "已有训练记录" if run["updates_recorded"] else "待运行"
        scores = "，".join(f"{method} BLEU {m['bleu']:.3f} ({m['samples']} 句)"
                          for method, m in run["tests"].items()) or "测试待完成"
        print(f"{run['run']}: {run['updates_recorded']}/{run['update_budget'] or '?'} 更新，{state}；{scores}")
    print("完整核验：python -m mt.summary；报告：python -m mt.report --all-seeds")


if __name__ == "__main__":
    main()
