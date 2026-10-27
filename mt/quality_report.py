"""Report one training-budget extension from saved, verified files.

This module never loads a model, decodes translations, or selects test settings.
"""
import argparse
import json
import math
import os
from pathlib import Path

import sacrebleu
import sentencepiece as spm

from .common import write_json
from .data import EXPECTED
from .report import number
from .summary import (artifact_path, read_jsonl, read_object, require,
                      valid_hash, validate_result, validate_training_log, verify_hash)


METHODS = ("greedy", "beam")
GROUPS = tuple(f"{run}_{method}" for run in ("baseline", "long") for method in METHODS)
BUCKETS = ("<=20", "21–40", "41–80", ">80")
PROJECT = Path(__file__).resolve().parents[1]
CORE_SOURCES = ("train.py", "model.py", "batching.py", "tokenizer.py", "decoding.py", "common.py", "data.py")


def bucket_name(length):
    require(type(length) is int and length >= 0, "source BPE length must be a nonnegative integer")
    return BUCKETS[0 if length <= 20 else 1 if length <= 40 else 2 if length <= 80 else 3]


def read_training(run):
    """Read the saved training state; an absent or incomplete run stays pending."""
    run = Path(run)
    config_path = run / "config.json"
    if not config_path.exists():
        planned = PROJECT / "experiments" / f"{run.name}.json"
        config_path = planned if planned.exists() else config_path
    config = read_object(config_path, str(run)) if config_path.exists() else None
    summary_path = run / "summary.json"
    summary = read_object(summary_path, str(run)) if summary_path.exists() else None
    log_path = run / "train.jsonl"
    rows = read_jsonl(log_path, str(run)) if log_path.exists() else []
    for index, row in enumerate(rows, 1):
        require(type(row.get("update")) is int and row["update"] == index,
                f"{run}: training log must contain ordered updates 1..N; bad row {index}")
    complete = bool(summary and summary.get("complete") is True)
    if complete:
        require(config is not None and summary.get("updates") == config.get("max_updates"),
                f"{run}: completed training summary/config budget mismatch")
        validate_training_log(run, config, str(run))
    validation = []
    for row in rows:
        if "validation_bleu_greedy" not in row:
            continue
        score = row["validation_bleu_greedy"]
        require(type(score) in (float, int) and math.isfinite(score)
                and row.get("validation_samples") == EXPECTED["validation"],
                f"{run}: validation at update {row['update']} is not the complete validation corpus")
        validation.append({"update": row["update"], "bleu": score,
                           "signature": row.get("validation_signature")})
    best = max(validation, key=lambda row: row["bleu"]) if validation else None
    if complete:
        summary_score = summary.get("best_validation_bleu")
        require(best is not None and type(summary_score) in (float, int) and math.isfinite(summary_score)
                and math.isclose(summary_score, best["bleu"], rel_tol=0, abs_tol=1e-8),
                f"{run}: best validation score differs from training log")
        interval = config.get("validate_every")
        require(type(interval) is int and interval > 0, f"{run}: invalid validation interval")
        expected_points = sorted(set(range(interval, config["max_updates"] + 1, interval)) | {config["max_updates"]})
        require([row["update"] for row in validation] == expected_points,
                f"{run}: missing or unexpected full-validation points")
    return {"run": str(run), "status": "complete" if complete else "pending",
            "config": config, "config_source": str(config_path) if config else None,
            "summary": summary, "last_logged_update": len(rows),
            "validation": validation, "best_validation": best}


def read_source_provenance(run, require_snapshot=False):
    """Verify saved seven-module provenance without comparing to today's source."""
    run = Path(run)
    hashes_path, manifest_path = run / "source_hashes.json", run / "source_snapshot" / "manifest.json"
    hashes = read_object(hashes_path, str(run)) if hashes_path.exists() else {}
    recorded = {name: hashes.get(name) for name in CORE_SOURCES}
    missing = [name for name, value in recorded.items() if value is None]
    for name, value in recorded.items():
        if value is not None:
            require(valid_hash(value), f"{run}: invalid saved source hash for {name}")
    snapshot = "missing_historical_snapshot"
    if manifest_path.exists():
        manifest = read_object(manifest_path, str(run))
        require(not missing and manifest.get("core_source_hashes") == recorded,
                f"{run}: source snapshot manifest differs from seven-module source record")
        for name, expected in recorded.items():
            verify_hash(artifact_path(manifest_path.parent / name, str(run)), expected, f"{run}: source snapshot {name}")
        snapshot = "verified"
    if require_snapshot:
        require(not missing and snapshot == "verified", f"{run}: complete seven-module source snapshot is required")
    return {"source_hashes_path": str(hashes_path) if hashes_path.exists() else None,
            "snapshot_manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "core_source_hashes": recorded, "missing_source_hashes": missing, "snapshot_status": snapshot,
            "note": "Missing historical source files are reported as missing, never reconstructed as original evidence."}


def compare_training_prefix(training):
    if not all(entry["status"] == "complete" for entry in training.values()):
        return "pending"
    counts = [training[name]["summary"].get("parameters") for name in ("baseline", "long")]
    require(isinstance(counts[0], dict) and counts[0] == counts[1],
            "quality extension parameter counts differ from baseline or are missing")
    a, b = (read_jsonl(Path(training[name]["run"]) / "train.jsonl", name)
            for name in ("baseline", "long"))
    require(len(b) >= len(a), "long training log is shorter than baseline")
    for left, right in zip(a, b):
        for field in ("data_order_sha256", "target_tokens", "sentences", "lr_used"):
            require(field in left and left[field] == right.get(field),
                    f"quality extension training prefix differs at update {left['update']}: {field}")
    return "verified"


def prefix_numerical_differences(training):
    """Record floating-point history, without mistaking it for a control change."""
    if not all(entry["status"] == "complete" for entry in training.values()):
        return None
    a, b = (read_jsonl(Path(training[name]["run"]) / "train.jsonl", name)
            for name in ("baseline", "long"))
    fields = ("loss", "validation_bleu_greedy", "validation_generation_limit_fraction")
    result = {"compared_updates": len(a), "fields": {}}
    for field in fields:
        differences, compared, missing = [], 0, []
        for left, right in zip(a, b):
            if field not in left and field not in right:
                continue
            if field not in left or field not in right:
                missing.append(left["update"])
                continue
            compared += 1
            if left[field] != right[field]:
                differences.append({"update": left["update"], "baseline": left[field],
                                    "long": right[field], "long_minus_baseline": right[field] - left[field]})
        result["fields"][field] = {"compared_points": compared, "missing_on_one_side": missing,
            "different_points": len(differences), "differences": differences,
            "max_absolute_difference": max((abs(row["long_minus_baseline"]) for row in differences), default=0)
                if compared else None}
    return result


def compare_configs(baseline, long):
    if baseline is None or long is None:
        return "pending"
    ignored = {"run_dir", "max_updates"}
    a, b = ({key: value for key, value in cfg.items() if key not in ignored} for cfg in (baseline, long))
    changed = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
    require(not changed, "quality extension config differs beyond update budget/run_dir: " + ", ".join(changed))
    require(type(long.get("max_updates")) is int and type(baseline.get("max_updates")) is int
            and long["max_updates"] > baseline["max_updates"], "quality extension must have a larger update budget")
    return "verified"


def validate_comparison(records, seed=42):
    """Keep data, initialization and the frozen test decoder identical."""
    anchor = next(iter(records.values()))
    fields = ("data_fingerprints", "initial_model_sha256", "inference_source_sha256",
              "length_penalty", "max_generation_tokens", "signature",
              "inference_precision", "inference_batch_size")
    for name, metrics in records.items():
        for field in fields:
            require(metrics.get(field) == anchor.get(field), f"{name}: {field} differs from baseline")
    for run in ("baseline", "long"):
        greedy, beam = (records.get(f"{run}_{method}") for method in METHODS)
        if greedy and beam:
            require(greedy["checkpoint_sha256"] == beam["checkpoint_sha256"],
                    f"{run}: greedy and beam must use the same validation-best checkpoint")
    left = records.get("baseline_greedy") or records.get("baseline_beam")
    right = records.get("long_greedy") or records.get("long_beam")
    if left and right:
        compare_configs(left["config"], right["config"])
    # Historical seed 42 precedes the stricter checkpoint recovery I/O. Record
    # source differences instead of claiming byte-identical training programs.
    a = left.get("training_source_hashes", {}) if left else {}
    b = right.get("training_source_hashes", {}) if right else {}
    changed = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key)) if left and right else None
    if changed:
        allowed = {"train.py", "decoding.py"} if seed == 42 else set()
        require(not (set(changed) - allowed),
                "quality extension has unexpected training source changes: " + ", ".join(changed))
    return changed


def source_lengths(rows, metrics):
    model = artifact_path(Path(metrics["config"]["data_dir"]) / "spm.model", "source length tokenizer")
    verify_hash(model, metrics["data_fingerprints"]["tokenizer_sha256"], "source length tokenizer")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    # SentencePiece.encode alone adds no BOS/EOS. No length filter or truncation.
    return [len(tokenizer.encode(row["source"], out_type=int)) for row in rows]


def corpus_scores(predictions, references, chrf=False):
    if not predictions:
        return {"bleu": None, "chrf": None}
    result = {"bleu": sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
              .corpus_score(predictions, [references]).score, "chrf": None}
    if chrf:
        result["chrf"] = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2)
        result["chrf"] = result["chrf"].corpus_score(predictions, [references]).score
    return result


def compare_examples(examples, lengths, chrf=False):
    anchor = next(iter(examples.values()))
    require(len(anchor) == len(lengths) == EXPECTED["test"], "length buckets require the complete test set")
    require([row["id"] for row in anchor] == list(range(EXPECTED["test"])),
            "quality report requires the original complete ordered test IDs 0..N-1")
    for name, rows in examples.items():
        require(len(rows) == len(anchor), f"{name}: example count differs from baseline")
        for left, right in zip(anchor, rows):
            require(all(left[key] == right[key] for key in ("id", "source", "reference")),
                    f"{name}: example alignment differs at ID {left['id']}")
    membership = [{"id": row["id"], "source_bpe_tokens": length, "bucket": bucket_name(length)}
                  for row, length in zip(anchor, lengths)]
    buckets = []
    for label in BUCKETS:
        indices = [index for index, value in enumerate(membership) if value["bucket"] == label]
        scores = {name: corpus_scores([examples[name][index]["prediction"] for index in indices],
                  [anchor[index]["reference"] for index in indices], chrf) if name in examples else None
                  for name in GROUPS}
        buckets.append({"bucket": label, "count": len(indices), "scores": scores})
    require(sum(item["count"] for item in buckets) == EXPECTED["test"], "length buckets do not cover the full test set")
    fixed = []
    for index, row in enumerate(anchor):
        if row["id"] % 400 == 0:
            fixed.append({"id": row["id"], "source": row["source"], "reference": row["reference"],
                          "source_bpe_tokens": lengths[index], "bucket": bucket_name(lengths[index]),
                          "predictions": {name: examples[name][index]["prediction"] if name in examples else None
                                          for name in GROUPS}})
    return buckets, membership, fixed


def collect_quality(baseline, long, chrf=False, seed=42):
    require(type(seed) is int and seed >= 0, "quality report seed must be a nonnegative integer")
    runs = {"baseline": Path(baseline), "long": Path(long)}
    require(runs["baseline"].resolve() != runs["long"].resolve(), "baseline and long run must be distinct")
    training = {name: read_training(run) for name, run in runs.items()}
    for name, entry in training.items():
        if entry["config"] is not None:
            require(entry["config"].get("seed") == seed and entry["config"].get("label_smoothing") == .1,
                    f"{name}: seed or label smoothing differs from requested quality experiment")
    controls = compare_configs(training["baseline"]["config"], training["long"]["config"])
    prefix = compare_training_prefix(training)
    provenance = {name: read_source_provenance(run, require_snapshot=((name == "long" or seed != 42)
                   and training[name]["status"] == "complete")) for name, run in runs.items()}
    records, examples, scores = {}, {}, {}
    for name, run in runs.items():
        for method in METHODS:
            group = f"{name}_{method}"
            path = run / method / "metrics.json"
            scores[group] = None
            if not path.exists():
                continue
            require(training[name]["status"] == "complete", f"{group}: test metrics exist before training budget completion")
            metrics = read_object(path, group)
            # validate_result also calls validate_saved_outputs: source/checkpoint
            # hashes, every saved row and independently recomputed corpus BLEU.
            examples[group] = validate_result(metrics, run, "A", seed, method)
            require(all(row["signature"] == metrics["signature"] for row in training[name]["validation"]),
                    f"{group}: validation signature differs from the verified BLEU scoring protocol")
            best = training[name]["best_validation"]
            require(metrics.get("checkpoint_update") == best["update"],
                    f"{group}: checkpoint update is not the earliest best full-validation checkpoint ({best['update']})")
            checkpoint = artifact_path(metrics.get("checkpoint"), group)
            require(checkpoint.resolve() == (run / "best.pt").resolve(),
                    f"{group}: evaluation must use this run's best.pt")
            records[group] = metrics
            for filename, value in metrics.get("training_source_hashes", {}).items():
                recorded = provenance[name]["core_source_hashes"].get(filename)
                if recorded is not None:
                    require(recorded == value, f"{group}: checkpoint source metadata differs from saved {filename}")
            scores[group] = {"bleu": metrics["bleu"], "chrf": None,
                             "checkpoint_update": metrics["checkpoint_update"], "samples": metrics["samples"]}
            if chrf:
                scores[group]["chrf"] = corpus_scores([row["prediction"] for row in examples[group]],
                    [row["reference"] for row in examples[group]], True)["chrf"]
    require(any(name.startswith("baseline_") for name in records),
            "quality report requires at least one completed baseline test result to establish the full source corpus")
    changed_sources = validate_comparison(records, seed)
    if changed_sources is not None:
        a, b = (provenance[name]["core_source_hashes"] for name in ("baseline", "long"))
        changed_sources = sorted(set(changed_sources) | {name for name in CORE_SOURCES
                                if a[name] is not None and b[name] is not None and a[name] != b[name]})
        allowed = {"train.py", "decoding.py"} if seed == 42 else set()
        require(not (set(changed_sources) - allowed),
                "quality extension has unexpected training source changes: " + ", ".join(changed_sources))
    anchor = next(iter(records))
    lengths = source_lengths(examples[anchor], records[anchor])
    buckets, membership, fixed = compare_examples(examples, lengths, chrf)
    differences = {method: {measure: scores[f"long_{method}"][measure] - scores[f"baseline_{method}"][measure]
                    if scores[f"long_{method}"] and scores[f"baseline_{method}"]
                    and scores[f"long_{method}"][measure] is not None else None
                    for measure in ("bleu", "chrf")} for method in METHODS}
    result = {"status": "complete" if len(records) == 4 else "pending", "seed": seed,
              "config_controls": controls, "training_prefix_controls": prefix, "training": training, "scores": scores,
              "long_minus_baseline": differences, "buckets": buckets,
              "bucket_definition": "source SentencePiece pieces without BOS/EOS; no filtering or truncation",
              "bucket_diagnostic_status": "fixed supplementary diagnostic added in this follow-up, not an original predeclared metric",
              "covered_test_samples": sum(item["count"] for item in buckets),
              "expected_test_samples": EXPECTED["test"], "fixed_example_ids": [row["id"] for row in fixed],
              "signature": records[anchor]["signature"], "chrf_enabled": chrf,
              "chrf_definition": "case-sensitive chrF2: char_order=6, word_order=0, beta=2" if chrf else None,
              "training_source_differences": changed_sources,
              "core_source_provenance": provenance,
              "training_prefix_numerical_differences": prefix_numerical_differences(training),
              "evaluation_metadata": records}
    return result, membership, fixed


def report_lines(result):
    training = result["training"]
    status = "全部完成" if result["status"] == "complete" else "待完成（pending）"
    budgets = {name: (training[name]["config"] or {}).get("max_updates") for name in ("baseline", "long")}
    lines = ["# 单种子训练预算扩展报告", "", f"状态：{status}。seed={result['seed']}，标签平滑 0.1。", "",
             "本报告独立于核心多种子标签平滑消融。扩展仅增加训练更新预算，配置中除 run_dir/max_updates 外的项逐项核验；从同一种子的相同初始化重新训练，按完整验证集 greedy BLEU 选取最早达到最高分的 best checkpoint。seed=43、44 属于已知 seed=42 长训练结果后增加的重复实验，各自计划在新训练前保存；不能将新增计划描述为全部历史实验的事前声明。每个完成结果须通过对应验证。",
             f"基线预算：{budgets['baseline'] or '未记录'} 更新；扩展预算：{budgets['long'] or '尚未记录'} 更新。配置控制检查：{result['config_controls']}。", "",
             f"共同训练前缀的数据顺序、token/句数及每步学习率检查：{result['training_prefix_controls']}。", "",
             "test 仅用于报告保存的最终预测；本模块不加载模型、不生成预测、不搜索或修改任何解码参数。两种预算沿用基线已冻结的解码配置。未完成结果以待完成/null 表示，不能解释为零分。", "",
             "## 完整测试集结果", "", "| 解码 | 基线 BLEU | 长训练 BLEU | 长训练 − 基线 | 基线 best 更新 | 长训练 best 更新 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        a, b = (result["scores"][f"{name}_{method}"] or {} for name in ("baseline", "long"))
        lines.append(f"| {method} | {number(a.get('bleu'), missing='待完成')} | {number(b.get('bleu'), missing='待完成')} | {number(result['long_minus_baseline'][method]['bleu'], signed=True, missing='待完成')} | {a.get('checkpoint_update', '待完成')} | {b.get('checkpoint_update', '待完成')} |")
    lines += ["", f"每项完整评测包含 {result['expected_test_samples']} 句，SacreBLEU signature：`{result['signature']}`。保存的源数据、参考、预测、实例与 checkpoint 已验证，BLEU 由保存文本独立复算。"]
    if result["chrf_enabled"]:
        lines += ["", "### 补充 chrF2", "", result["chrf_definition"], "",
                  "| 解码 | 基线 chrF2 | 长训练 chrF2 | 长训练 − 基线 |", "|---|---:|---:|---:|"]
        for method in METHODS:
            a, b = (result["scores"][f"{name}_{method}"] or {} for name in ("baseline", "long"))
            lines.append(f"| {method} | {number(a.get('chrf'), missing='待完成')} | {number(b.get('chrf'), missing='待完成')} | {number(result['long_minus_baseline'][method]['chrf'], signed=True, missing='待完成')} |")
    lines += ["", "## 验证曲线与训练成本", "", "![两种预算的完整验证集 greedy BLEU](validation.png)", "",
              "| 训练 | 状态 | 已记录更新 / 预算 | 总运行秒数（含验证） | 峰值显存 MiB | 参数量 | 最佳验证 BLEU |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, label in (("baseline", "基线"), ("long", "长训练")):
        entry = training[name]
        summary = entry["summary"] or {}
        memory = summary.get("peak_cuda_bytes")
        lines.append(f"| {label} | {'完成' if entry['status'] == 'complete' else '待完成'} | {entry['last_logged_update']} / {budgets[name] or '未记录'} | {number(summary.get('training_seconds_total'), 1, '待完成')} | {number(memory / 2**20 if memory is not None else None, 1, '未记录')} | {summary.get('parameters', {}).get('total', '未记录')} | {number((entry['best_validation'] or {}).get('bleu'), missing='待完成')} |")
    lines += ["", "训练总耗时采用保存的累计计时，包含验证与保存，不含数据准备和模型初始化；未完成运行的 summary 若存在，只代表该次保存时的部分累计成本。显存为 PyTorch 峰值已分配量，非整卡占用。图中每个点来自保存日志中的完整验证集评测，星标为验证最佳点。固定预算与验证选优均不保证收敛。", "",
              "基线与扩展跨日期运行。相同硬件、驱动和软件版本不能控制电源、功率、温度及频率；现有硬件快照不足以还原两次运行的持续状态。跨日耗时差不能全归因于训练预算，预算倍数仅指 optimizer 更新数，不代表同倍数运行时间或单位计算量的因果比较。", "",
              "## 推理成本与生成结束情况", "",
              "| 组别 | 推理耗时（秒） | 吞吐（句/秒） | 平均批服务时间（ms） | 摊销每句时间（ms） | 达生成上限句数 | 未生成 EOS 句数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    cost_fields = (("elapsed_seconds", 2), ("sentences_per_second", 2),
                   ("mean_batch_latency_ms", 2), ("mean_latency_ms", 2),
                   ("generation_limit_hits", 0), ("unfinished_generations", 0))
    settings = set()
    for name, label in (("baseline", "基线"), ("long", "长训练")):
        for method in METHODS:
            metadata = result["evaluation_metadata"].get(f"{name}_{method}") or {}
            values = [number(metadata.get(field), digits, missing="待完成") for field, digits in cost_fields]
            lines.append(f"| {label} {method} | " + " | ".join(values) + " |")
            if metadata:
                precision = metadata.get("inference_precision", "未记录")
                settings.add(f"{'FP32' if precision == 'float32' else precision} / batch={metadata.get('inference_batch_size', '未记录')}")
    lines += ["", "已保存评测的精度与批大小：" + "；".join(sorted(settings)) + "（末批可不足设定批大小）。缺失组或字段显示待完成，不以零代替。", "",
              "推理计时包含源文分词与目标文本还原，GPU 每批同步；先预热再计时，预热不计入表中耗时。计时不含 checkpoint 加载、结果写盘和 BLEU 计算。摊销每句时间是批处理耗时除以句数，不等于单请求独占设备时的响应延迟。达到生成上限包括最后一步恰好生成 EOS 的情况；未生成 EOS 单独计数。", "",
              "基线与长训练的推理跨日期进行，电源、功率、温度及频率未持续控制，生成长度与停止时机也会影响速度。表中速度仅描述这些实际运行，不能据此把速度差归因于训练预算或作因果比较。", "",
              "## 固定源长度分桶", "",
              "本次增加固定长度分桶作为补充诊断，不属于原始训练计划的预声明指标，也不改变任何 test 调参规则。按同一已校验的 SentencePiece 分词器计算英文源 BPE 片段数，**不含 BOS/EOS**，不截断、不按参考或预测长度筛选。阈值固定为 ≤20、21–40、41–80、>80；各桶分数为该桶全部句子的 corpus BLEU，不是 sentence BLEU 平均值。",
              f"覆盖 {result['covered_test_samples']} / {result['expected_test_samples']} 句，每句恰好属于一桶。空桶记 null，不伪造为 0；分桶 BLEU 不能加权还原全语料 BLEU。完整归属见 [source_lengths.jsonl](source_lengths.jsonl)。", "",
              "| 源 BPE 长度 | 句数 | 基线 greedy | 基线 beam | 长训练 greedy | 长训练 beam |", "|---|---:|---:|---:|---:|---:|"]
    for bucket in result["buckets"]:
        values = [number((bucket["scores"][name] or {}).get("bleu"), missing='待完成' if bucket['scores'][name] is None else '空桶') for name in GROUPS]
        lines.append(f"| {bucket['bucket']} | {bucket['count']} | " + " | ".join(values) + " |")
    lines += ["", "## 固定样例与解释边界", "",
              f"[examples.jsonl](examples.jsonl) 保存原测试 ID % 400 == 0 的同一组 {len(result['fixed_example_ids'])} 条样例，每条包含英文、德文参考、源长度与基线/长训练的 greedy/beam 四路译文。缺失路径为 null，未按译文质量挑选。人工分析应分别检查成功、漏译、重复、专名与长句语序，并区分原文已有重复。", "",
              "这是单个 seed 的预算扩展，不能代替核心多种子消融，也不能估计跨种子方差或声明稳定优势。BLEU/chrF 反映参考重合度，不足以证明语义、事实和流畅度均正确。后续分析不得据此测试集重新选择 checkpoint、训练预算或解码参数。"]
    changed = result["training_source_differences"]
    provenance = result.get("core_source_provenance", {})
    if provenance:
        lines += ["", "七个核心模块（train/model/batching/tokenizer/decoding/common/data）的记录与快照逐项核验："]
        for name, item in provenance.items():
            lines.append(f"- {name}：快照 {item['snapshot_status']}；缺少的历史哈希 {item['missing_source_hashes']}。")
        lines += ["", "历史 seed=42 基线缺失的原始快照如实保留为缺失；未拿当前源码补写历史证据。原始记录路径和七模块哈希保存在 core_source_provenance。"]
    numerical = result.get("training_prefix_numerical_differences")
    if numerical:
        lines += ["", "### 共同训练前缀的数值核验", "",
                  f"共同前 {numerical['compared_updates']} 次更新的数据顺序、学习率、token 和句数须精确一致。loss 与验证指标的浮点数差异如实记录，不据此改动测试设置。", "",
                  "| 字段 | 比较点数 | 值不同的点数 | 仅一侧缺失点数 | 最大绝对差 |", "|---|---:|---:|---:|---:|"]
        for field, item in numerical["fields"].items():
            lines.append(f"| {field} | {item['compared_points']} | {item['different_points']} | {len(item['missing_on_one_side'])} | {number(item['max_absolute_difference'], 10, '无记录')} |")
        lines += ["", "逐点差异及原始训练日志路径保存在 metrics.json；日志数值相同不等于最终权重逐字节相同，也不构成跨硬件复现保证。"]
    if changed:
        lines += ["", "源码追溯：保存的训练源码哈希存在差异：" + ", ".join(f"`{name}`" for name in changed) + "。配置一致不等于训练程序逐字节相同；历史基线与后续运行包含恢复/持久化实现及解码源码版本差异，详见 metrics.json 中的原始记录。两组正式测试的解码源码哈希已严格核验相同。"]
    elif changed is None:
        lines += ["", "长训练正式评测尚未保存，跨预算初始化、数据和正式解码源码的最终一致性检查待其完成后执行。"]
    lines += ["", "机器可读分数、差值、成本、分桶与原始评测追溯信息见 [metrics.json](metrics.json)。"]
    return lines


def plot_validation(training, output):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    for name, label, color in (("baseline", "Baseline", "#2166ac"), ("long", "Long budget", "#b35806")):
        entry = training[name]
        budget = (entry["config"] or {}).get("max_updates")
        points = entry["validation"]
        suffix = f" ({budget:,} updates)" if budget else " (budget pending)"
        if points:
            axis.plot([row["update"] for row in points], [row["bleu"] for row in points],
                      marker="o", markersize=4, label=label + suffix, color=color)
            best = entry["best_validation"]
            axis.scatter([best["update"]], [best["bleu"]], marker="*", s=155,
                         color=color, edgecolor="white", zorder=4)
        else:
            axis.plot([], [], color=color, label=label + suffix + ": pending")
    axis.set(title="Single-seed budget extension: full validation greedy BLEU",
             xlabel="Optimizer updates", ylabel="Corpus BLEU (13a)", ylim=(0, None))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.18)
    axis.legend(fontsize=9)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def generate_report(baseline=None, long=None, out=None, chrf=False, seed=42):
    baseline = baseline or f"artifacts/runs/A{seed}"
    long = long or f"artifacts/runs/quality_A{seed}_12000"
    out = out or ("artifacts/quality_report" if seed == 42 else f"artifacts/quality_seed{seed}/report")
    result, membership, fixed = collect_quality(baseline, long, chrf, seed)
    return save_report(result, membership, fixed, out)


def save_report(result, membership, fixed, out):
    """Write already-verified collected results; useful for a multi-seed report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    write_json(out / "metrics.json", result)
    for name, rows in (("examples.jsonl", fixed), ("source_lengths.jsonl", membership)):
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    plot_validation(result["training"], out / "validation")
    (out / "report.md").write_text("\n".join(report_lines(result)) + "\n", encoding="utf-8")
    return out / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline")
    parser.add_argument("--long-run")
    parser.add_argument("--out")
    parser.add_argument("--chrf", action="store_true", help="also compute case-sensitive chrF2 from saved translations")
    args = parser.parse_args()
    print(generate_report(args.baseline or f"artifacts/runs/A{args.seed}",
          args.long_run or f"artifacts/runs/quality_A{args.seed}_12000",
          args.out or ("artifacts/quality_report" if args.seed == 42 else f"artifacts/quality_seed{args.seed}/report"),
          args.chrf, args.seed))


if __name__ == "__main__":
    main()
