"""Verify and explain the frozen paired BPE8k/unigram8k experiment.

This module only reads saved training/translation evidence. Group membership
comes exclusively from the pre-training raw-ID tables, never a new tokenizer.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import sacrebleu

from .common import read_json, sha256, write_json
from .data import read_pairs
from .representation import DEFAULT_ROOT, REPRESENTATIONS, SEEDS, validate_plan
from .representation_train import EXECUTION_SOURCES, validate_completed_run
from .summary import require, validate_saved_outputs

SPLITS = ("validation", "test")
METHODS = ("greedy", "beam")
MEASURES = ("bleu", "chrf")
FEATURES = ("source_words", "source_rare_fraction", "anchor_fragmentation")


def stats(values):
    values = list(values)
    require(all(type(value) in (int, float) and math.isfinite(value) for value in values),
            "statistics require finite numeric observations; missing is not zero")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def corpus_scores(predictions, references):
    require(len(predictions) == len(references), "prediction/reference sample count differs")
    if not predictions:
        return {"n": 0, "bleu": None, "chrf": None, "signature": None, "chrf_signature": None}
    bleu = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    chrf = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2, lowercase=False)
    return {"n": len(predictions), "bleu": bleu.corpus_score(predictions, [references]).score,
            "chrf": chrf.corpus_score(predictions, [references]).score,
            "signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


def aggregate_pairs(per_seed, seeds=SEEDS):
    selected = list(seeds)
    require(len(set(selected)) == len(selected) and selected, "requested seeds must be nonempty and unique")
    included = []
    for seed in selected:
        entry = per_seed.get(str(seed), per_seed.get(seed))
        if not entry or entry.get("status") != "complete":
            continue
        require(entry.get("seed") == seed and entry.get("controls", {}).get("status") == "verified",
                f"seed {seed}: completed pair lacks verified identity/controls")
        for rep in REPRESENTATIONS:
            run = entry["representations"][rep]
            require(run.get("status") == "complete", f"seed {seed}: incomplete representation")
            for split in SPLITS:
                for method in METHODS:
                    scores = run["evaluations"][split + "_" + method]
                    for measure in MEASURES:
                        value = scores.get(measure)
                        require(type(value) in (int, float) and math.isfinite(value),
                                f"seed {seed}: invalid {rep}/{split}/{method}/{measure}")
        included.append(seed)

    def value(seed, rep, split, method, measure):
        entry = per_seed.get(str(seed), per_seed.get(seed))
        return entry["representations"][rep]["evaluations"][split + "_" + method][measure]

    conditions, paired = {}, {}
    for split in SPLITS:
        conditions[split], paired[split] = {}, {}
        for method in METHODS:
            conditions[split][method] = {rep: {measure: stats(value(s, rep, split, method, measure) for s in included)
                                              for measure in MEASURES} for rep in REPRESENTATIONS}
            paired[split][method] = {}
            for measure in MEASURES:
                differences = {str(s): value(s, "unigram", split, method, measure)
                               - value(s, "bpe", split, method, measure) for s in included}
                paired[split][method][measure] = {**stats(differences.values()), "per_seed": differences}
    return {"selected_seeds": selected, "included_seeds": included, "n": len(included),
            "excluded_incomplete_seeds": [s for s in selected if s not in included],
            "conditions": conditions, "paired_unigram_minus_bpe": paired,
            "definition": "All aggregates share the same fully verified paired seeds. Sample SD ddof=1; n<2 SD=null. Paired differences computed per seed before aggregation; not significance tests."}


def verify_pair_controls(seed, runs, configs, logs, plan):
    require(all(configs[r]["seed"] == seed and configs[r]["representation"] == r for r in REPRESENTATIONS),
            "representation/seed labels differ from configs")
    ignored = {"data_dir", "run_dir", "representation"}
    require({k: v for k, v in configs["bpe"].items() if k not in ignored}
            == {k: v for k, v in configs["unigram"].items() if k not in ignored},
            "paired configurations differ beyond representation/data/output paths")
    expected_init = plan["initial_model_sha256"][str(seed)]
    require(all(runs[r]["initial_model_sha256"] == expected_init for r in REPRESENTATIONS),
            "paired initialization differs from frozen pre-training audit")
    require(runs["bpe"]["training"]["parameters"] == runs["unigram"]["training"]["parameters"],
            "paired parameter counts differ")
    require(len(logs["bpe"]) == len(logs["unigram"]) == configs["bpe"]["max_updates"],
            "paired logs do not contain the complete fixed update budget")
    fields = ("update", "epoch", "batch_cursor", "lr_used", "data_order_sha256", "microbatch_sha256", "sentences")
    for left, right in zip(logs["bpe"], logs["unigram"]):
        require(all(left.get(k) == right.get(k) for k in fields),
                f"paired sentence exposure/microbatch boundaries/LR differ at update {left.get('update')}")
    return {"status": "verified", "updates": len(logs["bpe"]), "identical_fields": list(fields),
            "initial_model_sha256": expected_init,
            "separately_verified_fields": ["source_tokens", "target_tokens", "padded_tokens"],
            "note": "Each representation's counts checked against the frozen common schedule; tokens and computation are not required to be equal."}


def validate_membership(table, examples, source_file_sha256=None):
    require(table.get("n") == len(table["rows"]) == len(examples), "frozen membership sample count differs")
    require(len({r["id"] for r in table["rows"]}) == len(table["rows"]), "duplicate ID in frozen membership")
    if source_file_sha256 is not None:
        require(table.get("source_file_sha256") == source_file_sha256, "grouping belongs to different source data")
    for member, example in zip(table["rows"], examples):
        require(member["id"] == example["id"], "frozen group ID order differs from evaluation")
        for field, key in (("source", "source_sha256"), ("reference", "reference_sha256")):
            require(hashlib.sha256(example[field].encode()).hexdigest() == member[key],
                    f"frozen group {field} differs for ID {member['id']}")


def read_evaluation(directory, cfg, split, method, protocol, best_update, table):
    directory = Path(directory)
    path = directory / "metrics.json"
    if not path.exists():
        return None, None
    metadata = read_json(path)
    expected = {"config": cfg, "split": split, "method": method, "smoke": False, "pilot": False,
                "checkpoint_update": best_update, "checkpoint_sha256": sha256(Path(cfg["run_dir"]) / "best.pt"),
                "inference_source_sha256": sha256(Path(__file__).with_name("decoding.py")),
                "beam_size": 1 if method == "greedy" else protocol["beam_size"],
                "length_penalty": protocol["length_penalty"], "max_generation_tokens": protocol["max_generation_tokens"],
                "inference_precision": "float32", "inference_batch_size": cfg["decode_batch_size"],
                "samples": table["n"], "expected_samples": table["n"]}
    require(all(metadata.get(k) == value for k, value in expected.items()),
            f"{directory}: evaluation identity/configuration/checkpoint/protocol differs")
    data = Path(cfg["data_dir"])
    fingerprints = {"manifest_sha256": sha256(data / "manifest.json"), "tokenizer_sha256": sha256(data / "spm.model"),
                    **{s + "_sha256": sha256(data / (s + ".jsonl")) for s in ("train", "validation", "test")}}
    require(metadata.get("data_fingerprints") == fingerprints, "evaluation data/tokenizer fingerprints differ")
    require(metadata.get("training_source_hashes") == {name: sha256(Path(__file__).with_name(name)) for name in
                ("train.py", "model.py", "batching.py", "tokenizer.py", "decoding.py")},
            "evaluation's legacy source subset differs from frozen live sources")
    require(isinstance(metadata.get("environment"), dict) and bool(metadata["environment"]),
            "evaluation runtime environment is missing")
    require(all(type(metadata.get(k)) in (int, float) and math.isfinite(metadata[k]) and metadata[k] > 0 for k in
                ("elapsed_seconds", "sentences_per_second", "mean_latency_ms", "mean_batch_latency_ms")),
            "evaluation runtime measurements are invalid")
    examples = validate_saved_outputs(metadata, directory, str(directory))
    validate_membership(table, examples, metadata["reference_sha256"])
    scores = corpus_scores([r["prediction"] for r in examples], [r["reference"] for r in examples])
    require(math.isclose(scores["bleu"], metadata["bleu"], rel_tol=0, abs_tol=1e-8), "independent BLEU differs")
    return {**scores, "metadata": metadata, "metrics_sha256": sha256(path),
            "examples_sha256": sha256(directory / "examples.jsonl")}, examples


def _read_run(root, plan, name, tables, protocol):
    info = plan["configs"][name]
    cfg = read_json(info["file"])
    rep, seed = info["representation"], info["seed"]
    require(cfg["representation"] == rep and cfg["seed"] == seed, "frozen run label differs from configuration")
    directory = Path(cfg["run_dir"])
    summary_path = directory / "summary.json"
    pending = {"status": "pending", "seed": seed, "representation": rep, "training": None,
               "evaluations": {split + "_" + method: None for split in SPLITS for method in METHODS}}
    done = root / "completed" / (name + ".json")
    if not summary_path.exists() or read_json(summary_path).get("complete") is not True:
        require(not done.exists(), "completed run marker exists without completed training")
        require(not any((directory / (s + "_" + m) / "metrics.json").exists() for s in SPLITS for m in METHODS),
                "completed evaluation exists without completed training")
        return pending, cfg, [], {}
    summary = validate_completed_run(cfg)
    sources = read_json(directory / "source_hashes.json")
    require(sources == {key: plan["source_hashes"][key] for key in EXECUTION_SOURCES},
            "training execution sources differ from the frozen plan")
    link = read_json(directory / "data_link.json")
    logs = read_pairs(directory / "train.jsonl")
    validation = [row for row in logs if "validation_bleu_greedy" in row]
    best = max(validation, key=lambda row: row["validation_bleu_greedy"])
    result = {**pending, "training": summary, "initial_model_sha256": link["initial_model_sha256"],
              "best_update": best["update"], "best_validation_bleu": best["validation_bleu_greedy"],
              "training_totals": {field: sum(row[field] for row in logs)
                                  for field in ("source_tokens", "target_tokens", "padded_tokens", "sentences")},
              "validation_curve": [{k: row[k] for k in ("update", "validation_bleu_greedy", "validation_samples", "validation_seconds")}
                                   for row in validation], "source_hashes": sources,
              "run_dir": str(directory.resolve()), "config_sha256": sha256(info["file"]),
              "train_log_sha256": sha256(directory / "train.jsonl")}
    example_sets = {}
    for split in SPLITS:
        for method in METHODS:
            key = split + "_" + method
            value, examples = read_evaluation(directory / key, cfg, split, method, protocol, best["update"], tables[split])
            if value:
                require(value["metadata"].get("initial_model_sha256") == link["initial_model_sha256"],
                        "evaluation initialization differs from training")
                result["evaluations"][key], example_sets[key] = value, examples
    if done.exists():
        marker = read_json(done)
        require(all(result["evaluations"].values()), "completed marker exists with missing evaluation")
        require(marker.get("run") == name and marker.get("training") == summary
                and marker.get("config_sha256") == result["config_sha256"]
                and marker.get("initial_model_sha256") == result["initial_model_sha256"]
                and marker.get("results") == {k: v["metadata"] for k, v in result["evaluations"].items()},
                "completed marker differs from independently verified evidence")
        result["status"] = "complete"
    else:
        result["pending_reason"] = "runner completion marker not yet present"
    return result, cfg, logs, example_sets


def grouped_scores(tables, example_sets, included_seeds):
    """Recompute corpus metrics on the same saved IDs for every paired arm."""
    result = {}
    for split, table in tables.items():
        result[split] = {}
        names = FEATURES + (("cluster",) if split == "validation" else ())
        for feature in names:
            count = 5 if feature == "cluster" else len(table["definitions"][feature + "_boundaries"]) + 1
            require(all(type(r["groups"].get(feature)) is int and 0 <= r["groups"][feature] < count for r in table["rows"]),
                    "invalid frozen group label")
            output = []
            for group in range(count):
                indices = [i for i, row in enumerate(table["rows"]) if row["groups"][feature] == group]
                members = [table["rows"][i] for i in indices]
                per_seed = {}
                for seed in included_seeds:
                    arms = {}
                    for rep in REPRESENTATIONS:
                        arms[rep] = {}
                        for method in METHODS:
                            examples = example_sets[str(seed)][rep][split + "_" + method]
                            validate_membership(table, examples)
                            selected = [examples[i] for i in indices]
                            arms[rep][method] = corpus_scores([r["prediction"] for r in selected], [r["reference"] for r in selected])
                    per_seed[str(seed)] = arms
                conditions, paired = {}, {}
                for method in METHODS:
                    conditions[method] = {rep: {metric: stats(per_seed[str(s)][rep][method][metric] for s in included_seeds)
                                                          if members else stats([]) for metric in MEASURES}
                                          for rep in REPRESENTATIONS}
                    paired[method] = {}
                    for metric in MEASURES:
                        diffs = {str(s): per_seed[str(s)]["unigram"][method][metric] - per_seed[str(s)]["bpe"][method][metric]
                                 for s in included_seeds} if members else {}
                        paired[method][metric] = {**stats(diffs.values()), "per_seed": diffs}
                output.append({"group": group, "n": len(members), "ids": [r["id"] for r in members],
                    "small": 0 < len(members) < 30, "empty": not members, "included_seeds": list(included_seeds),
                    "features": {key: {**stats(r["features"][key] for r in members),
                                      "median": statistics.median(r["features"][key] for r in members) if members else None}
                                 for key in FEATURES}, "per_seed": per_seed, "conditions": conditions,
                    "paired_unigram_minus_bpe": paired})
            result[split][feature] = output
    return result


def fixed_examples(tables, example_sets, included_seeds):
    selected = {"test": [r["id"] for r in tables["test"]["rows"] if r["id"] % 400 == 0], "validation": []}
    for cluster in range(5):
        selected["validation"] += sorted(r["id"] for r in tables["validation"]["rows"] if r["groups"]["cluster"] == cluster)[:3]
    records = []
    for split in SPLITS:
        lookup = {r["id"]: i for i, r in enumerate(tables[split]["rows"])}
        for seed in included_seeds:
            for identifier in selected[split]:
                index = lookup[identifier]
                source = example_sets[str(seed)]["bpe"][split + "_greedy"][index]
                records.append({"split": split, "seed": seed, "id": identifier,
                    "groups": tables[split]["rows"][index]["groups"], "source": source["source"],
                    "reference": source["reference"], "translations": {rep + "_" + method:
                        example_sets[str(seed)][rep][split + "_" + method][index]["prediction"]
                        for rep in REPRESENTATIONS for method in METHODS},
                    "selection": "test: original ID % 400 == 0; validation: lowest 3 original IDs per frozen cluster",
                    "annotation": "saved text only; any subsequent interpretation is agent review, not independent human annotation"})
    return records


def collect(root=DEFAULT_ROOT):
    root = Path(root)
    plan = validate_plan(root)
    protocol = read_json(plan["protocol"])
    tables = {split: read_json(plan["groups"][split]["file"]) for split in SPLITS}
    per_seed, examples = {}, {}
    for seed in SEEDS:
        runs, configs, logs, examples[str(seed)] = {}, {}, {}, {}
        for rep in REPRESENTATIONS:
            runs[rep], configs[rep], logs[rep], examples[str(seed)][rep] = _read_run(root, plan, f"{rep}_{seed}", tables, protocol)
        complete = all(runs[r]["status"] == "complete" for r in REPRESENTATIONS)
        controls = verify_pair_controls(seed, runs, configs, logs, plan) if complete else {"status": "pending"}
        per_seed[str(seed)] = {"seed": seed, "status": "complete" if complete else "pending", "controls": controls,
                               "representations": runs}
    aggregate = aggregate_pairs(per_seed)
    statistics_path = root / "representation_statistics.json"
    metrics = {"schema_version": 1, "status": "complete" if aggregate["n"] == len(SEEDS) else "pending",
               "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "plan_sha256": sha256(root / "plan.json"), "plan_created_at_utc": plan["created_at_utc"],
               "per_seed": per_seed, "representation_statistics": read_json(statistics_path),
               "provenance": {"plan": str((root / "plan.json").resolve()),
                   "representation_statistics_sha256": sha256(statistics_path),
                   "groups": plan["groups"], "report_source_sha256": sha256(__file__),
                   "training_execution_sources": {key: plan["source_hashes"][key] for key in EXECUTION_SOURCES}},
               "historical_reference": {"path": "artifacts/report/report.md",
                   "scope": "Historical BPE runs use different training sentence/batch schedules and are excluded from paired statistics."}}
    runtime_observation = root / "runtime_gpu_observation.csv"
    if runtime_observation.is_file():
        metrics["provenance"]["runtime_gpu_observation"] = {
            "path": str(runtime_observation.resolve()), "sha256": sha256(runtime_observation),
            "scope": "Opportunistic hardware snapshots, not a continuous controlled thermal/power experiment."}
    grouped = grouped_scores(tables, examples, aggregate["included_seeds"])
    fixed = fixed_examples(tables, examples, aggregate["included_seeds"])
    return {"metrics": metrics, "aggregate": aggregate, "groups": grouped, "examples": fixed}


def _number(value, signed=False):
    return "待完成/不可估计" if value is None else (f"{value:+.3f}" if signed else f"{value:.3f}")


def _stat(value, signed=False):
    return _number(value["mean"], signed) + " ± " + _number(value["sample_std"])


def report_text(result):
    metrics, aggregate, groups = result["metrics"], result["aggregate"], result["groups"]
    lines = ["# BPE 8k / unigram 8k 共同批次表示对照", "",
        f"状态：{metrics['status']}；完整核验后纳入种子 {aggregate['included_seeds']}，n={aggregate['n']}；待完成 {aggregate['excluded_incomplete_seeds']}。缺失值保持 null，不计零分。", "",
        "## 问题与方法", "",
        "在相同训练句对暴露、模型形状和 3,000 更新预算下，BPE 与 unigram 表示是否改变序列长度、翻译质量及计算成本？本实验在历史结果和探索性分析之后新建，并在六次训练前冻结设计、输入、训练代码、共同批次、分组和解码规则。旧 BPE 仅作为历史参照，不进入本轮严格配对组。", "",
        "两个 tokenizer 各只拟合同一完整清洗官方训练语料，实际词表均为 8,000；新模型均从随机初始化训练，11,681,600 参数。两种表示的源/目标含 BOS/EOS 长度均 ≤128 的交集作为共同训练集；每句取两种表示源长最大值和目标长最大值生成 4,096 padded-token 上界的共同微批，每更新累积四个完全相同原始 ID 与边界的微批。三个种子分别为 42/43/44。", "",
        "同种子数值初始化、更新数、每步句数、ID 顺序、微批边界、学习率和验证时点严格相同；token ID 的词义不同，实际 source/target/padded token、时间和计算量不宣称相等。每 500 更新在全部 888 句验证集上运行 Greedy BLEU，精确同分保留最早最佳检查点。所有最终方法使用该表示自己的同一最佳检查点。", "",
        "最终 Greedy/Beam=4 共用历史冻结协议（长度惩罚 0.6、生成上限 256、float32、batch=16），验证/测试分别为完整 888/8,079 句。BLEU 为大小写敏感 SacreBLEU tokenizer=13a；chrF2 为 char_order=6、word_order=0、beta=2。全部文本、原始 ID、顺序、哈希与指标独立复核。", "",
        "该共享协议的长度惩罚0.6、Beam=4和生成上限256来自历史BPE模型的验证选参；本轮没有分别为新BPE和unigram重新调参。因此结论限定在这套共同历史协议下，不代表两种表示各自最优解码设置的比较。相同的token级长度惩罚与生成上限也不等于相同的文本级约束。", "",
        "## 结果：完整语料与配对种子", "",
        "| 划分 | 解码 | 表示 | n种子 | BLEU均值 ± 样本SD | chrF2均值 ± 样本SD |", "|---|---|---|---:|---:|---:|"]
    for split in SPLITS:
        for method in METHODS:
            for rep in REPRESENTATIONS:
                values = aggregate["conditions"][split][method][rep]
                lines.append(f"| {split} | {method} | {rep} | {aggregate['n']} | {_stat(values['bleu'])} | {_stat(values['chrf'])} |")
    lines += ["", "| 划分 | 解码 | unigram − BPE ΔBLEU | ΔchrF2 |", "|---|---|---:|---:|"]
    for split in SPLITS:
        for method in METHODS:
            values = aggregate["paired_unigram_minus_bpe"][split][method]
            lines.append(f"| {split} | {method} | {_stat(values['bleu'], True)} | {_stat(values['chrf'], True)} |")
    lines += ["", "所有均值及差值仅使用两种表示、两种解码、两种划分均完整且控制核验通过的同一组种子。先按种子计算 unigram−BPE，再取均值和样本标准差（ddof=1）；n<2 的标准差为 null。标准差不是置信区间，三种子不能直接支持统计显著或普遍优势。", "", "![分数与配对差值](figures/quality.png)", "",
        "### 逐种子测试结果", "", "| seed | 解码 | BPE BLEU / chrF2 | unigram BLEU / chrF2 | ΔBLEU / ΔchrF2 |", "|---|---|---:|---:|---:|"]
    for seed in SEEDS:
        entry = metrics["per_seed"][str(seed)]
        for method in METHODS:
            arms = [(entry["representations"][r]["evaluations"]["test_" + method] or {}) for r in REPRESENTATIONS]
            vals = [" / ".join(_number(a.get(k)) for k in MEASURES) for a in arms]
            delta = " / ".join(_number(arms[1][k] - arms[0][k], True) if k in arms[0] and k in arms[1] else "待完成" for k in MEASURES)
            lines.append(f"| {seed} | {method} | {vals[0]} | {vals[1]} | {delta} |")
    lines += ["", "### 方向与解释", ""]
    for method in METHODS:
        values = aggregate["paired_unigram_minus_bpe"]["test"][method]["bleu"]["per_seed"]
        if values:
            positive, negative = sum(v > 0 for v in values.values()), sum(v < 0 for v in values.values())
            lines.append(f"- 测试 {method}：unigram−BPE 的 BLEU 差值在 {positive}/{len(values)} 个种子为正，{negative}/{len(values)} 个为负；均值 {_number(statistics.mean(values.values()), True)}。该方向只描述当前共同句对、更新预算及解码流程，不单独证明某种分词机制的因果作用。")
            character_values = aggregate["paired_unigram_minus_bpe"]["test"][method]["chrf"]["per_seed"]
            character_positive = sum(v > 0 for v in character_values.values())
            lines.append(f"- 同一 {method} 的 chrF2 差值均值 {_number(statistics.mean(character_values.values()), True)}，"
                         f"其中 {character_positive}/{len(character_values)} 个种子为正。BLEU与chrF2关注不同粒度的参考文本重合；应并列解释二者，不能仅凭BLEU方向宣称一种表示在所有质量维度上全面优于另一种。")
        else:
            lines.append(f"- 测试 {method}：尚无完整配对种子，暂不解释分数方向。")
    statistics_data = metrics["representation_statistics"]
    common = statistics_data["common_training_set"]
    lines += ["", "## 序列表示与实际成本", "",
        f"清洗训练集 {common['cleaned_pairs']:,} 对；共同保留 {common['retained_pairs']:,} 对，排除 {common['excluded_pairs']:,} 对。各自超长的句子不能只在自身条件删除。两个 tokenizer 都拟合完整清洗训练语料，模型训练使用共同交集。", "",
        "| 划分/表示 | 源均长 | 目标均长 | 源片段/空格词 | 目标片段/空格词 | 源/目标UNK数 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        for rep in REPRESENTATIONS:
            values = statistics_data[split][rep]
            lines.append(f"| {split}/{rep} | {values['src']['encoded_length_including_bos_eos']['mean']:.3f} | {values['tgt']['encoded_length_including_bos_eos']['mean']:.3f} | {values['src']['corpus_pieces_per_whitespace_word']:.3f} | {values['tgt']['corpus_pieces_per_whitespace_word']:.3f} | {values['src']['unk_tokens']}/{values['tgt']['unk_tokens']} |")
    lines += ["", "固定表示的观察："]
    for side, label in (("src", "英文源端"), ("tgt", "德文目标端")):
        a, b = (statistics_data["train"][rep][side] for rep in REPRESENTATIONS)
        length_delta = b["encoded_length_including_bos_eos"]["mean"] - a["encoded_length_including_bos_eos"]["mean"]
        pieces_delta = b["corpus_pieces_per_whitespace_word"] - a["corpus_pieces_per_whitespace_word"]
        lines.append(f"- {label}：完整清洗训练集 unigram−BPE 平均含特殊token长度差 {length_delta:+.3f}，每空格词片段数差 {pieces_delta:+.3f}。长度变化是表示结果，不能直接等同于质量改善。")
    lines += ["", "长度含 BOS/EOS；片段排除 BOS/EOS/PAD、保留并单独计数 UNK；词为 Python str.split 空格词，大小写和标点不再规范化。测试表示统计仅作已冻结设计的描述，不用于重新定义方案。", "",
        "| run | 最佳更新 | 最佳验证BLEU | 训练秒 | 峰值已分配MiB | 句子暴露 | 源tokens | 目标损失tokens | padded tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for rep in REPRESENTATIONS:
            run = metrics["per_seed"][str(seed)]["representations"][rep]
            if run["training"]:
                totals = run["training_totals"]
                peak = run['training']['peak_cuda_bytes']
                lines.append(f"| {rep}_{seed} | {run['best_update']} | {run['best_validation_bleu']:.3f} | {run['training']['training_seconds_total']:.3f} | {_number(peak / 2**20 if peak is not None else None)} | {totals['sentences']:,} | {totals['source_tokens']:,} | {totals['target_tokens']:,} | {totals['padded_tokens']:,} |")
            else:
                lines.append(f"| {rep}_{seed} | 待完成 | — | — | — | — | — | — | — |")
    lines += ["", "训练总时长包括训练循环内验证和检查点处理；恢复时累积各调用时间。峰值为已分配显存，不是驱动显示或reserved显存。源token含BOS/EOS；目标损失token排除BOS、含EOS；padded token按每个微批的源与完整目标矩形计数。", "",
        "执行顺序固定为 BPE42→unigram42→BPE43→unigram43→BPE44→unigram44，没有随机化或交错平衡。功率、温度和前后台负载未受控；运行期间观察到较高GPU温度和更新耗时波动。现场证据见 [runtime_gpu_observation.csv](../runtime_gpu_observation.csv)，它是实际快照而非连续温控记录，不能据此单独判定波动原因。训练时间和推理吞吐只能作为本轮实测成本；不能根据它们的均值宣称某表示具有因果性的速度优势。", "",
        "![序列长度与训练成本](figures/representation_cost.png)", "", "![验证曲线与最佳检查点](figures/validation_curves.png)", "",
        "### 推理实测", "", "| run | 划分/解码 | 秒 | 句/秒 | 摊销毫秒/句 | 平均毫秒/批 | 触限数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for rep in REPRESENTATIONS:
            for key, scores in metrics["per_seed"][str(seed)]["representations"][rep]["evaluations"].items():
                if scores:
                    m = scores["metadata"]
                    lines.append(f"| {rep}_{seed} | {key} | {m['elapsed_seconds']:.3f} | {m['sentences_per_second']:.3f} | {m['mean_latency_ms']:.3f} | {m['mean_batch_latency_ms']:.3f} | {m['generation_limit_hits']} |")
    lines += ["", "所有推理为float32、batch=16（末批可小于16），排除2个热身批，保留原句顺序；计时含源SentencePiece编码和目标反分词，每批CUDA同步。句/秒=总句数/总计时；摊销毫秒/句=各批服务时长之和/总句数，不是batch=1交互延迟；平均毫秒/批=批服务时长均值。GPU顺序执行，但日期、电源、温度和缓存条件未完全控制，速度差异仅作描述。", "",
        "## 固定相同句子集合的群体分析", "",
        "问题：差异是否集中于较长、稀有或历史BPE切分较碎的句子？方法：所有表示/种子/解码共用训练前保存的原始ID分组。源词数边界10/20/40；训练词频≤5的稀有比例边界0/0.05/0.15；历史锚定BPE每词片段数边界1.25/1.5/2。边界上限归低组。验证额外复用原探索PCA50/K5的五簇ID，测试不推断新簇。", "",
        "每个组、种子、表示和解码都重新计算 corpus BLEU/chrF2。图中展示Beam；完整Greedy与Beam及各组描述统计见groups.json。小组n<30标记small，空组指标null；不平均sentence BLEU，也不把各簇BLEU加权冒充总体BLEU。", "",
        "| 划分 | 分组 | n句 | 小组 | Beam BPE BLEU | Beam unigram BLEU | ΔBLEU | ΔchrF2 |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for feature, records in groups[split].items():
            for row in records:
                lines.append(f"| {split} | {feature}:{row['group']} | {row['n']} | {'空组' if row['empty'] else '是' if row['small'] else '否'} | {_stat(row['conditions']['beam']['bpe']['bleu'])} | {_stat(row['conditions']['beam']['unigram']['bleu'])} | {_stat(row['paired_unigram_minus_bpe']['beam']['bleu'], True)} | {_stat(row['paired_unigram_minus_bpe']['beam']['chrf'], True)} |")
    lines += ["", "### 固定两端分组的观察", ""]
    for feature in FEATURES:
        low, high = groups["test"][feature][0], groups["test"][feature][-1]
        if aggregate["n"] and low["n"] and high["n"]:
            contrasts = []
            for rep in REPRESENTATIONS:
                a, b = (r["conditions"]["beam"][rep]["bleu"]["mean"] for r in (low, high))
                contrasts.append(f"{rep} {_number(a)}→{_number(b)}")
            lines.append(f"- {feature}最低组→最高组（n={low['n']}→{high['n']}）：Beam BLEU均值 "
                         + "；".join(contrasts) + f"。最高组 unigram−BPE 为 {_stat(high['paired_unigram_minus_bpe']['beam']['bleu'], True)}；这些组在其他特征上也不同，不能把组间差距单独归因于该特征。")
    lines += ["", "![相同句子集合上的Beam差值](figures/groups.png)", "",
        "解释：这些分组具有相关性而非独立随机处理；长度、稀有度和碎片化可能同时变化。旧模型句向量形成的簇也不等于语义主题。群体差值用于定位后续可核查的错误，不证明某特征导致质量变化。", "",
        "历史BPE碎片化与旧BPE模型的簇是事先固定的诊断视角，并非表示无关的语义真值；它们确保每组评测的是相同句子，却不能消除原表示作为锚点的限制。", "",
        "## 固定样例与复核", "",
        "[examples.jsonl](examples.jsonl)保存每个完整种子的四路实际译文：测试原始ID%400==0，验证为每个旧簇最小3个原始ID。样例选择与质量分数无关；并列展示改善与失败，不能只挑有利译文。任何语言学判断须标为代理复核，不代表独立人工标注。", "",
        "## 局限与后续", "",
        "- 该实验使用同一官方划分上的三个随机种子，不是交叉验证；CV课程含义仍待确认。", 
        "- 既有多轮历史测试结果已经公开；本轮冻结可防止当前实现按新测试分数调整，但不使整个项目成为从未观察测试的研究。验证也用于选择checkpoint，验证分组结果是诊断证据。",
        "- 最大生成256 token和长度惩罚以各自子词计数，文本容量不同；须连同触限比例解释。共同句对/更新数不等于等token数、等FLOPs或等时长。",
        "- 统计不含旧BPE模型。历史训练及12,000更新实验在各自报告中独立呈现，不能替代本轮共同批次BPE控制组。",
        "- 独立预训练MT直接推理应单列方法、原生分词器、数据重叠风险及成本；与本轮从零训练结果之间的差异包含数据、规模和解码实现等因素。LoRA与CV未在本报告执行。", "",
        "## 可复现证据", "",
        f"计划SHA-256：`{metrics['plan_sha256']}`。训练入口的9个执行模块均保留于各run/source_snapshot，整套编排/推理源码保留于上级source_snapshot。旧metrics内的五文件训练来源字段不是本轮完整执行源码列表。", "",
        "命令：`./project-python.ps1 -m mt.representation_report --root artifacts/representation_2026_10_27`。机器可读输出为metrics.json、aggregate.json、groups.json、examples.jsonl。图同时保存PNG/PDF。报告生成不会训练、解码或改写历史评测。", ""]
    return "\n".join(lines)


def figures(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "savefig.dpi": 160})
    folder = Path(out) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    colors = {"bpe": "#2563a6", "unigram": "#db7a28"}

    def save(fig, name):
        fig.savefig(folder / (name + ".png"), bbox_inches="tight")
        fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    agg, metrics = result["aggregate"], result["metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    for row, split in enumerate(SPLITS):
        ax, contrast = axes[row]
        for rep, offset in (("bpe", -.18), ("unigram", .18)):
            vals = [agg["conditions"][split][m][rep]["bleu"] for m in METHODS]
            y = [v["mean"] if v["mean"] is not None else np.nan for v in vals]
            ax.bar(np.arange(2) + offset, y, .34, color=colors[rep], label=rep,
                   yerr=[v["sample_std"] or 0 for v in vals], capsize=4)
        ax.set(xticks=range(2), xticklabels=METHODS, ylabel="Corpus BLEU", title=f"{split}: paired seeds n={agg['n']}")
        ax.legend()
        for index, method in enumerate(METHODS):
            item = agg["paired_unigram_minus_bpe"][split][method]["bleu"]
            values = list(item["per_seed"].values())
            contrast.scatter([index] * len(values), values, s=45, color="#155e75", zorder=3)
            if values:
                contrast.errorbar(index + .12, item["mean"], yerr=item["sample_std"] or 0, fmt="D", color="#111827", capsize=4)
        contrast.axhline(0, color="#64748b", linewidth=1)
        contrast.set(xticks=range(2), xticklabels=METHODS, ylabel="Unigram - BPE BLEU", title="Per-seed differences; mean +/- sample SD")
    save(fig, "quality")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    values = metrics["representation_statistics"]
    for rep, offset in (("bpe", -.18), ("unigram", .18)):
        axes[0].bar(np.arange(2) + offset, [values["train"][rep][s]["encoded_length_including_bos_eos"]["mean"] for s in ("src", "tgt")], .34, label=rep, color=colors[rep])
        rows = [metrics["per_seed"][str(s)]["representations"][rep] for s in SEEDS]
        axes[1].bar(np.arange(3) + offset, [r["training_totals"]["target_tokens"] / 1e6 if r["training"] else np.nan for r in rows], .34, color=colors[rep])
        axes[2].bar(np.arange(3) + offset, [r["training"]["training_seconds_total"] / 60 if r["training"] else np.nan for r in rows], .34, color=colors[rep])
    axes[0].set(xticks=range(2), xticklabels=["Source", "Target"], ylabel="Mean encoded tokens", title="Full cleaned training corpus")
    axes[0].legend()
    axes[1].set(xticks=range(3), xticklabels=SEEDS, xlabel="Seed", ylabel="Million target loss tokens", title="Actual exposure; same raw sentences")
    axes[2].set(xticks=range(3), xticklabels=SEEDS, xlabel="Seed", ylabel="Training minutes", title="Observed cost incl. validation")
    fig.suptitle("Fixed BPE/U order; power, temperature and background load uncontrolled. Timing is descriptive.", fontsize=11)
    save(fig, "representation_cost")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        for rep in REPRESENTATIONS:
            run = metrics["per_seed"][str(seed)]["representations"][rep]
            if run["training"]:
                curve = run["validation_curve"]
                ax.plot([p["update"] for p in curve], [p["validation_bleu_greedy"] for p in curve], "o-", color=colors[rep], label=rep)
                ax.scatter(run["best_update"], run["best_validation_bleu"], marker="*", s=150, color=colors[rep], edgecolor="black", zorder=4)
        ax.set(title=f"seed={seed}", xlabel="Optimizer updates")
    axes[0].set_ylabel("Full validation greedy BLEU")
    if axes[0].lines:
        axes[0].legend()
    save(fig, "validation_curves")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    fig.suptitle(f"Beam on fixed source-ID groups; mean +/- sample SD across {agg['n']} paired seeds", fontsize=12)
    for row, split in enumerate(SPLITS):
        for col, feature in enumerate(FEATURES + ("cluster",)):
            ax = axes[row, col]
            if feature not in result["groups"][split]:
                ax.axis("off"); ax.text(.05, .5, "No test clustering\n(pre-frozen design)", transform=ax.transAxes)
                continue
            records = result["groups"][split][feature]
            values = [r["paired_unigram_minus_bpe"]["beam"]["bleu"] for r in records]
            ax.bar(range(len(records)), [v["mean"] if v["mean"] is not None else np.nan for v in values],
                   color="#155e75", yerr=[v["sample_std"] or 0 for v in values], capsize=3)
            ax.axhline(0, color="#64748b", linewidth=1)
            ax.set(xticks=range(len(records)), xticklabels=[f"{r['group']}\nn={r['n']}{'*' if r['small'] else ''}" for r in records],
                   title=f"{split}: {feature}", ylabel="Unigram - BPE BLEU")
    save(fig, "groups")


def write_report(root=DEFAULT_ROOT, out=None):
    root = Path(root)
    out = Path(out) if out else root / "report"
    os.environ.setdefault("MPLCONFIGDIR", str((out / "matplotlib").resolve()))
    result = collect(root)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "aggregate", "groups"):
        write_json(out / (name + ".json"), result[name])
    (out / "examples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["examples"]), encoding="utf-8")
    (out / "report.md").write_text(report_text(result), encoding="utf-8")
    figures(result, out)
    files = [p for p in out.rglob("*") if p.is_file() and "matplotlib" not in p.parts and p.name != "verification.json"]
    write_json(out / "verification.json", {"status": result["metrics"]["status"],
        "included_seeds": result["aggregate"]["included_seeds"], "report_source_sha256": sha256(__file__),
        "files_sha256": {str(p.relative_to(out)): sha256(p) for p in files}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = write_report(args.root, args.out)
    print(json.dumps({"status": result["metrics"]["status"], "included_seeds": result["aggregate"]["included_seeds"],
                      "out": args.out or str(Path(args.root) / "report")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
