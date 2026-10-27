"""Independently verify and report paired 3,000/12,000-update quality runs.

All calculations use saved text/logs. No model is loaded and no test setting is
selected here. Incomplete seeds are displayed but excluded from every aggregate.
"""
import argparse
import json
import math
import os
from pathlib import Path

from .common import sha256, write_json
from .quality_report import CORE_SOURCES, METHODS, GROUPS, collect_quality, save_report
from .report import number, sample_statistics
from .summary import artifact_path, read_object, require, valid_hash


DEFAULT_PROTOCOL = "artifacts/decode_protocol_seed42.json"
MEASURES = ("bleu", "chrf")
READING_IDS = (0, 6800, 7600)  # Fixed subset of ID % 400 == 0; never selected by scores.


def paired_aggregates(per_seed, seeds):
    """Use identical, complete paired seeds for all scores and differences."""
    selected = list(seeds)
    require(selected and len(selected) == len(set(selected)), "seeds must be nonempty and unique")
    included = []
    for seed in selected:
        require(type(seed) is int and seed >= 0, "seeds must be nonnegative integers")
        entry = per_seed.get(seed)
        if not entry or entry["status"] != "complete":
            continue
        require(entry.get("seed") == seed, f"seed {seed}: result label mismatch")
        require(entry.get("config_controls") == entry.get("training_prefix_controls") == "verified",
                f"seed {seed}: complete result lacks verified training controls")
        for group in GROUPS:
            for measure in MEASURES:
                value = (entry["scores"].get(group) or {}).get(measure)
                require(type(value) in (int, float) and math.isfinite(value),
                        f"seed {seed}: completed {group} is missing {measure}")
        included.append(seed)
    groups = {group: {measure: sample_statistics([per_seed[s]["scores"][group][measure] for s in included])
                     for measure in MEASURES} for group in GROUPS}
    paired = {}
    for method in METHODS:
        paired[method] = {}
        for measure in MEASURES:
            values = {s: per_seed[s]["scores"][f"long_{method}"][measure]
                      - per_seed[s]["scores"][f"baseline_{method}"][measure] for s in included}
            paired[method][measure] = {"per_seed": values, **sample_statistics(list(values.values()))}
    return {"selected_seeds": selected, "included_seeds": included, "n": len(included),
            "excluded_incomplete_seeds": [s for s in selected if s not in included],
            "groups": groups, "paired_long_minus_baseline": paired,
            "uncertainty": "sample standard deviation across complete paired seeds, ddof=1; null for n<2; not a confidence interval"}


def validate_cross_seed(per_seed, protocol):
    """Require shared data/model/decoder, while initialization varies by seed."""
    require(protocol.get("selection_split") == "validation" and protocol.get("beam_size") == 4,
            "shared decoding protocol must be validation-frozen beam=4")
    anchor = None
    configs, initializations, long_sources = {}, {}, None
    fields = ("data_fingerprints", "inference_source_sha256", "signature",
              "inference_precision", "inference_batch_size")
    for seed, result in per_seed.items():
        require(result.get("seed") == seed, f"seed {seed}: result label mismatch")
        provenance = result.get("core_source_provenance", {}).get("long", {})
        has_long_results = any(name.startswith("long_") for name in result["evaluation_metadata"])
        if has_long_results:
            sources = provenance.get("core_source_hashes", {})
            require(provenance.get("snapshot_status") == "verified"
                    and set(sources) == set(CORE_SOURCES) and all(valid_hash(value) for value in sources.values()),
                    f"seed {seed}: long run requires verified seven-module source provenance")
            if long_sources is None:
                long_sources = sources
            require(sources == long_sources, f"seed {seed}: cross-seed long core source hashes differ")
        for name, metrics in result["evaluation_metadata"].items():
            require(metrics["config"].get("seed") == seed, f"seed {seed}/{name}: wrong checkpoint seed")
            for key in ("length_penalty", "max_generation_tokens"):
                require(metrics.get(key) == protocol.get(key), f"seed {seed}/{name}: frozen protocol {key} differs")
            require(metrics.get("beam_size") == (1 if name.endswith("greedy") else protocol["beam_size"]),
                    f"seed {seed}/{name}: frozen protocol beam size differs")
            if anchor is None:
                anchor = metrics
            for field in fields:
                require(metrics.get(field) == anchor.get(field), f"seed {seed}/{name}: cross-seed {field} differs")
            normalized = {k: v for k, v in metrics["config"].items() if k not in ("seed", "run_dir")}
            budget = "baseline" if name.startswith("baseline_") else "long"
            require(metrics["config"].get("max_updates") == (3000 if budget == "baseline" else 12000),
                    f"seed {seed}/{name}: expected 3000/12000 fixed budgets")
            require(metrics["config"].get("validate_every") == 500,
                    f"seed {seed}/{name}: expected full validation every 500 updates")
            if budget in configs:
                require(normalized == configs[budget], f"seed {seed}: cross-seed {budget} config differs")
            configs[budget] = normalized
            init = metrics["initial_model_sha256"]
            if seed in initializations:
                require(init == initializations[seed], f"seed {seed}: initialization differs within seed")
            initializations[seed] = init
    require(len(set(initializations.values())) == len(initializations),
            "different seeds unexpectedly share an initialization fingerprint")
    return {"shared_data_model_decoder": "verified" if anchor else "pending",
            "long_core_source_hashes": long_sources,
            "initialization_sha256_by_seed": initializations,
            "initialization_rule": "same seed: baseline/long equal; different seeds: distinct",
            "environment_rule": "environments are recorded per evaluation; cross-date speed is descriptive only"}


def collect_multiseed(runs="artifacts/runs", seeds=(42, 43, 44), protocol=DEFAULT_PROTOCOL):
    root, selected = Path(runs), list(seeds)
    require(selected and len(selected) == len(set(selected)), "seeds must be nonempty and unique")
    protocol_path = artifact_path(protocol, "shared decode protocol")
    frozen = read_object(protocol_path, "shared decode protocol")
    per_seed, memberships, examples = {}, {}, {}
    for seed in selected:
        result, membership, fixed = collect_quality(root / f"A{seed}", root / f"quality_A{seed}_12000", True, seed)
        per_seed[seed], memberships[seed], examples[seed] = result, membership, fixed
    controls = validate_cross_seed(per_seed, frozen)
    aggregates = paired_aggregates(per_seed, selected)
    return {"status": "complete" if aggregates["n"] == len(selected) else "pending",
            "design": "seed 43/44 extensions were added after observing the seed 42 12000-update result; each new plan frozen before its training; not a retrospective preregistration of all historical runs",
            "protocol": {"path": str(protocol_path.resolve()), "sha256": sha256(protocol_path),
                         "beam_size": frozen["beam_size"], "length_penalty": frozen["length_penalty"],
                         "max_generation_tokens": frozen["max_generation_tokens"]},
            "cross_seed_controls": controls, "aggregate": aggregates, "seeds": per_seed}, memberships, examples


def report_lines(result):
    aggregate = result["aggregate"]
    selected, included = aggregate["selected_seeds"], aggregate["included_seeds"]
    lines = ["# 三种子长训练预算扩展报告", "",
             "状态：" + ("全部完成" if result["status"] == "complete" else "待完成（pending）") + "。", "",
             "本报告独立于 3,000 更新的标签平滑 A/B 核心实验。本次在已知 seed=42 长训练结果后增加 seed=43、44 重复实验，各新实验在训练前保存独立计划；新增计划不是全部历史实验的事前声明。所有长训练从各自种子初始化开始，固定标签平滑 0.1、12,000 次更新和每 500 次更新的完整 888 句验证；按验证 greedy BLEU 取最高分，平分取最早检查点。测试只用于最终报告。", "",
             f"请求种子：{selected}；完整核验后纳入：{included}；n={aggregate['n']}；待完成而排除：{aggregate['excluded_incomplete_seeds']}。仅长短两种预算、两种解码全部完整并通过校验的种子进入任何均值、标准差和配对差值汇总。缺失值为 null/待完成，不能作为零分。", "",
             "## 各种子完整测试分数与配对差值", "",
             "| seed | 解码 | 基线 BLEU | 长训练 BLEU | Δ BLEU | 基线 chrF2 | 长训练 chrF2 | Δ chrF2 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for seed in selected:
        entry = result["seeds"][seed]
        for method in METHODS:
            a, b = (entry["scores"][f"{name}_{method}"] or {} for name in ("baseline", "long"))
            values = []
            for measure in MEASURES:
                values += [number(a.get(measure), missing="待完成"), number(b.get(measure), missing="待完成"),
                           number(entry["long_minus_baseline"][method][measure], missing="待完成", signed=True)]
            lines.append(f"| {seed} | {method} | " + " | ".join(values) + " |")
    lines += ["", "BLEU 为完整 8,079 句测试集的大小写敏感 SacreBLEU、tokenizer=13a；chrF2 为 char_order=6、word_order=0、beta=2、大小写敏感。源句/参考 ID、数量、顺序和对应关系、数据与分词器指纹、文件哈希及 checkpoint 均校验；分数从保存文本独立复算。", "",
              "## 完整配对种子的均值与样本标准差", "", "| 预算 / 解码 | n | BLEU 均值 ± SD | chrF2 均值 ± SD |", "|---|---:|---:|---:|"]
    def show_stat(item):
        return number(item["mean"], missing="待完成") + " ± " + number(item["sample_std"], missing="不可估计")
    for group in GROUPS:
        item = aggregate["groups"][group]
        lines.append(f"| {group} | {aggregate['n']} | {show_stat(item['bleu'])} | {show_stat(item['chrf'])} |")
    lines += ["", "| 同种子长训练 − 基线 | n | Δ BLEU 均值 ± SD | Δ chrF2 均值 ± SD |", "|---|---:|---:|---:|"]
    for method in METHODS:
        item = aggregate["paired_long_minus_baseline"][method]
        lines.append(f"| {method} | {aggregate['n']} | {show_stat(item['bleu'])} | {show_stat(item['chrf'])} |")
    lines += ["", "SD 为跨完整种子的样本标准差（ddof=1）；n<2 时为 null，不是置信区间。配对差值先按每个种子相减再计算其均值和 SD，不能用两个 SD 相减。", "", "![完整种子的分数与同种子差值](summary.png)", "", "### 结果方向与解释范围", ""]
    for method in METHODS:
        for measure in MEASURES:
            values = aggregate["paired_long_minus_baseline"][method][measure]["per_seed"]
            up, tie, down = (sum(value > 0 for value in values.values()), sum(value == 0 for value in values.values()), sum(value < 0 for value in values.values()))
            lines.append(f"- {method} {measure}：纳入的 {len(values)} 个种子中，增加 {up}、相等 {tie}、下降 {down}。")
    lines += ["", "这些方向仅描述已纳入种子与当前数据/模型/预算。三次重复不足以直接声明统计显著或普遍优势；若未完成全部种子，应等待剩余结果后再判断本轮一致性。BLEU/chrF2 上升也不保证专名、事实、完整性和语法正确。", "",
              "## 每次训练的验证曲线与成本", "", "![各个种子的完整验证集曲线](validation.png)", "",
              "| seed | 预算 | 状态 | 更新数 / 预算 | best 更新 | best 验证 BLEU | 训练秒数 | 峰值已分配 MiB |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for seed in selected:
        for budget, item in result["seeds"][seed]["training"].items():
            summary, config, best = item["summary"] or {}, item["config"] or {}, item["best_validation"] or {}
            memory = summary.get("peak_cuda_bytes")
            lines.append(f"| {seed} | {budget} | {item['status']} | {item['last_logged_update']} / {config.get('max_updates', '待完成')} | {best.get('update', '待完成')} | {number(best.get('bleu'), missing='待完成')} | {number(summary.get('training_seconds_total'), 2, '待完成')} | {number(memory / 2**20 if memory is not None else None, 2, '未记录')} |")
    lines += ["", "星标表示各次训练最早的验证最佳点。累计训练时间包含验证与保存，不含数据准备和模型初始化；峰值显存采用 PyTorch 已分配量，不是整卡使用量。固定预算和验证选优不意味着已收敛。", "",
              "## 推理实测与定义", "", "| seed | 预算 / 解码 | 精度 | batch | 预热批数 | 秒数 | 句/秒 | 批服务 ms | 摊销每句 ms | 达上限 / 无 EOS |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for seed in selected:
        for group in GROUPS:
            item = result["seeds"][seed]["evaluation_metadata"].get(group) or {}
            values = [number(item.get(key), 2, "待完成") for key in ("elapsed_seconds", "sentences_per_second", "mean_batch_latency_ms", "mean_latency_ms")]
            lines.append(f"| {seed} | {group} | {item.get('inference_precision', '待完成')} | {item.get('inference_batch_size', '待完成')} | {item.get('warmup_batches', '未记录')} | " + " | ".join(values) + f" | {item.get('generation_limit_hits', '待完成')} / {item.get('unfinished_generations', '待完成')} |")
    lines += ["", "推理计时包含源文 SentencePiece 分词和目标文本还原；GPU 每批同步，固定句数 batch，末批可不足；预热排除。计时不含 checkpoint 加载、结果写盘和评分。吞吐=句数/总秒数；平均批服务时间=总秒数/批数；摊销每句时间=总秒数/句数，并非单请求独占 GPU 的响应延迟。达生成上限包含最后一步生成 EOS 的情况，无 EOS 另计。", "",
              "历史与本轮训练、推理跨日期进行，电源、功率、温度和频率未持续控制；运行环境原始记录保存在 metrics.json。生成长度和停止时机也影响耗时。速度比较仅描述实际运行，不能将差异全归因于预算、种子或算法。", "",
              "## 同种子控制变量、前缀及源码核验", "", "| seed | 配置 | 前缀顺序 / LR / token / 句数 | loss 不同点数 | 验证 BLEU 不同点数 | 历史训练源码差异 |", "|---|---|---|---:|---:|---|"]
    for seed in selected:
        item = result["seeds"][seed]
        fields = (item.get("training_prefix_numerical_differences") or {}).get("fields", {})
        changes = item["training_source_differences"]
        lines.append(f"| {seed} | {item['config_controls']} | {item['training_prefix_controls']} | {fields.get('loss', {}).get('different_points', '待完成')} | {fields.get('validation_bleu_greedy', {}).get('different_points', '待完成')} | {', '.join(changes) if changes else ('无已记录差异' if changes is not None else '待完成')} |")
    lines += ["", "同种子长短预算配置除 run_dir/max_updates 外须一致，初始化与数据指纹须相同；不同种子初始化须不同。共同前 3,000 更新的数据顺序、学习率、token 和句数逐步精确核验。loss、验证数值的全部差异保存在每种子的 metrics.json，不能把控制一致误写为浮点轨迹一致。仅 seed=42 保留其历史 train.py/decoding.py 差异例外；seed=43、44 基线与长训练的七核心源码必须相同，三个长训练的七核心源码也必须共同一致。全部完成的长训练均核验完整快照，包括已有 seed=42 长训练；历史快照缺失例外仅适用于 A42 基线。正式评测解码源码要求共同一致。训练日志一致也不是最终权重逐字节一致的证明。", "",
              f"共同协议：beam={result['protocol']['beam_size']}、length_penalty={result['protocol']['length_penalty']}、生成上限={result['protocol']['max_generation_tokens']}；协议 SHA-256：`{result['protocol']['sha256']}`。完整 BLEU signature 与 chrF2 定义、源码哈希和原文件路径见 metrics.json。", "",
              "## 固定样例与长度诊断", "", "每个种子固定保存原测试 ID % 400 == 0 的全部 21 条样例，见 [examples.jsonl](examples.jsonl)，含源句、参考、长短预算两种解码。报告中的 [固定阅读窗口](examples.md) 固定使用其中 ID=0、6800、7600，不按新增种子的结果调整；窗口仅辅助阅读，不代表整体质量。应结合全部固定样例检查改善、漏译、重复、专名和长句语序，并区分原文已有重复。", "",
              "各单种子报告保留源 BPE 长度 ≤20、21–40、41–80、>80 的补充描述性分桶；不含 BOS/EOS，不截断、不按译文质量筛选。分桶不是原始预声明指标，桶内 corpus BLEU 不能按句数加权还原全语料分数。", ""]
    for seed in selected:
        lines.append(f"- [seed={seed} 完整报告、曲线、前缀数值差异与长度分桶](seed{seed}/report.md)")
    lines += ["", "机器可读全部结果：[metrics.json](metrics.json)；完整配对统计：[aggregate.json](aggregate.json)。原 3,000 更新 A/B 报告和历史 seed=42 长训练产物保持独立。"]
    return lines


def plot_results(result, output):
    import matplotlib.pyplot as plt
    aggregate = result["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    colors = {"greedy": "#2166ac", "beam": "#b35806"}
    included = aggregate["included_seeds"]
    for index, method in enumerate(METHODS):
        for budget, offset, marker in (("baseline", -.12, "o"), ("long", .12, "s")):
            values = [result["seeds"][seed]["scores"][f"{budget}_{method}"]["bleu"] for seed in included]
            axes[0].scatter([pos + index * .30 + offset for pos in range(len(included))], values,
                            color=colors[method], marker=marker, label=f"{budget} {method}")
        diffs = list(aggregate["paired_long_minus_baseline"][method]["bleu"]["per_seed"].values())
        axes[1].plot(range(len(included)), diffs, marker="o", color=colors[method], label=method)
    for axis in axes:
        axis.set_xticks(range(len(included)), [str(seed) for seed in included])
        axis.set_xlabel("Complete paired seed")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
        if not included:
            axis.text(.5, .5, "No complete paired seed yet", ha="center", transform=axis.transAxes)
    axes[0].set(title=f"Verified test scores (n={len(included)})", ylabel="Corpus BLEU (13a)")
    axes[1].axhline(0, color="#777777", lw=.8)
    axes[1].set(title="Within-seed long minus baseline", ylabel="BLEU difference")
    fig.savefig(output / "summary.png", dpi=180)
    fig.savefig(output / "summary.pdf")
    plt.close(fig)
    selected = aggregate["selected_seeds"]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.4 * len(selected)), squeeze=False, layout="constrained")
    for axis, seed in zip(axes[:, 0], selected):
        for budget, color in (("baseline", "#2166ac"), ("long", "#b35806")):
            training = result["seeds"][seed]["training"][budget]
            points = training["validation"]
            if points:
                axis.plot([row["update"] for row in points], [row["bleu"] for row in points],
                          marker="o", markersize=3, color=color, label=budget)
                best = training["best_validation"]
                axis.scatter([best["update"]], [best["bleu"]], marker="*", s=140, color=color, edgecolor="white", zorder=4)
        axis.set(title=f"Seed {seed}: full validation greedy BLEU", xlabel="Optimizer updates", ylabel="Corpus BLEU (13a)", ylim=(0, None))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.savefig(output / "validation.png", dpi=180)
    fig.savefig(output / "validation.pdf")
    plt.close(fig)


def generate_report(runs="artifacts/runs", seeds=(42, 43, 44), protocol=DEFAULT_PROTOCOL,
                    out="artifacts/quality_multiseed_report"):
    result, memberships, examples = collect_multiseed(runs, seeds, protocol)
    output = Path(out)
    run_root = Path(runs).resolve()
    require(output.resolve() != run_root and run_root not in output.resolve().parents,
            "report output must be outside training directories")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    write_json(output / "metrics.json", result)
    write_json(output / "aggregate.json", result["aggregate"])
    rows, reading = [], ["# 固定样例阅读窗口", "", "固定 ID=0、6800、7600，来自全部 ID % 400 == 0 样例；不按新增种子的结果调整，窗口仅辅助阅读，不代表整体质量。", ""]
    for seed in result["aggregate"]["selected_seeds"]:
        save_report(result["seeds"][seed], memberships[seed], examples[seed], output / f"seed{seed}")
        rows += [{"seed": seed, **row} for row in examples[seed]]
        for row in examples[seed]:
            if row["id"] not in READING_IDS:
                continue
            reading += [f"## seed={seed} / ID={row['id']}", "", "英文：" + row["source"], "", "参考：" + row["reference"], ""]
            for group in GROUPS:
                reading += [f"- {group}：{row['predictions'][group] if row['predictions'][group] is not None else '待完成'}"]
            reading.append("")
    (output / "examples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "examples.md").write_text("\n".join(reading) + "\n", encoding="utf-8")
    plot_results(result, output)
    lines = report_lines(result)
    if (output / "examples_review.md").is_file():
        lines += ["", "[固定样例逐句复核](examples_review.md)：包含改善与失败分析，并注明复核日期及适用运行；其适用范围以该文件所列运行和检查点为准。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", default="artifacts/quality_multiseed_report")
    args = parser.parse_args()
    print(generate_report(args.runs, args.seeds, args.protocol, args.out))


if __name__ == "__main__":
    main()
