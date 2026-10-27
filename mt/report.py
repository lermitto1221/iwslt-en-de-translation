"""Generate per-seed results and paired across-seed statistics from saved runs."""
import argparse
import json
import os
import statistics
from pathlib import Path

from .common import read_json, write_json
from .summary import collect


GROUPS = ("A1", "A2", "B1", "B2")
COLORS = {"A1": "#2166ac", "A2": "#67a9cf", "B1": "#b35806", "B2": "#f1a340"}
MEASURES = ("bleu", "sentences_per_second", "mean_batch_latency_ms", "mean_latency_ms")
CONTRASTS = {
    "A_beam_minus_greedy": ("A2", "A1", "A：beam − greedy"),
    "B_beam_minus_greedy": ("B2", "B1", "B：beam − greedy"),
    "greedy_smoothing_effect": ("A1", "B1", "greedy：平滑 0.1 − 平滑 0"),
    "beam_smoothing_effect": ("A2", "B2", "beam：平滑 0.1 − 平滑 0"),
}


def read_log(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_statistics(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def paired_aggregates(result, seeds):
    """Compute every mean and within-seed contrast on the same complete pairs."""
    selected = sorted(set(seeds))
    paired = [seed for seed in selected
              if all(isinstance(result.get(f"{group}_seed{seed}"), dict) for group in GROUPS)]
    groups = {group: {measure: sample_statistics([result[f"{group}_seed{seed}"][measure] for seed in paired])
                      for measure in MEASURES} for group in GROUPS}
    contrasts = {}
    for key, (positive, negative, label) in CONTRASTS.items():
        differences = {seed: result[f"{positive}_seed{seed}"]["bleu"] - result[f"{negative}_seed{seed}"]["bleu"]
                       for seed in paired}
        contrasts[key] = {"label": label, "per_seed": differences,
                          **sample_statistics(list(differences.values()))}
    return {"selected_seeds": selected, "paired_seeds": paired,
            "excluded_incomplete_seeds": [seed for seed in selected if seed not in paired],
            "groups": groups, "contrasts": contrasts,
            "uncertainty": "sample standard deviation across complete paired seeds (ddof=1), not a confidence interval; null when n<2"}


def select_seeds(result, root, seed=42, all_seeds=False):
    if not all_seeds:
        if seed not in (42, 43, 44):
            raise ValueError("reports currently support experiment seeds 42, 43, and 44")
        return [seed]
    # Include unfinished runs in their own tables/curves, but not aggregate statistics.
    present = [value for value in (42, 43, 44)
               if any(isinstance(result.get(f"{group}_seed{value}"), dict) for group in GROUPS)
               or any((Path(root) / f"{condition}{value}" / name).exists()
                      for condition in ("A", "B") for name in ("summary.json", "train.jsonl"))]
    return present or [42]


def number(value, digits=3, missing="待运行", signed=False):
    return missing if value is None else format(value, f"{'+' if signed else ''}.{digits}f")


def relative_link(path, out):
    target = Path(path).resolve()
    try:
        return os.path.relpath(target, Path(out).resolve()).replace("\\", "/")
    except ValueError:  # Windows output may be on a different drive.
        return target.as_posix()


def plot_summary(result, root, seeds, aggregates, output, per_seed=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), layout="constrained")
    styles = ("-", "--", ":")
    for index, seed in enumerate(seeds):
        for condition in ("A", "B"):
            validation = [row for row in read_log(root / f"{condition}{seed}" / "train.jsonl")
                          if "validation_bleu_greedy" in row]
            if validation:
                axes[0].plot([row["update"] for row in validation],
                             [row["validation_bleu_greedy"] for row in validation],
                             marker="o", markersize=3, linestyle=styles[index % len(styles)],
                             color=COLORS[condition + "1"], label=f"{condition}, seed {seed}")
    axes[0].set(title="Validation: individual training runs", xlabel="Optimizer updates", ylabel="Corpus BLEU (13a)")
    axes[0].set_ylim(bottom=0)
    if axes[0].lines:
        axes[0].legend(fontsize=8, ncol=2)
    else:
        axes[0].text(.5, .5, "Validation pending", ha="center", va="center", transform=axes[0].transAxes)
    n = len(aggregates["paired_seeds"])
    plotted_groups = aggregates["groups"]
    if per_seed:
        # Per-seed plots may show available groups before the pair is complete.
        # Only the overview and aggregate tables require all four groups.
        plotted_groups = {group: {measure: sample_statistics(
            [result[f"{group}_seed{seeds[0]}"][measure]]
            if isinstance(result.get(f"{group}_seed{seeds[0]}"), dict) else [])
            for measure in MEASURES} for group in GROUPS}
    for axis, measure, title, ylabel, digits in (
            (axes[1], "bleu", "Full test corpus BLEU", "BLEU (13a)", 2),
            (axes[2], "sentences_per_second", "Inference throughput", "Sentences / second", 1)):
        names = [group for group in GROUPS if plotted_groups[group][measure]["n"]]
        values = [plotted_groups[group][measure]["mean"] for group in names]
        deviations = [plotted_groups[group][measure]["sample_std"] for group in names]
        if names:
            bars = axis.bar(names, values, color=[COLORS[name] for name in names],
                            yerr=deviations if n > 1 else None, capsize=5,
                            error_kw={"elinewidth": 1.3})
            tops = [value + (sd if sd is not None else 0) for value, sd in zip(values, deviations)]
            for bar, value, top in zip(bars, values, tops):
                axis.annotate(f"{value:.{digits}f}", (bar.get_x() + bar.get_width()/2, top),
                              xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
            axis.set_ylim(0, max(tops) * 1.23 if max(tops) > 0 else 1)
        else:
            axis.text(.5, .5, "No complete paired seed", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_ylim(0, 1)
        axis.set(title=title, ylabel=ylabel)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    if per_seed:
        caption = f"Seed {seeds[0]}: saved results; no across-seed error bars"
    elif n > 1:
        caption = f"{n} complete paired seeds: mean +/- sample SD (not a confidence interval)"
    elif n == 1:
        caption = "1 complete paired seed: mean only; sample SD is unavailable"
    else:
        caption = "No complete paired seeds: aggregate results are pending"
    fig.suptitle(caption, fontsize=12)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def report_lines(result, root, out, seeds, aggregates):
    paired = aggregates["paired_seeds"]
    lines = ["# 英→德受控实验报告", "", f"展示种子：{', '.join(map(str, seeds))}。所有结果来自已校验的保存文件。", "",
             "## 完整配对种子的总体结果", "",
             f"总体统计仅使用 A1/A2/B1/B2 四组全部完成的同一组种子：{', '.join(map(str, paired)) or '暂无'}（n={len(paired)}）。",
             "均值与样本标准差按种子计算，标准差使用 n−1 分母；不是置信区间。n<2 时标准差记为不可估计，JSON 中保存 null。", ""]
    if aggregates["excluded_incomplete_seeds"]:
        lines += [f"以下种子仍有未完成组，仅展示已有单次结果，不进入任何总体均值或差值统计：{', '.join(map(str, aggregates['excluded_incomplete_seeds']))}。", ""]
    lines += ["| 组别 | 配对种子数 | BLEU 均值 | BLEU 样本 SD | 句/秒均值 | 句/秒样本 SD |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        scores = aggregates["groups"][group]["bleu"]
        speeds = aggregates["groups"][group]["sentences_per_second"]
        lines.append(f"| {group} | {scores['n']} | {number(scores['mean'])} | {number(scores['sample_std'], missing='不可估计')} | {number(speeds['mean'], 2)} | {number(speeds['sample_std'], 2, '不可估计')} |")
    lines += ["", "### 同一种子内的 BLEU 差值", "",
              "先在每个种子内计算差值，再计算这些差值的均值与样本标准差；不把两组独立标准差相加。正值分别表示 beam 或标签平滑 0.1 得分更高。", "",
              "| 对照 | 配对种子数 | 差值均值 | 差值样本 SD | 各种子差值 |", "|---|---:|---:|---:|---|"]
    for item in aggregates["contrasts"].values():
        detail = "; ".join(f"{seed}: {value:+.3f}" for seed, value in item["per_seed"].items()) or "待运行"
        lines.append(f"| {item['label']} | {item['n']} | {number(item['mean'], signed=True)} | {number(item['sample_std'], missing='不可估计')} | {detail} |")
    lines += ["", "![逐种子验证曲线与配对总体统计](experiment.png)", "",
              "误差条仅在至少两个完整配对种子时绘制，表示样本标准差，不代表显著性检验或置信区间。吞吐的均值与 SD 描述已保存运行的实际差异，其中也可能包含跨日期设备运行状态变化，不能解释为单纯的种子波动。"]
    for seed in seeds:
        lines += ["", f"## 种子 {seed}", "",
                  "| 组别 | 标签平滑 | 解码 | 测试 BLEU | 句/秒 | 批延迟 ms | 摊销每句 ms | 达上限句数 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        available = {}
        for group in GROUPS:
            item = result.get(f"{group}_seed{seed}")
            if isinstance(item, dict):
                available[group] = item
                lines.append(f"| {group} | {item['config']['label_smoothing']} | {item['method']} | {item['bleu']:.3f} | {item['sentences_per_second']:.2f} | {item['mean_batch_latency_ms']:.2f} | {item['mean_latency_ms']:.2f} | {item['generation_limit_hits']}/{item['samples']} |")
            else:
                lines.append(f"| {group} | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |")
        lines += ["", f"![种子 {seed} 的验证曲线与结果](seed{seed}.png)", "",
                  "| 训练 | 状态 | 更新数 | 总参数量 | 可训练参数量 | 总运行秒数（含验证） | 峰值显存 MiB | 最佳验证 BLEU |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for condition in ("A", "B"):
            path = root / f"{condition}{seed}" / "summary.json"
            if path.exists():
                summary = read_json(path)
                memory = number(summary["peak_cuda_bytes"] / 2**20, 1) if summary.get("peak_cuda_bytes") is not None else "不适用"
                status = "完成" if summary.get("complete") else "未完成"
                lines.append(f"| {condition} | {status} | {summary['updates']} | {summary['parameters']['total']:,} | {summary['parameters']['trainable']:,} | {summary['training_seconds_total']:.1f} | {memory} | {number(summary.get('best_validation_bleu'), missing='待完成')} |")
            else:
                lines.append(f"| {condition} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
        protocol = root.parent / f"decode_protocol_seed{seed}.json"
        if protocol.exists():
            value = read_json(protocol)
            lines += ["", f"验证集冻结的解码参数：beam={value['beam_size']}，长度惩罚={value['length_penalty']}，最大生成长度={value['max_generation_tokens']}。"]
        if available:
            first = next(iter(available.values()))
            env = first["environment"]
            lines += ["", f"SacreBLEU signature：`{first['signature']}`。", "",
                      f"硬件：{env.get('gpu') or env.get('cpu', 'CPU')}；PyTorch {env.get('torch', '未记录')}；CUDA {env.get('torch_cuda')}；推理精度 {first['inference_precision']}；batch={first['inference_batch_size']}。"]
        if len(available) == 4:
            lines += ["", f"A/B 最佳验证 checkpoint 的更新数：{available['A1']['checkpoint_update']} / {available['B1']['checkpoint_update']}。"]
        sample_path = root.parent / f"fixed_examples_seed{seed}.jsonl"
        if seed == 42 and not sample_path.exists():
            sample_path = root.parent / "fixed_examples.jsonl"
        if sample_path.exists():
            sample_count = len(read_log(sample_path))
            lines += ["", f"已保存该种子的 [固定对齐样例]({relative_link(sample_path, out)})，实际 {sample_count} 条。"]
        else:
            lines += ["", "尚未保存该种子的固定样例文件；可用 mt.examples 指定 --seed 和独立 --out 生成。"]
    lines += ["", "## 解释与复现边界", "",
              "A1/A2 共用 A checkpoint，B1/B2 共用 B checkpoint；同一种子内 A/B 只改变标签平滑，均按完整 validation greedy BLEU 选择 checkpoint。只比较生成式 BLEU，不横向比较不同损失函数的训练 loss。", "",
              "固定更新预算下的结果不等于模型已充分收敛，应结合逐种子验证曲线判断。单种子不能估计跨种子波动；少量种子的样本标准差也不足以独立证明稳定优势，本文不作统计显著性声明。", "",
              "训练计时从训练循环开始，包含验证和 checkpoint 保存，不含下载、分词器拟合、数据编码及模型初始化。显存为 PyTorch 峰值已分配量，不等于整卡占用。", "",
              "推理速度包含分词与文本还原，不含 checkpoint 载入、结果写盘及 BLEU 计算；预热后计时，GPU 每批同步。批时延是整批服务时间；摊销每句时间不能当作单请求时延。", "",
              "性能比较边界：seed 42 的原始实验与后续 seed 43/44 跨日期运行。虽然 GPU、驱动、软件、推理精度、batch 和输入样本保持相同，电源模式、功耗、温度与时钟频率没有持续控制；硬件快照不足以还原各次运行全程的状态。因此训练耗时、吞吐与时延的跨日期差异不能归因于单一因素，也不能仅由种子或标签平滑解释。同日各组还受生成长度、beam 停止时机和运行波动影响。", "",
              "BLEU 衡量与参考译文的 n-gram 重合度，不能完整替代语义、事实与流畅度评价。数据为 IWSLT-17 英德发布划分，validation=dev2010，test=tst2010–tst2015；不可与不同测试集公开 BLEU 直接比较。", "",
              f"固定样例规则为原始测试 ID % 400 == 0；人工错误分析、复现命令和设计取舍见 [README]({relative_link(Path(__file__).resolve().parents[1] / 'README.md', out)}#结果讨论与固定样例)。"]
    return lines


def generate_report(root, out, seed=42, all_seeds=False):
    root, out = Path(root), Path(out)
    result = collect(root)
    seeds = select_seeds(result, root, seed, all_seeds)
    aggregates = paired_aggregates(result, seeds)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    write_json(out / "results.json", result)
    write_json(out / "aggregate.json", aggregates)
    plot_summary(result, root, seeds, aggregates, out / "experiment")
    for value in seeds:
        plot_summary(result, root, [value], paired_aggregates(result, [value]), out / f"seed{value}", per_seed=True)
    (out / "report.md").write_text("\n".join(report_lines(result, root, out, seeds, aggregates)) + "\n", encoding="utf-8")
    return out / "report.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--out", default="artifacts/report")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--seed", type=int, default=42)
    choice.add_argument("--all-seeds", action="store_true", help="show all available seeds; aggregate only complete four-way pairs")
    args = parser.parse_args()
    print(generate_report(args.runs, args.out, args.seed, args.all_seeds))


if __name__ == "__main__":
    main()
