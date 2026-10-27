"""Build the exploratory course report exclusively from saved analysis evidence."""
import argparse
from pathlib import Path

from .common import read_json, write_json
from .explore import check_hashes


def n(value, digits=3):
    return "不可估计" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "不可估计" if value is None else f"{100 * value:.2f}%"


def generate_report(out):
    out = Path(out)
    for stage in ("data", "vectors", "quality"):
        check_hashes(read_json(out / "stages" / f"{stage}.json")["files_sha256"])
    data = read_json(out / "data/summary.json")
    vectors = read_json(out / "vectors/summary.json")
    quality = read_json(out / "quality/summary.json")
    fit = read_json(out / "vectors/fit_meta.json")
    lexical = read_json(out / "quality/lexical_profiles.json")
    plan = read_json(out / "plan.json")
    cfg = plan["config"]
    dists = data["distributions"]
    lines = ["# 数据、句向量与验证集质量探索", "",
        "本轮日期：2026-10-27（Asia/Shanghai）。状态：数据统计、句向量、PCA/KMeans、完整验证质量分析均完成。", "",
        "本轮使用已有九次训练的成果，没有重新训练模型。主分析模型预先固定为 seed=42、12,000 更新的验证最佳检查点；对照为同 seed 的 3,000 更新最佳检查点。原有三种子测试结果仅作历史背景，没有用于本轮选模型、改 K 或改分桶。", "",
        f"探索方案在 {plan['created_at_utc']} 保存，早于本轮主要计算，但晚于历史训练及其结果，属于新增探索性分析。见 [冻结计划](plan.json)、[配置](../../experiments/exploration_2026_10_27.json)、[分析源码指纹](analysis_sources.json)。验证集曾用于选择检查点和共同解码参数，因此本报告的验证分数不是独立留出集上的无偏效果确认。", "",
        "## 问题一：数据的长度、稀有词与分词碎片化有什么特点？", "",
        "### 方法", "",
        "完整清洗训练集和完整验证集参与统计；不抽样计算词频、长度、清洗或过滤数量。原始 Parquet 与 manifest 核对后，按 empty → duplicate_pair → long_words → length_ratio 顺序复算清洗，各删除数为互斥首个命中规则。最终每行原始 ID、文本与顺序均和保存 JSONL 对齐。测试正文及其分布没有参与探索。", "",
        "词定义为 Python str.split() 后 casefold()；标点仍附着在词上或独立成词，不另外去标点。英德各建一份词频表，只用完整清洗训练集（在模型 BPE 长度过滤前）。低频为训练出现次数 ≤5；验证未见词为次数 0，同时属于低频。以下词汇比例按 token 出现次数加权；类型比例单列。标点与词形会增加类型数，不能将这些数字当作语言学词元数或 BPE 的 UNK 率。", "",
        "BPE 长度取整句常规 SentencePiece 编码，排除 BOS/EOS/PAD、保留 UNK；碎片化=整句 BPE 数/空格分隔词数。不是每个词单独重编码后的均值。句级特征统计对每句等权；因此句级稀有度均值与 token 加权词汇比例不一定相等。源目标长度比为德文词数/英文词数。分布 SD 为总体 ddof=0。", "",
        "### 结果", "", "| 划分 | 原始句对 | 空文本 | 重复句对 | 过长词数 | 长度比异常 | 清洗后 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, audit in data["cleaning_audit"].items():
        r = audit.get("exclusive_removals", {})
        lines.append(f"| {split} | {audit.get('rows_raw', '原始未核验')} | {r.get('empty', '未核验')} | {r.get('duplicate_pair', '未核验')} | {r.get('long_words', '未核验')} | {r.get('length_ratio', '未核验')} | {audit['rows_saved']} |")
    lines += ["", "| 特征 | train 均值 / 中位数 / P95 | validation 均值 / 中位数 / P95 |", "|---|---:|---:|"]
    for field in ("src_words", "tgt_words", "src_bpe", "tgt_bpe", "length_ratio", "src_fragmentation", "tgt_fragmentation"):
        vals = []
        for split in ("train", "validation"):
            stats = dists[split][field]
            vals.append(" / ".join(n(x, 2) for x in (stats["mean"], stats["quantiles"]["p50"], stats["quantiles"]["p95"])))
        lines.append(f"| {field} | " + " | ".join(vals) + " |")
    lines += ["", "![完整长度分布](figures/data_lengths.png)", "",
              "| 语言 | train token 数 | train 类型数 | 低频类型比例 | 低频 token 比例 | validation 未见 token 比例 | validation 低频 token 比例 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lang, v in data["vocabulary"].items():
        lines.append(f"| {lang} | {v['training_token_count']:,} | {v['training_type_count']:,} | {pct(v['rare_training_type_fraction'])} | {pct(v['rare_training_token_fraction'])} | {pct(v['validation_unseen_token_fraction'])} | {pct(v['validation_rare_token_fraction'])} |")
    filt = data["training_length_filter"]
    lines += ["", "![词频、稀有度与碎片化](figures/data_vocabulary.png)", "",
        f"模型实际训练还要求源和目标的 BPE 长度各加 BOS/EOS 后均 ≤{cfg['max_train_tokens']}。清洗后训练集再排除 {filt['excluded']:,} 对（{pct(filt['excluded_fraction'])}），保留 {filt['retained']:,} 对。若将该规则误用于验证集会删除 {data['validation_counterfactual_length_exclusions']} 对；本轮保留全部 {data['counts']['validation']} 对，不截断。向量拟合样本也从过滤前的完整清洗训练集抽取。", "",
        "![清洗与训练长度过滤](figures/data_filtering.png)", "",
        "### 解释", "",
        f"验证集英文平均 {n(dists['validation']['src_words']['mean'],2)} 词，训练集为 {n(dists['train']['src_words']['mean'],2)} 词，说明本次划分在长度上存在分布差异。德文的稀有 token、验证未见 token 和每词 BPE 片段比例均高于英文，支持将表示对照聚焦于子词分解及稀有表达。数量很少的长度过滤仍集中影响长尾，不能只因删除比例小就认为没有偏差。", "",
        "### 局限", "",
        "这些是当前语料和词定义下的描述；不能推出所有德语文本都更难，也不能由未见空格词推出模型无法编码该词。验证分布差异可能与演讲内容有关，现有 JSONL 只有 id/en/de，未提供可直接验证的演讲或主题标签。BPE 与 unigram 哪种更好需要对应模型的受控训练对照。", "",
        "## 问题二：句向量的群体主要反映什么？", "", "### 方法", "",
        f"使用预先固定的 seed=42 长训练检查点，float32、eval、关闭梯度；取最后一层编码器输出，masked mean 排除 PAD/BOS/EOS，保留 UNK。空有效序列报错；完整输入，不静默截断。原向量 256 维。random.Random({cfg['sampling']['seed']}) 从完整清洗训练集选取 {vectors['fit_sample_size']:,} 句，再按原始 ID 排序，保存 [选择文件](vectors/selection.json)。全部 {vectors['validation_size']} 句验证数据独立处理。", "",
        "标准化均值/标准差仅由这 10,000 个训练向量拟合（ddof=0，常量维度 scale=1）；同一训练样本做 full-SVD PCA，保留 50 维、不 whitening；KMeans 在 50 维空间拟合，K=5、kmeans++、n_init=10、max_iter=300、固定 seed=20261027。停止规则是最大中心位移 ≤1e-4。仅以训练 inertia 选择重启；未扫 K。验证集只变换并归到最近中心。全部参数见 [fit.npz](vectors/fit.npz) 与 [拟合元数据](vectors/fit_meta.json)。", "",
        "### 结果", "",
        f"PC1 / PC2 分别解释 {pct(vectors['pca']['pc1'])} / {pct(vectors['pca']['pc2'])} 的训练标准化方差，前两维合计 {pct(vectors['pca']['first_two'])}，前 50 维合计 {pct(vectors['pca']['total_retained'])}。二维图只是显示，不是聚类计算空间。", "",
        "![训练拟合与验证投影](figures/pca_clusters.png)", "",
        f"训练 PCA50 空间诊断：固定随机 {vectors['diagnostics']['silhouette']['sample_size']} 句子集、仅子集内部欧氏距离的 silhouette={n(vectors['diagnostics']['silhouette']['score'])}；全 10,000 句 CH={n(vectors['diagnostics']['calinski_harabasz'],2)}、DB={n(vectors['diagnostics']['davies_bouldin'])}。它们描述本次空间中的紧凑/分离程度，不是语义质量或译文质量分数，也不用于跨不同表示空间直接排优劣。", "",
        f"所选 KMeans 重启 {fit['selected_restart']['restart']}，迭代 {fit['selected_restart']['iterations']} 次，收敛标记={fit['selected_restart']['converged']}；所有重启记录保留。", "",
        "| 簇 | train n | validation n | validation 平均英文词数 | 平均英文 BPE | 稀有词比例（句均值） | 每词 BPE（句均值） |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in vectors["clusters"]:
        f = c["validation"]["features"]
        values = [(f[key] or {}).get("mean") for key in ("src_words", "src_bpe", "src_rare_fraction", "src_fragmentation")]
        lines.append(f"| C{c['cluster']} | {c['train']['n']} | {c['validation']['n']} | {n(values[0],2)} | {n(values[1],2)} | {pct(values[2])} | {n(values[3],2)} |")
    lines += ["", "![簇与简单数据特征](figures/cluster_features.png)", "",
        "| 特征 | train η² | validation η² | train corr(PC1,特征) | validation corr(PC1,特征) |", "|---|---:|---:|---:|---:|"]
    for field in ("src_words", "src_bpe", "src_rare_fraction", "src_fragmentation"):
        a, b = (vectors["feature_associations"][s] for s in ("train", "validation"))
        lines.append(f"| {field} | {n(a['eta_squared'][field])} | {n(b['eta_squared'][field])} | {n(a['pearson_pc1'][field])} | {n(b['pearson_pc1'][field])} |")
    lines += ["", "η²=簇间特征平方和/总体特征平方和，只描述簇标签区分该特征的程度；不是预测准确率或因果效应。训练未见词比例恒为零，相关无法估计，机器结果记 null。", "",
              "训练抽样中的词汇线索如下：每句只计一次，固定停用词排除、仅 alphabetic token，簇内文档支持数至少 10；按加一平滑的簇内文档出现率/总体出现率排序取 8 项。同一长句拥有更多词汇机会，这也是线索的混杂来源。", "", "| 簇 | 训练词汇富集线索（不当作主题名） |", "|---|---|"]
    for c in lexical["clusters"]:
        lines.append(f"| C{c['cluster']} | " + "; ".join(f"{t['term']} ({t['cluster_documents']}句, {t['prevalence_ratio']:.2f}倍)" for t in c["terms"]) + " |")
    lines += ["", "### 解释与局限", "",
        "这次聚类首先体现强烈的长度结构：源词数/BPE 长度的 η² 明显高于稀有度和碎片化，PC1 也与长度强相关。平均池化不会保证消除编码器上下文和位置中的长度信息。词汇富集只提供阅读线索；没有主题真值、对长度的干预或专门消混杂对照，不能将这些簇命名为已经验证的主题，不能说长度是唯一成因。簇编号任意，二维分离不代表语义正确。", "",
        "## 问题三：哪些验证群体的质量不同，长训练在哪里改善？", "", "### 方法", "",
        "同 seed=42 的短/长训练最佳检查点，均用共同已冻结协议 beam=4、length_penalty=0.6、max_tokens=256；同时补充 greedy。短训练两组完整历史验证输出严格核验后复制到独立目录，保留原计时；长训练两组在本轮新生成。两种预算都使用同一 888 句源文和参考。逐条 ID、源文、参考、归簇与预测核验，再从保存文本重新计算大小写敏感 tokenizer=13a 的 corpus SacreBLEU 和 chrF2（char_order=6、word_order=0、beta=2），完整 signature 保存在 JSON。", "",
        "分组包括预先固定的五簇、英文 BPE 长度（≤20、21–40、41–80、>80）、稀有比例（0、(0,.05]、(.05,.15]、>.15）和碎片化（≤1.25、(1.25,1.5]、(1.5,2]、>2）。空组分数 null，n<30 标为小组。各组分别计算 corpus 指标，不平均 sentence BLEU，不加权拼回总体 BLEU。", "", "### 结果", "",
        "| 解码 | n | 短 BLEU | 长 BLEU | Δ BLEU | 短 chrF2 | 长 chrF2 | Δ chrF2 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    overall = quality["overall"]
    for method in cfg["methods"]:
        a, b = (overall["scores"][f"{budget}_{method}"] for budget in ("baseline", "long"))
        diff = overall["long_minus_baseline"][method]
        lines.append(f"| {method} | {overall['n']} | {n(a['bleu'])} | {n(b['bleu'])} | {n(diff['bleu'])} | {n(a['chrf'])} | {n(b['chrf'])} | {n(diff['chrf'])} |")
    for category in ("cluster", *cfg["quality_bins"]):
        lines += ["", f"#### {category}：Beam 主对照", "", "| 群体 | n | 标记 | 短 BLEU | 长 BLEU | Δ BLEU | 短 chrF2 | 长 chrF2 | Δ chrF2 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for group in (g for g in quality["groups"] if g["category"] == category):
            a, b = (group["scores"][f"{budget}_beam"] for budget in ("baseline", "long"))
            diff = group["long_minus_baseline"]["beam"]
            lines.append(f"| {group['group']} | {group['n']} | {'空' if group['empty'] else '小组' if group['small'] else '完整'} | {n(a['bleu'])} | {n(b['bleu'])} | {n(diff['bleu'])} | {n(a['chrf'])} | {n(b['chrf'])} | {n(diff['chrf'])} |")
    lines += ["", "全部群体的两种解码、ID 成员表及句长/稀有度均值和中位数见 [group_scores.json](quality/group_scores.json)。", "",
              "![簇与验证翻译质量](figures/quality_clusters.png)", "", "![固定特征分组与验证质量](figures/quality_features.png)", "",
              "### 解释与局限", ""]
    cluster_groups = [g for g in quality["groups"] if g["category"] == "cluster" and g["n"]]
    for method in cfg["methods"]:
        deltas = [g["long_minus_baseline"][method]["bleu"] for g in cluster_groups]
        low = min(cluster_groups, key=lambda g: g["long_minus_baseline"][method]["bleu"])
        high = max(cluster_groups, key=lambda g: g["long_minus_baseline"][method]["bleu"])
        lines += [f"- {method}：{sum(x > 0 for x in deltas)}/{len(deltas)} 个非空簇 BLEU 提升；最大增益 C{high['group']}（+{n(max(deltas))}），最小 C{low['group']}（{n(min(deltas))}）。这只说明当前固定分组下的观测差异。"]
    for feature, label in (("src_bpe", "英文 BPE 长度"), ("src_rare_fraction", "英文稀有词比例"),
                           ("src_fragmentation", "英文分词碎片化")):
        rows = [g for g in quality["groups"] if g["category"] == feature and g["n"]]
        first, last = rows[0], rows[-1]
        lines += [f"- {label}：最低区间 {first['group']}（n={first['n']}）的长训练 Beam BLEU 为 {n(first['scores']['long_beam']['bleu'])}，最高区间 {last['group']}（n={last['n']}）为 {n(last['scores']['long_beam']['bleu'])}。后者的长短训练差值为 BLEU +{n(last['long_minus_baseline']['beam']['bleu'])}、chrF2 +{n(last['long_minus_baseline']['beam']['chrf'])}。"]
    lines += ["", "本次最高 BPE 长度区间只有 27 句，标为小组；即便增益较大，也不足以认定长训练对所有长句更有效。最高碎片化组只有 45 句，长训练 Beam BLEU 仍较低、BLEU 增益也较小，支持把后续表示对照聚焦于该问题，但不能据此预言 unigram 会胜出。C0 与 C3 的平均词长相近，稀有度、碎片化和质量却不同，说明长度不足以概括全部差异。", ""]
    lines += ["", "句长、词汇稀有度、碎片化、主题和参考表达可能同时变化，分组之间不构成随机对照。少量或很短句的 corpus BLEU 对 n-gram 匹配和参考措辞敏感，不能把高低直接当作该群体的难度真值，也不能将差值差异视为显著的交互效应。验证集曾用于选 checkpoint/协议；当前分组还来自长模型的表示，它提供统一群体划分，并不证明短模型也形成同样的簇。", "",
        "每簇按原始 ID 最小的 3 句固定选例，共 15 句，四路实际译文见 [cluster_examples.jsonl](quality/cluster_examples.jsonl)。该规则可复现但可能集中在验证集前部演讲，不保证主题代表性。改善、失败及不确定解释见 [代理样例复核](sample_review.md)；没有完成独立人工标注，不估计总体错误率。", "",
        "### 推理计时与资源", "", "| 预算/解码 | 来源 | 精度/batch | 秒 | 句/秒 | 摊销每句 ms | 批服务 ms | 达上限/无EOS |", "|---|---|---|---:|---:|---:|---:|---|"]
    for name, source in quality["sources"].items():
        m = source["metrics"]
        lines.append(f"| {name} | {'历史复用' if source['historical_timing_reused'] else '本轮生成'} | {m['inference_precision']}/{m['inference_batch_size']} | {n(m['elapsed_seconds'],2)} | {n(m['sentences_per_second'],2)} | {n(m['mean_latency_ms'],2)} | {n(m['mean_batch_latency_ms'],2)} | {m['generation_limit_hits']}/{m['unfinished_generations']} |")
    lines += ["", "计时含源端分词和目标文本还原，排除两批预热、checkpoint 加载、评分和写盘；GPU 每批同步。每句延迟为批处理总时长/句数的摊销值，不是独占单请求延迟。历史与本轮跨日期，功率和温度未完整控制，不把速度差异归因于训练预算。GPU 编码和两组新解码顺序执行；本轮训练更新数为 0。", "",
        "## 问题四：这些观察如何形成下一轮对照？", "",
        "当前证据支持优先研究子词表示：德文稀有词与碎片化更高，而验证簇大部分与长度有关。下一步 BPE8k/unigram8k 计划将同时控制训练句对、词表和参数量，记录表示导致的长度与 token 暴露差异。它需要分别训练对应模型，不能套用旧权重。本轮不启动新训练。独立预训练 MT 直接推理可以提供外部参照，但规模、预训练数据与分词器不同，不能把差异解释为 LoRA 的效果。", "",
        "完整预算和官方模型来源见 [后续实验计划](future_experiments.md)；已完成证据、缺口与待确认课程解释见 [课程评分项对应表](course_alignment.md)。三个随机种子不是交叉验证。是否要求 MT 做 k-fold、什么算新方法/新表示仍需向老师确认；该待确认事项未阻塞本轮探索。", "",
        "## 复现、检查与证据索引", "",
        "```powershell", ".\\project-python.ps1 -m mt.explore run", ".\\project-python.ps1 -m mt.explore_report", ".\\project-python.ps1 -m mt.explore_verify", ".\\project-python.ps1 -m mt.explore verify-history", ".\\project-python.ps1 -m unittest discover -s tests -v", "```", "",
        "已有计划时直接 run：逐阶段核验哈希后复用，部分阶段按保存的向量/拟合元数据或评测完成标记恢复。计划、配置或源码变动会拒绝混用，不能覆盖旧计划来解释已有结果。新工作目录使用 plan 命令先冻结；当前目录再次 plan 会拒绝覆盖。异常中断若留下没有配套元数据的 NPZ，会明确报错，先核查保留该不完整文件再恢复，不能把它冒充完成结果。", "",
        "- [数据统计与定义](data/summary.json)、[清洗复算](data/cleaning_audit.json)、[EN词频](data/word_frequency_en.jsonl)、[DE词频](data/word_frequency_de.jsonl)。",
        "- [向量元数据](vectors/embeddings_meta.json)、[嵌入与ID](vectors/embeddings.npz)、[拟合参数](vectors/fit.npz)、[归簇表](vectors/assignments.jsonl)、[聚类诊断](vectors/summary.json)。",
        "- [验证对应关系及译文](quality/predictions_joined.jsonl)、[质量统计](quality/summary.json)、[来源与计时](quality/evaluation_sources.json)。",
        "- [全量保存结果复核](verification.json)、[历史文件核验](historical_verification.json)、[最终回归日志](regressions_final.log)、[完成记录](completion.json)。",
        "- 所有图在 figures 下同时保存 PNG 与 PDF；[完整机器结果](analysis_metrics.json)。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "analysis_metrics.json", {"plan_created_at_utc": plan["created_at_utc"],
               "data": data, "vectors": vectors, "quality": quality})
    return out / "report.md"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/exploration_2026_10_27")
    args = p.parse_args()
    print(generate_report(args.out))


if __name__ == "__main__":
    main()
