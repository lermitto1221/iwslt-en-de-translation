"""Combine independently verified representation and pretrained course evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sacrebleu

from .common import read_json, sha256, write_json
from .data import read_pairs
from .pretrained_mt import evaluate, verify_plan
from .representation import validate_plan

DEFAULT_OUT="artifacts/course_report_2026_10_27"
REP_ROOT=Path("artifacts/representation_2026_10_27")
PRE_ROOT=Path("artifacts/pretrained_mt_2026_10_27")


def number(x):return "待完成" if x is None else f"{x:.3f}"


def statistic(item):
    return number(item["mean"])+" ± "+number(item["sample_std"])


def verified_representation_cache(root):
    from .representation_report import collect
    root=Path(root)
    verified=collect(root)
    for key in ("aggregate","groups","metrics"):
        saved=read_json(root/"report"/(key+".json"))
        expected=verified[key]
        if key=="metrics":
            saved={k:v for k,v in saved.items() if k!="generated_at_utc"}
            expected={k:v for k,v in expected.items() if k!="generated_at_utc"}
        if saved!=expected:
            raise ValueError("saved representation report is stale or damaged: "+key)
    return verified


def native_group_scores(group_file, examples):
    members=read_json(group_file)
    if [r["id"] for r in examples]!=[r["id"] for r in members["rows"]]:
        raise ValueError("pretrained examples and common membership order differ")
    for row,member in zip(examples,members["rows"]):
        for text,key in ((row["source"],"source_sha256"),(row["reference"],"reference_sha256")):
            if hashlib.sha256(text.encode()).hexdigest()!=member[key]:
                raise ValueError("pretrained group source/reference identity differs")
    output={}
    for category in members["rows"][0]["groups"]:
        output[category]=[]
        size=5 if category=="cluster" else len(members["definitions"][category+"_boundaries"])+1
        for label in range(size):
            selected=[r for r,m in zip(examples,members["rows"]) if m["groups"][category]==label]
            bleu=sacrebleu.metrics.BLEU(tokenize="13a",lowercase=False)
            chrf=sacrebleu.metrics.CHRF(char_order=6,word_order=0,beta=2)
            predictions=[r["prediction"] for r in selected]; references=[r["reference"] for r in selected]
            bleu_score=bleu.corpus_score(predictions,[references]).score if selected else None
            chrf_score=chrf.corpus_score(predictions,[references]).score if selected else None
            output[category].append({"group":label,"n":len(selected),"ids":[r["id"] for r in selected],
                "small":0<len(selected)<30,"empty":not selected,
                "bleu":bleu_score,"chrf":chrf_score,
                "bleu_signature":str(bleu.get_signature()) if selected else None,
                "chrf_signature":str(chrf.get_signature()) if selected else None})
    return output


def generate(out=DEFAULT_OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    validate_plan(REP_ROOT)
    verified=verified_representation_cache(REP_ROOT)
    rep,aggregate,groups=(verified[k] for k in ("metrics","aggregate","groups"))
    if aggregate["n"]!=3 or aggregate["included_seeds"]!=[42,43,44]:
        raise ValueError("formal report requires all three complete paired seeds")
    config,pre_plan=verify_plan(PRE_ROOT)
    native={}; native_groups={};examples={}
    for split in ("validation","test"):
        native[split]={};native_groups[split]={};examples[split]={}
        for method in ("greedy","beam"):
            directory=PRE_ROOT/split/method
            if not (directory/"metrics.json").is_file():
                raise ValueError("pretrained inference incomplete; report never triggers generation")
            native[split][method]=evaluate(PRE_ROOT,split,method,config,device="cuda")
            examples[split][method]=read_pairs(directory/"examples.jsonl")
            native_groups[split][method]=native_group_scores(REP_ROOT/"groups"/(split+".json"),examples[split][method])
    # The same ID membership, not separately selected populations, must be compared.
    for split,methods in native_groups.items():
        for method,features in methods.items():
            for feature,items in features.items():
                expected={str(g["group"]):g for g in groups[split][feature]}
                for item in items:
                    if item["ids"]!=expected[str(item["group"])]["ids"]:
                        raise ValueError("representation/pretrained group membership differs")
    historical=read_json("artifacts/quality_multiseed_report/aggregate.json")
    exploration=read_json("artifacts/exploration_2026_10_27/analysis_metrics.json")
    model=read_json(PRE_ROOT/"model/config.json")
    metrics={"status":"complete","representations":aggregate,"representation_groups":groups,
             "pretrained":{"model_id":config["model_id"],"revision":config["revision"],"parameters":config["parameters"],
                           "config":config,"metrics":native,"groups":native_groups},
             "historical_long_training":historical,
             "exploration_findings":{"pca":exploration["vectors"]["pca"],"diagnostics":exploration["vectors"]["diagnostics"],
                                    "validation_feature_associations":exploration["vectors"]["feature_associations"]["validation"]},
             "input_report_hashes":{str(p):sha256(p) for p in (REP_ROOT/"report/metrics.json",REP_ROOT/"report/aggregate.json",REP_ROOT/"report/groups.json",PRE_ROOT/"plan.json")}}
    write_json(out/"metrics.json",metrics)
    write_json(out/"pretrained_groups.json",native_groups)
    fixed=[]
    for row_g,row_b in zip(examples["test"]["greedy"],examples["test"]["beam"]):
        if row_g["id"]%400==0:
            fixed.append({"id":row_g["id"],"source":row_g["source"],"reference":row_g["reference"],
                          "pretrained_greedy":row_g["prediction"],"pretrained_beam":row_b["prediction"]})
    (out/"pretrained_examples.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in fixed),encoding="utf-8")
    lines=["# 英德机器翻译课程项目：数据、表示与方法比较", "", "## 摘要", "",
        "本项目使用 IWSLT-17 英德数据，研究标签平滑、训练预算、子词表示与独立预训练翻译方法。已有核心六次3,000更新和长训练三次12,000更新；本轮另完成 BPE8k/unigram8k × 三个种子的六次新3,000更新训练，以及一个固定预训练模型的直接推理。所有正式训练均从各自新随机初始化开始；预训练基线没有本项目训练或适配更新。", "",
        "本报告重点是控制变量、数据特点与结果解释。表示比较使用共同训练句对和逐更新微批计划，组内质量使用相同原始句子集合。本文不把少量种子差值直接解释为统计显著，不将预训练模型的效果差异归因于某一单独机制。", "",
        "## 1. 数据与分析问题", "",
        "官方原始训练206,112对，清洗排除1,695个重复句对和132个长度比异常，保留204,285对；验证888对、测试8,079对完整保留。使用已固定发布版本及逐行ID/哈希证据。词频仅由训练集建立；空格分词后casefold且保留标点，因此未见词率不能等同于BPE UNK率。", "",
        "已有探索显示：英文/德文验证未见空格token比例为2.14%/4.07%；德文训练低频token比例8.17%，英文4.35%。固定seed42编码器的PCA50解释85.71%方差，二维仅20.05%；K=5的训练silhouette=0.0462，验证簇与源BPE长度η²=0.729，提示强长度关联和显著重叠。聚类提供描述性群体，未验证为主题。", "",
        "研究问题因此是：改变子词划分后，长度、碎片化和群体翻译表现是否改变；增加独立预训练方法后，质量与成本的差异如何解释。数据/PCA的完整方法、PNG/PDF图与局限见 [探索报告](../exploration_2026_10_27/report.md)。", "",
        "## 2. 实验设计与控制", "",
        "| 实验 | 实际运行 | 主要控制 | 解释范围 |", "|---|---|---|---|",
        "| 历史标签平滑 | 2条件×3seed，各3000更新 | seed/初始化/数据/预算配对 | 平滑在该训练设置下的差异 |",
        "| 历史长训练 | 3seed，各12000更新 | 同seed短训练基线，保留历史源码差异 | 预算扩展的重复结果 |",
        "| 新表示对照 | BPE/unigram×3seed，各3000更新 | 共同句对、共同微批边界、同数值初始化、LR/更新数 | 固定暴露计划下的表示方案差异 |",
        "| 预训练直接推理 | 固定OPUS-MT一次，无适配 | 相同评测句、参考及计分 | 独立完整方法比较，非纯架构/LoRA效应 |", "",
        "两个新分词器仅用同一完整清洗训练语料拟合；实际共享词表均为8000，PAD/UNK/BOS/EOS为0/1/2/3。共同训练集合为两种表示两侧含BOS/EOS都≤128的交集，共204,115对；新BPE不能由旧BPE模型代替。每句取两表示较大源长、较大目标长，构造每微批≤4096 padded token的共同计划，累积4微批。每seed两表示的所有原ID顺序和边界完全一致；实际非PAD token数和算力不相等。", "",
        "小型Transformer两侧各3层、256维、4头、FFN1024，总参数11,681,600。AdamW、学习率/warmup、标签平滑0.1、bfloat16训练、每500更新完整验证保持一致。按验证Greedy BLEU取最早最高分checkpoint。使用旧验证冻结协议：Greedy及Beam4、长度惩罚0.6、生成上限256、float32推理batch16。不同子词下256 token表示的文本容量不同，触限数及原始token成本随结果报告。", "",
        "该共同解码协议来自历史BPE模型的验证选参，本轮未按表示分别重新调参。相同token级长度惩罚和上限也不保证相同文本级约束；结论限定为共享旧协议下的对照，不是两种表示各自最优解码性能的比较。", "",
        "计划、配置、完整执行源码和共同分组均在新训练/测试前冻结。分组使用同一个原始ID表：空格句长、训练频率定义的稀有度、历史锚定BPE碎片化，验证另沿用旧聚类；不按新tokenizer各自分桶。组内直接计算corpus BLEU/chrF2，不平均sentence BLEU，不加权拼回总BLEU。", "",
        "执行顺序为BPE42→unigram42→BPE43→unigram43→BPE44→unigram44，没有随机化运行时段。曾观测GPU温度86℃及每步耗时明显波动，见 [运行时GPU快照](../representation_2026_10_27/runtime_gpu_observation.csv)。电源、温度和前后台负载未完全受控，耗时/吞吐是本次真实运行记录，不能单独证明表示的速度优势或解释为纯算法效应。", "",
        "## 3. 表示对照结果", "", "| 划分/解码 | BPE BLEU | unigram BLEU | 配对Δ BLEU | BPE chrF2 | unigram chrF2 | 配对Δ chrF2 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation","test"):
        for method in ("greedy","beam"):
            condition=aggregate["conditions"][split][method]; diff=aggregate["paired_unigram_minus_bpe"][split][method]
            lines.append(f"| {split}/{method} | {statistic(condition['bpe']['bleu'])} | {statistic(condition['unigram']['bleu'])} | {statistic(diff['bleu'])} | {statistic(condition['bpe']['chrf'])} | {statistic(condition['unigram']['chrf'])} | {statistic(diff['chrf'])} |")
    lines += ["", "均值±样本标准差，n=3、ddof=1；Δ为同seed unigram减BPE后再汇总，不能相减两个SD。验证已用于检查点选择，不能当作独立无偏确认。", ""]
    for method in ("greedy","beam"):
        delta=aggregate["paired_unigram_minus_bpe"]["test"][method]["bleu"]
        values=list(delta["per_seed"].values())
        lines.append(f"- 测试{method}：3个种子中unigram BLEU增加{sum(x>0 for x in values)}个、下降{sum(x<0 for x in values)}个、持平{sum(x==0 for x in values)}个。其均值差为{number(delta['mean'])}。这是当前共同暴露预算下的观测，不能推为普遍优劣。")
    lines += ["", "与此同时，测试chrF2的配对均值差为Greedy +0.238、Beam +0.250；seed42为负，seed43/44为正。BLEU关注词级n-gram的精确参考重合，chrF2关注字符n-gram且更重视召回。两项指标在本轮给出不同方向，不能只用BLEU宣布BPE全面占优，也不能由chrF2上升直接断言形态泛化或语义忠实度已经改善。", "",
        "每seed成绩、配对SD、全部分组、实际token/句数、训练时间、显存、验证曲线和PNG/PDF见 [表示对照报告](../representation_2026_10_27/report/report.md)。旧BPE历史结果放在独立章节，不混入本次配对统计。", "",
        "## 4. 独立预训练MT方法", "",
        f"Helsinki-NLP/opus-mt-en-de，固定revision `{config['revision']}`，Marian编码/解码各{model['encoder_layers']}/{model['decoder_layers']}层、{model['d_model']}维，原生词表{model['vocab_size']:,}，实测参数 **{config['parameters']:,}**。保留原生源/目标SentencePiece，未使用课程模型词表。模型和依赖文件均保存哈希，新增依赖在独立目录，原Torch/CUDA环境未升级。", "",
        "原生生成设定显式冻结：Beam4/Greedy、max_length512（包含decoder起始token）、renormalize_logits=true、原生EOS/PAD及完成规则；float32、batch16。全部输入经原生分词器审计均未超位置容量，没有截断或分句。它与自写解码并不完全同义，因而比较的是各自固定推理流程。", "",
        "| 划分 | 解码 | 句数 | BLEU | chrF2 | 秒 | 句/秒 | 摊销每句ms | 批服务ms | 峰值已分配MiB | 触限数 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in native:
        for method,m in native[split].items():
            lines.append(f"| {split} | {method} | {m['samples']} | {number(m['bleu'])} | {number(m['chrf2'])} | {m['elapsed_seconds']:.2f} | {m['sentences_per_second']:.2f} | {m['mean_latency_ms']:.2f} | {m['mean_batch_latency_ms']:.2f} | {m['peak_cuda_bytes']/2**20:.2f} | {m['generation_limit_hits']} |")
    lines += ["", "### 完整测试 Beam 方法比较", "", "| 方法 | 训练/适配说明 | 重复数 | BLEU | chrF2 | 参数量 |", "|---|---|---:|---:|---:|---:|"]
    for representation in ("bpe","unigram"):
        current=aggregate["conditions"]["test"]["beam"][representation]
        lines.append(f"| 新{representation}8k | 每seed新训练3000更新，共同批次 | 3 | {statistic(current['bleu'])} | {statistic(current['chrf'])} | 11,681,600 |")
    old=historical["groups"]["long_beam"]
    lines.append(f"| 历史BPE长训练（独立参照） | 每seed12000更新，非本次共同批次 | 3 | {statistic(old['bleu'])} | {statistic(old['chrf'])} | 11,681,600 |")
    lines.append(f"| OPUS-MT原生直接推理 | 项目适配0更新，含既有外部预训练 | 1固定模型 | {number(native['test']['beam']['bleu'])} | {number(native['test']['beam']['chrf2'])} | {config['parameters']:,} |")
    lines += ["", "计时含原生分词、生成、目标还原和GPU传输，每批同步；排除模型加载、两批预热、评分与写盘。摊销每句延迟不是独占单请求时延。批次日志保存中断恢复前的原计时；与旧模型跨日期的电源/温度/频率未完全控制，生成长度及实现差异也影响耗时。", "",
        "补充生成上限审计：预训练测试Beam有1句触及上限（原ID=3213），出现Oh后持续h的退化输出。原生forced_eos在含起始token的decoder长度512处强制结束，因此unfinished=0不等于没有触限或没有失败。该句是按触限标记定位的诊断案例，不替换ID%400==0的固定21句，也未用于更改生成协议。", "",
        "模型卡将预训练数据标为OPUS，未提供这一checkpoint可核查的完整子语料版本和逐句清单，因此不能排除IWSLT/TED或评测句的预训练重叠。质量差异同时包含规模、外部训练数据、分词器及解码实现差异，不归因于LoRA，也不能宣称同样训练预算下更优。本项目适配预算为0，不代表它的预训练成本为0。", "",
        f"来源：[固定模型卡](https://huggingface.co/Helsinki-NLP/opus-mt-en-de/blob/{config['revision']}/README.md)、[固定模型配置](https://huggingface.co/Helsinki-NLP/opus-mt-en-de/blob/{config['revision']}/config.json)、[本地下载/依赖与语义说明](../pretrained_mt_2026_10_27/preparation.md)。", "",
        "## 5. 相同句子群体的质量比较", "",
        "下表以完整测试Beam为主展示。每行所有方法的原始句ID完全一致；BPE/unigram为三个seed均值，预训练为一个固定checkpoint、禁用采样的固定流程单次推理，未虚构种子SD，也不承诺跨设备逐位一致。分组的token尺度是历史锚定表示，而不是各条件重新计算的分组。", ""]
    for category in ("source_words","source_rare_fraction","anchor_fragmentation"):
        lines += [f"### {category}", "", "| 固定组 | n | BPE BLEU均值 | unigram BLEU均值 | 预训练BLEU | BPE chrF2均值 | unigram chrF2均值 | 预训练chrF2 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for native_item,rep_item in zip(native_groups["test"]["beam"][category],groups["test"][category]):
            cond=rep_item["conditions"]["beam"]
            labels={"source_words":["≤10词","11–20词","21–40词",">40词"],
                    "source_rare_fraction":["0","(0,5%]","(5%,15%]",">15%"],
                    "anchor_fragmentation":["≤1.25","(1.25,1.5]","(1.5,2]",">2"]}
            label=labels[category][int(rep_item['group'])]
            lines.append(f"| {label} | {rep_item['n']} | {number(cond['bpe']['bleu']['mean'])} | {number(cond['unigram']['bleu']['mean'])} | {number(native_item['bleu'])} | {number(cond['bpe']['chrf']['mean'])} | {number(cond['unigram']['chrf']['mean'])} | {number(native_item['chrf'])} |")
    lines += ["", "分组解读：测试Beam的>40词组（328句）BPE/unigram为13.557/13.288；稀有比例>15%组（860句）为14.227/13.975。最高锚定碎片化>2组（441句）为14.003/13.686，配对ΔBLEU=-0.317±0.454，其中seed42为正、另两个为负。11–20词组的平均ΔBLEU却为+0.092±0.386，方向混合。由此可见总体小幅下降并不等于每个群体都下降，序列略短也没有带来一致的高碎片化群体收益。", "",
        "具体区间、所有组ID和小组标记见 [共同分组定义](../representation_2026_10_27/groups/test.json)；预训练的两划分/两解码完整分组见 [pretrained_groups.json](pretrained_groups.json)。分组差异存在长度、稀有度、主题及参考措辞混杂，不能据组均值推断因果。", "",
        "## 6. 固定样例、局限与课程证据", "",
        "测试统一按原始ID%400==0取21句；三个新训练seed各保存四路真实译文，预训练保存两路。验证沿用每簇最小3个ID。代理复核同时呈现改善与失败，见 [样例复核](sample_review.md)；它不是独立人工标注，不计算虚构的总体错误率。", "",
        "三个seed是固定官方划分上的重复，不是交叉验证。已有多个实验重复使用同一公开测试集，本次先冻结设置可防本轮结果驱动的修改，但不能把该测试集当作从未观察过的秘密留出集。PCA/KMeans为单个固定模型的探索，未验证跨模型稳定主题；表示对照每组只3000更新，不据此断言收敛或更长预算下结论不变。", "",
        "LoRA及k-fold CV仍未执行；MT是否需要字面k-fold，以及课程对新方法/新表示的认定仍待老师确认。已完成的新表示对照和独立预训练方法提供明确实验依据，但不承诺课程得分。见 [更新后的课程对应表](course_alignment.md)。", "",
        "## 7. 复现与证据", "",
        "```powershell", ".\\project-python.ps1 -m mt.representation run", ".\\project-python.ps1 -m mt.representation_report", ".\\project-pretrained-python.ps1 -m mt.pretrained_mt run", ".\\project-pretrained-python.ps1 -m mt.course_report", "```", "",
        "当前所有计划已冻结，完成结果须严格校验后只读复用，不重新prepare/freeze。训练中断仅从对应运行last.pt恢复，预训练推理按已提交批次journal继续；不改配置或源码后混用旧状态。", "",
        "- [表示设计](../representation_2026_10_27/design.json)、[执行计划](../representation_2026_10_27/plan.json)、[严格配对统计](../representation_2026_10_27/report/aggregate.json)。",
        "- [预训练计划](../pretrained_mt_2026_10_27/plan.json)、[原生输入长度审计](../pretrained_mt_2026_10_27/length_audit.json)、[推理汇总](../pretrained_mt_2026_10_27/summary.json)。",
        "- [本报告机器结果](metrics.json)、[完成记录](completion.json)、[项目README](../../README.md)。", ""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    alignment=["# 课程要求与已完成证据", "",
        "本表依据用户转述的课程关注点盘点证据，不代表教师已认可每项分类，也不承诺得分。", "",
        "| 关注点 | 已完成证据 | 仍需说明或后续工作 |", "|---|---|---|",
        "| 数据探索 | 完整清洗复算、词长/频率/OOV/碎片化、过滤影响与PNG/PDF | 空格词不是词元，OOV不是BPE UNK；测试不用于调整本轮方案 |",
        "| 降维与聚类 | 训练样本专用标准化/PCA50/KMeans5，完整验证归簇与诊断 | 强长度关联、低silhouette，不当作主题真值；未做跨模型聚类稳定性 |",
        "| 不同表示 | BPE8k/unigram8k各3个seed的新3000更新训练 | 共同句对和微批相同，token/时间不相等；结论受预算限制 |",
        "| 独立方法 | 固定版本OPUS-MT原生直接推理，完整验证/测试 | 外部预训练、规模、词表及解码不同，不能排除评测重叠 |",
        "| 实验设置 | 事前冻结本轮执行规则、同seed初始化、共同计划、共享ID分组 | 历史结果已经知道，本轮不是所有历史实验的事前声明 |",
        "| 结果解释 | 三seed配对均值/样本SD、同句分组、成本及实际样例 | 不将n=3均值差当作统计显著，不从顺序计时推断算法速度因果 |",
        "| 报告/图表/展示 | 本综合报告、独立明细、机器结果、原生可编辑演示文稿 | 样例为代理复核，未完成独立人工标注或现场演练 |",
        "| 代码/复现 | 严格计划/源/输入校验、训练及批次推理恢复、损坏拒绝检查 | 精确通过数与历史完整性以本轮completion记录为准 |",
        "| 新方法/新表示认定 | 本轮已提供新的受控表示训练和独立预训练方法证据 | 教师如何定义课程中的“新”，仍待确认 |",
        "| 交叉验证 | 已保存官方train内部按演讲/重复源连通组分折方案 | 尚未执行；三个seed不是CV；MT是否必须k-fold仍待确认 |",
        "| LoRA | 仅有后续设计 | 尚未执行；其效果须同一底座适配前后比较 |", "",
        "索引：[综合报告](report.md)、[表示对照](../representation_2026_10_27/report/report.md)、[探索](../exploration_2026_10_27/report.md)、[保存的后续设计](../exploration_2026_10_27/future_experiments.md)、[完成记录](completion.json)。", ""]
    (out/"course_alignment.md").write_text("\n".join(alignment),encoding="utf-8")
    return out/"report.md"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",default=DEFAULT_OUT)
    a=p.parse_args();print(generate(a.out))


if __name__=="__main__":main()
