# IWSLT-17 英德机器翻译

[模型下载](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) · [English](README.md) · [实验结果](docs/RESULTS.md) · [CV 与适配研究](docs/CV_ADAPTATION.md) · [复现说明](docs/REPRODUCING.md) · [第三方来源](docs/THIRD_PARTY.md)

[![CPU regression tests](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml)

基于 PyTorch 的英→德机器翻译项目，重点是控制变量、结果稳定性与错误分析。项目从零训练小型 Transformer，使用三个随机种子研究训练预算和子词表示，完成分组交叉验证调参，并比较同一 OPUS-MT 底座的直接推理、LoRA 与全参数微调。

## 模型下载与试译

**[下载模型 ZIP（约 43.5 MB）](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.0.0/iwslt-en-de-seed42-12000-inference-v1.0.0.zip) · [发布说明](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) · [SHA-256 校验文件](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.0.0/iwslt-en-de-seed42-12000-inference-v1.0.0.zip.sha256)**

先克隆或下载本仓库源码，进入仓库根目录；使用 Python 3.13 在 Windows PowerShell 中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

将模型 ZIP 放到仓库根目录，然后解压并试译：

```powershell
Expand-Archive -LiteralPath .\iwslt-en-de-seed42-12000-inference-v1.0.0.zip -DestinationPath .\models
python -m mt.translate --bundle .\models\quality_A42_12000 --device cpu --method beam --text "Thank you very much."
```

已验证输出：`Vielen Dank.` 模型包包含权重、原分词器和解码设置，试译无需下载训练数据或重新训练。`--bundle` 指向解压后的模型目录，不是 ZIP 文件。

下载包是 **seed=42、12,000 次更新**的单模型，测试 Beam BLEU 为 **22.889**；下文 **23.137 ± 0.321** 是三个种子的汇总，不是这个单模型或集成模型的成绩。漏译、实体和长句错误仍存在，详见[结果说明](docs/RESULTS.md)。

## 主要结果

以下均在完整 **8,079 句 IWSLT-17 英德测试集**上评测；训练结果为 seed=42、43、44 的均值±样本标准差，直接推理为单个固定模型。

| 实验 | Beam BLEU | chrF2 |
|---|---:|---:|
| 历史 BPE，3,000 更新 | 17.146 ± 0.191 | 42.887 ± 0.299 |
| 历史 BPE，12,000 更新 | **23.137 ± 0.321** | 49.778 ± 0.394 |
| 共同批次 BPE8k，3,000 更新 | 17.222 ± 0.087 | 42.932 ± 0.208 |
| 共同批次 unigram8k，3,000 更新 | 17.036 ± 0.218 | 43.182 ± 0.329 |
| OPUS-MT 原生直接推理 | 32.503 | 59.371 |
| OPUS-MT＋LoRA | **33.450 ± 0.052** | 59.994 ± 0.023 |
| OPUS-MT＋全参数微调 | **34.032 ± 0.094** | 60.421 ± 0.042 |

增加训练预算的同种子配对提升为 **5.992 ± 0.232 BLEU**。表示对照使用独立的共同句对和批次计划，其 BPE 不能与历史基线混合统计。OPUS-MT 直接推理行使用外部预训练的大模型及其原生分词器和解码，项目适配更新为零；与从零训练的数据和计算预算不同。下文两种适配均从该固定底座出发。

### 分组 CV 与同底座适配

三折 CV 将同演讲和重复规范化源句的连通分量整体分折，各折仅用训练部分拟合 BPE。按平均 Greedy BLEU 选择标签平滑 **0.1**，但仅领先 **0.021**，一折方向相反，chrF2 偏好 0。随后三次新的 3,000 步重训精确复现历史对应模型；CV 补充了选参与划分敏感性证据，没有新增翻译质量收益。

LoRA 训练 **294,912 / 74,705,408 个参数（0.395%）**，较底座提高 **0.948 Beam BLEU**；全参训练 **73,886,208 / 74,410,496 个参数（99.295%）**，提高 **1.530**。实测训练峰值分配显存分别约 **2.53 / 3.52 GiB**。两方法完成共同的 3,000 步有效批次轨迹，但验证选择的 LoRA 最佳步为 1,000–2,000，全参为 3,000，因此实际评测模型的训练前缀暴露不同。

![同底座适配质量](results/figures/adaptation_quality.png)

2026-10-28 研究完成 **19 次训练、57,000 次净更新及 102 遍完整评测**；资源探测和中断回放额外计算另行记录。详见[研究设计与局限](docs/CV_ADAPTATION.md)、[便携复现说明](docs/REPRODUCING_CV_ADAPTATION.md)及[机器可读结果](results/cv_adaptation.json)。这次源码更新和现有 v1.0.0 模型 ZIP 均不包含新增适配权重。

## 实现与实验

- 编码器和解码器各 3 层、隐藏维度 256、4 个注意力头，共 **11,681,600 个参数**。
- 训练集拟合 SentencePiece、动态批处理、混合精度、验证集选检查点与严格断点恢复。
- Greedy／Beam Search、评测复算与结果复用、命令行翻译和推理包导出。
- 标签平滑、训练预算和 BPE／unigram 对照，以及固定预训练模型直接推理。
- 演讲／重复源句连通分组 CV、折内拟合分词器、固定终点选参与共同预算重训。
- 原生 Marian LoRA／全参适配、共同有效批次、验证选学习率和检查点、严格恢复。LoRA 由本项目实现，无需安装 PEFT。
- 句长、词汇稀有度和分词碎片化分组诊断；句向量降维和聚类用于探索，不解释为已验证的语义主题。

CV／适配研究前，本地已完成 **15 次从零训练、194 项回归检查**。扩展研究的本地验收记录为 **349 项检查通过、1,393 个历史文件指纹不变**。以上是研究工作区的实际历史记录；本公开副本的验证另见 `RELEASE_CHECKS.json`，GitHub Actions 徽章仅代表线上已运行的提交。

## 快速开始

当前项目要求 **Python 3.13**，已验证环境为 Windows。以下在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m mt.data --out artifacts/data
python -m mt.tokenizer --data-dir artifacts/data --vocab-size 8000
python -m mt.checks --data-dir artifacts/data --config smoke.json
python -m mt.train --config smoke.json --smoke
```

以上会下载数据并运行小规模 CPU 诊断，不代表执行全部实验。扩展回归集合还会导入预训练依赖，完整检查前安装：

```powershell
python -m pip install -r requirements-pretrained.txt -c requirements.txt
python -m unittest discover -s tests -v
```

先安装 `requirements.txt`，再使用约束安装附加依赖，保留核心版本。已有运行目录受到防覆盖检查保护。

CUDA、完整实验、恢复、评测和翻译入口见 [复现说明](docs/REPRODUCING.md)。部分高级编排使用 Windows 专用文件锁；默认正式训练使用 BF16，换设备或精度需要单独的配置与实验记录。

新的 CV／适配输入准备与底层执行命令见[专项复现说明](docs/REPRODUCING_CV_ADAPTATION.md)，准备入口为 `scripts/prepare_cv_adaptation.py`。`mt.study` 与 `mt.study_complete` 用于原研究工作区的编排和验收，不能直接当作新机器的一键复现入口，也不要求新机器拥有原工作区的 1,393 个历史文件。

## 发布范围

本仓库包含代码、配置、便携配方、精选图表、[历史聚合结果](results/summary.json)和[CV／适配结果](results/cv_adaptation.json)。**源码仓库不包含数据集、模型权重、分词器二进制、完整预测、本地依赖环境及大型运行产物。** 已训练的 seed42 推理包在 [Releases](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) 单独提供。复现实验需要自行下载资源并运行；精选汇总不能代替原始预测逐条审计。

三种子衡量固定划分上的变化，另行执行的三折 CV 衡量 3,000 步预算下的划分敏感性。n=3 不代表统计显著。CV／官方验证参与选择，其分数用于诊断；公开测试集已在历史多轮研究中观察。OPUS 预训练与 IWSLT／TED 重叠无法排除，历史直接推理的后端工作区设置记录不完整。总体指标提高不代表逐句正确，漏译、否定反转、实体及长句错误仍存在。时间比较没有完全控制电源、温度、频率和执行顺序；LoRA 可训练参数少也不保证推理更快。

数据、预训练模型及依赖遵循各自上游条款，见[第三方来源](docs/THIRD_PARTY.md)。此副本尚未选定项目级开源许可证，公开可见不等于已经授予任意使用和再分发权限。
