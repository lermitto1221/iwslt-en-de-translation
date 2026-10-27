# IWSLT-17 英德机器翻译

[模型下载](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) · [English](README.md) · [实验结果](docs/RESULTS.md) · [复现说明](docs/REPRODUCING.md) · [第三方来源](docs/THIRD_PARTY.md)

[![CPU regression tests](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml)

基于 PyTorch 的英→德机器翻译项目，重点是控制变量、结果稳定性与错误分析。项目从零训练小型 Transformer，使用三个随机种子研究训练预算和子词表示，并将固定的预训练 OPUS-MT 直接推理作为独立方法参照。

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

以下均在完整 **8,079 句 IWSLT-17 英德测试集**上评测；从零训练结果为 seed=42、43、44 的均值±样本标准差。

| 实验 | Beam BLEU | chrF2 |
|---|---:|---:|
| 历史 BPE，3,000 更新 | 17.146 ± 0.191 | 42.887 ± 0.299 |
| 历史 BPE，12,000 更新 | **23.137 ± 0.321** | 49.778 ± 0.394 |
| 共同批次 BPE8k，3,000 更新 | 17.222 ± 0.087 | 42.932 ± 0.208 |
| 共同批次 unigram8k，3,000 更新 | 17.036 ± 0.218 | 43.182 ± 0.329 |
| OPUS-MT 原生直接推理 | 32.503 | 59.371 |

增加训练预算的同种子配对提升为 **5.992 ± 0.232 BLEU**。表示对照使用独立的共同句对和批次计划，其 BPE 不能与历史基线混合统计。OPUS-MT 包含外部预训练，规模、分词器和解码不同；它不是本项目自行训练的模型，也不是同训练预算比较。

![训练预算对照](results/figures/training_budget.png)

## 实现与实验

- 编码器和解码器各 3 层、隐藏维度 256、4 个注意力头，共 **11,681,600 个参数**。
- 训练集拟合 SentencePiece、动态批处理、混合精度、验证集选检查点与严格断点恢复。
- Greedy／Beam Search、评测复算与结果复用、命令行翻译和推理包导出。
- 标签平滑、训练预算和 BPE／unigram 对照，以及固定预训练模型直接推理。
- 句长、词汇稀有度和分词碎片化分组诊断；句向量降维和聚类用于探索，不解释为已验证的语义主题。

原本地实验累计完成 **15 次从零训练**。历史最终记录为 **194 项回归检查通过**，覆盖数据与分词完整性、恢复一致性、解码及指标核验、计划冻结与结果复用。本发布副本的检查另行记录，不能将历史数字当作对任意环境的通过保证。

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

以上会下载数据并运行小规模 CPU 诊断，不代表执行全部实验。回归检查使用 `python -m unittest discover -s tests -v`。已有运行目录受到防覆盖检查保护。

CUDA、完整实验、恢复、评测和翻译入口见 [复现说明](docs/REPRODUCING.md)。部分高级编排使用 Windows 专用文件锁；默认正式训练使用 BF16，换设备或精度需要单独的配置与实验记录。

## 发布范围

本仓库包含代码、配置、便携配方、精选图表和[聚合结果 JSON](results/summary.json)。**源码仓库不包含数据集、模型权重、分词器二进制、完整预测、本地依赖环境及大型运行产物。** 已训练的 seed42 推理包在 [Releases](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) 单独提供。 复现实验需要自行下载资源并运行；精选汇总不能代替原始预测逐条审计。

三种子重复不是交叉验证；LoRA 和 k-fold CV 尚未执行。验证诊断使用了参与模型选择的数据。总体指标提高不代表逐句正确，漏译、否定反转、实体及长句错误仍存在。跨日期耗时没有完全控制电源、温度和频率，不能单独作为速度优势的因果证据。

数据、预训练模型及依赖遵循各自上游条款，见[第三方来源](docs/THIRD_PARTY.md)。此副本尚未选定项目级开源许可证，公开可见不等于已经授予任意使用和再分发权限。
