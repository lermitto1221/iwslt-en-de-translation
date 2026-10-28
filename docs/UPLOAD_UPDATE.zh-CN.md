# 将本次更新上传到已有 GitHub 仓库

本文件描述 v1.1.0 模型 Release 发布后的仓库文档更新步骤。**先发布两个模型 ZIP 和各自的 `.sha256`，确认可以下载，再上传这组文档。** 这些文档已准备好新链接，本地准备文件不代表已经提交、推送或发布 Release。

## 上传前

1. 查看根目录 `RELEASE_CHECKS.json` 的本地测试结果和 `PUBLIC_UPDATE.json` 的文件清单。
2. 使用整理后的文档副本作为本次上传来源，确认 v1.1.0 四个附件已发布；保留现有仓库和 `v1.0.0` Release。
3. 源码仓库包含 CV、适配训练、CPU 测试、复现配方和精选统计，不包含原始数据、训练权重或完整运行环境。本次仓库更新主要补充模型下载与使用说明，模型 ZIP 通过 Release 单独提供。

## 通过网页上传

1. 打开 [现有仓库](https://github.com/lermitto1221/iwslt-en-de-translation)。
2. 选择 **Add file → Upload files**。
3. 按 `PUBLIC_UPDATE.json` 上传新增和修改的文件，保持相对于仓库根目录的原目录结构。上传的是目录中的内容，不额外套一层项目目录。
4. 以 `Document v1.1.0 standalone adaptation model bundles` 作为提交说明。
5. 这组发布后文档不修改 `.github` 工作流；现有源码与路径兼容修复保留。
6. 提交后查看 [Actions](https://github.com/lermitto1221/iwslt-en-de-translation/actions)，确认新的 CPU 工作流通过。线上实际结果以对应提交的 Actions 为准。

网页每次最多上传 100 个文件，单文件上限 25 MiB。可按目录分批上传，或使用 GitHub Desktop 克隆现有仓库、复制整理后的文件、检查差异后提交与推送。[GitHub 文件上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## v1.1.0 模型附件

在 [v1.1.0 Release](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0) 的独立二进制附件／Assets 区域上传以下四个文件，不要上传到仓库源码或正文附件区域：

- `iwslt-en-de-full-seed42-inference-v1.1.0.zip`；
- `iwslt-en-de-full-seed42-inference-v1.1.0.zip.sha256`；
- `iwslt-en-de-lora-seed42-inference-v1.1.0.zip`；
- `iwslt-en-de-lora-seed42-inference-v1.1.0.zip.sha256`。

全参 ZIP **277,564,485 字节（277.6 MB，十进制）**，解压目录 `full_seed42`，包含完整权重，单模型测试 Beam BLEU **33.924**。LoRA ZIP **2,339,896 字节（2.34 MB）**，解压目录 `lora_seed42`，单模型成绩 **33.402**；首次使用需下载约 **298 MB** 的固定原始底座，或指定本地原始底座目录。LoRA 为项目自定义格式，不是 PEFT adapter。

两个包均自带 `translate.py`、`inference.py`、`requirements.txt`、原生分词器与配置，无需原研究目录或源码仓库。使用包内 README 的 CPU 安装和试译命令；全参包安装依赖后可离线运行，LoRA 需先准备底座。

报告中的全参 **34.032 ± 0.094**、LoRA **33.450 ± 0.052** 为三个种子的均值±样本标准差，必须与下载包的 seed42 单模型分数分别说明。导出使用已完成训练的检查点，不重新训练，也不宣称重新运行了完整测试集。

原有 [v1.0.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) 保留 seed42、12,000 步小型 Transformer，单模型 **22.889 Beam BLEU**。原 `mt.translate --bundle` 对应这个旧包；新适配包使用其独立 `translate.py` 入口。
