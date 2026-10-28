# 将本次更新上传到已有 GitHub 仓库

本文件描述本地公开副本的上传步骤。准备本地文件不代表已提交或推送。

## 上传前

1. 查看根目录 `RELEASE_CHECKS.json` 的本地测试结果和 `PUBLIC_UPDATE.json` 的文件清单。
2. 使用整理后的公开副本作为上传来源。保留现有仓库和 `v1.0.0` Release。
3. 源码仓库的本次更新包含 CV、适配训练、CPU 测试、复现配方和精选统计，不包含原始数据、训练权重或完整运行环境。

## 通过网页上传

1. 打开 [现有仓库](https://github.com/lermitto1221/iwslt-en-de-translation)。
2. 选择 **Add file → Upload files**。
3. 按 `PUBLIC_UPDATE.json` 上传新增和修改的文件，保持相对于仓库根目录的原目录结构。上传的是目录中的内容，不额外套一层项目目录。
4. 以 `Add grouped CV and same-base adaptation research` 作为提交说明。
5. 隐藏目录 `.github` 中的工作流也有更新；确认 `tests.yml` 已同步。
6. 提交后查看 [Actions](https://github.com/lermitto1221/iwslt-en-de-translation/actions)，确认新的 CPU 工作流通过。线上实际结果以对应提交的 Actions 为准。

网页每次最多上传 100 个文件，单文件上限 25 MiB。可按目录分批上传，或使用 GitHub Desktop 克隆现有仓库、复制整理后的文件、检查差异后提交与推送。[GitHub 文件上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## 本次模型状态

原有 `v1.0.0` 下载包仍是 seed42、12,000 次更新的小型 Transformer，单模型测试 Beam BLEU 为 **22.889**。

本次报告中的 LoRA **33.450 ± 0.052**、全参 **34.032 ± 0.094** 是本地三种子研究结果。此次源码整理没有发布新的适配模型包。现有 `mt.translate --bundle` 入口针对原小型 Transformer；新的适配权重需要另行制作可移植的导出和推理入口。

完成新的模型包后，在独立的新 Release 中提供权重、校验文件、固定底座版本及所选单模型的实际分数。LoRA 的 adapter 文件仍需要原底座。三种子均值与下载包单模型成绩应分别说明。
