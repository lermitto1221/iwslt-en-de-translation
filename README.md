# IWSLT-17 English–German Translation

[Download trained models — v1.1.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0) · [简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [CV and adaptation study](docs/CV_ADAPTATION.md) · [Reproduction](docs/REPRODUCING.md) · [Third-party sources](docs/THIRD_PARTY.md)

[![CPU regression tests](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml)

A PyTorch machine translation project focused on controlled experiments and error analysis. It trains a compact Transformer from scratch, compares training budgets and subword representations across three seeds, performs grouped cross-validation for parameter selection, and compares direct inference, LoRA and full-parameter fine-tuning on the same OPUS-MT base.

## Download and try the adapted models

The [v1.1.0 Release](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0) provides two independent inference bundles. Each includes `translate.py`, `inference.py`, pinned requirements, the native tokenizer/configuration and usage instructions. **No cloned repository, training data or retraining is needed.** Sizes below are decimal MB.

| Download | ZIP size | Single seed42 test Beam BLEU | Runtime requirement |
|---|---:|---:|---|
| [Full fine-tuning ZIP](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.1.0/iwslt-en-de-full-seed42-inference-v1.1.0.zip) · [SHA-256](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.1.0/iwslt-en-de-full-seed42-inference-v1.1.0.zip.sha256) | **277.6 MB** | **33.924** | Complete weights included; offline translation after dependency installation |
| [LoRA ZIP](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.1.0/iwslt-en-de-lora-seed42-inference-v1.1.0.zip) · [SHA-256](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.1.0/iwslt-en-de-lora-seed42-inference-v1.1.0.zip.sha256) | **2.34 MB** | **33.402** | First use downloads the pinned original base weight, about 298 MB |

For the main full-model demo, download the full ZIP and compare its SHA-256 with the accompanying checksum file. From the download directory, use Python 3.13 and Windows PowerShell:

```powershell
Expand-Archive -LiteralPath .\iwslt-en-de-full-seed42-inference-v1.1.0.zip -DestinationPath .\models
Set-Location .\models\full_seed42
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe translate.py --device cpu --method beam --offline --text "Thank you very much."
```

The LoRA ZIP extracts to `lora_seed42`; use its bundled README and the same installation steps, omitting `--offline` on the first run so the fixed base can be downloaded. An existing original base can be supplied with `--base-model-dir PATH --offline`. Its ID, revision and weight SHA-256 are checked. This is the project's own LoRA format, **not a PEFT adapter**; the 2.34 MB ZIP is not the total deployment size.

The downloadable seed42 checkpoints score **33.924 / 33.402** for full/LoRA. The **34.032 ± 0.094 / 33.450 ± 0.052** values below are three-seed means ± sample SD, not these individual checkpoints or ensembles. Recorded test scores use the original FP32/GPU protocol; CPU export checks do not promise bitwise equality across hardware. Inputs over 512 native source tokens are rejected without truncation, and translation errors remain.

### Older scratch-model demo

The separate [v1.0.0 Release](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) still provides the 43.5 MB scratch seed42/12,000-update bundle, scoring **22.889 Beam BLEU**. Its **23.137 ± 0.321** research score is the three-seed mean. With this source repository's dependencies installed and that old ZIP extracted under `models`, its original command remains:

```powershell
python -m mt.translate --bundle .\models\quality_A42_12000 --device cpu --method beam --text "Thank you very much."
```

The new v1.1.0 adapted bundles use their own `translate.py` entry point. See [Results](docs/RESULTS.md) for scoring scope and limitations.

## Results at a glance

All rows below use the same complete **8,079-sentence IWSLT-17 English→German test split**. Trained results are mean ± sample standard deviation across seeds 42, 43 and 44. Direct inference is one fixed model.

| Experiment | Beam BLEU | chrF2 |
|---|---:|---:|
| Historical BPE, 3,000 updates | 17.146 ± 0.191 | 42.887 ± 0.299 |
| Historical BPE, 12,000 updates | **23.137 ± 0.321** | 49.778 ± 0.394 |
| Controlled BPE8k, 3,000 updates | 17.222 ± 0.087 | 42.932 ± 0.208 |
| Controlled unigram8k, 3,000 updates | 17.036 ± 0.218 | 43.182 ± 0.329 |
| OPUS-MT, native direct inference | 32.503 | 59.371 |
| OPUS-MT + LoRA | **33.450 ± 0.052** | 59.994 ± 0.023 |
| OPUS-MT + full-parameter fine-tuning | **34.032 ± 0.094** | 60.421 ± 0.042 |

Increasing the scratch budget gave a paired **+5.992 ± 0.232 BLEU**. The representation comparison used a separate shared-sentence, shared-batch design; its BPE runs are distinct from the historical baseline. The direct-inference OPUS-MT row uses an externally pretrained, larger model with its native tokenizer/decoder and no project adaptation. Its comparison with scratch training has different data and computation budgets; the separate adaptations below start from that same fixed base.

### Grouped CV and same-base adaptation

Three-fold CV keeps talks and duplicate normalized source sentences within the same fold and fits BPE separately on each training fold. It selects label smoothing **0.1**, but its mean greedy BLEU lead is only **0.021**, one fold prefers 0, and chrF2 prefers 0. Three fresh 3,000-update retrainings reproduce the corresponding historical models; CV adds selection evidence without a new quality gain.

LoRA trains **294,912 / 74,705,408 parameters (0.395%)** and achieves **33.450 ± 0.052 Beam BLEU**, compared with **34.032 ± 0.094** for full fine-tuning (three-seed mean ± sample SD). The within-seed LoRA-minus-full difference is **−0.582 ± 0.064 BLEU**. In this setup, mean training peak allocation is **2.535 vs 3.521 GiB**, about **28% lower** for LoRA. Both complete matched 3,000-update trajectories, but validation selects LoRA at steps 1,000–2,000 and full models at step 3,000; the evaluated models therefore have different prefix exposures.

![Same-base adaptation quality](results/figures/adaptation_quality.png)

The 2026-10-28 study completed **19 trainings, 57,000 net updates and 102 complete evaluation passes**. Resource probes and replay after an interruption are accounted for separately. See [study design and limitations](docs/CV_ADAPTATION.md), [portable reproduction](docs/REPRODUCING_CV_ADAPTATION.md) and [selected machine-readable results](results/cv_adaptation.json). Adaptation weights are provided separately in [v1.1.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0); they are excluded from this source repository and the old v1.0.0 scratch-model ZIP.

## What is implemented

- A 3-layer encoder/3-layer decoder Transformer: 256-dimensional hidden states, 4 attention heads and **11,681,600 parameters**.
- Training-only SentencePiece fitting, token-budget batches, mixed precision, validation-based checkpoint selection and strict checkpoint recovery.
- Greedy/beam decoding, verified evaluation reuse, command-line translation and portable inference-bundle export.
- Paired label-smoothing and budget experiments, BPE/unigram experiments with shared batch plans, and pretrained direct inference.
- Talk/source-connected grouped CV, fold-local tokenizer fitting, fixed-endpoint CV selection and matched-budget final retraining.
- Native Marian LoRA and full fine-tuning, shared effective-batch plans, validation-based learning-rate/checkpoint selection and strict recovery. LoRA is implemented in this project and does not require PEFT.
- Length, lexical-rarity and subword-fragmentation diagnostics; exploratory sentence-vector analysis with explicit limitations.

The public source package passed **359 CPU regression checks: 349 research checks + 10 portable-recipe checks**. The standalone inference bundles separately passed **19 focused runtime checks**, which are not included in the 359. The explicit records are in [RELEASE_CHECKS.json](RELEASE_CHECKS.json), under `regression_tests` and `post_release_documentation_update.bundle_runtime_checks`.

The original expanded study's **349-test** acceptance record and **1,393 unchanged historical file fingerprints** are retained as historical research evidence. Check counts describe the named suites, not the number of times tests were rerun. The GitHub Actions badge describes the commits actually evaluated online.

## Quick start

Python **3.13** is required by the current package metadata. These commands use Windows PowerShell, the tested platform:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m mt.data --out artifacts/data
python -m mt.tokenizer --data-dir artifacts/data --vocab-size 8000
python -m mt.checks --data-dir artifacts/data --config smoke.json
python -m mt.train --config smoke.json --smoke
```

This downloads the dataset and runs a small CPU diagnostic, not the full experiment. The expanded regression suite also imports the optional pretrained stack. Install it before running all tests:

```powershell
python -m pip install -r requirements-pretrained.txt -c requirements.txt
python -m unittest discover -s tests -v
```

The constraints preserve pinned core dependencies; install `requirements.txt` first. Existing completed run directories are protected against accidental overwriting.

For CUDA setup, full experiments, evaluation, recovery and translation, follow [REPRODUCING.md](docs/REPRODUCING.md). Some advanced runners use a Windows-specific execution lock and currently require Windows. GPU defaults use BF16; other precision/device settings require a separate configuration and experiment.

For fresh CV/adaptation preparation and the lower-level commands, use [REPRODUCING_CV_ADAPTATION.md](docs/REPRODUCING_CV_ADAPTATION.md). `mt.study` and `mt.study_complete` orchestrate and accept the original research workspace; they are not a one-command fresh-machine recipe and do not require another machine to possess its 1,393 historical files.

## Repository contents

| Path | Purpose |
|---|---|
| `mt/` | Data, model, training, inference, experiment and analysis modules |
| `tests/` | Regression checks |
| `experiments/` | Baseline, budget-extension and analysis configurations |
| `recipes/` | Portable historical settings for a fresh execution |
| `scripts/prepare_recipes.py` | Initialize machine-local recipe inputs without replacing existing evidence |
| `scripts/prepare_cv_adaptation.py` | Prepare inputs for a fresh grouped-CV and same-base adaptation execution |
| [results/summary.json](results/summary.json) | Selected historical aggregate results |
| [results/cv_adaptation.json](results/cv_adaptation.json) | Curated CV, adaptation and resource results from the completed study |
| [docs/RESULTS.md](docs/RESULTS.md) | Findings, experimental controls and interpretation limits |

This repository is a source-and-results release. **Datasets, trained weights, tokenizer binaries, full predictions, local environments and large execution artifacts are not stored in Git.** Adapted seed42 bundles are provided separately in [v1.1.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0); the original scratch bundle remains in [v1.0.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0). Reproducing scores requires downloading the external resources and running the experiments. The selected JSON and figures document prior runs; they cannot independently reconstruct or re-audit every original prediction.

## Limitations and attribution

Three seeds measure variation on a fixed split; the separate three-fold CV measures split sensitivity at 3,000 updates. Three seeds do not establish statistical significance. CV/official validation scores used for selection are diagnostic, and the public test has been observed across successive studies. OPUS pretraining overlap with IWSLT/TED cannot be excluded. Historical direct-inference backend workspace settings were not fully recorded. BLEU and chrF2 do not establish semantic correctness; omissions, entity errors, negation reversals and long-sentence failures remain. Timing is not fully controlled for power, temperature, clocks or execution order; lower LoRA parameter count does not establish faster inference.

Data, pretrained models and libraries retain their upstream terms; see [THIRD_PARTY.md](docs/THIRD_PARTY.md). A project-wide code license has not yet been selected. See [PUBLISHING.md](docs/PUBLISHING.md) for upload instructions.
