# IWSLT-17 English–German Translation

[Download trained model](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) · [简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [CV and adaptation study](docs/CV_ADAPTATION.md) · [Reproduction](docs/REPRODUCING.md) · [Third-party sources](docs/THIRD_PARTY.md)

[![CPU regression tests](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lermitto1221/iwslt-en-de-translation/actions/workflows/tests.yml)

A PyTorch machine translation project focused on controlled experiments and error analysis. It trains a compact Transformer from scratch, compares training budgets and subword representations across three seeds, performs grouped cross-validation for parameter selection, and compares direct inference, LoRA and full-parameter fine-tuning on the same OPUS-MT base.

## Try the trained model

**[Download model ZIP (43.5 MB)](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.0.0/iwslt-en-de-seed42-12000-inference-v1.0.0.zip) · [Release notes](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) · [SHA-256](https://github.com/lermitto1221/iwslt-en-de-translation/releases/download/v1.0.0/iwslt-en-de-seed42-12000-inference-v1.0.0.zip.sha256)**

Clone or download this source repository, enter its root directory, and set up the dependencies using Python 3.13 on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Save the model ZIP in the repository root. Extract and translate:

```powershell
Expand-Archive -LiteralPath .\iwslt-en-de-seed42-12000-inference-v1.0.0.zip -DestinationPath .\models
python -m mt.translate --bundle .\models\quality_A42_12000 --device cpu --method beam --text "Thank you very much."
```

Verified output: `Vielen Dank.` The bundle includes weights, its original tokenizer and decoder settings, so this demo requires no dataset download or retraining. `--bundle` points to the extracted model directory, not the ZIP.

These weights are the single **seed42 / 12,000-update** model: beam test BLEU **22.889**. The **23.137 ± 0.321** reported below is the three-seed aggregate, not this individual checkpoint or an ensemble. Omissions and entity/long-sentence errors remain; see [Results](docs/RESULTS.md).

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

LoRA trains **294,912 / 74,705,408 parameters (0.395%)** and gains **0.948 Beam BLEU** over the fixed base. Full fine-tuning trains **73,886,208 / 74,410,496 parameters (99.295%)** and gains **1.530**. Measured training peak allocation is about **2.53 / 3.52 GiB**, respectively. Both complete matched 3,000-update trajectories, but validation selects LoRA at steps 1,000–2,000 and full models at step 3,000; the evaluated models therefore have different prefix exposures.

![Same-base adaptation quality](results/figures/adaptation_quality.png)

The 2026-10-28 study completed **19 trainings, 57,000 net updates and 102 complete evaluation passes**. Resource probes and replay after an interruption are accounted for separately. See [study design and limitations](docs/CV_ADAPTATION.md), [portable reproduction](docs/REPRODUCING_CV_ADAPTATION.md) and [selected machine-readable results](results/cv_adaptation.json). Adaptation weights are not included in this source update or in the existing v1.0.0 model ZIP.

## What is implemented

- A 3-layer encoder/3-layer decoder Transformer: 256-dimensional hidden states, 4 attention heads and **11,681,600 parameters**.
- Training-only SentencePiece fitting, token-budget batches, mixed precision, validation-based checkpoint selection and strict checkpoint recovery.
- Greedy/beam decoding, verified evaluation reuse, command-line translation and portable inference-bundle export.
- Paired label-smoothing and budget experiments, BPE/unigram experiments with shared batch plans, and pretrained direct inference.
- Talk/source-connected grouped CV, fold-local tokenizer fitting, fixed-endpoint CV selection and matched-budget final retraining.
- Native Marian LoRA and full fine-tuning, shared effective-batch plans, validation-based learning-rate/checkpoint selection and strict recovery. LoRA is implemented in this project and does not require PEFT.
- Length, lexical-rarity and subword-fragmentation diagnostics; exploratory sentence-vector analysis with explicit limitations.

Before the CV/adaptation study, local work completed **15 scratch training runs** and passed **194 regression tests**. The expanded study's local acceptance record passed **349 tests** and preserved **1,393 historical file fingerprints**. These are recorded research-workspace observations. Test results for this curated copy are recorded separately in `RELEASE_CHECKS.json`; its GitHub Actions badge only describes commits actually run online.

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

This repository is a source-and-results release. **Datasets, trained weights, tokenizer binaries, full predictions, local environments and large execution artifacts are not stored in Git.** The trained seed42 inference bundle is available separately in [Releases](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0). Reproducing scores requires downloading the external resources and running the experiments. The selected JSON and figures document prior runs; they cannot independently reconstruct or re-audit every original prediction.

## Limitations and attribution

Three seeds measure variation on a fixed split; the separate three-fold CV measures split sensitivity at 3,000 updates. Three seeds do not establish statistical significance. CV/official validation scores used for selection are diagnostic, and the public test has been observed across successive studies. OPUS pretraining overlap with IWSLT/TED cannot be excluded. Historical direct-inference backend workspace settings were not fully recorded. BLEU and chrF2 do not establish semantic correctness; omissions, entity errors, negation reversals and long-sentence failures remain. Timing is not fully controlled for power, temperature, clocks or execution order; lower LoRA parameter count does not establish faster inference.

Data, pretrained models and libraries retain their upstream terms; see [THIRD_PARTY.md](docs/THIRD_PARTY.md). A project-wide code license has not yet been selected. See [PUBLISHING.md](docs/PUBLISHING.md) for upload instructions.
