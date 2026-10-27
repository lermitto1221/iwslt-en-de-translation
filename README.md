# IWSLT-17 English–German Translation

[简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [Third-party sources](docs/THIRD_PARTY.md)

A PyTorch machine translation project focused on controlled experiments and error analysis. It trains a compact Transformer from scratch, compares training budgets and subword representations across three seeds, and evaluates a fixed pretrained OPUS-MT model as an external reference.

## Results at a glance

All rows below use the same complete **8,079-sentence IWSLT-17 English→German test split**. Scratch results are mean ± sample standard deviation across seeds 42, 43 and 44.

| Experiment | Beam BLEU | chrF2 |
|---|---:|---:|
| Historical BPE, 3,000 updates | 17.146 ± 0.191 | 42.887 ± 0.299 |
| Historical BPE, 12,000 updates | **23.137 ± 0.321** | 49.778 ± 0.394 |
| Controlled BPE8k, 3,000 updates | 17.222 ± 0.087 | 42.932 ± 0.208 |
| Controlled unigram8k, 3,000 updates | 17.036 ± 0.218 | 43.182 ± 0.329 |
| OPUS-MT, native direct inference | 32.503 | 59.371 |

Increasing the scratch budget gave a paired **+5.992 ± 0.232 BLEU**. The representation comparison used a separate shared-sentence, shared-batch design; its BPE runs are distinct from the historical baseline. OPUS-MT uses external pretraining, a larger model and its native tokenizer/decoder. Its scores are **not** a same-budget comparison or a model trained by this project.

![Training-budget comparison](results/figures/training_budget.png)

## What is implemented

- A 3-layer encoder/3-layer decoder Transformer: 256-dimensional hidden states, 4 attention heads and **11,681,600 parameters**.
- Training-only SentencePiece fitting, token-budget batches, mixed precision, validation-based checkpoint selection and strict checkpoint recovery.
- Greedy/beam decoding, verified evaluation reuse, command-line translation and portable inference-bundle export.
- Paired label-smoothing and budget experiments, BPE/unigram experiments with shared batch plans, and pretrained direct inference.
- Length, lexical-rarity and subword-fragmentation diagnostics; exploratory sentence-vector analysis with explicit limitations.

The original local work completed **15 scratch training runs**. Its recorded final regression suite passed **194 tests**, covering data/tokenizer integrity, recovery consistency, decoding/metric verification, and frozen-plan/result-reuse checks. This is historical evidence; verification of this upload copy is recorded separately in its release checks.

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

This downloads the dataset and runs a small CPU diagnostic, not the full experiment. Run `python -m unittest discover -s tests -v` for the regression suite. Existing completed run directories are protected against accidental overwriting.

For CUDA setup, full experiments, evaluation, recovery and translation, follow [REPRODUCING.md](docs/REPRODUCING.md). Some advanced runners use a Windows-specific execution lock and currently require Windows. GPU defaults use BF16; other precision/device settings require a separate configuration and experiment.

## Repository contents

| Path | Purpose |
|---|---|
| `mt/` | Data, model, training, inference, experiment and analysis modules |
| `tests/` | Regression checks |
| `experiments/` | Baseline, budget-extension and analysis configurations |
| `recipes/` | Portable historical settings for a fresh execution |
| `scripts/prepare_recipes.py` | Initialize machine-local recipe inputs without replacing existing evidence |
| [results/summary.json](results/summary.json) | Selected historical aggregate results |
| [docs/RESULTS.md](docs/RESULTS.md) | Findings, experimental controls and interpretation limits |

This is a source-and-results release. **Datasets, trained weights, tokenizer binaries, full predictions, local environments and large execution artifacts are not bundled.** Reproducing scores requires downloading the external resources and running the experiments. The selected JSON and figures document prior runs; they cannot independently reconstruct or re-audit every original prediction.

## Limitations and attribution

Three seeds are repeated runs on a fixed split, **not cross-validation**. LoRA and k-fold CV were not performed. Validation diagnostics use data already involved in model selection. BLEU and chrF2 do not establish semantic correctness; omissions, entity errors, negation reversals and long-sentence failures remain. Cross-date runtime comparisons were not fully controlled for power, temperature or clocks.

Data, pretrained models and libraries retain their upstream terms; see [THIRD_PARTY.md](docs/THIRD_PARTY.md). A project-wide code license has not yet been selected. See [PUBLISHING.md](docs/PUBLISHING.md) for upload instructions.
