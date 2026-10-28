# Reproducing the project

Run commands from the repository root. This source release contains historical aggregate results and portable recipes, **not** the original datasets, trained checkpoints, full predictions, tokenizer binaries or frozen machine-local execution plans. A fresh run must download resources and create its own evidence with its actual timestamp and paths.

## Environment and platform

The historical environment was Windows, Python 3.13.14, PyTorch 2.7.1+cu128 and an RTX 3060 Laptop GPU with 6 GB VRAM. `pyproject.toml` currently requires Python 3.13. The basic CPU paths are separate from the CUDA experiment configuration. Advanced exploration, representation and pretrained runners use `msvcrt` execution locks and currently require Windows; they are not a ready-to-run Colab/Linux package.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

For the historically used CUDA wheel, install it explicitly on a compatible machine:

```powershell
python -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The lock file records one historical environment; it is not a portable guarantee for other operating systems or Python versions. Scratch GPU defaults use BF16. Adapting precision, architecture or device creates a different experiment and must not silently reuse a frozen configuration or incompatible optimizer/scaler state.

## 1. Data and CPU diagnostics

```powershell
python -m mt.data --out artifacts/data
python -m mt.tokenizer --data-dir artifacts/data --vocab-size 8000
python -m mt.checks --data-dir artifacts/data --config smoke.json
python -m mt.train --config smoke.json --smoke
python -m pip install -r requirements-pretrained.txt -c requirements.txt
python -m unittest discover -s tests -v
```

Data access requires a network connection and compliance with upstream terms. The loader verifies the pinned source files and writes manifests. The tokenizer is fitted only on training data. The smoke run is a CPU diagnostic and its score is not a formal result. Completed directories are protected; to rerun a smoke experiment, use a new `run_dir` in a separate configuration.

The expanded suite uses the optional pretrained dependencies for its native Marian label-shifting check; install them to run the complete suite without skips. No model download is required for these CPU tests. Results for this curated copy are recorded in `RELEASE_CHECKS.json`.

For the new grouped CV and same-base LoRA/full fine-tuning workflow, use [REPRODUCING_CV_ADAPTATION.md](REPRODUCING_CV_ADAPTATION.md). The remaining sections below document the earlier scratch, representation and direct-inference recipes.

## 2. Initialize portable recipes

```powershell
python scripts/prepare_recipes.py
```

The helper initializes the known historical decoder settings and the representation design in their expected `artifacts/` locations. It uses [the decoder recipe](../recipes/decode_protocol_seed42.json) and [the representation recipe](../recipes/representation_design.json). It retains existing matching settings and rejects conflicting execution evidence rather than overwriting a run.

This is a **fixed historical recipe**, not a claim that a new machine independently selected the same validation parameters. The compact protocol omits the original validation predictions and machine paths. New plans bind the actual local recipe hash, files and execution time.

## 3. Baseline experiments

The six runs train two smoothing conditions for each seed. Run GPU jobs sequentially:

```powershell
python -m mt.run_experiments --seed 42 --protocol artifacts/decode_protocol_seed42.json
python -m mt.run_experiments --seed 43 --protocol artifacts/decode_protocol_seed42.json
python -m mt.run_experiments --seed 44 --protocol artifacts/decode_protocol_seed42.json
python -m mt.report --all-seeds
```

Supplying the fixed protocol skips the original validation grid search. The exploration stage also needs two explicit A42 validation outputs. Generate these once after A42 completes:

```powershell
python -m mt.evaluate --checkpoint artifacts/runs/A42/best.pt --split validation --method greedy --out artifacts/runs/A42/validation_greedy --max-generation-tokens 256
python -m mt.evaluate --checkpoint artifacts/runs/A42/best.pt --split validation --method beam --out artifacts/runs/A42/validation_beam_alpha0.6 --beam-size 4 --length-penalty 0.6 --max-generation-tokens 256
```

`mt.evaluate` refuses to replace a completed evaluation; do not rerun these two commands blindly when their completion markers already exist. The experiment runners provide strict verified reuse. A genuinely different evaluation request needs a new output directory.

## 4. Three-seed budget extension

Freeze each new 12,000-update plan after its corresponding baseline is complete and before its long run starts:

```powershell
python -m mt.plan_quality --seed 42
python -m mt.run_quality --seed 42 --plan artifacts/quality_plan.json
python -m mt.plan_quality --seed 43
python -m mt.run_quality --seed 43
python -m mt.plan_quality --seed 44
python -m mt.run_quality --seed 44
python -m mt.quality_multiseed_report
```

The supplied `experiments/quality_A{seed}_12000.json` files define fresh runs from their own initialization, not continuation of the 3,000-update checkpoint. Each baseline/extension pair shares the configured data and initialization while changing the update budget and output directory. Exact historical scores are not guaranteed across hardware or library changes.

For an interrupted seed-42 run, re-enter `python -m mt.run_quality --seed 42 --plan artifacts/quality_plan.json`; for seeds 43/44, re-enter the corresponding command above. Do not recreate the plan. The runner verifies the plan and restores its matching `last.pt` when necessary. Do not remove fingerprint checks or create a new plan over started results.

## 5. Exploration and validation diagnostics

This stage needs the trained A42 baseline, its two validation outputs, the trained long A42 model and prepared data. It produces training-fitted frequency tables and vector assignments required by the representation experiment.

```powershell
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
python -m mt.explore plan
python -m mt.explore run
python -m mt.explore_report
python -m mt.explore_verify
```

Create the plan only once, then resume with `run`. The configuration is [experiments/exploration_2026_10_27.json](../experiments/exploration_2026_10_27.json). The date in the filename identifies the recipe; new evidence records its real execution date. PCA and clustering are fitted on a fixed training sample, not on validation/test data. Quality analysis on validation remains diagnostic after model selection.

## 6. Paired representation experiments

The recipe helper creates the design file, but tokenizers, shared batch plans, group membership and execution evidence are generated locally:

```powershell
.\project-python.ps1 -m mt.representation_prepare
.\project-python.ps1 -m mt.representation freeze
.\project-python.ps1 -m mt.representation run
.\project-python.ps1 -m mt.representation_report
```

Use the supplied `project-python.ps1` entry point for this stage: preparation launches a `-S` Python subprocess and relies on the wrapper's explicit environment setup. Preparation fits both 8k tokenizers on training data and materializes a common eligible set and shared batch schedules. Freezing requires the frequency table and validation cluster assignments from exploration. It binds the current data, decoder, configuration and source hashes. Resume with `run`, preserving the original plan; do not re-prepare or re-freeze a started experiment.

The inherited beam protocol is shared across representations. It does not make token counts, text-level length constraints or actual computation equal. The old BPE model is a historical reference, not a replacement for the fresh paired BPE runs.

## 7. Optional pretrained direct inference

Install [requirements-pretrained.txt](../requirements-pretrained.txt) only when running this extension, after the base dependencies are installed. Keep the pinned base dependencies while allowing pip to install missing transitive dependencies in a fresh environment:

```powershell
python -m pip install -r requirements-pretrained.txt -c requirements.txt
python -m pip check
python -m mt.pretrained_mt download
python -m mt.pretrained_mt prepare
python -m mt.pretrained_mt audit_lengths
python -m mt.pretrained_mt smoke
python -m mt.pretrained_mt freeze
python -m mt.pretrained_mt run
```

The model uses its fixed upstream revision and its own tokenizer. Downloading and native length auditing precede freezing. This is direct inference with no project training updates. The local model snapshot is ignored by version control and must not be uploaded as part of this source repository. The historical `project-pretrained-python.ps1` entry point is retained because provenance code fingerprints it; the active environment's Python can execute the module directly.

The historical publication-level report also included agent-reviewed sample text and presentation assets. Those are not required to run the individual experiment/report modules and are not reconstructed by these commands. The portable [RESULTS.md](RESULTS.md) summarizes the selected published findings.

## Translation and export after training

Once a checkpoint exists:

```powershell
python -m mt.translate --checkpoint artifacts/runs/quality_A42_12000/best.pt --protocol artifacts/decode_protocol_seed42.json --text "Machine translation helps people communicate." --device cpu
python -m mt.export --checkpoint artifacts/runs/quality_A42_12000/best.pt --protocol artifacts/decode_protocol_seed42.json --out artifacts/bundles/quality_A42_12000
python -m mt.translate --bundle artifacts/bundles/quality_A42_12000 --text "Machine translation helps people communicate." --device cpu
```

The bundle includes inference weights, tokenizer and protocol, with checksums; it still needs this package and dependencies. It is not a complete optimizer checkpoint or standalone executable. Export refuses to overwrite an existing destination.

## Evidence and recovery rules

- Plans bind absolute local paths and hashes. Generate them on the machine running the experiment; copying an old plan to a new location is not migration support.
- A complete result is reused only after configuration, data, decoder, IDs and recomputed metrics agree. Missing/inconsistent evidence should be investigated, not bypassed.
- Keep configuration and execution source fixed during a frozen run. Use a separate experiment for changed precision, budgets or architectures.
- Save complete checkpoints and evidence to persistent storage. A console disconnect alone is not proof that a training process stopped; inspect its committed state before retrying.
- Historical figures and aggregates are references. Fresh execution creates a new result set and must report its actual environment, timings and scores.
