# Grouped CV and same-base adaptation: new-machine reproduction

This route creates **new local evidence** from the public source package. It does not require the author's 1,393 historical files, old checkpoints, old exploration clusters, or machine-local research plan. It does require the pinned dataset and pretrained model downloads, a compatible Python 3.13 environment, and substantial GPU time. Run every command from the repository root. Follow [REPRODUCING.md](REPRODUCING.md) for base dependencies and the historically used CUDA wheel.

The original study completed 19 training runs and 57,000 net optimizer updates. Resource probes, validation, and complete CV holdout decoding add work beyond that count. This publication update validates recipe generation and CPU regression behavior; it does **not** rerun the 19 GPU trainings. Hardware/library changes can change results.

## Scope and fixed decisions

- CV: three grouped folds × smoothing `0` / `0.1`, each fixed at update 3,000. The fold holdout never selects a checkpoint. Select by mean fold Greedy corpus BLEU, then chrF2, then candidate order. Refit a fresh full-training BPE and retrain three seeds.
- Grouping: recover explicit TED talk IDs from pinned original tagged text, validate every raw pair against pinned Parquet, then combine same-talk and normalized-source links. Source-only fallback is permitted only when joint components violate predefined feasibility bounds. Missing or discordant talk recovery is a blocker in this implementation; it is not silently replaced by row adjacency.
- Adaptation: one pinned OPUS-MT base, native tokenizer, four seed-7 learning-rate candidates, then LoRA/full three-seed repeats. Each run has 3,000 updates. Validation every 500 updates selects the checkpoint. LoRA is the project's audited low-rank linear implementation; PEFT is not required.
- Test outcomes were already known before the original research was planned. New runs must freeze configuration before new test evaluation. Do not use test scores to change candidates, budgets, rank, or seed count.
- `mt.study freeze/run` and `mt.study_complete` retain dependencies on the original historical research package. **Do not use those whole-study controllers for this portable route.** The helper only reuses the isolated adaptation selector, which verifies the local run evidence without the historical manifest.

## 1. Initialize a new root

Install optional native dependencies in the active `.venv`:

```powershell
python -m pip install -r requirements-pretrained.txt -c requirements.txt
python -m pip check
python -m mt.data --out artifacts/data
python scripts/prepare_cv_adaptation.py init --root artifacts/cv_adaptation --source artifacts/data
```

The helper reads [the path-free design](../recipes/cv_adaptation_design.json), saves the actual timestamp, and resolves source/output paths on this machine. It writes its own fixed scratch decoder recipe inside the new root. It refuses a nonempty destination. Later stages reject a changed design/protocol and do not replace existing config files. Keep the initialized root in place for the complete study.

No full-data scratch tokenizer is needed before CV: each fold creates its own. For a local partial reproduction, the CV and native sections can be run separately; report the actually executed scope.

## 2. Recover groups and fit fold-only tokenizers

```powershell
.\project-python.ps1 -m mt.cv_prepare --root artifacts/cv_adaptation --source artifacts/data --stage groups
.\project-python.ps1 -m mt.cv_prepare --root artifacts/cv_adaptation --source artifacts/data --stage folds
python scripts/prepare_cv_adaptation.py cv-configs --root artifacts/cv_adaptation
```

Use `project-python.ps1` for this preparation: SentencePiece is launched in a `-S` child process, which needs the wrapper's explicit package path. Inspect `data/cv/group_audit.json`, `grouping_complete.json`, all three `tokenizer_fit_ids.json` files, and `cv/configs/stage_plan.json`. Each candidate in a fold shares the tokenizer, initialization seed, original sentence IDs, and batch schedule. Only fold-training examples are eligible for filtering; holdouts remain complete.

Train and evaluate sequentially:

```powershell
$studyRoot = 'artifacts/cv_adaptation'
foreach ($fold in 0, 1, 2) {
    foreach ($candidate in 'ls0', 'ls01') {
        $name = "fold${fold}_${candidate}"
        .\project-python.ps1 -m mt.cv_train --config "$studyRoot/cv/configs/$name.json"
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $name" }
        .\project-python.ps1 -m mt.cv_evaluate --checkpoint "$studyRoot/cv/runs/$name/last.pt" --split holdout --method greedy --out "$studyRoot/cv/runs/$name/holdout_greedy" --protocol "$studyRoot/decode_protocol.json" --decode-batch-size 16 --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Holdout evaluation failed: $name" }
    }
}
.\project-python.ps1 -m mt.cv_select --root artifacts/cv_adaptation --source artifacts/data
.\project-python.ps1 -m mt.cv_prepare --root artifacts/cv_adaptation --source artifacts/data --stage final
python scripts/prepare_cv_adaptation.py cv-final-configs --root artifacts/cv_adaptation
```

The selector validates all six complete runs and paired schedules, recomputes corpus metrics, and reconstructs OOF predictions in original training-row order. Fold means and whole-OOF corpus scores are distinct statistics. `cv-final-configs` calls that validator again; dropping a handwritten smoothing value into a file does not unlock final configs.

```powershell
foreach ($seed in 42, 43, 44) {
    .\project-python.ps1 -m mt.cv_train --config "$studyRoot/cv/final_configs/seed$seed.json"
    if ($LASTEXITCODE -ne 0) { throw "Final CV training failed: $seed" }
    foreach ($split in 'validation', 'test') {
        foreach ($method in 'greedy', 'beam') {
            .\project-python.ps1 -m mt.cv_evaluate --checkpoint "$studyRoot/cv/final_runs/seed$seed/last.pt" --split $split --method $method --out "$studyRoot/cv/final_runs/seed$seed/${split}_${method}" --protocol "$studyRoot/decode_protocol.json" --decode-batch-size 16 --device cuda
            if ($LASTEXITCODE -ne 0) { throw "Final CV evaluation failed: $seed $split $method" }
        }
    }
}
```

Never substitute the old global BPE for fold-fitted tokenizers. The selected smoothing is supported at this 3,000-update budget only.

## 3. Download and measure the fixed native baseline

Use a new explicit native output root, not the date-specific module default:

```powershell
python -m mt.pretrained_mt download --out artifacts/cv_adaptation/baseline
python -m mt.pretrained_mt prepare --out artifacts/cv_adaptation/baseline --data-dir artifacts/data
python -m mt.pretrained_mt audit_lengths --out artifacts/cv_adaptation/baseline
python -m mt.pretrained_mt smoke --out artifacts/cv_adaptation/baseline --device cuda
python -m mt.pretrained_mt freeze --out artifacts/cv_adaptation/baseline
python -m mt.pretrained_mt run --out artifacts/cv_adaptation/baseline --device cuda
```

Download pins `Helsinki-NLP/opus-mt-en-de` revision `6183067f769a302e3861815543b9f312c71b0ca4`. New direct-baseline predictions are evaluated under the same native FP32 protocol used for adaptation. This route does not copy a historical score into a new baseline. The native and scratch decoding implementations differ; comparisons are descriptive.

## 4. Prepare common native batches and check resources

The published defaults are common microbatch 8, effective batch 64, native training length limit 256, and BF16. Before executing this block confirm CUDA and BF16 support. Otherwise use the same FP16 choice for **both** methods and preserve that difference in the report.

```powershell
python -m mt.adapt_data --out artifacts/cv_adaptation/adaptation/data --model-dir artifacts/cv_adaptation/baseline/model --data-dir artifacts/data --max-tokens 256 --micro-batch-size 8
python -m mt.adapt_train smoke --model-dir artifacts/cv_adaptation/baseline/model --data-root artifacts/cv_adaptation/adaptation/data --out artifacts/cv_adaptation/adaptation/resource_smoke.json --micro-batch-size 8 --precision bfloat16 --device cuda
python scripts/prepare_cv_adaptation.py adapt-configs --root artifacts/cv_adaptation --native-root artifacts/cv_adaptation/baseline --model-dir artifacts/cv_adaptation/baseline/model --smoke artifacts/cv_adaptation/adaptation/resource_smoke.json
```

This probe checks forward/backward/optimizer steps, the longest retained pair, exact zero-adapter equivalence, unchanged fixed positional tables, and export/reload equivalence. `adapt-configs` requires passed evidence with matching source, base, data, batch and precision identities. It generates the four tune configs plus a new local `plan.json`; it does not require the old historical fingerprint list.

If the probe fails, retain its failure evidence. Test a smaller common microbatch (8 → 4 → 2 → 1), retaining effective batch 64; use a **new root/data preparation and new probe output**, because prepared batch settings are immutable. Do not overwrite an existing manifest or let LoRA/full use different effective sentence batches. A successful resource probe still does not guarantee absence of later OOM. Stop and record any remaining hardware blocker.

## 5. Tune learning rates using validation only

```powershell
$tuneConfigs = Get-ChildItem -LiteralPath "$studyRoot/adaptation/configs" -Filter 'tune_*.json' | Sort-Object Name
foreach ($configuration in $tuneConfigs) {
    python -m mt.adapt_train freeze --config $configuration.FullName
    if ($LASTEXITCODE -ne 0) { throw 'Run freeze failed' }
    python -m mt.adapt_train train --config $configuration.FullName --device cuda
    if ($LASTEXITCODE -ne 0) { throw 'Adaptation tuning failed' }
}
python scripts/prepare_cv_adaptation.py adapt-final-configs --root artifacts/cv_adaptation
```

The four candidates are LoRA `1e-4` / `3e-4` and full `1e-5` / `3e-5`, seed 7. Each trains from the same original base. Training evaluates full validation Greedy every 500 updates. Selection checks each complete `last.pt`, best export, validation journals, input identity and metrics. It does not read test metrics. `adaptation/selection.json` and `adaptation/final_plan.json` record the decision before six formal runs.

```powershell
foreach ($seed in 42, 43, 44) {
    foreach ($adaptation in 'lora', 'full') {
        $configuration = "$studyRoot/adaptation/configs/final_${adaptation}_seed$seed.json"
        python -m mt.adapt_train freeze --config $configuration
        if ($LASTEXITCODE -ne 0) { throw 'Formal run freeze failed' }
        python -m mt.adapt_train train --config $configuration --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'Formal adaptation training failed' }
        foreach ($split in 'validation', 'test') {
            foreach ($method in 'greedy', 'beam') {
                python -m mt.adapt_train evaluate --config $configuration --split $split --method $method --device cuda
                if ($LASTEXITCODE -ne 0) { throw 'Formal adaptation evaluation failed' }
            }
        }
    }
}
```

The example tuning loop sorts file names; actual order must be recorded because timing depends on thermal/power state. Each formal seed starts from the base, not a tuned checkpoint. `best.pt` is a learned-state export interpreted by `adapt_train`; it is not compatible with the old scratch `mt.translate --bundle` loader. This update does not publish a new inference ZIP or add an end-user adapter translation interface.

## Recovery and result interpretation

- Check whether the previous process is still running before starting another. Keep GPU jobs sequential.
- CV recovery: rerun the **same config** with `--resume <matching-run>/last.pt`. Never recreate configs over a started run. `--stop-after` supports explicit partial training for diagnosis.
- Adaptation recovery: use `python -m mt.adapt_train verify --config <same-config>`, then repeat `train` with that config. Do not repeat `freeze`; a frozen run correctly refuses a second freeze.
- Evaluations retain hashed progress journals and reuse complete outputs only after verification. Repeat their same evaluation command to resume. Do not manually splice predictions.
- Whole PowerShell loops above are first-run examples. After an interruption, restart at the specific failed stage; a freeze refusal for an existing run is expected protection, not a reason to delete its plan.
- CV outputs are under `cv/selection.json`, `cv/oof/`, and `cv/final_runs/seed*/{validation,test}_{greedy,beam}/metrics.json`. Adaptation outputs are under `adaptation/runs/final_*/evaluation/{validation,test}/{greedy,beam}/metrics.json`, plus each run's `summary.json`.
- For each final family, aggregate corpus BLEU/chrF2 over seeds 42/43/44 with sample SD (`ddof=1`). Pair LoRA/full scores within seed for differences. The direct model is one fixed baseline, not three independent training seeds. Report parameter counts, train peaks, wall time, selected checkpoint exposures and export sizes from actual run records.
- Full-model parameter updates exclude Marian's intentionally fixed sinusoidal positional embeddings. Preserve exact native PAD/start/EOS handling. Neither three seeds nor a higher average establishes significance or universal superiority.
- The historical report's 102 complete evaluations includes its particular internal and formal accounting. Do not assert that count for a fresh execution unless its own journals demonstrate it.
- Frozen sentence-level analysis groups from the old representation study are not bundled. This portable route reproduces core CV/adaptation training and full-set metrics; rerunning its old cluster-based analysis and presentation requires separately rebuilding those analysis inputs.

Keep complete local execution evidence outside Git. Publish selected aggregate statistics and figures only after removing machine paths, private identifiers and raw text not intended for redistribution. See [RESULTS.md](RESULTS.md) for the published historical findings.
