# Results and interpretation

These are selected results from completed research runs, not new runs performed while preparing the public copy. Earlier aggregates are in [summary.json](../results/summary.json); the 2026-10-28 CV/adaptation study is in [cv_adaptation.json](../results/cv_adaptation.json). The source release excludes full corpora, weights and prediction files; rerun the pipelines to obtain a complete machine-local evidence chain.

## Dataset and scoring

The pinned IWSLT-17 English→German source has 206,112/888/8,079 train/validation/test pairs. Training cleaning removes 1,695 duplicate pairs and 132 length-ratio outliers, retaining **204,285 pairs**. The historical training-length filter removes another 139 pairs. Validation and test sentences are retained in full, without silent truncation.

The source is the `iwslt2017-en-de` configuration at revision `3fc158c8125b4d798f2696c0bb5c8928a8fbb332`. The combined test split covers the source release's TED test sets from 2010–2015; scores should not be directly compared with a different IWSLT edition or test subset.

BLEU is case-sensitive SacreBLEU with tokenizer `13a`. chrF2 uses character order 6, word order 0 and beta 2. Statistics across seeds use the sample standard deviation (`ddof=1`), not a confidence interval. Every subgroup score is a corpus metric on its own sentence set, not an average of sentence BLEU scores.

## Historical label-smoothing experiment

Six independent 3,000-update trainings pair label smoothing 0.1 versus 0.0 within seeds 42, 43 and 44. Initial numeric parameters, data, ordering and budgets are matched within each pair. Greedy and beam share each run's validation-best checkpoint.

| Label smoothing | Decoder | Test BLEU, mean ± sample SD |
|---|---|---:|
| 0.1 | Greedy | 15.848 ± 0.079 |
| 0.1 | Beam 4 | 17.146 ± 0.191 |
| 0.0 | Greedy | 15.785 ± 0.163 |
| 0.0 | Beam 4 | 17.050 ± 0.296 |

Beam improves each paired run. Label smoothing has a small mean difference; its greedy effect reverses in seed 44. This does not establish a general or statistically significant advantage.

## Training-budget extension

Three additional fresh trainings extend label-smoothing-0.1 runs from 3,000 to 12,000 updates. Checkpoints are selected by full validation greedy BLEU, retaining the earliest exact tie. The test protocol is fixed to beam 4, length penalty 0.6 and generation cap 256.

| Budget | Greedy BLEU | Beam BLEU | Beam chrF2 |
|---|---:|---:|---:|
| 3,000 updates | 15.848 ± 0.079 | 17.146 ± 0.191 | 42.887 ± 0.299 |
| 12,000 updates | 21.917 ± 0.250 | 23.137 ± 0.321 | 49.778 ± 0.394 |

Within-seed long-minus-short BLEU gains are **6.069 ± 0.181** for greedy and **5.992 ± 0.232** for beam. Historical seed-42 baseline source snapshots are incomplete, and later training versions include persistence/recovery changes. The original comparisons record those differences rather than asserting byte-identical historical training programs. Formal comparison decoders were checked for consistency.

![Training-budget results](../results/figures/training_budget.png)

## Controlled subword representations

Six fresh trainings compare BPE8k and unigram8k across the same three seeds. Both tokenizers are fitted only on the same cleaned training corpus. The common length-eligible intersection contains **204,115 pairs**. A shared envelope produces identical original sentence IDs and batch boundaries within a seed, with matched initial numeric weights, update count, learning rate and validation schedule. Actual token counts and computation remain representation-dependent.

| Representation | Greedy BLEU | Beam BLEU | Greedy chrF2 | Beam chrF2 |
|---|---:|---:|---:|---:|
| BPE8k | 15.762 ± 0.329 | 17.222 ± 0.087 | 42.049 ± 0.153 | 42.932 ± 0.208 |
| unigram8k | 15.723 ± 0.337 | 17.036 ± 0.218 | 42.288 ± 0.354 | 43.182 ± 0.329 |

The paired unigram-minus-BPE beam difference is **−0.186 ± 0.163 BLEU** and **+0.250 ± 0.373 chrF2**. All three BLEU differences are negative, while chrF2 directions vary by seed. Neither representation is uniformly better under both metrics. The inherited decoder was selected on historical BPE validation results; these are shared-protocol results, not individually optimized decoder comparisons.

![Representation comparison](../results/figures/representation.png)

## External pretrained reference

`Helsinki-NLP/opus-mt-en-de`, revision `6183067f769a302e3861815543b9f312c71b0ca4`, is evaluated with its native tokenizer and a frozen native generation procedure. It has **74,410,496 parameters**, versus 11,681,600 in the scratch Transformer. The direct-inference reference has **zero project training or adaptation updates**; its external pretraining cost is unknown. Separate adaptations are described below.

| Native decoder | Test BLEU | Test chrF2 |
|---|---:|---:|
| Greedy | 31.825 | 58.458 |
| Beam 4 | 32.503 | 59.371 |

Its pretraining data, size, vocabulary and decoder differ. The available provenance does not rule out overlap with IWSLT/TED evaluation material. The result therefore compares complete methods, not architecture alone or equal training budgets. One beam test output reached the generation limit with repetition; forced EOS does not imply the translation succeeded.

## Grouped CV and same-base adaptation (2026-10-28)

The completed extension comprises **19 trainings and 57,000 net optimizer updates**: six CV candidates, three post-selection scratch retrainings, four adaptation learning-rate candidates and six final adaptations. It records **102 complete evaluation passes**. Resource probes and interruption replay are additional work, not extra formal seeds or net budget. The original test results were already known when this extension was planned; its settings and selection rules were frozen before viewing new test results.

### CV answers a selection question

Three folds partition the cleaned official training corpus using connected components of TED talk identity and normalized duplicate source sentences. Original tagged text was aligned to all 206,112 source rows to recover 1,698 talks. No talk or normalized source crosses folds. Each fold fits BPE8k on its own training portion, and same-fold candidates share initialization, data order and batch boundaries. Both smoothing candidates use fixed step 3,000, with no held-fold checkpoint selection.

| Label smoothing | Fold-mean greedy BLEU ± fold SD | Fold-mean chrF2 ± fold SD | OOF corpus BLEU |
|---|---:|---:|---:|
| 0 | 13.970 ± 0.305 | 40.949 ± 0.186 | 13.966 |
| 0.1 | 13.991 ± 0.266 | 40.757 ± 0.173 | 13.990 |

The frozen primary rule selects **0.1**, with only a **0.021 BLEU** mean lead. One fold prefers 0 by BLEU, and all three prefer 0 by chrF2. This is limited selection evidence at 3,000 updates, not stable or universal optimality. Fold SD is different from seed SD, and overlapping fold training sets are not independent replications. The largest connected component contains 59,843 sentences (29.29%); balanced sentence counts do not guarantee balanced content.

![CV candidate scores](../results/figures/cv_candidate_scores.png)

Three fresh full-training-set retrainings at the selected setting obtain test greedy BLEU **15.848 ± 0.079**, beam BLEU **17.146 ± 0.191**, and beam chrF2 **42.887 ± 0.299**. Audited endpoint tensors and aligned test predictions reproduce the corresponding historical models. This adds a properly grouped selection procedure without a new quality gain. New training was actually executed; these are not reused checkpoints or six independent seeds. No claim about CV-optimal smoothing at 12,000 updates is made.

### Adaptation answers a quality/cost question

LoRA and full fine-tuning start from the same pinned OPUS-MT model and retain its native tokenizer and special IDs. Both train for 3,000 updates per run, with the same effective batch of 64 and complete sentence-ID trajectory within each seed. Each method receives two learning-rate candidates at tuning seed 7, followed by fresh seeds 42/43/44. Official-validation greedy BLEU selects learning rate and checkpoint; test scores are not used for selection.

| Method | Greedy BLEU | Beam BLEU | Beam chrF2 |
|---|---:|---:|---:|
| Fixed base, direct inference | 31.825 | 32.503 | 59.371 |
| LoRA | 32.919 ± 0.084 | 33.450 ± 0.052 | 59.994 ± 0.023 |
| Full fine-tuning | 33.436 ± 0.084 | 34.032 ± 0.094 | 60.421 ± 0.042 |

The repeated rows use three seeds and sample SD. Beam gains over the fixed base are **0.948 ± 0.052 BLEU** for LoRA and **1.530 ± 0.094** for full fine-tuning. The within-seed LoRA-minus-full difference is **−0.582 ± 0.064 BLEU**; n=3 does not establish statistical significance.

| Resource | LoRA | Full fine-tuning |
|---|---:|---:|
| Trainable parameters / total | 294,912 / 74,705,408 (0.395%) | 73,886,208 / 74,410,496 (99.295%) |
| Formal-run training peak allocation | 2,593.5–2,597.3 MiB | 3,603.3–3,606.9 MiB |
| Selected checkpoint updates (seeds 42/43/44) | 1,500 / 2,000 / 1,000 | 3,000 / 3,000 / 3,000 |
| Saved selected learned weights | about 1.16 MiB; base still required | about 284.17 MiB |

LoRA is the project's own r=8/alpha=16 implementation on 36 q_proj/v_proj modules; PEFT is not required. Full fine-tuning leaves native non-gradient sinusoidal position tables unchanged, explaining its trainable/total count. LoRA adds adapter parameters to the base, explaining its different denominator. The reported file sizes exclude optimizer recovery state; adapter size is not total deployment size.

![Adaptation training costs](../results/figures/adaptation_cost.png)

Both methods complete the same planned trajectory, but validation chooses different prefixes. Their evaluated checkpoints have therefore seen different cumulative sentence/token exposures. Peak allocation is a measured training value on the recorded hardware, not total device use or a universal memory guarantee. LoRA evaluation uses unmerged adapters; fewer trainable parameters do not establish faster inference.

All adaptation evaluation uses FP32 native greedy/beam 4, batch 16 and max_length=512. The fixed direct-inference outputs and original timings were read-only reused after identity/configuration audit. Its historical `CUBLAS_WORKSPACE_CONFIG` was not recorded, so backend identity and bitwise reproducibility are not established. Neither the later adaptation nor matched decoding removes possible OPUS pretraining overlap.

![Fixed-group adaptation gains](../results/figures/adaptation_groups.png)

Fixed length, rarity and anchor-tokenizer fragmentation groups use the same original sentence IDs for every method. For source length ≤10 versus >40 whitespace words, test beam BLEU is **37.957 / 31.773** for LoRA and **38.509 / 32.185** for full fine-tuning (three-seed means). Group associations include confounding; they do not isolate causes. Agent-assisted examples retain both improvements and failures, including grammatical, long-range relation and number-context errors.

See [CV_ADAPTATION.md](CV_ADAPTATION.md) for the full public study description and [REPRODUCING_CV_ADAPTATION.md](REPRODUCING_CV_ADAPTATION.md) for fresh-machine commands. New adaptation weights are not released by this source-only update. Existing [v1.0.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) remains the single scratch seed42 model with **22.889 Beam BLEU**.

## Data and group diagnostics

Training-frequency definitions use whitespace tokens with case-folding and retained punctuation. Validation unseen-token rates are 2.14% in English and 4.07% in German; these are **not SentencePiece UNK rates**.

![Sentence length distributions](../results/figures/exploration_lengths.png)

Fixed same-sentence groups expose quality differences. In the controlled 3,000-update BPE experiment, beam BLEU is 21.662 for source sentences of at most 10 whitespace words and 13.557 for sentences over 40 words. Corresponding low/high fragmentation groups score 20.353/14.003. These associations include multiple confounders; they are not isolated causal effects.

Exploratory sentence-vector analysis fits standardization, PCA50 and KMeans5 on 10,000 training vectors only, then assigns the 888 validation sentences. PCA50 explains 85.71% of variance. Silhouette is 0.0462, and validation source-BPE length has cluster association η²=0.729. The clusters overlap substantially and mainly track length; they are not validated semantic themes.

![Exploratory group quality](../results/figures/exploration_groups.png)

Validation scores in this exploration are diagnostic because validation was already used for checkpoint and decoding choices. Fixed sample review records both improvements and failures, including omissions, entity substitutions, repetition and negation reversal. It is agent-assisted review, not independent human annotation or a measured overall error rate.

## Reliability and remaining limitations

The earlier local suite passed **194 regression tests**, spanning data/tokenizer integrity, checkpoint recovery, decoding/metric checks, and plan freezing/result reuse; that execution audited 518 pre-existing files as unchanged. The expanded CV/adaptation study passed **349 local regression checks** and preserved **1,393 historical file fingerprints**. Those are research-workspace observations, not a statement that all original evidence is included here or that GitHub Actions/another machine has passed 349 tests. Public-copy verification is recorded separately.

- Fifteen scratch trainings comprise six label-smoothing runs, three budget extensions and six representation runs. Diagnostic smoke/overfit runs are excluded from this count.
- Three seeds on fixed splits measure run variation. The separately executed grouped three-fold CV measures split sensitivity under a fixed model seed and 3,000-update budget.
- The public test set has been observed across successive experiments. Freezing a later experiment prevents its own result-driven changes but does not make the public test set a previously unseen holdout again.
- Runtime measurements span dates and were not fully controlled for power, temperature and clocks. Batch-amortized time is not isolated single-request latency.
- Cross-hardware bitwise reproducibility, convergence and universal translation correctness are not established.
