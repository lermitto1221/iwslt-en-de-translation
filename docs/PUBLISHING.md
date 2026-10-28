# Publishing this copy

Existing repository: [lermitto1221/iwslt-en-de-translation](https://github.com/lermitto1221/iwslt-en-de-translation).

This folder is a curated source-and-results copy. The original research workspace remains separate. Preparing or validating this folder does not upload it. The original v1.0.0 source/model release remains available; this update adds completed grouped CV and same-base adaptation research without replacing that model ZIP.

## Update the existing repository

Review `RELEASE_CHECKS.json` and run the package checks before publishing. Keep the existing folder structure when transferring modified/new files. Use either the GitHub upload interface or a clone of the existing repository; do not create another repository or initialize a replacement history for this update.

For a local clone, copy the curated update into the checkout, inspect all changes, then commit/push:

```powershell
python scripts/verify_public_package.py
git status --short
git diff --stat
git add .
git diff --cached --stat
git commit -m "Add grouped CV and same-base adaptation study"
git push origin main
```

These commands assume an already configured clone and do not configure identity or credentials. Inspect staged files before the commit. If using GitHub's web interface, upload the changed/new source and documentation at the matching paths, and remove obsolete files only if the update inventory explicitly requires it. Do not drag an extra enclosing folder into the repository root.

After the upload, check the actual commit's Actions run. The earlier online CPU workflow passed for the previous version. The expanded research workspace separately passed 349 tests; this folder's current local suite and counts are in `RELEASE_CHECKS.json`. Preparing the update does not establish that its new online CI has run.

## Model assets remain separate

The existing [v1.0.0 model Release](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) contains the scratch seed42/12,000-update model with 22.889 Beam BLEU. Do not label it with the new adaptation aggregate scores. New LoRA/full inference packaging and a separate asset release are outside this source update. A future adapter release must identify its required fixed base, adapter format, chosen seed and that single model's actual score.

The included GitHub Actions workflow runs the CPU regression suite on Windows with Python 3.13. The expanded suite installs `requirements.txt` first and then `requirements-pretrained.txt` constrained by the core requirements. Local checks and previous online runs must be reported separately.

## What belongs in Git

Commit source, tests, configurations, recipes, documentation, selected aggregate JSON and figures. The ignore rules exclude environments, raw/generated corpora, checkpoints, tokenizers, model downloads and experiment state. Before a later upload, check `git status` to see what will actually be committed.

Choose a project-wide code license separately if you want to grant reuse permissions. Dataset and model terms remain separate; see [THIRD\_PARTY.md](THIRD_PARTY.md).
