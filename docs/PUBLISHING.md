# Publishing this copy

Existing repository: [lermitto1221/iwslt-en-de-translation](https://github.com/lermitto1221/iwslt-en-de-translation).

This folder is a curated source-and-results copy. The original research workspace remains separate. Preparing or validating this folder does not upload it. The original v1.0.0 model release remains available; v1.1.0 provides separate full/LoRA inference assets without replacing that scratch-model ZIP.

**Publication order:** upload these documentation updates only after the v1.1.0 Release is published with both ZIPs and both `.sha256` files. Their links are prepared in this copy and become usable when those assets are published. A local prepared asset folder is not evidence of an online Release.

## Update the existing repository

Review `RELEASE_CHECKS.json` and run the package checks before publishing. Keep the existing folder structure when transferring modified/new files. Use either the GitHub upload interface or a clone of the existing repository; do not create another repository or initialize a replacement history for this update.

For a local clone, copy the curated update into the checkout, inspect all changes, then commit/push:

```powershell
python scripts/verify_public_package.py
git status --short
git diff --stat
git add .
git diff --cached --stat
git commit -m "Document v1.1.0 standalone adaptation model bundles"
git push origin main
```

These commands assume an already configured clone and do not configure identity or credentials. Inspect staged files before the commit. If using GitHub's web interface, upload the changed/new source and documentation at the matching paths, and remove obsolete files only if the update inventory explicitly requires it. Do not drag an extra enclosing folder into the repository root.

After the upload, check the actual commit's Actions run. The expanded research workspace separately passed 349 tests; this folder's current local suite and counts are in `RELEASE_CHECKS.json`. The badge and online logs apply to the commits they actually evaluated. Preparing this documentation update does not establish that its new online CI has run.

## Model assets remain separate

Create or complete [v1.1.0](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.1.0) with these four binary attachments:

- `iwslt-en-de-full-seed42-inference-v1.1.0.zip` and its `.zip.sha256` file;
- `iwslt-en-de-lora-seed42-inference-v1.1.0.zip` and its `.zip.sha256` file.

Attach them in the Release's binary/Assets upload area. They are separate from GitHub's automatic source-code archives and from attachments inserted into the text description. Preserve the exact filenames used by the README links. Do not commit the model ZIPs into Git.

The full ZIP is 277,564,485 bytes (277.6 MB decimal), contains complete weights and extracts to `full_seed42`; its single-model Beam BLEU is **33.924**. The LoRA ZIP is 2,339,896 bytes (2.34 MB decimal), extracts to `lora_seed42` and scores **33.402**. LoRA needs the exact original `Helsinki-NLP/opus-mt-en-de` base at revision `6183067f769a302e3861815543b9f312c71b0ca4` (about 298 MB of base weights). Its format is custom to this project, not PEFT.

The packages include their standalone translation runtime and pinned dependencies. They reuse trained checkpoints and do not require the source repository for inference. In Release notes, keep single-model scores distinct from the three-seed means **34.032 ± 0.094 / 33.450 ± 0.052** for full/LoRA. Use the packaged `translate.py` instructions; the repository's existing `mt.translate` remains the old scratch-model interface.

After publication, open the Release and verify both named assets and checksum files. Then upload this documentation set and its refreshed metadata, preserving paths. The existing [v1.0.0 model Release](https://github.com/lermitto1221/iwslt-en-de-translation/releases/tag/v1.0.0) remains the scratch seed42/12,000-update model with **22.889 Beam BLEU**.

The included GitHub Actions workflow runs the CPU regression suite on Windows with Python 3.13. The expanded suite installs `requirements.txt` first and then `requirements-pretrained.txt` constrained by the core requirements. Local checks and previous online runs must be reported separately.

## What belongs in Git

Commit source, tests, configurations, recipes, documentation, selected aggregate JSON and figures. The ignore rules exclude environments, raw/generated corpora, checkpoints, tokenizers, model downloads and experiment state. Before a later upload, check `git status` to see what will actually be committed.

Choose a project-wide code license separately if you want to grant reuse permissions. Dataset and model terms remain separate; see [THIRD\_PARTY.md](THIRD_PARTY.md).
