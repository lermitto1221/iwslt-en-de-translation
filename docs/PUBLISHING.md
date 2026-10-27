# Publishing this copy

Suggested repository name: **iwslt-en-de-translation**.

This folder is a curated source-and-results copy. The original research workspace remains separate. No remote repository is configured in this copy, and preparing the folder does not upload it.

## Upload with Git

Create an empty repository on GitHub with the desired visibility. For the first push, leave GitHub's automatic README, license and gitignore options empty because this folder already contains its documentation and ignore rules.

Open a terminal in this folder and run:

```powershell
python scripts/verify_public_package.py
git init -b main
git add .
git status --short
git commit -m "Publish translation experiments and reproducibility tools"
git remote add origin https://github.com/YOUR_USERNAME/iwslt-en-de-translation.git
git push -u origin main
```

Replace `YOUR_USERNAME` with the repository owner. Git may ask you to configure your commit name/email and sign in. Keep credentials out of committed files.

The included GitHub Actions workflow runs the CPU regression suite on Windows with Python 3.13. This workflow has not been run on GitHub as part of preparing the local copy.

## What belongs in Git

Commit source, tests, configurations, recipes, documentation, selected aggregate JSON and figures. The ignore rules exclude environments, raw/generated corpora, checkpoints, tokenizers, model downloads and experiment state. Before a later upload, check `git status` to see what will actually be committed.

Choose a project-wide code license separately if you want to grant reuse permissions. Dataset and model terms remain separate; see [THIRD\_PARTY.md](THIRD_PARTY.md).
