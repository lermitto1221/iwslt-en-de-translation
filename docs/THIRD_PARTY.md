# Third-party sources and release scope

This project implements and evaluates machine translation experiments using external research, software, data and pretrained resources. Links below identify their upstream sources. They do not transfer ownership of those resources or replace their terms.

## Dataset

- Dataset: [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017), configuration `iwslt2017-en-de`.
- Pinned conversion revision: `3fc158c8125b4d798f2696c0bb5c8928a8fbb332`; download URLs and expected digests are defined in `mt/data.py`.
- Original context: [IWSLT](https://iwslt.org/) and TED translation material referenced by the dataset release.

The repository does **not** include sentence corpora, raw Parquet files or full reference/prediction dumps. The data preparation command downloads them from the upstream source. Review the dataset card and relevant original-use terms before accessing or redistributing the material. Do not infer a dataset-wide license from this project's code or from public download availability.

## Pretrained reference model

- Model: [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de).
- Exact model-card revision: [6183067f769a302e3861815543b9f312c71b0ca4](https://huggingface.co/Helsinki-NLP/opus-mt-en-de/blob/6183067f769a302e3861815543b9f312c71b0ca4/README.md).
- Upstream project: [OPUS-MT training resources](https://github.com/Helsinki-NLP/OPUS-MT-train).

No pretrained weights or upstream tokenizer files are bundled in this source copy. The optional pipelines download the fixed revision and record hashes locally. Results distinguish the unchanged externally pretrained base from project-trained LoRA and full fine-tuning adaptations; they do not imply this project performed the original OPUS-MT pretraining. Pretraining/evaluation overlap cannot be excluded from available provenance. The existing v1.0.0 download is a separate scratch-model bundle, not OPUS-MT or a newly released adaptation.

## Software and methods

Direct dependencies and versions are recorded in [requirements.txt](../requirements.txt); optional pretrained inference/adaptation dependencies are in [requirements-pretrained.txt](../requirements-pretrained.txt). Install core requirements first, then the optional file with `-c requirements.txt` to retain pinned core versions. LoRA is implemented in this project's `mt/adapt_model.py`; PEFT is not required. Dependencies' licenses and notices remain those supplied by each upstream project.

- [PyTorch](https://github.com/pytorch/pytorch): tensor operations, model building and optimization.
- [SentencePiece](https://github.com/google/sentencepiece): BPE and unigram subword models.
- [SacreBLEU](https://github.com/mjpost/sacrebleu): corpus BLEU and chrF evaluation.
- [Apache Arrow / PyArrow](https://github.com/apache/arrow): reading the pinned Parquet source.
- [Matplotlib](https://github.com/matplotlib/matplotlib): figures.
- [NumPy](https://github.com/numpy/numpy): numerical analysis.
- [Hugging Face Transformers](https://github.com/huggingface/transformers): native Marian pretrained inference and adaptation base.

Relevant research:

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/).
- Kudo and Richardson, [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/).
- Post, [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/).

## This repository's licensing status

The project owner has not selected a project-wide open-source license for this release. No MIT, Apache or other license is assigned on their behalf. Making a GitHub repository public does not by itself grant unrestricted reuse or redistribution rights.

Downloaded data/models, generated full-corpus outputs, dependency installations and local checkpoints are excluded from this source copy. Before adding them to a future release, check their respective terms and attribution requirements. The repository's selected aggregate metrics and figures should be understood as experiment documentation, with the limitations stated in [RESULTS.md](RESULTS.md).
