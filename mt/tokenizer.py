import argparse
from pathlib import Path

import sentencepiece as spm

from .common import sha256, write_json
from .data import read_pairs

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def train(data_dir, vocab_size=8000):
    data_dir = Path(data_dir)
    pairs = read_pairs(data_dir / "train.jsonl")
    if not pairs:
        raise ValueError("empty training corpus")
    corpus = data_dir / "spm_train.txt"
    with corpus.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(row["en"].replace("\n", " ") + "\n")
            f.write(row["de"].replace("\n", " ") + "\n")
    prefix = str(data_dir / "spm")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix,
        vocab_size=vocab_size, model_type="bpe", pad_id=PAD, unk_id=UNK,
        bos_id=BOS, eos_id=EOS, hard_vocab_limit=False,
        character_coverage=1.0, num_threads=1)
    model = data_dir / "spm.model"
    tok = spm.SentencePieceProcessor(model_file=str(model))
    if [tok.pad_id(), tok.unk_id(), tok.bos_id(), tok.eos_id()] != [PAD, UNK, BOS, EOS]:
        raise AssertionError("special token IDs changed")
    write_json(data_dir / "tokenizer.json", {"model_type": "bpe", "requested_vocab_size": vocab_size,
        "actual_vocab_size": tok.vocab_size(), "special_ids": {"pad": PAD, "unk": UNK, "bos": BOS, "eos": EOS},
        "training_pairs": len(pairs), "training_file_sha256": sha256(data_dir / "train.jsonl"),
        "model_sha256": sha256(model), "vocab_sha256": sha256(data_dir / "spm.vocab")})


def load(data_dir):
    return spm.SentencePieceProcessor(model_file=str(Path(data_dir) / "spm.model"))


def encode(tok, text):
    return [BOS] + tok.encode(text, out_type=int) + [EOS]


def decode(tok, ids):
    clean = []
    for token in ids:
        token = int(token)
        if token == EOS:
            break
        if token not in (PAD, BOS):
            clean.append(token)
    return tok.decode(clean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/data")
    p.add_argument("--vocab-size", type=int, default=8000)
    args = p.parse_args()
    train(args.data_dir, args.vocab_size)


if __name__ == "__main__":
    main()
