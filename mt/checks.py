import argparse
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .batching import collate
from .common import read_json, sha256, write_json
from .data import read_pairs
from .decoding import generate_batch_ids, generate_ids
from .model import TranslationTransformer
from .tokenizer import PAD, UNK, BOS, EOS, decode, encode, load


class _ScriptedDecoder(torch.nn.Module):
    """Tiny conditional model whose exact best paths are known in advance."""

    def __init__(self, mode):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.mode = mode
        self.calls = []
        self.last_sources = None

    def encode(self, src):
        self.last_sources = src.clone()
        return src.float().unsqueeze(-1), src.eq(PAD)

    def decode_logits(self, memory, src_pad, tgt):
        assert not tgt.eq(EOS).any(), "finished hypotheses must never be expanded"
        self.calls.append(tgt.size(0))
        logits = torch.full((len(tgt), tgt.size(1), 9), -float("inf"), device=tgt.device)
        for index, seq in enumerate(tgt.tolist()):
            marker = int(memory[index, 1, 0])
            if self.mode == "batch":
                token = EOS if marker == 4 or (marker == 5 and len(seq) == 2) else 6
                probabilities = {token: 1.0}
            elif self.mode == "completed":
                probabilities = ({EOS: 0.49, 5: 0.50, 6: 0.01} if len(seq) == 1
                                 else {EOS: 0.4, 6: 0.6})
            elif self.mode == "pruned_eos":
                # With K=2, root EOS ranks third and must be discarded.
                # At step 2, EOS ranks second within each parent but third/
                # fourth globally, so neither may become a finished hypothesis.
                # Keeping those pruned EOS paths would select the empty output
                # (probability .30) over the legal winner (.36*.51=.1836).
                probabilities = ({5: 0.36, 6: 0.34, EOS: 0.30} if len(seq) == 1 else
                                 ({7: 0.51, EOS: 0.49} if seq[1] == 5 else {7: 0.52, EOS: 0.48})
                                 if len(seq) == 2 else {EOS: 1.0})
            elif self.mode == "refill":
                # EOS ranks second at the root. Top 2K must refill the second
                # live slot with token 6, although it ranks below the first K.
                probabilities = ({5: 0.50, EOS: 0.30, 6: 0.20} if len(seq) == 1
                                 else {EOS: 1.0})
            else:
                # Greedy takes 5 then EOS (0.55*0.51), while beam retains 6
                # and reaches EOS with probability 0.45*0.99.
                probabilities = ({5: 0.55, 6: 0.45} if len(seq) == 1 else
                                 ({EOS: 0.51, 7: 0.49} if seq[1] == 5 else {EOS: 0.99, 7: 0.01}))
            for token, probability in probabilities.items():
                logits[index, -1, token] = math.log(probability)
        return logits


def check_decoding():
    model = _ScriptedDecoder("batch").train()
    sources = [[BOS, 4, EOS], [BOS, 5, EOS], [BOS, 6] + [7] * 257 + [EOS]]
    sequences, hits = generate_batch_ids(model, sources, max_tokens=2)
    assert sequences == [[BOS, EOS], [BOS, 6, EOS], [BOS, 6, 6]]
    assert hits == [False, True, True]  # EOS on final step still reaches limit.
    assert [seq[-1] != EOS for seq in sequences] == [False, False, True]
    assert model.training and model.calls == [3, 2]
    assert model.last_sources[2].tolist() == sources[2]  # no source truncation
    beam1, beam1_hits = generate_batch_ids(model, sources, "beam", 1, max_tokens=2)
    assert (beam1, beam1_hits) == (sequences, hits)
    assert generate_batch_ids(model, []) == ([], [])
    model = _ScriptedDecoder("completed")
    seq, _ = generate_ids(model, [BOS, 4, EOS], "beam", 2, alpha=0, max_tokens=5)
    assert seq == [BOS, EOS] and len(model.calls) > 1
    model = _ScriptedDecoder("pruned_eos")
    seq, hit = generate_ids(model, [BOS, 4, EOS], "beam", 2, alpha=0, max_tokens=3)
    assert seq == [BOS, 5, 7, EOS] and hit
    assert model.calls == [1, 2, 2]
    batched, _ = generate_batch_ids(model, sources[:2], "beam", 2, alpha=0, max_tokens=3)
    assert batched == [[BOS, 5, 7, EOS], [BOS, 5, 7, EOS]]
    model = _ScriptedDecoder("refill")
    seq, _ = generate_ids(model, [BOS, 4, EOS], "beam", 2, alpha=0, max_tokens=3)
    assert seq == [BOS, 5, EOS] and model.calls == [1, 2]
    model = _ScriptedDecoder("branch")
    seq, _ = generate_ids(model, [BOS, 4, EOS], "beam", 2, alpha=0, max_tokens=3)
    assert seq == [BOS, 6, EOS]
    greedy, _ = generate_ids(model, [BOS, 4, EOS], "greedy", max_tokens=3)
    assert greedy == [BOS, 5, EOS]
    batched, _ = generate_batch_ids(model, sources[:2], "beam", 2, alpha=0, max_tokens=3)
    assert batched == [[BOS, 6, EOS], [BOS, 6, EOS]]
    # eval mode is restored even if encoding fails (e.g. a positional limit).
    class Failing(_ScriptedDecoder):
        def encode(self, src):
            raise ValueError("intentional source encoding failure")
    failing = Failing("batch").train()
    try:
        generate_ids(failing, sources[0])
    except ValueError:
        pass
    assert failing.training


def run(data_dir, cfg_path, splits=("train", "validation", "test")):
    check_decoding()
    data = Path(data_dir)
    cfg = read_json(cfg_path)
    tok = load(data)
    assert [tok.pad_id(), tok.unk_id(), tok.bos_id(), tok.eos_id()] == [PAD, UNK, BOS, EOS]
    assert decode(tok, encode(tok, "Hello world")) == "Hello world"
    for split in splits:
        rows = read_pairs(data / f"{split}.jsonl")
        assert rows and all(isinstance(r["en"], str) and isinstance(r["de"], str) and r["en"] and r["de"] for r in rows)
    examples = [([BOS, 5, EOS], [BOS, 6, EOS], 0),
                ([BOS, 5, 7, EOS], [BOS, 6, 7, EOS], 1)]
    src, tgt, _ = collate(examples, [0, 1], torch.device("cpu"))
    assert src.shape == (2, 4) and tgt.shape == (2, 4)
    assert tgt[0].tolist() == [BOS, 6, EOS, PAD]
    model = TranslationTransformer(tok.vocab_size(), cfg).eval()
    with torch.no_grad():
        logits = model(src, tgt[:, :-1])
        assert logits.shape == (2, 3, tok.vocab_size())
        # Future decoder tokens must not affect earlier logits.
        a = torch.tensor([[BOS, 6, 7]], dtype=torch.long)
        b = torch.tensor([[BOS, 6, 8]], dtype=torch.long)
        la = model(src[:1], a)
        lb = model(src[:1], b)
        assert torch.allclose(la[:, :2], lb[:, :2], atol=1e-5)
        # Extra source PAD positions must be invisible to cross attention.
        extra = F.pad(src[:1, :3], (0, 2), value=PAD)
        base = src[:1, :3]
        assert torch.allclose(model(base, a), model(extra, a), atol=1e-5)
        assert torch.allclose(model(base, a), model(base, F.pad(a, (0, 2), value=PAD))[:, :3], atol=1e-5)
        # PAD target positions do not contribute to loss.
        flat = logits.reshape(-1, logits.size(-1))
        labels = tgt[:, 1:].reshape(-1)
        loss1 = F.cross_entropy(flat, labels, ignore_index=PAD, reduction="sum")
        changed = flat.clone()
        changed[labels.eq(PAD)] = 0
        changed[labels.eq(PAD), PAD] = 100
        loss2 = F.cross_entropy(changed, labels, ignore_index=PAD, reduction="sum")
        assert torch.allclose(loss1, loss2)
        greedy, _ = generate_ids(model, [BOS, 5, EOS], "greedy", max_tokens=8)
        beam1, _ = generate_ids(model, [BOS, 5, EOS], "beam", beam_size=1, max_tokens=8)
        assert greedy == beam1
        assert greedy[0] == BOS and (greedy[-1] == EOS or len(greedy) == 9)
        batch_ids, _ = generate_batch_ids(model, [[BOS, 5, EOS], [BOS, 5, 7, EOS]], max_tokens=8)
        assert batch_ids[0] == greedy
        assert batch_ids[1] == generate_ids(model, [BOS, 5, 7, EOS], max_tokens=8)[0]
        if hasattr(model, "decode_step_logits"):
            memory, padding = model.encode(src)
            assert torch.allclose(model.decode_step_logits(memory, padding, tgt[:, :-1]),
                                  model.decode_logits(memory, padding, tgt[:, :-1])[:, -1:], atol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.pt"
            torch.save(model.state_dict(), path)
            restored = TranslationTransformer(tok.vocab_size(), cfg).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            assert torch.equal(model(src, tgt[:, :-1]), restored(src, tgt[:, :-1]))
    print("PASS: pairs, special IDs, shift, causal/source/target padding, PAD loss, batched decoding, global beam/EOS pruning/refill, EOS retention/stopping, beam=1, final-step EOS limit, full sources, reload")


def check_split_overlap(data_dir, output):
    """Describe exact overlap without altering the official evaluation splits."""
    data = Path(data_dir)
    splits = {name: read_pairs(data / f"{name}.jsonl") for name in ("train", "validation", "test")}
    train_pairs = {(row["en"], row["de"]) for row in splits["train"]}
    train_source = {row["en"] for row in splits["train"]}
    train_target = {row["de"] for row in splits["train"]}
    result = {"definition": "Exact strings in the existing prepared data; official splits remain unchanged. "
              "Repeated short utterances do not by themselves establish talk-level leakage.",
              "training_pairs": len(splits["train"]),
              "files_sha256": {name: sha256(data / f"{name}.jsonl") for name in splits}}
    for name in ("validation", "test"):
        rows = splits[name]
        repeated = [row for row in rows if (row["en"], row["de"]) in train_pairs]
        result[name] = {"samples": len(rows), "exact_pair_overlap_with_train": len(repeated),
                        "source_overlap_with_train": sum(row["en"] in train_source for row in rows),
                        "target_overlap_with_train": sum(row["de"] in train_target for row in rows),
                        "exact_pair_ids": [row["id"] for row in repeated],
                        "exact_pairs_over_5_source_words": sum(len(row["en"].split()) > 5 for row in repeated)}
    write_json(output, result)
    print(f"Saved exact split-overlap audit to {output}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="artifacts/data")
    p.add_argument("--config", default="smoke.json")
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                   default=["train", "validation", "test"],
                   help="corpus splits to inspect; model/decoding checks always run")
    p.add_argument("--overlap-out", help="also audit exact train/validation/test overlap; no data is removed")
    args = p.parse_args()
    run(args.data_dir, args.config, args.splits)
    if args.overlap_out:
        check_split_overlap(args.data_dir, args.overlap_out)


if __name__ == "__main__":
    main()
