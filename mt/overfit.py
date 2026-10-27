import argparse
import json
from pathlib import Path

import sacrebleu
import torch
import torch.nn.functional as F

from .batching import collate, tokenize_pairs
from .common import read_json, seed_all, write_json
from .data import read_pairs
from .decoding import translate
from .model import TranslationTransformer
from .tokenizer import PAD, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="smoke.json")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--out", default="artifacts/overfit.json")
    args = p.parse_args()
    cfg = read_json(args.config)
    seed_all(cfg["seed"])
    tok = load(cfg["data_dir"])
    rows = read_pairs(Path(cfg["data_dir"]) / "train.jsonl")[:16]
    examples, excluded = tokenize_pairs(rows, tok, cfg["max_train_tokens"])
    assert len(examples) >= 8, f"too many selected pairs filtered: {excluded}"
    by_id = {row["id"]: row for row in rows}
    rows = [by_id[e[2]] for e in examples]
    requested = cfg.get("device", "auto")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    tiny = dict(cfg, dropout=0.0, label_smoothing=0.0)
    model = TranslationTransformer(tok.vocab_size(), tiny).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    src, tgt, _ = collate(examples, list(range(len(examples))), device)
    for _ in range(args.updates):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    preds = [translate(model, tok, row["en"], "greedy", max_tokens=cfg["max_generation_tokens"])[0] for row in rows]
    metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    score = metric.corpus_score(preds, [[r["de"] for r in rows]]).score
    result = {"pairs": len(rows), "updates": args.updates, "final_teacher_forced_loss": float(loss.detach()),
              "training_set_greedy_bleu": score, "signature": str(metric.get_signature()),
              "note": "diagnostic on memorized training pairs; not a validation or test result"}
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
