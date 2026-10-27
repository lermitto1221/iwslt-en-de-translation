import random

import torch

from .tokenizer import PAD, encode


def tokenize_pairs(rows, tok, max_train_tokens=None):
    examples = []
    excluded = 0
    for row in rows:
        src, tgt = encode(tok, row["en"]), encode(tok, row["de"])
        if max_train_tokens and max(len(src), len(tgt)) > max_train_tokens:
            excluded += 1
            continue
        examples.append((src, tgt, row["id"]))
    return examples, excluded


def batches(examples, token_budget, seed=None, shuffle=False):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
        pools = [order[i:i + 512] for i in range(0, len(order), 512)]
        order = [j for pool in pools for j in sorted(pool, key=lambda j: max(len(examples[j][0]), len(examples[j][1])))]
    else:
        order.sort(key=lambda j: max(len(examples[j][0]), len(examples[j][1])))
    batch, max_src, max_tgt = [], 0, 0
    for index in order:
        src, tgt, _ = examples[index]
        new_src, new_tgt = max(max_src, len(src)), max(max_tgt, len(tgt))
        # Budget counts padded encoder tokens plus padded decoder tokens.
        cost = (len(batch) + 1) * (new_src + new_tgt)
        if batch and cost > token_budget:
            yield batch
            batch, max_src, max_tgt = [], 0, 0
        batch.append(index)
        max_src, max_tgt = max(max_src, len(src)), max(max_tgt, len(tgt))
    if batch:
        yield batch


def collate(examples, indices, device):
    items = [examples[i] for i in indices]
    src = torch.full((len(items), max(len(x[0]) for x in items)), PAD, dtype=torch.long)
    tgt = torch.full((len(items), max(len(x[1]) for x in items)), PAD, dtype=torch.long)
    for i, (s, t, _) in enumerate(items):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src.to(device), tgt.to(device), [x[2] for x in items]
