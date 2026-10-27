"""A shared, immutable sentence exposure schedule for tokenizer comparisons.

The budget uses each sentence's larger BPE/unigram lengths. Actual token counts
remain representation specific; sharing sentence IDs does not equate FLOPs.
"""
import hashlib
import json
import random
from pathlib import Path

from .common import read_json, sha256

REPRESENTATIONS = ("bpe", "unigram")


def ids_digest(ids):
    # Same serialization as the historical training data-order digest.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def common_rows(length_rows, max_tokens=128):
    if len({row["id"] for row in length_rows}) != len(length_rows):
        raise ValueError("duplicate original IDs in length table")
    return [row for row in length_rows
            if all(2 <= row[f"{rep}_{side}"] <= max_tokens
                   for rep in REPRESENTATIONS for side in ("src", "tgt"))]


def _epoch_batches(rows, seed, token_budget, pool_size):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order = [index for start in range(0, len(order), pool_size)
             for index in sorted(order[start:start + pool_size],
                                 key=lambda i: max(rows[i][k] for k in
                                     ("bpe_src", "bpe_tgt", "unigram_src", "unigram_tgt")))]
    batch, max_src, max_tgt = [], 0, 0
    for index in order:
        row = rows[index]
        src = max(row[f"{rep}_src"] for rep in REPRESENTATIONS)
        tgt = max(row[f"{rep}_tgt"] for rep in REPRESENTATIONS)
        if src + tgt > token_budget:
            raise ValueError("a sentence exceeds the common padded-token budget")
        if batch and (len(batch) + 1) * (max(max_src, src) + max(max_tgt, tgt)) > token_budget:
            yield batch
            batch, max_src, max_tgt = [], 0, 0
        batch.append(row["id"])
        max_src, max_tgt = max(max_src, src), max(max_tgt, tgt)
    if batch:
        yield batch


def schedule_records(length_rows, seed, max_updates=3000, token_budget=4096,
                     accumulation=4, pool_size=512):
    """Yield updates whose four microbatch boundaries are shared by both arms."""
    if not length_rows or min(max_updates, token_budget, accumulation, pool_size) < 1:
        raise ValueError("nonempty rows and positive schedule settings are required")
    by_id = {row["id"]: row for row in length_rows}
    if len(by_id) != len(length_rows):
        raise ValueError("duplicate original IDs in shared schedule")
    if any(type(row["id"]) is not int or any(type(row[f"{rep}_{side}"]) is not int
               or row[f"{rep}_{side}"] < 2 for rep in REPRESENTATIONS for side in ("src", "tgt"))
           for row in length_rows):
        raise ValueError("schedule requires integer IDs and encoded lengths including BOS/EOS")
    epoch, cursor = 0, 0
    groups = list(_epoch_batches(length_rows, seed, token_budget, pool_size))
    for update in range(1, max_updates + 1):
        micro_batches = []
        for _ in range(accumulation):
            if cursor == len(groups):
                epoch += 1
                cursor = 0
                groups = list(_epoch_batches(length_rows, seed + epoch, token_budget, pool_size))
            micro_batches.append(groups[cursor])
            cursor += 1
        flat = [i for group in micro_batches for i in group]
        yield {"seed": seed, "update": update, "epoch": epoch, "batch_cursor": cursor,
               "micro_batches": micro_batches, "data_order_sha256": ids_digest(flat),
               "microbatch_sha256": ids_digest(micro_batches), "sentences": len(flat),
               "source_tokens": {rep: sum(by_id[i][f"{rep}_src"] for i in flat)
                                 for rep in REPRESENTATIONS},
               "target_tokens": {rep: sum(by_id[i][f"{rep}_tgt"] - 1 for i in flat)
                                 for rep in REPRESENTATIONS},
               "padded_tokens": {rep: sum(len(group) * (max(by_id[i][f"{rep}_src"] for i in group)
                                      + max(by_id[i][f"{rep}_tgt"] for i in group))
                                      for group in micro_batches) for rep in REPRESENTATIONS}}


def load_schedule(cfg, examples):
    """Validate provenance, deterministic boundaries and actual encoded lengths.

    ``examples`` may contain the complete training corpus or exactly the common
    subset. The return preserves all on-disk evidence and adds index aliases for
    the existing collator. No file is modified by this function.
    """
    rep = cfg["representation"]
    if rep not in REPRESENTATIONS:
        raise ValueError("unknown representation")
    for key in ("batch_plan", "common_ids"):
        if sha256(cfg[key]) != cfg[key + "_sha256"]:
            raise ValueError(f"{key} SHA-256 differs from frozen configuration")
    meta = read_json(cfg["common_ids"])
    lengths_path = Path(meta["lengths_file"])
    if not lengths_path.is_absolute():
        lengths_path = Path(cfg["common_ids"]).parent / lengths_path
    if sha256(lengths_path) != meta["lengths_sha256"]:
        raise ValueError("length table SHA-256 differs from common IDs evidence")
    with lengths_path.open(encoding="utf-8") as handle:
        lengths = [json.loads(line) for line in handle]
    retained = common_rows(lengths, meta["max_tokens"])
    ids = [row["id"] for row in retained]
    id_set = set(ids)
    if meta["ids"] != ids or meta["excluded_ids"] != [row["id"] for row in lengths if row["id"] not in id_set]:
        raise ValueError("common ID set does not match both representations' length filter")
    if "data_dir" in cfg:
        data = Path(cfg["data_dir"])
        if sha256(data / "train.jsonl") != meta["source_train_sha256"]:
            raise ValueError("schedule belongs to a different training corpus")
        if sha256(data / "spm.model") != meta["tokenizer_sha256"][rep]:
            raise ValueError("schedule belongs to a different tokenizer")
    mapping = {item[2]: index for index, item in enumerate(examples)}
    if len(mapping) != len(examples) or not id_set.issubset(mapping):
        raise ValueError("encoded examples have duplicate IDs or omit a common sentence")
    for row in retained:
        src, tgt, _ = examples[mapping[row["id"]]]
        if (len(src), len(tgt)) != (row[f"{rep}_src"], row[f"{rep}_tgt"]):
            raise ValueError("encoded lengths differ from the frozen shared schedule")
    budget = cfg.get("batch_tokens", 4096)
    accumulation = cfg.get("accumulation_steps", cfg.get("gradient_accumulation", 4))
    # The established training configuration uses accumulation_steps.
    if "accum_steps" in cfg:
        accumulation = cfg["accum_steps"]
    expected = schedule_records(retained, cfg["seed"], cfg["max_updates"], budget,
                                accumulation, meta.get("pool_size", 512))
    result = []
    with Path(cfg["batch_plan"]).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            reference = next(expected, None)
            if reference is None or row != reference:
                raise ValueError(f"shared batch schedule differs at update {index + 1}")
            indices = [[mapping[i] for i in group] for group in row["micro_batches"]]
            result.append({**row, "indices": indices, "microbatches": indices,
                           "microbatch_ids": row["micro_batches"]})
    if next(expected, None) is not None:
        raise ValueError("shared batch schedule is incomplete")
    return result
