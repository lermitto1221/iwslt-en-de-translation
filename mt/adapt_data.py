"""Immutable native-token caches and common raw-ID schedules for adaptation."""
import argparse
import json
import random
from pathlib import Path

from .common import read_json, sha256
from .data import read_pairs
from .pretrained_mt import exclusive_json, now, digest, model_evidence


def make_schedule(ids, seed, updates, effective_batch_size):
    if not ids or len(ids) != len(set(ids)) or min(updates, effective_batch_size) < 1:
        raise ValueError("Schedule needs unique IDs and positive dimensions")
    rng, output, pending = random.Random(seed), [], []
    while len(output) < updates:
        epoch = list(ids)
        rng.shuffle(epoch)
        pending.extend(epoch)
        while len(pending) >= effective_batch_size and len(output) < updates:
            output.append(pending[:effective_batch_size])
            del pending[:effective_batch_size]
    return output


def prepare_data(out, model_dir, data_dir, max_tokens=256, seeds=(7, 42, 43, 44),
                 updates=3000, effective_batch_size=64, micro_batch_size=8):
    from transformers import MarianTokenizer
    if effective_batch_size % micro_batch_size:
        raise ValueError("Effective batch must be divisible by common microbatch")
    out, model_dir, data_dir = Path(out), Path(model_dir), Path(data_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Adaptation data preparation refuses to overwrite existing outputs")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = MarianTokenizer.from_pretrained(model_dir, local_files_only=True)
    manifest = {"created_at": now(), "model_files": model_evidence(model_dir),
                "data_dir": str(data_dir.resolve()), "model_dir": str(model_dir.resolve()),
                "max_train_tokens": max_tokens, "truncation": False,
                "length_definition": "native token sequences, includes EOS, no PAD; filter training only",
                "effective_batch_size": effective_batch_size, "micro_batch_size": micro_batch_size,
                "microbatch_rule": "contiguous chunks of the same saved effective raw-ID batch",
                "updates": updates, "splits": {}, "schedules": {}}
    retained = []
    for split in ("train", "validation", "test"):
        rows = read_pairs(data_dir / f"{split}.jsonl")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate native input IDs")
        encoded_rows, excluded, src_lengths, tgt_lengths = [], [], [], []
        with (out / f"{split}.jsonl").open("x", encoding="utf-8") as f:
            for offset in range(0, len(rows), 256):
                batch = rows[offset:offset + 256]
                encoded = tokenizer([r["en"] for r in batch], text_target=[r["de"] for r in batch],
                                    truncation=False, padding=False)
                for row, source, target in zip(batch, encoded["input_ids"], encoded["labels"]):
                    src_lengths.append(len(source)); tgt_lengths.append(len(target))
                    record = {"id": row["id"], "source_ids": source, "target_ids": target}
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if split == "train":
                        if max(len(source), len(target)) <= max_tokens:
                            retained.append(row["id"])
                        else:
                            excluded.append({"id": row["id"], "source_tokens": len(source), "target_tokens": len(target)})
        if split != "train" and max(src_lengths) > 512:
            raise ValueError("Formal source exceeds native positional capacity; no truncation permitted")
        manifest["splits"][split] = {"samples": len(rows), "input_sha256": sha256(data_dir / f"{split}.jsonl"),
            "cache_sha256": sha256(out / f"{split}.jsonl"), "ids_sha256": digest([r["id"] for r in rows]),
            "source_token_total": sum(src_lengths), "target_token_total": sum(tgt_lengths),
            "source_max": max(src_lengths), "target_max": max(tgt_lengths),
            "source_mean": sum(src_lengths) / len(rows), "target_mean": sum(tgt_lengths) / len(rows)}
        if split == "train":
            exclusive_json(out / "excluded_train.json", excluded)
    exclusive_json(out / "retained_train_ids.json", retained)
    manifest["retained_train_ids_sha256"] = sha256(out / "retained_train_ids.json")
    manifest["excluded_train_sha256"] = sha256(out / "excluded_train.json")
    manifest["retained_train_samples"] = len(retained)
    for seed in seeds:
        schedule = make_schedule(retained, seed, updates, effective_batch_size)
        path = out / f"schedule_seed{seed}.json"
        exclusive_json(path, {"seed": seed, "updates": updates, "effective_batch_size": effective_batch_size,
                              "micro_batch_size": micro_batch_size, "ids": schedule})
        manifest["schedules"][str(seed)] = {"path": path.name, "sha256": sha256(path),
            "sentence_exposures": updates * effective_batch_size, "ids_sha256": digest(schedule)}
    exclusive_json(out / "manifest.json", manifest)
    return manifest


def load_data(data_root, seed, verify=True):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    schedule_path = data_root / manifest["schedules"][str(seed)]["path"]
    if verify:
        for split, item in manifest["splits"].items():
            if sha256(data_root / f"{split}.jsonl") != item["cache_sha256"]:
                raise ValueError("Native token cache changed")
            if sha256(Path(manifest["data_dir"]) / f"{split}.jsonl") != item["input_sha256"]:
                raise ValueError("Native raw data changed")
        if sha256(schedule_path) != manifest["schedules"][str(seed)]["sha256"]:
            raise ValueError("Common effective-batch schedule changed")
        for name, key in (("retained_train_ids.json", "retained_train_ids_sha256"),
                          ("excluded_train.json", "excluded_train_sha256")):
            if sha256(data_root / name) != manifest[key]:
                raise ValueError("Training selection evidence changed")
    rows = [json.loads(line) for line in (data_root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    indexed = {r["id"]: r for r in rows}
    schedule = read_json(schedule_path)
    retained_order = read_json(data_root / "retained_train_ids.json")
    retained = set(retained_order)
    if (schedule.get("seed") != seed or any(schedule.get(k) != manifest[k] for k in
        ("updates", "effective_batch_size", "micro_batch_size"))):
        raise ValueError("Common schedule seed or batch configuration is mislabeled")
    if schedule["ids"] != make_schedule(retained_order, seed, manifest["updates"], manifest["effective_batch_size"]):
        raise ValueError("Saved effective batch IDs do not reproduce from the declared seed")
    if len(indexed) != len(rows) or len(schedule["ids"]) != manifest["updates"]:
        raise ValueError("Native data or schedule shape mismatch")
    for batch in schedule["ids"]:
        if len(batch) != manifest["effective_batch_size"] or any(i not in retained or i not in indexed for i in batch):
            raise ValueError("Scheduled IDs differ from retained training data")
    return manifest, indexed, schedule["ids"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True); p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True); p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=8)
    a = p.parse_args()
    print(json.dumps(prepare_data(a.out, a.model_dir, a.data_dir, a.max_tokens,
                                 micro_batch_size=a.micro_batch_size), indent=2), flush=True)


if __name__ == "__main__":
    main()
