import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from .common import sha256, write_json


REVISION = "3fc158c8125b4d798f2696c0bb5c8928a8fbb332"
BASE = f"https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/{REVISION}/iwslt2017-en-de"
EXPECTED = {"train": 206112, "validation": 888, "test": 8079}
# SHA-256 values measured from the downloaded files at the immutable revision above.
PARQUET_SHA256 = {
    "train": "39f518537ee06492a56f01366044454f14fd8dc50434f74181905dead0b4bd2a",
    "validation": "fdaa92327a2780b7c2bd44768870aafbd1fa4e3eda4312292319bf06088f059a",
    "test": "3b06faa5a242e8588f82244cf06753c3788520b2fd109f8cb513b70ee7234b3d",
}
ORIGINS = {
    "train": ["train.tags.en-de.en", "train.tags.en-de.de"],
    "validation": ["IWSLT17.TED.dev2010.en-de.en.xml", "IWSLT17.TED.dev2010.en-de.de.xml"],
    "test": [f"IWSLT17.TED.tst{y}.en-de.{lang}.xml" for y in range(2010, 2016) for lang in ("en", "de")],
}


def prepare(out, train_limit=None, seed=42, max_words=300, max_ratio=3.0):
    import pyarrow.parquet as pq
    if max_words < 1 or max_ratio < 1:
        raise ValueError("max_words and max_ratio must be positive length limits")
    out = Path(out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "IWSLT/iwslt2017", "configuration": "iwslt2017-en-de",
                "parquet_revision": REVISION, "source_builder": "https://huggingface.co/datasets/IWSLT/iwslt2017/blob/main/iwslt2017.py",
                "files": {}, "cleaning": {}, "train_limit": train_limit, "subset_seed": seed,
                "cleaning_rules": {
                    "all_splits": {"strip_outer_whitespace": True, "require_aligned_string_pairs": True},
                    "train": {"order": ["empty", "duplicate_pair", "long_words", "length_ratio"],
                              "deduplicate": "exact English/German pair after stripping outer whitespace",
                              "word_count": "Python str.split() whitespace-separated tokens",
                              "max_words_per_side": max_words, "max_length_ratio": max_ratio,
                              "length_ratio": "max(source_words, target_words) / min(source_words, target_words)",
                              "counts": "exclusive first-match removals before optional fixed-seed sampling"},
                    "validation_and_test": {"empty_pair": "error", "filter_length": False,
                                            "deduplicate": False, "truncate": False}}}
    for split in EXPECTED:
        name = f"iwslt2017-{split}.parquet"
        path = raw / name
        url = f"{BASE}/{name}"
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, temporary)
                if sha256(temporary) != PARQUET_SHA256[split]:
                    raise ValueError(f"downloaded {split} Parquet SHA-256 does not match revision {REVISION}")
                temporary.replace(path)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"IWSLT-17 download failed: {url}. No substitute corpus is used.") from exc
        parquet_hash = sha256(path)
        if parquet_hash != PARQUET_SHA256[split]:
            raise ValueError(f"cached {split} Parquet failed SHA-256 verification: {path}; "
                             "remove the corrupt cache file and rerun data preparation")
        table = pq.read_table(path, columns=["translation"])
        if table.num_rows != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} rows, got {table.num_rows}")
        rows = table.column("translation").to_pylist()
        pairs = []
        counts = Counter()
        seen = set()
        for idx, row in enumerate(rows):
            if isinstance(row, str):
                row = json.loads(row)
            if not isinstance(row, dict) or "en" not in row or "de" not in row:
                raise ValueError(f"unaligned row in {split} at {idx}")
            en, de = row["en"], row["de"]
            if not isinstance(en, str) or not isinstance(de, str):
                raise ValueError(f"non-string pair in {split} at {idx}")
            en, de = en.strip(), de.strip()
            if not en or not de:
                counts["empty"] += 1
                if split != "train":
                    raise ValueError(f"empty evaluation pair in {split} at {idx}; refusing silent removal")
                continue
            if split == "train":
                key = (en, de)
                if key in seen:
                    counts["duplicate_pair"] += 1
                    continue
                seen.add(key)
                a, b = len(en.split()), len(de.split())
                if max(a, b) > max_words:
                    counts["long_words"] += 1
                    continue
                if max(a, b) / max(1, min(a, b)) > max_ratio:
                    counts["length_ratio"] += 1
                    continue
            pairs.append({"id": idx, "en": en, "de": de})
        if split == "train" and train_limit is not None:
            if train_limit <= 0 or train_limit > len(pairs):
                raise ValueError("train_limit outside cleaned training size")
            selected = sorted(random.Random(seed).sample(range(len(pairs)), train_limit))
            write_json(out / "train_indices.json", {"cleaned_positions": selected, "original_ids": [pairs[i]["id"] for i in selected]})
            pairs = [pairs[i] for i in selected]
        output = out / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        manifest["files"][split] = {"url": url, "parquet_sha256": parquet_hash,
            "rows_raw": table.num_rows, "rows_saved": len(pairs), "jsonl_sha256": sha256(output),
            "original_files": ORIGINS[split]}
        manifest["cleaning"][split] = {key: counts[key] for key in
            ("empty", "duplicate_pair", "long_words", "length_ratio")}
    write_json(out / "manifest.json", manifest)
    return manifest


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/data")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(prepare(args.out, args.train_limit, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
