"""Prepare the frozen BPE/unigram experiment without changing historical data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import sentencepiece as spm

from .common import read_json, sha256
from .data import read_pairs
from .representation_batches import REPRESENTATIONS, common_rows, schedule_records
from .tokenizer import BOS, EOS, PAD, UNK, encode

DEFAULT_ROOT = "artifacts/representation_2026_10_27"


def _immutable_bytes(path, content):
    """Publish a complete file without replacing any prior content."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-link creation is atomic and fails if the target already exists.
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise FileExistsError(f"concurrent different evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_json(path, value):
    _immutable_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _verify_files(root, manifest):
    for relative, digest in manifest["files"].items():
        path = Path(root) / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"completed preparation evidence is missing or changed: {path}")


def trainer_options(representation, corpus, prefix):
    return {"input": str(corpus), "model_prefix": str(prefix),
            "model_type": representation, "vocab_size": 8000,
            "pad_id": PAD, "unk_id": UNK, "bos_id": BOS, "eos_id": EOS,
            "character_coverage": 1.0, "normalization_rule_name": "nmt_nfkc",
            "input_sentence_size": 0, "shuffle_input_sentence": False,
            "num_threads": 1, "hard_vocab_limit": True,
            "split_by_whitespace": True, "split_by_unicode_script": True,
            "split_by_number": True, "split_digits": False, "byte_fallback": False,
            "add_dummy_prefix": True, "remove_extra_whitespaces": True,
            "escape_whitespaces": True, "max_sentence_length": 4192}


def _tokenizer(root, source, representation, input_hashes, rows):
    target = root / "data" / representation
    if target.exists():
        marker = target / "preparation_complete.json"
        if not marker.is_file():
            raise ValueError(f"existing tokenizer directory lacks completion evidence: {target}")
        complete = read_json(marker)
        if complete["input_hashes"] != input_hashes or complete["representation"] != representation:
            raise ValueError("existing tokenizer was prepared from different inputs")
        _verify_files(target, complete)
        metadata = read_json(target / "tokenizer.json")
        expected_options = trainer_options(representation, metadata["trainer_arguments"]["input"],
                                           metadata["trainer_arguments"]["model_prefix"])
        if metadata["trainer_arguments"] != expected_options or metadata["actual_vocab_size"] != 8000:
            raise ValueError("existing tokenizer training parameters differ")
        return spm.SentencePieceProcessor(model_file=str(target / "spm.model"))
    # Interrupted attempts remain explicitly partial and are never promoted.
    stage = root / "partial" / (representation + "-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    for name in input_hashes:
        shutil.copyfile(source / name, stage / name)
    corpus = stage / "spm_train.txt"
    longest_bytes = 0
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            for language in ("en", "de"):
                line = row[language].replace("\n", " ")
                longest_bytes = max(longest_bytes, len(line.encode("utf-8")))
                handle.write(line + "\n")
    options = trainer_options(representation, corpus.resolve(), (stage / "spm").resolve())
    if longest_bytes > options["max_sentence_length"]:
        raise ValueError("tokenizer max_sentence_length would silently exclude training text")
    print(json.dumps({"preparation_stage": "fit_tokenizer", "representation": representation,
                      "training_pairs": len(rows), "started_at_utc": _timestamp()}), flush=True)
    with (stage / "sentencepiece_training.log").open("wb") as log:
        process = subprocess.run([sys.executable, "-S", "-c",
            "import json,sys,sentencepiece as s;s.SentencePieceTrainer.train(**json.loads(sys.argv[1]))",
            json.dumps(options)], stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"SentencePiece {representation} failed; partial evidence retained at {stage}")
    processor = spm.SentencePieceProcessor(model_file=str(stage / "spm.model"))
    if processor.vocab_size() != 8000:
        raise ValueError("actual tokenizer vocabulary is not 8000; stop and amend the design before training")
    if [processor.pad_id(), processor.unk_id(), processor.bos_id(), processor.eos_id()] != [0, 1, 2, 3]:
        raise ValueError("special IDs differ from the frozen tokenizer design")
    log_text = (stage / "sentencepiece_training.log").read_text(encoding="utf-8", errors="replace")
    spec_start, spec_end = log_text.find("trainer_spec {"), log_text.find("denormalizer_spec {}")
    resolved = log_text[spec_start:spec_end + len("denormalizer_spec {}")] if spec_start >= 0 and spec_end >= 0 else log_text
    metadata = {"model_type": representation, "requested_vocab_size": 8000,
        "actual_vocab_size": processor.vocab_size(), "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "training_pairs": len(rows), "training_file_sha256": input_hashes["train.jsonl"],
        "model_sha256": sha256(stage / "spm.model"), "vocab_sha256": sha256(stage / "spm.vocab"),
        "sentencepiece_version": importlib.metadata.version("sentencepiece"),
        "training_corpus_sha256": sha256(corpus), "training_corpus_lines": 2 * len(rows),
        "corpus_order": "original cleaned training row order; English then German per row",
        "longest_input_utf8_bytes": longest_bytes, "trainer_arguments": options,
        "resolved_trainer_normalizer_spec": resolved,
        "training_log_sha256": sha256(stage / "sentencepiece_training.log"),
        "path_note": "trainer input/model_prefix record the original partial staging directory; published files are here"}
    _immutable_json(stage / "tokenizer.json", metadata)
    complete = {"schema_version": 1, "representation": representation, "input_hashes": input_hashes,
                "created_at_utc": _timestamp(), "files": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}}
    _immutable_json(stage / "preparation_complete.json", complete)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return spm.SentencePieceProcessor(model_file=str(target / "spm.model"))


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "min": float(array.min()),
            "p50": float(np.quantile(array, .5)), "p90": float(np.quantile(array, .9)),
            "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99)), "max": float(array.max())}


def _encode_statistics(rows, tokenizers):
    lengths, stats = [], {}
    values = {rep: {side: {"lengths": [], "fragmentation": [], "tokens": 0, "unknown": 0, "words": 0}
                        for side in ("src", "tgt")} for rep in REPRESENTATIONS}
    for row in rows:
        record = {"id": row["id"]}
        for rep, tok in tokenizers.items():
            for side, language in (("src", "en"), ("tgt", "de")):
                ids = encode(tok, row[language])
                words = len(row[language].split())
                if not words:
                    raise ValueError("empty whitespace word sequence in cleaned corpus")
                record[f"{rep}_{side}"] = len(ids)
                v = values[rep][side]
                v["lengths"].append(len(ids))
                v["fragmentation"].append((len(ids) - 2) / words)
                v["tokens"] += len(ids) - 2
                v["unknown"] += ids.count(UNK)
                v["words"] += words
        lengths.append(record)
    for rep in REPRESENTATIONS:
        stats[rep] = {}
        for side, value in values[rep].items():
            stats[rep][side] = {"encoded_length_including_bos_eos": _distribution(value["lengths"]),
                "per_sentence_pieces_per_whitespace_word": _distribution(value["fragmentation"]),
                "content_tokens": value["tokens"], "whitespace_words": value["words"],
                "corpus_pieces_per_whitespace_word": value["tokens"] / value["words"],
                "unk_tokens": value["unknown"], "unk_content_token_fraction": value["unknown"] / value["tokens"]}
    return lengths, stats


def _validate_design(design):
    expected = {"vocab_size": 8000, "character_coverage": 1.0, "normalization_rule_name": "nmt_nfkc",
                "input_sentence_size": 0, "shuffle_input_sentence": False, "num_threads": 1,
                "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}}
    if (design["representations"] != list(REPRESENTATIONS) or design["seeds"] != [42, 43, 44]
            or design["max_updates"] != 3000 or design["batching"]["token_budget"] != 4096
            or design["batching"]["accum_steps"] != 4
            or any(design["tokenizers"].get(k) != v for k, v in expected.items())):
        raise ValueError("design differs from this implementation's fixed experiment settings")


def prepare(root=DEFAULT_ROOT, source="artifacts/data"):
    root, source = Path(root), Path(source)
    design = read_json(root / "design.json")
    _validate_design(design)
    original = read_json(source / "manifest.json")
    if original.get("train_limit") is not None:
        raise ValueError("representation tokenizers must fit the full cleaned training corpus")
    inputs = {name: sha256(source / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")}
    for split in ("train", "validation", "test"):
        if inputs[split + ".jsonl"] != original["files"][split]["jsonl_sha256"]:
            raise ValueError("source data differs from the historical manifest")
    request = {"schema_version": 1, "design_sha256": sha256(root / "design.json"), "input_hashes": inputs,
               "source_directory": str(source.resolve()), "sentencepiece_version": importlib.metadata.version("sentencepiece"),
               "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
                                         for name in ("representation_prepare.py", "representation_batches.py")}}
    _immutable_json(root / "preparation_request.json", request)
    completed = root / "prepare_manifest.json"
    if completed.exists():
        result = read_json(completed)
        if result["request"] != request:
            raise ValueError("completed preparation belongs to a different request")
        _verify_files(root, result)
        return result
    rows = {split: read_pairs(source / (split + ".jsonl")) for split in ("train", "validation", "test")}
    for split, values in rows.items():
        if len(values) != original["files"][split]["rows_saved"] or len({r["id"] for r in values}) != len(values):
            raise ValueError("saved source row counts or unique IDs differ from manifest")
    tokenizers = {rep: _tokenizer(root, source, rep, inputs, rows["train"]) for rep in REPRESENTATIONS}
    all_stats, train_lengths = {}, None
    for split, values in rows.items():
        print(json.dumps({"preparation_stage": "encode_statistics", "split": split, "rows": len(values)}), flush=True)
        length_rows, all_stats[split] = _encode_statistics(values, tokenizers)
        _immutable_bytes(root / f"{split}_lengths.jsonl", _jsonl_bytes(length_rows))
        if split == "train":
            train_lengths = length_rows
    retained = common_rows(train_lengths)
    ids = [row["id"] for row in retained]
    keep = set(ids)
    metadata = {"schema_version": 1, "ids": ids,
        "excluded_ids": [row["id"] for row in train_lengths if row["id"] not in keep],
        "max_tokens": 128, "pool_size": 512, "lengths_file": "train_lengths.jsonl",
        "lengths_sha256": sha256(root / "train_lengths.jsonl"), "source_train_sha256": inputs["train.jsonl"],
        "tokenizer_sha256": {rep: sha256(root / "data" / rep / "spm.model") for rep in REPRESENTATIONS},
        "filter": "both source and target encoded lengths including BOS/EOS <=128 for both representations",
        "order": "original cleaned train row order"}
    _immutable_json(root / "common_ids.json", metadata)
    all_stats["common_training_set"] = {"cleaned_pairs": len(train_lengths), "retained_pairs": len(ids),
        "excluded_pairs": len(train_lengths) - len(ids),
        "excluded_by_representation": {rep: sum(max(row[f"{rep}_src"], row[f"{rep}_tgt"]) > 128
                                                for row in train_lengths) for rep in REPRESENTATIONS},
        "definitions": {"words": "Python str.split whitespace tokens, case and punctuation unchanged",
                        "pieces": "SentencePiece IDs excluding BOS/EOS/PAD; UNK included and reported",
                        "length": "source/target encoded lengths include BOS and EOS"}}
    _immutable_json(root / "representation_statistics.json", all_stats)
    schedules = {}
    for seed in (42, 43, 44):
        records = list(schedule_records(retained, seed))
        path = root / "batches" / f"seed{seed}.jsonl"
        _immutable_bytes(path, _jsonl_bytes(records))
        schedules[str(seed)] = {"file": str(path), "sha256": sha256(path), "updates": len(records),
            "sentences": sum(record["sentences"] for record in records),
            "final_epoch": records[-1]["epoch"],
            "source_tokens": {rep: sum(record["source_tokens"][rep] for record in records) for rep in REPRESENTATIONS},
            "target_tokens": {rep: sum(record["target_tokens"][rep] for record in records) for rep in REPRESENTATIONS},
            "padded_tokens": {rep: sum(record["padded_tokens"][rep] for record in records) for rep in REPRESENTATIONS}}
    published = [p for directory in (root / "data", root / "batches") for p in directory.rglob("*") if p.is_file()]
    published += [root / name for name in ("preparation_request.json", "common_ids.json", "representation_statistics.json",
                                          "train_lengths.jsonl", "validation_lengths.jsonl", "test_lengths.jsonl")]
    result = {"schema_version": 1, "complete": True, "completed_at_utc": _timestamp(), "request": request,
              "common_training_pairs": len(ids), "excluded_training_pairs": len(train_lengths) - len(ids),
              "batch_schedules": schedules,
              "files": {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sorted(published)}}
    _immutable_json(completed, result)
    print(json.dumps({"preparation_stage": "complete", "common_pairs": len(ids), "manifest": str(completed)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--source", default="artifacts/data")
    args = parser.parse_args()
    prepare(args.root, args.source)


if __name__ == "__main__":
    main()
