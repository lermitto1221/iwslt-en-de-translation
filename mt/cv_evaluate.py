"""Resumable full-fold/final evaluation with immutable batch evidence.

Each atomic batch file links to the preceding file's content digest. Progress
may lag one committed batch after interruption. Finished results are verified
and reused read-only; no shortened fold or silently truncated source is used.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import sacrebleu
import torch

from .common import environment, read_json, sha256
from .data import read_pairs
from .decoding import generate_batch_ids
from .inference import build_model, resolve_data_dir, select_device, tokenizer_fingerprint, validate_protocol
from .tokenizer import EOS, decode, encode, load


INFERENCE_SOURCES = ("cv_evaluate.py", "inference.py", "decoding.py", "model.py", "tokenizer.py", "common.py", "data.py")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _immutable_bytes(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError("existing output differs from journal: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def corpus_metrics(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("nonempty aligned predictions/references required")
    bleu = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    chrf = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2, lowercase=False,
                                whitespace=False, eps_smoothing=False)
    return {"bleu": bleu.corpus_score(predictions, [references]).score,
            "chrf2": chrf.corpus_score(predictions, [references]).score,
            "signature": str(bleu.get_signature()), "chrf2_signature": str(chrf.get_signature()),
            "sacrebleu_version": sacrebleu.__version__}


def _load_inputs(checkpoint, split, method, protocol, batch_size, device):
    if method not in ("greedy", "beam") or type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid method or inference batch size")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if (state.get("checkpoint_selection") != "fixed_last"
            or cfg.get("checkpoint_selection") != "fixed_last"
            or state.get("update") != cfg["max_updates"]
            or Path(checkpoint).name != "last.pt"):
        raise ValueError("evaluation requires the fixed final last.pt checkpoint")
    if cfg.get("stage") == "cv":
        if split != "holdout" or method != "greedy":
            raise ValueError("CV selection evaluates only the complete holdout with greedy")
    elif cfg.get("stage") != "final" or split not in ("validation", "test"):
        raise ValueError("final retraining evaluates only official validation/test")
    tensor_digest = hashlib.sha256()
    for name, tensor in state["model"].items():
        tensor_digest.update(name.encode())
        tensor_digest.update(tensor.detach().cpu().numpy().tobytes())
    if tensor_digest.hexdigest() != state.get("model_sha256"):
        raise ValueError("checkpoint tensor digest differs")
    data = resolve_data_dir(checkpoint, cfg, tokenizer_fingerprint(state))
    manifest = read_json(data / "manifest.json")
    metadata = read_json(data / "tokenizer.json")
    if (sha256(data / "manifest.json") != state["fingerprints"]["manifest_sha256"]
            or sha256(data / "tokenizer.json") != state["fingerprints"]["tokenizer_metadata_sha256"]
            or sha256(data / "train.jsonl") != state["fingerprints"]["splits"]["train"]
            or metadata["training_file_sha256"] != state["fingerprints"]["splits"]["train"]):
        raise ValueError("checkpoint data/tokenizer provenance differs")
    rows_path = data / (split + ".jsonl")
    row_hash = sha256(rows_path)
    if row_hash != manifest["files"][split]["jsonl_sha256"]:
        raise ValueError("evaluation rows differ from manifest")
    rows = read_pairs(rows_path)
    if (not rows or len(rows) != manifest["files"][split]["rows_saved"]
            or len({row["id"] for row in rows}) != len(rows)
            or any(type(row["id"]) is not int or not row["en"].strip() or not row["de"].strip()
                   or "\n" in row["en"] or "\r" in row["en"]
                   or "\n" in row["de"] or "\r" in row["de"] for row in rows)):
        raise ValueError("evaluation IDs/counts/texts are invalid")
    selected = read_json(protocol) if protocol is not None else cfg
    settings = validate_protocol(selected)
    expected_settings = validate_protocol(cfg)
    if settings != expected_settings:
        raise ValueError("evaluation protocol differs from frozen training configuration")
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    runtime = environment()
    identity = {"schema_version": 1, "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": sha256(checkpoint), "checkpoint_update": state["update"],
                "model_sha256": state["model_sha256"], "config_sha256": _digest(cfg),
                "split": split, "method": method, "samples": len(rows),
                "input_path": str(rows_path.resolve()), "input_sha256": row_hash,
                "ids_sha256": _digest([row["id"] for row in rows]),
                "tokenizer_sha256": tokenizer_fingerprint(state),
                "protocol": settings, "protocol_file_sha256": sha256(protocol) if protocol else None,
                "batch_size": batch_size, "precision": "float32", "device": str(device),
                "environment": runtime, "source_hashes": {name: sha256(Path(__file__).with_name(name))
                                                              for name in INFERENCE_SOURCES}}
    return rows, cfg, state, load(data), identity


def _read_journal(out, identity, rows):
    folder = out / "batches"
    files = sorted(folder.glob("batch_*.json")) if folder.exists() else []
    previous = _digest(identity)
    records = []
    size = identity["batch_size"]
    for index, path in enumerate(files):
        if path.name != f"batch_{index:06d}.json":
            raise ValueError("evaluation journal has a missing/noncontinuous batch")
        try:
            saved = read_json(path)
            payload = {key: value for key, value in saved.items() if key != "sha256"}
            if (saved.get("sha256") != _digest(payload) or saved.get("previous_sha256") != previous
                    or saved.get("index") != index or saved.get("offset") != index * size):
                raise ValueError("evaluation batch checksum/chain differs")
            chunk = rows[index * size:(index + 1) * size]
            if not chunk or len(saved["examples"]) != len(chunk):
                raise ValueError("evaluation batch count differs")
            for row, example in zip(chunk, saved["examples"]):
                if (example.get("id") != row["id"] or example.get("source") != row["en"]
                        or example.get("reference") != row["de"]
                        or not isinstance(example.get("prediction"), str)
                        or "\n" in example["prediction"] or "\r" in example["prediction"]
                        or type(example.get("generation_limit")) is not bool
                        or type(example.get("unfinished")) is not bool):
                    raise ValueError("evaluation batch ID/source/reference/prediction alignment differs")
            if not math.isfinite(saved["seconds"]) or saved["seconds"] <= 0:
                raise ValueError("invalid evaluation batch timing")
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("damaged committed evaluation batch: " + str(path)) from error
        records.append(saved)
        previous = saved["sha256"]
    progress_path = out / "progress.json"
    if progress_path.exists():
        progress = read_json(progress_path)
        count = progress.get("committed_batches")
        if type(count) is not int or not max(0, len(records) - 1) <= count <= len(records):
            raise ValueError("evaluation progress differs from committed journal")
        expected = records[count - 1]["sha256"] if count else _digest(identity)
        if (progress.get("identity_sha256") != _digest(identity)
                or progress.get("last_batch_sha256") != expected
                or progress.get("samples_completed") != min(count * size, len(rows))):
            raise ValueError("evaluation progress checksum differs from journal")
    elif records:
        raise ValueError("evaluation progress is missing for committed journal")
    return records


def _progress(out, identity, records):
    value = {"identity_sha256": _digest(identity), "committed_batches": len(records),
             "samples_completed": sum(len(record["examples"]) for record in records),
             "expected_samples": identity["samples"],
             "last_batch_sha256": records[-1]["sha256"] if records else _digest(identity),
             "complete": sum(len(record["examples"]) for record in records) == identity["samples"]}
    _atomic_json(out / "progress.json", value)
    return value


def _derived_metrics(identity, records):
    examples = [example for record in records for example in record["examples"]]
    if len(examples) != identity["samples"]:
        raise ValueError("incomplete journal cannot produce final metrics")
    seconds = sum(record["seconds"] for record in records)
    hits = sum(example["generation_limit"] for example in examples)
    unfinished = sum(example["unfinished"] for example in examples)
    peaks = [record["peak_cuda_bytes"] for record in records if record["peak_cuda_bytes"] is not None]
    return {**corpus_metrics([example["prediction"] for example in examples],
                            [example["reference"] for example in examples]),
            "complete": True, "samples": len(examples), "split": identity["split"],
            "method": identity["method"], "checkpoint": identity["checkpoint"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "checkpoint_update": identity["checkpoint_update"],
            "input_sha256": identity["input_sha256"], "identity_sha256": _digest(identity),
            "journal_final_sha256": records[-1]["sha256"],
            "generation_limit_hits": hits, "generation_limit_fraction": hits / len(examples),
            "unfinished_generations": unfinished, "unfinished_fraction": unfinished / len(examples),
            "elapsed_seconds": seconds, "sentences_per_second": len(examples) / seconds,
            "mean_latency_ms": 1000 * seconds / len(examples),
            "mean_batch_latency_ms": 1000 * seconds / len(records),
            "inference_batch_size": identity["batch_size"], "inference_precision": "float32",
            "beam_size": 1 if identity["method"] == "greedy" else identity["protocol"]["beam_size"],
            "length_penalty": identity["protocol"]["length_penalty"],
            "max_generation_tokens": identity["protocol"]["max_generation_tokens"],
            "peak_cuda_bytes": max(peaks) if peaks else None, "warmup_batches_per_invocation": 2,
            "timing": "sum of synchronized batch service times, including full source tokenization and target decoding; excludes model loading, two warmup batches per invocation, journal writes and restart idle time; mean_latency_ms is amortized per sentence, mean_batch_latency_ms is batch service latency",
            "generation_limit_definition": "selected output generated max_generation_tokens tokens including final-step EOS; BOS excluded",
            "environment": identity["environment"]}


def _output_bytes(records):
    examples = [example for record in records for example in record["examples"]]
    return {"predictions.txt": ("\n".join(example["prediction"] for example in examples) + "\n").encode("utf-8"),
            "references.txt": ("\n".join(example["reference"] for example in examples) + "\n").encode("utf-8"),
            "examples.jsonl": "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples).encode("utf-8")}


def verify_results(out, rows=None):
    """Read-only corpus rescore and full journal/text/ID verification."""
    out = Path(out)
    identity = read_json(out / "identity.json")
    source = Path(identity["input_path"])
    if sha256(source) != identity["input_sha256"]:
        raise ValueError("evaluation input hash differs")
    actual_rows = read_pairs(source)
    if rows is not None and rows != actual_rows:
        raise ValueError("evaluation input rows differ")
    rows = actual_rows
    if (len(rows) != identity["samples"] or len({row["id"] for row in rows}) != len(rows)
            or _digest([row["id"] for row in rows]) != identity["ids_sha256"]):
        raise ValueError("evaluation input IDs/counts differ")
    records = _read_journal(out, identity, rows)
    progress = read_json(out / "progress.json")
    if (progress.get("complete") is not True or progress.get("committed_batches") != len(records)
            or progress.get("samples_completed") != len(rows)):
        raise ValueError("completed evaluation has inconsistent progress")
    expected = _derived_metrics(identity, records)
    for name, value in _output_bytes(records).items():
        if not (out / name).is_file() or (out / name).read_bytes() != value:
            raise ValueError("saved evaluation text differs from verified journal: " + name)
        expected[name.removesuffix(".txt").removesuffix(".jsonl") + "_sha256"] = sha256(out / name)
    actual = read_json(out / "metrics.json")
    if actual != expected:
        raise ValueError("saved metrics differ from independent corpus recomputation")
    return actual


def evaluate_checkpoint(checkpoint, split, method, out, protocol=None, decode_batch_size=16,
                        device="auto", stop_after_batches=None):
    if stop_after_batches is not None and (type(stop_after_batches) is not int or stop_after_batches < 1):
        raise ValueError("stop_after_batches must be a positive integer")
    device = select_device(device)
    rows, cfg, state, tok, identity = _load_inputs(checkpoint, split, method, protocol, decode_batch_size, device)
    out = Path(out)
    if (out / "identity.json").exists():
        if read_json(out / "identity.json") != identity:
            raise ValueError("evaluation identity differs; use the original inputs/settings or a new directory")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("output directory has no valid evaluation identity")
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / "identity.json", identity)
        _progress(out, identity, [])
    records = _read_journal(out, identity, rows)
    if (out / "metrics.json").exists():
        return verify_results(out, rows)
    first = len(records)
    total = (len(rows) + decode_batch_size - 1) // decode_batch_size
    if first < total:
        model = build_model(cfg, state["model"], tok, device)
        del state

        @torch.inference_mode()
        def run_batch(chunk):
            sources = [encode(tok, row["en"]) for row in chunk]
            # Let the encoder's explicit positional-capacity error surface;
            # never slice long inputs to fit a preselected maximum.
            with torch.autocast(device_type=device.type, enabled=False):
                sequences, limits = generate_batch_ids(model, sources, method,
                    identity["protocol"]["beam_size"], identity["protocol"]["length_penalty"],
                    identity["protocol"]["max_generation_tokens"])
            if len(sequences) != len(chunk) or len(limits) != len(chunk) or any(not seq for seq in sequences):
                raise ValueError("decoder returned incomplete batch")
            return [{"id": row["id"], "source": row["en"], "reference": row["de"],
                     "prediction": decode(tok, sequence), "generation_limit": bool(limit),
                     "unfinished": sequence[-1] != EOS}
                    for row, sequence, limit in zip(chunk, sequences, limits)]

        warmup_rows = rows[first * decode_batch_size:(first + 1) * decode_batch_size]
        for _ in range(2):
            run_batch(warmup_rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        target = min(total, first + stop_after_batches) if stop_after_batches else total
        for index in range(first, target):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            chunk = rows[index * decode_batch_size:(index + 1) * decode_batch_size]
            examples = run_batch(chunk)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            payload = {"index": index, "offset": index * decode_batch_size,
                       "previous_sha256": records[-1]["sha256"] if records else _digest(identity),
                       "examples": examples, "seconds": seconds,
                       "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            saved = {**payload, "sha256": _digest(payload)}
            path = out / "batches" / f"batch_{index:06d}.json"
            if path.exists():
                raise FileExistsError("refusing to replace a committed batch")
            _atomic_json(path, saved)
            records.append(saved)
            progress = _progress(out, identity, records)
            if (index + 1) % 128 == 0 or index + 1 == target:
                print(json.dumps({"decode_progress": progress["samples_completed"],
                                  "samples": len(rows), "method": method}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(records) < total:
        return {"complete": False, "samples_completed": sum(len(record["examples"]) for record in records),
                "expected_samples": len(rows)}
    metrics = _derived_metrics(identity, records)
    for name, value in _output_bytes(records).items():
        _immutable_bytes(out / name, value)
        metrics[name.removesuffix(".txt").removesuffix(".jsonl") + "_sha256"] = sha256(out / name)
    _progress(out, identity, records)
    _atomic_json(out / "metrics.json", metrics)
    return verify_results(out, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("holdout", "validation", "test"))
    parser.add_argument("--method", required=True, choices=("greedy", "beam"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stop-after-batches", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.split, args.method, args.out,
        args.protocol, args.decode_batch_size, args.device, args.stop_after_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
