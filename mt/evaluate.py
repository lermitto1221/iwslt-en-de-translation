import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch

from .common import environment, read_json, sha256, write_json
from .data import read_pairs
from .decoding import generate_batch_ids
from .inference import load_checkpoint, resolve_data_dir, tokenizer_fingerprint
from .tokenizer import EOS, decode, encode


def evaluate_model(model, tok, rows, method, beam_size, alpha, max_tokens,
                   warmup=0, decode_batch_size=16):
    if not rows:
        raise ValueError("empty evaluation split")
    if decode_batch_size < 1 or warmup < 0:
        raise ValueError("decode_batch_size must be positive and warmup nonnegative")
    device = next(model.parameters()).device

    def run_batch(batch):
        # Includes full source tokenization and target detokenization in timing.
        sources = [encode(tok, row["en"]) for row in batch]
        with torch.autocast(device_type=device.type, enabled=False):
            sequences, limits = generate_batch_ids(model, sources, method,
                                                    beam_size, alpha, max_tokens)
        return ([decode(tok, sequence) for sequence in sequences], sum(limits),
                sum(sequence[-1] != EOS for sequence in sequences))

    for _ in range(warmup):
        run_batch(rows[:decode_batch_size])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    predictions, hits, unfinished, latencies = [], 0, 0, []
    for offset in range(0, len(rows), decode_batch_size):
        t0 = time.perf_counter()
        preds, batch_hits, batch_unfinished = run_batch(rows[offset:offset + decode_batch_size])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - t0)
        predictions.extend(preds)
        hits += batch_hits
        unfinished += batch_unfinished
        if len(rows) > 1000 and (len(latencies) % 128 == 0 or len(predictions) == len(rows)):
            print(json.dumps({"decode_progress": len(predictions), "samples": len(rows),
                              "method": method}), flush=True)
    elapsed = time.perf_counter() - start
    refs = [row["de"] for row in rows]
    metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    bleu = metric.corpus_score(predictions, [refs])
    runtime = environment()
    runtime.update({"device": str(device), "cpu_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
    if device.type == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name(device)
        runtime["gpu_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return predictions, {
        "bleu": bleu.score, "signature": str(metric.get_signature()),
        "sacrebleu_version": sacrebleu.__version__, "samples": len(rows),
        "generation_limit_hits": hits, "generation_limit_fraction": hits / len(rows),
        "unfinished_generations": unfinished, "unfinished_fraction": unfinished / len(rows),
        "generation_limit_definition": "selected output generated max_tokens tokens, including a final-step EOS; BOS excluded",
        "elapsed_seconds": elapsed, "sentences_per_second": len(rows) / elapsed,
        "mean_latency_ms": 1000 * sum(latencies) / len(rows),
        "mean_batch_latency_ms": 1000 * sum(latencies) / len(latencies),
        "inference_batch_size": decode_batch_size,
        "inference_precision": str(next(model.parameters()).dtype).removeprefix("torch."),
        "environment": runtime, "warmup_batches": warmup,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "timing": "includes full source SentencePiece encoding and target decoding; original sample order; fixed sentence batch size (last batch may be smaller); GPU synchronized per batch; warmup batches excluded; mean_latency_ms is amortized per sentence, mean_batch_latency_ms is batch service latency",
    }


def evaluate_checkpoint(checkpoint, split, method, out, beam_size=None, alpha=None,
                        max_tokens=None, decode_batch_size=None):
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    out = Path(out)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"{out} already contains a completed evaluation; use a new output directory, "
                              "or mt.run_experiments to verify and reuse it")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg, state = load_checkpoint(checkpoint, device)
    data = resolve_data_dir(checkpoint, cfg, tokenizer_fingerprint(state))
    manifest = read_json(data / "manifest.json")
    tokenizer = read_json(data / "tokenizer.json")
    fingerprints = {"manifest_sha256": sha256(data / "manifest.json"),
                    "tokenizer_sha256": sha256(data / "spm.model")}
    for name in ("train", "validation", "test"):
        fingerprints[name + "_sha256"] = sha256(data / f"{name}.jsonl")
        if fingerprints[name + "_sha256"] != manifest["files"][name]["jsonl_sha256"]:
            raise ValueError(f"{name} data does not match its manifest")
    if fingerprints["tokenizer_sha256"] != tokenizer["model_sha256"]:
        raise ValueError("tokenizer does not match its recorded SHA-256")
    if tokenizer["training_file_sha256"] != fingerprints["train_sha256"]:
        raise ValueError("tokenizer was trained on different training data")
    if state.get("data_fingerprints") is not None and state["data_fingerprints"] != fingerprints:
        raise ValueError("checkpoint data/tokenizer fingerprints do not match current artifacts")
    if state.get("fingerprints") is not None:
        saved = state["fingerprints"]
        normalized = {"manifest_sha256": saved["manifest_sha256"],
                      "tokenizer_sha256": saved["tokenizer_sha256"],
                      **{name + "_sha256": value for name, value in saved["splits"].items()}}
        if normalized != fingerprints:
            raise ValueError("checkpoint data/tokenizer fingerprints do not match current artifacts")
    rows = read_pairs(data / f"{split}.jsonl")
    expected_samples = manifest["files"][split]["rows_saved"]
    if not rows or len(rows) != expected_samples:
        raise ValueError("evaluation split sample count differs from manifest")
    beam_size = cfg["beam_size"] if beam_size is None else beam_size
    alpha = cfg["length_penalty"] if alpha is None else alpha
    max_tokens = cfg["max_generation_tokens"] if max_tokens is None else max_tokens
    decode_batch_size = cfg.get("decode_batch_size", 16) if decode_batch_size is None else decode_batch_size
    preds, metrics = evaluate_model(model, tok, rows, method, beam_size, alpha,
                                   max_tokens, warmup=2, decode_batch_size=decode_batch_size)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (out / "references.txt").write_text("\n".join(row["de"] for row in rows) + "\n", encoding="utf-8")
    with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, preds):
            f.write(json.dumps({"id": row["id"], "source": row["en"], "reference": row["de"], "prediction": pred}, ensure_ascii=False) + "\n")
    metrics.update({"split": split, "method": method, "beam_size": 1 if method == "greedy" else beam_size,
        "length_penalty": alpha, "max_generation_tokens": max_tokens,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "inference_source_sha256": sha256(Path(__file__).with_name("decoding.py")),
        "config": cfg, "smoke": bool(state.get("smoke", False)), "pilot": bool(state.get("pilot", False)),
        "initial_model_sha256": state.get("initial_model_sha256"),
        "training_source_hashes": {name: state.get("source_hashes", {}).get(name) for name in
            ("train.py", "model.py", "batching.py", "tokenizer.py", "decoding.py")},
        "data_fingerprints": fingerprints, "expected_samples": expected_samples,
        "reference_source": str(data / f"{split}.jsonl"),
        "reference_sha256": fingerprints[split + "_sha256"],
        "references_sha256": sha256(out / "references.txt"),
        "predictions_sha256": sha256(out / "predictions.txt"),
        "checkpoint_update": state.get("update")})
    # metrics.json is the completion marker. A failed/incomplete write must not
    # look like a reusable evaluation to the experiment runner.
    temporary = out / "metrics.json.tmp"
    write_json(temporary, metrics)
    temporary.replace(out / "metrics.json")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--method", choices=["greedy", "beam"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--beam-size", type=int)
    p.add_argument("--length-penalty", type=float)
    p.add_argument("--max-generation-tokens", type=int)
    p.add_argument("--decode-batch-size", type=int)
    args = p.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.split, args.method, args.out,
        args.beam_size, args.length_penalty, args.max_generation_tokens, args.decode_batch_size),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
