import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F

from .batching import batches, collate, tokenize_pairs
from .common import environment, read_json, seed_all, sha256, write_json
from .data import read_pairs
from .model import TranslationTransformer, parameter_counts
from .tokenizer import PAD, load


CORE_TRAINING_SOURCES = ("train.py", "model.py", "batching.py", "tokenizer.py",
                         "decoding.py", "common.py", "data.py")


def _validate_resume_sources(state, current, run):
    """Reject changed execution code before touching a prior run's artifacts."""
    saved = state.get("source_hashes", {})
    changed = [name for name in CORE_TRAINING_SOURCES if saved.get(name) != current.get(name)]
    if changed:
        raise ValueError("core training source differs from checkpoint: " + ", ".join(changed)
                         + "; restore the recorded source snapshot or start a new run")
    metadata = Path(run) / "source_hashes.json"
    if not metadata.is_file() or read_json(metadata) != saved:
        raise ValueError("run source_hashes.json is missing or differs from checkpoint provenance")
    snapshot = Path(run) / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("resume requires the original source_snapshot/manifest.json; "
                         "a missing snapshot cannot be reconstructed as verified history")
    expected = {name: saved[name] for name in CORE_TRAINING_SOURCES}
    if read_json(manifest_path).get("core_source_hashes") != expected:
        raise ValueError("source snapshot manifest differs from checkpoint")
    damaged = [name for name in CORE_TRAINING_SOURCES
               if not (snapshot / name).is_file() or sha256(snapshot / name) != expected[name]]
    if damaged:
        raise ValueError("source snapshot is missing or changed: " + ", ".join(damaged))


def _save_source_snapshot(run, source_hashes):
    """Save immutable-by-convention source copies; never replace an old copy."""
    snapshot = Path(run) / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    expected = {name: source_hashes[name] for name in CORE_TRAINING_SOURCES}
    manifest = {"schema_version": 1, "core_source_hashes": expected}
    manifest_path = snapshot / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("existing source snapshot belongs to different training code")
    for name in CORE_TRAINING_SOURCES:
        target = snapshot / name
        if target.exists():
            if sha256(target) != expected[name]:
                raise ValueError(f"existing source snapshot differs: {name}")
        else:
            content = Path(__file__).with_name(name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError(f"training source changed during startup: {name}")
            with target.open("xb") as handle:
                handle.write(content)
    if not manifest_path.exists():
        temporary = snapshot / "manifest.json.tmp"
        write_json(temporary, manifest)
        temporary.replace(manifest_path)


def _record_bytes(record):
    return (json.dumps(record) + "\n").encode("utf-8")


def reconcile_training_log(run, state):
    """Retain exactly the committed checkpoint prefix, backing up any edit.

    A checkpoint embeds its final record and prefix hash. Only a missing or
    incomplete final record can be repaired; older committed corruption fails.
    Work after the checkpoint, including a torn UTF-8/JSON tail, is discarded.
    """
    path = Path(run) / "train.jsonl"
    original = path.read_bytes() if path.exists() else b""
    lines = original.splitlines(keepends=True)
    update = state["update"]
    retained, records = [], []
    for index, line in enumerate(lines):
        if len(records) == update:
            break
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError("training log has malformed data inside the committed prefix") from None
            break
        if not isinstance(record, dict) or type(record.get("update")) is not int:
            raise ValueError("training log contains a record without an integer update")
        if record["update"] != len(records) + 1:
            raise ValueError("training log updates are not a continuous checkpoint prefix")
        records.append(record)
        retained.append(line if line.endswith(b"\n") else line + b"\n")
    saved_record = state.get("log_record")
    repaired = False
    if len(records) == update - 1 and saved_record is not None:
        if saved_record.get("update") != update:
            raise ValueError("checkpoint log_record does not match its update")
        retained.append(_record_bytes(saved_record))
        records.append(saved_record)
        repaired = True
    if len(records) != update:
        raise ValueError("training log is missing committed updates; cannot fabricate training history")
    if saved_record is not None and update and records[-1] != saved_record:
        raise ValueError("checkpoint log_record differs from the committed training log")
    recovered = b"".join(retained)
    expected_hash = state.get("log_sha256")
    if expected_hash is not None and hashlib.sha256(recovered).hexdigest() != expected_hash:
        raise ValueError("committed training log SHA-256 differs from checkpoint")
    result = {"checkpoint_update": update, "changed": recovered != original,
              "repaired_checkpoint_record": repaired, "backup": None}
    if recovered != original:
        if path.exists():
            backup = path.with_name(f"train.jsonl.before-resume.{time.time_ns()}.bak")
            with backup.open("xb") as handle:
                handle.write(original)
                handle.flush()
                os.fsync(handle.fileno())
            result["backup"] = str(backup)
        temporary = path.with_name("train.jsonl.resume.tmp")
        with temporary.open("wb") as handle:
            handle.write(recovered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    return result


def check_inputs(cfg):
    data = Path(cfg["data_dir"])
    manifest = read_json(data / "manifest.json")
    tokenizer = read_json(data / "tokenizer.json")
    for split in ("train", "validation", "test"):
        if sha256(data / f"{split}.jsonl") != manifest["files"][split]["jsonl_sha256"]:
            raise ValueError(f"{split} file changed since data preparation")
    if sha256(data / "spm.model") != tokenizer["model_sha256"]:
        raise ValueError("tokenizer changed")
    if tokenizer["training_file_sha256"] != manifest["files"]["train"]["jsonl_sha256"]:
        raise ValueError("tokenizer trained on a different training subset")
    return {"manifest_sha256": sha256(data / "manifest.json"),
            "tokenizer_sha256": sha256(data / "spm.model"),
            "splits": {s: manifest["files"][s]["jsonl_sha256"] for s in manifest["files"]}}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validation_bleu(model, tok, rows, cfg):
    from .decoding import translate_batch
    predictions, hits = [], 0
    size = cfg.get("decode_batch_size", 16)
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        pred, limited = translate_batch(model, tok, [r["en"] for r in chunk], "greedy", 1,
                                       cfg["length_penalty"], cfg["max_generation_tokens"])
        predictions.extend(pred)
        hits += sum(limited)
    metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    score = metric.corpus_score(predictions, [[row["de"] for row in rows]]).score
    return {"validation_bleu_greedy": score, "validation_signature": str(metric.get_signature()),
            "validation_samples": len(rows), "validation_generation_limit_fraction": hits / len(rows)}


def train(cfg, resume=None, smoke=False, stop_after=None, pilot=False):
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    fingerprints = check_inputs(cfg)
    data, run = Path(cfg["data_dir"]), Path(cfg["run_dir"])
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    state = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state.get("fingerprints") != fingerprints:
            raise ValueError("resume configuration or input fingerprints differ from checkpoint")
        if state["smoke"] != smoke or state.get("pilot", False) != pilot:
            raise ValueError("resume mode differs from checkpoint")
        _validate_resume_sources(state, source_hashes, run)
        latest = run / "last.pt"
        if latest.is_file() and Path(resume).resolve() != latest.resolve():
            latest_state = torch.load(latest, map_location="cpu", weights_only=False)
            if latest_state["update"] > state["update"]:
                raise ValueError("resume checkpoint is older than last.pt; cannot rewind this run's history")
            del latest_state
        completed = run / "summary.json"
        if completed.exists() and read_json(completed).get("complete"):
            summary = read_json(completed)
            if state["update"] != cfg["max_updates"] or summary.get("updates") != state["update"]:
                raise ValueError("cannot rewind a completed run; preserve its artifacts and start a new run")
            return summary
        # Keep original source provenance, including non-core files that may
        # have since received unrelated reporting/UI changes.
        source_hashes = state["source_hashes"]
    elif any((run / name).exists() for name in ("last.pt", "best.pt", "train.jsonl", "summary.json")):
        raise FileExistsError(f"{run} already contains training history; use --resume or a new run_dir")
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(cfg.get("deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = False
    requested = cfg.get("device", "auto")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    run.mkdir(parents=True, exist_ok=True)
    tok = load(data)
    train_rows = read_pairs(data / "train.jsonl")
    val_rows = read_pairs(data / "validation.jsonl")
    if smoke:
        train_rows, val_rows = train_rows[:32], val_rows[:8]
    examples, excluded = tokenize_pairs(train_rows, tok, cfg["max_train_tokens"])
    if not examples or not val_rows:
        raise ValueError("training or validation examples empty")
    model = TranslationTransformer(tok.vocab_size(), cfg).to(device)
    initial_digest = model_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    warmup = cfg["warmup_updates"]
    if min(warmup, cfg["max_updates"], cfg["accum_steps"], cfg["batch_tokens"]) <= 0:
        raise ValueError("training budgets and warmup must be positive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, math.sqrt(warmup / (step + 1))))
    use_amp = bool(cfg["amp"] and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 unavailable; use a common float16 config for both experiments")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    update, epoch, batch_cursor, best_bleu = 0, 0, 0, -float("inf")
    accumulated_time, previous_peak = 0.0, 0
    if resume:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        update, epoch, batch_cursor = state["update"], state["epoch"], state["batch_cursor"]
        best_bleu = state["best_bleu"]
        accumulated_time = state.get("training_seconds_total", 0.0)
        previous_peak = state.get("peak_cuda_bytes", 0) or 0
        initial_digest = state["initial_model_sha256"]
        set_rng_state(state["rng"])
        recovery = reconcile_training_log(run, state)
        if recovery["changed"]:
            print(json.dumps({"resume_log_recovery": recovery}), flush=True)
    else:
        _save_source_snapshot(run, source_hashes)
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", environment())
    write_json(run / "data_link.json", {**fingerprints, "smoke": smoke, "pilot": pilot,
        "train_rows_used": len(train_rows), "train_rows_subword_excluded": excluded,
        "train_rows_after_filter": len(examples), "train_excluded_fraction": excluded / len(train_rows),
        "effective_token_budget": cfg["batch_tokens"] * cfg["accum_steps"],
        "initial_model_sha256": initial_digest, "parameter_counts": parameter_counts(model)})
    if not resume:
        write_json(run / "source_hashes.json", source_hashes)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def peak_memory():
        return max(previous_peak, torch.cuda.max_memory_allocated()) if device.type == "cuda" else None

    def save(name, record, log_hash):
        state = {"config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "update": update, "epoch": epoch, "batch_cursor": batch_cursor,
            "best_bleu": best_bleu, "smoke": smoke, "pilot": pilot,
            "fingerprints": fingerprints, "initial_model_sha256": initial_digest,
            "source_hashes": source_hashes, "training_seconds_total": accumulated_time + time.perf_counter() - start,
            "peak_cuda_bytes": peak_memory(), "checkpoint_schema_version": 2,
            "log_record": record, "log_sha256": log_hash}
        temporary = run / (name + ".tmp")
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(run / name)

    groups = list(batches(examples, cfg["batch_tokens"], seed=cfg["seed"] + epoch, shuffle=True))
    target_update = min(cfg["max_updates"], update + stop_after) if stop_after else cfg["max_updates"]
    with (run / "train.jsonl").open("a" if resume else "w", encoding="utf-8", newline="\n") as log:
        while update < target_update:
            step_start = time.perf_counter()
            microbatches = []
            for _ in range(cfg["accum_steps"]):
                if batch_cursor >= len(groups):
                    epoch += 1
                    groups = list(batches(examples, cfg["batch_tokens"], seed=cfg["seed"] + epoch, shuffle=True))
                    batch_cursor = 0
                microbatches.append(groups[batch_cursor])
                batch_cursor += 1
            token_count = sum(len(examples[i][1]) - 1 for group in microbatches for i in group)
            order_digest = hashlib.sha256(json.dumps([examples[i][2] for group in microbatches for i in group]).encode()).hexdigest()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for group in microbatches:
                src, tgt, _ = collate(examples, group, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(src, tgt[:, :-1])
                    summed = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                        ignore_index=PAD, label_smoothing=cfg["label_smoothing"], reduction="sum")
                scaler.scale(summed / token_count).backward()
                loss_sum += summed.detach().float().item()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("nonfinite gradients: no optimizer/scheduler update was counted; retry from checkpoint with a stable precision")
            lr_used = optimizer.param_groups[0]["lr"]
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise FloatingPointError("AMP skipped optimizer update; stopping to preserve the fixed update budget")
            scheduler.step()
            update += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            record = {"update": update, "epoch": epoch, "loss": loss_sum / token_count,
                "lr_used": lr_used, "target_tokens": token_count,
                "sentences": sum(map(len, microbatches)), "data_order_sha256": order_digest,
                "update_seconds": time.perf_counter() - step_start,
                "elapsed_seconds": accumulated_time + time.perf_counter() - start}
            should_validate = not pilot and (update % cfg["validate_every"] == 0 or update == cfg["max_updates"])
            improved = False
            if should_validate:
                val_start = time.perf_counter()
                record.update(validation_bleu(model, tok, val_rows, cfg))
                record["validation_seconds"] = time.perf_counter() - val_start
                if record["validation_bleu_greedy"] > best_bleu:
                    best_bleu = record["validation_bleu_greedy"]
                    improved = True
            log.write(json.dumps(record) + "\n")
            log.flush()
            if should_validate or update % cfg.get("save_every", 100) == 0 or update == target_update:
                # Commit the log before checkpoint replacement. A crash before
                # replacement leaves a removable log tail, never a missing step.
                os.fsync(log.fileno())
                log_hash = sha256(run / "train.jsonl")
                if improved:
                    save("best.pt", record, log_hash)
                save("last.pt", record, log_hash)
            if update % cfg.get("log_every", 25) == 0 or update == target_update or should_validate:
                print(json.dumps(record), flush=True)
    summary = {"updates": update, "best_validation_bleu": best_bleu if math.isfinite(best_bleu) else None,
        "training_seconds_this_invocation": time.perf_counter() - start,
        "training_seconds_total": accumulated_time + time.perf_counter() - start,
        "peak_cuda_bytes": peak_memory(), "parameters": parameter_counts(model),
        "smoke": smoke, "pilot": pilot, "complete": update == cfg["max_updates"]}
    write_json(run / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pilot", action="store_true", help="timing run without checkpoint selection or reported evaluation")
    p.add_argument("--stop-after", type=int, help="stop cleanly after this many additional updates")
    args = p.parse_args()
    cfg = read_json(args.config)
    if args.smoke and cfg["max_updates"] > 100:
        raise ValueError("smoke flag requires a short config")
    print(json.dumps(train(cfg, args.resume, args.smoke, args.stop_after, args.pilot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
