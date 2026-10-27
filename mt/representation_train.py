"""Train paired representations against an immutable, shared raw-ID schedule.

The optimizer/loss/validation algorithm is the existing mt.train algorithm.
Only training-example selection and microbatch scheduling differ.  In
particular, tokens are counted separately for each representation: equal
sentence exposure is not claimed to be equal token or compute exposure.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .batching import collate, tokenize_pairs
from .common import environment, read_json, seed_all, sha256, write_json
from .data import read_pairs
from .model import TranslationTransformer, parameter_counts
from .representation_batches import load_schedule
from .tokenizer import BOS, EOS, PAD, UNK, load
from .train import (CORE_TRAINING_SOURCES, check_inputs, model_digest,
                    reconcile_training_log, rng_state, set_rng_state,
                    validation_bleu)


EXECUTION_SOURCES = CORE_TRAINING_SOURCES + ("representation_train.py", "representation_batches.py")


def execution_source_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in EXECUTION_SOURCES}


def _save_source_snapshot(run, hashes):
    snapshot = Path(run) / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 2, "execution_source_hashes": hashes,
                "core_source_hashes": {name: hashes[name] for name in CORE_TRAINING_SOURCES}}
    path = snapshot / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("existing source snapshot belongs to different execution code")
    for name, digest in hashes.items():
        target = snapshot / name
        if target.exists():
            if sha256(target) != digest:
                raise ValueError("source snapshot is changed: " + name)
        else:
            content = Path(__file__).with_name(name).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("execution source changed during startup: " + name)
            with target.open("xb") as handle:
                handle.write(content)
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")


def _validate_sources(state, run, current=None):
    current = current or execution_source_hashes()
    saved = state.get("source_hashes", {})
    if saved != current:
        raise ValueError("execution source differs from checkpoint; restore its complete source snapshot")
    run = Path(run)
    if not (run / "source_hashes.json").is_file() or read_json(run / "source_hashes.json") != saved:
        raise ValueError("run source_hashes.json differs from checkpoint")
    snapshot = run / "source_snapshot"
    path = snapshot / "manifest.json"
    manifest = {"schema_version": 2, "execution_source_hashes": saved,
                "core_source_hashes": {name: saved[name] for name in CORE_TRAINING_SOURCES}}
    if not path.is_file() or read_json(path) != manifest:
        raise ValueError("complete execution source snapshot manifest is missing or changed")
    for name, digest in saved.items():
        if not (snapshot / name).is_file() or sha256(snapshot / name) != digest:
            raise ValueError("source snapshot is missing or changed: " + name)


def _inputs(cfg):
    if cfg.get("representation") not in ("bpe", "unigram"):
        raise ValueError("representation must be bpe or unigram")
    for key in ("max_updates", "warmup_updates", "batch_tokens", "validate_every", "save_every"):
        if type(cfg.get(key, 100 if key == "save_every" else None)) is not int or cfg.get(key, 100) < 1:
            raise ValueError("training budgets and validation/save periods must be positive integers")
    if cfg.get("accum_steps") != 4:
        raise ValueError("paired representation training requires exactly four common microbatches")
    fingerprints = check_inputs(cfg)
    for key in ("common_ids", "batch_plan"):
        if sha256(cfg[key]) != cfg.get(key + "_sha256"):
            raise ValueError(key + " SHA-256 differs from frozen configuration")
        fingerprints[key + "_sha256"] = cfg[key + "_sha256"]
    data = Path(cfg["data_dir"])
    token_metadata = read_json(data / "tokenizer.json")
    if token_metadata.get("model_type") != cfg["representation"]:
        raise ValueError("tokenizer representation label differs from configuration")
    tok = load(data)
    if [tok.pad_id(), tok.unk_id(), tok.bos_id(), tok.eos_id()] != [PAD, UNK, BOS, EOS]:
        raise ValueError("tokenizer special token IDs differ")
    if tok.vocab_size() != cfg["vocab_size"]:
        raise ValueError("actual tokenizer vocabulary differs from frozen size")
    rows = read_pairs(data / "train.jsonl")
    common = read_json(cfg["common_ids"])
    if common.get("max_tokens") != cfg["max_train_tokens"]:
        raise ValueError("common length filter differs from frozen configuration")
    ids = common["ids"]
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError("common training IDs must be unique nonempty integers")
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or any(i not in indexed for i in ids):
        raise ValueError("common training IDs do not map one-to-one to source rows")
    selected = [indexed[i] for i in ids]
    examples, excluded = tokenize_pairs(selected, tok, None)
    if excluded or any(max(len(src), len(tgt)) > cfg["max_train_tokens"] for src, tgt, _ in examples):
        raise ValueError("common training IDs exceed the frozen token length limit")
    validation = read_pairs(data / "validation.jsonl")
    if not validation or len({row["id"] for row in validation}) != len(validation):
        raise ValueError("validation IDs are empty or duplicated")
    expected_validation = cfg.get("expected_validation_samples")
    if expected_validation is not None and len(validation) != expected_validation:
        raise ValueError("validation sample count differs from frozen configuration")
    schedule = load_schedule(cfg, examples)
    return fingerprints, tok, examples, validation, schedule, len(rows)


def _schedule_record(item, representation):
    return {"epoch": item["epoch"], "batch_cursor": item["batch_cursor"],
            "target_tokens": item["target_tokens"][representation],
            "source_tokens": item["source_tokens"][representation],
            "padded_tokens": item["padded_tokens"][representation],
            "sentences": item["sentences"], "data_order_sha256": item["data_order_sha256"],
            "microbatch_sha256": item["microbatch_sha256"]}


def _validate_state(state, cfg, fingerprints, schedule):
    required = ("config", "model", "optimizer", "scheduler", "scaler", "rng", "update", "epoch",
                "batch_cursor", "best_bleu", "fingerprints", "initial_model_sha256",
                "source_hashes", "training_seconds_total", "log_record", "log_sha256", "model_sha256")
    if any(key not in state for key in required):
        raise ValueError("resume requires a complete training checkpoint")
    if state["config"] != cfg or state["fingerprints"] != fingerprints:
        raise ValueError("resume configuration or input fingerprints differ from checkpoint")
    if state.get("smoke") is not False or state.get("pilot") is not False:
        raise ValueError("checkpoint mode differs from paired representation training")
    digest = hashlib.sha256()
    for name, tensor in state["model"].items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    if digest.hexdigest() != state["model_sha256"]:
        raise ValueError("checkpoint model tensor SHA-256 differs from saved evidence")
    update = state["update"]
    if type(update) is not int or not 1 <= update <= cfg["max_updates"]:
        raise ValueError("checkpoint update is outside the frozen training budget")
    expected = _schedule_record(schedule[update - 1], cfg["representation"])
    if state["epoch"] != expected["epoch"] or state["batch_cursor"] != expected["batch_cursor"]:
        raise ValueError("checkpoint schedule cursor differs from frozen microbatch boundaries")
    if state["log_record"].get("update") != update or any(
            state["log_record"].get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint log record differs from frozen common schedule")
    _validate_sources(state, cfg["run_dir"])


def _records(path):
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("training log contains invalid committed records") from error


def _validate_run_metadata(state, cfg, fingerprints):
    run = Path(cfg["run_dir"])
    if read_json(run / "config.json") != cfg:
        raise ValueError("saved run configuration differs")
    link = read_json(run / "data_link.json")
    if any(link.get(key) != value for key, value in fingerprints.items()):
        raise ValueError("saved data evidence differs from frozen inputs")
    if link.get("initial_model_sha256") != state["initial_model_sha256"]:
        raise ValueError("initialization evidence differs from checkpoint")
    return link


def _validate_log(records, cfg, schedule, last_state):
    if [r.get("update") for r in records] != list(range(1, last_state["update"] + 1)):
        raise ValueError("training log is not the complete continuous checkpoint prefix")
    for record, item in zip(records, schedule):
        expected = _schedule_record(item, cfg["representation"])
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("training log differs from frozen common schedule")
        step = record["update"]
        expected_lr = cfg["lr"] * min(step / cfg["warmup_updates"], math.sqrt(cfg["warmup_updates"] / step))
        if not math.isclose(record.get("lr_used", -1), expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("training learning rate differs from the frozen schedule")
        required = step % cfg["validate_every"] == 0 or step == cfg["max_updates"]
        if ("validation_bleu_greedy" in record) != required:
            raise ValueError("validation checkpoint selection cadence differs")
        if required and (not math.isfinite(record["validation_bleu_greedy"])
                         or record.get("validation_samples") != cfg.get("expected_validation_samples", record.get("validation_samples"))):
            raise ValueError("validation score or full sample count is invalid")
    if not records or records[-1] != last_state["log_record"]:
        raise ValueError("checkpoint log record differs from final committed record")


def validate_completed_run(cfg, schedule=None, fingerprints=None):
    """Strict, read-only verification before reusing a completed training run."""
    if schedule is None or fingerprints is None:
        fingerprints, _, _, _, schedule, _ = _inputs(cfg)
    run = Path(cfg["run_dir"])
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    _validate_state(state, cfg, fingerprints, schedule)
    if state["update"] != cfg["max_updates"]:
        raise ValueError("completed run checkpoint has not exhausted its fixed budget")
    link = _validate_run_metadata(state, cfg, fingerprints)
    # Recreate initialization on CPU without changing the caller's RNG. Numeric
    # initialization equality is distinct from token-ID semantic equivalence.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"])
        initial_model = TranslationTransformer(cfg["vocab_size"], cfg)
        if (model_digest(initial_model) != state["initial_model_sha256"]
                or parameter_counts(initial_model) != link.get("parameter_counts")):
            raise ValueError("initialization or parameter count differs from configured model")
    del initial_model
    records = _records(run / "train.jsonl")
    _validate_log(records, cfg, schedule, state)
    if sha256(run / "train.jsonl") != state["log_sha256"]:
        raise ValueError("completed training log SHA-256 differs from checkpoint")
    validated = [record for record in records if "validation_bleu_greedy" in record]
    best_record = max(validated, key=lambda record: record["validation_bleu_greedy"])
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    _validate_state(best, cfg, fingerprints, schedule)
    prefix = b"".join((run / "train.jsonl").read_bytes().splitlines(keepends=True)[:best["update"]])
    if (best["update"] != best_record["update"] or best["log_record"] != best_record
            or best["best_bleu"] != best_record["validation_bleu_greedy"]
            or state["best_bleu"] != best_record["validation_bleu_greedy"]
            or hashlib.sha256(prefix).hexdigest() != best["log_sha256"]
            or best["initial_model_sha256"] != state["initial_model_sha256"]):
        raise ValueError("best checkpoint does not match earliest best full validation")
    summary = read_json(run / "summary.json")
    if (summary.get("complete") is not True or summary.get("updates") != cfg["max_updates"]
            or summary.get("best_validation_bleu") != best_record["validation_bleu_greedy"]
            or summary.get("parameters") != link.get("parameter_counts")
            or summary.get("representation") != cfg["representation"]
            or summary.get("batch_plan_sha256") != cfg["batch_plan_sha256"]
            or summary.get("common_ids_sha256") != cfg["common_ids_sha256"]):
        raise ValueError("completed summary differs from verified training evidence")
    if (not math.isfinite(summary.get("training_seconds_total", -1))
            or summary["training_seconds_total"] < state["training_seconds_total"]
            or summary.get("peak_cuda_bytes") != state.get("peak_cuda_bytes")):
        raise ValueError("completed summary cost differs from checkpoint evidence")
    return summary


def train(cfg, resume=None, stop_after=None):
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive integer")
    fingerprints, tok, examples, val_rows, schedule, full_train_count = _inputs(cfg)
    run = Path(cfg["run_dir"])
    source_hashes = execution_source_hashes()
    state = None
    if resume is not None:
        if Path(resume).resolve() != (run / "last.pt").resolve():
            raise ValueError("resume is only allowed from this run's complete last.pt")
        state = torch.load(resume, map_location="cpu", weights_only=False)
        _validate_state(state, cfg, fingerprints, schedule)
        _validate_run_metadata(state, cfg, fingerprints)
        if (run / "summary.json").exists() and read_json(run / "summary.json").get("complete"):
            return validate_completed_run(cfg, schedule, fingerprints)
    elif any((run / name).exists() for name in ("last.pt", "best.pt", "train.jsonl", "summary.json")):
        raise FileExistsError(f"{run} already contains training history; use --resume")
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(cfg.get("deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = False
    requested = cfg.get("device", "auto")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    model = TranslationTransformer(tok.vocab_size(), cfg).to(device)
    initial_digest = model_digest(model)
    counts = parameter_counts(model)
    expected_parameters = cfg.get("expected_parameters")
    if expected_parameters is not None and counts["total"] != expected_parameters:
        raise ValueError("model parameter count differs from frozen expectation")
    if state and state["initial_model_sha256"] != initial_digest:
        raise ValueError("checkpoint initialization differs from its configured seed and model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    warmup = cfg["warmup_updates"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, math.sqrt(warmup / (step + 1))))
    use_amp = bool(cfg["amp"] and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 unavailable; freeze a common supported precision before training")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    update, epoch, batch_cursor, best_bleu = 0, 0, 0, -float("inf")
    accumulated_time, previous_peak = 0.0, 0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        update, epoch, batch_cursor = state["update"], state["epoch"], state["batch_cursor"]
        best_bleu = state["best_bleu"]
        accumulated_time = state["training_seconds_total"]
        previous_peak = state.get("peak_cuda_bytes") or 0
        set_rng_state(state["rng"])
        recovery = reconcile_training_log(run, state)
        _validate_log(_records(run / "train.jsonl"), cfg, schedule, state)
        if recovery["changed"]:
            print(json.dumps({"resume_log_recovery": recovery}), flush=True)
    else:
        run.mkdir(parents=True, exist_ok=True)
        _save_source_snapshot(run, source_hashes)
        write_json(run / "source_hashes.json", source_hashes)
        write_json(run / "config.json", cfg)
        write_json(run / "environment.json", environment())
        write_json(run / "data_link.json", {**fingerprints, "smoke": False, "pilot": False,
            "representation": cfg["representation"], "tokenizer_training_rows": full_train_count,
            "train_rows_used": len(examples), "train_rows_after_filter": len(examples),
            "train_rows_subword_excluded": full_train_count - len(examples),
            "train_excluded_fraction": (full_train_count - len(examples)) / full_train_count,
            "effective_token_budget": cfg["batch_tokens"] * cfg["accum_steps"],
            "batch_budget_definition": "upper bound on sum of padded source and full target tokens",
            "initial_model_sha256": initial_digest, "parameter_counts": counts})
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def peak_memory():
        return max(previous_peak, torch.cuda.max_memory_allocated()) if device.type == "cuda" else None

    def save(name, record, log_hash):
        saved = {"config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "update": update, "epoch": epoch, "batch_cursor": batch_cursor, "best_bleu": best_bleu,
            "smoke": False, "pilot": False, "fingerprints": fingerprints,
            "initial_model_sha256": initial_digest, "model_sha256": model_digest(model),
            "source_hashes": source_hashes,
            "training_seconds_total": accumulated_time + time.perf_counter() - start,
            "peak_cuda_bytes": peak_memory(), "checkpoint_schema_version": 3,
            "log_record": record, "log_sha256": log_hash}
        temporary = run / (name + ".tmp")
        with temporary.open("wb") as handle:
            torch.save(saved, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(run / name)

    target_update = min(cfg["max_updates"], update + stop_after) if stop_after else cfg["max_updates"]
    with (run / "train.jsonl").open("a" if resume else "w", encoding="utf-8", newline="\n") as log:
        while update < target_update:
            step_start = time.perf_counter()
            item = schedule[update]
            microbatches = item["microbatches"]
            expected = _schedule_record(item, cfg["representation"])
            token_count = sum(len(examples[i][1]) - 1 for group in microbatches for i in group)
            actual_ids = [[examples[i][2] for i in group] for group in microbatches]
            if (len(microbatches) != 4 or token_count != expected["target_tokens"]
                    or hashlib.sha256(json.dumps(actual_ids).encode()).hexdigest() != expected["microbatch_sha256"]):
                raise ValueError("training consumption differs from frozen microbatch plan")
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
                raise FloatingPointError("nonfinite gradients: optimizer/scheduler update not counted")
            lr_used = optimizer.param_groups[0]["lr"]
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise FloatingPointError("AMP skipped optimizer update; stopping to preserve the fixed budget")
            scheduler.step()
            update += 1
            epoch, batch_cursor = expected["epoch"], expected["batch_cursor"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            record = {"update": update, "loss": loss_sum / token_count, "lr_used": lr_used,
                **expected, "update_seconds": time.perf_counter() - step_start,
                "elapsed_seconds": accumulated_time + time.perf_counter() - start}
            should_validate = update % cfg["validate_every"] == 0 or update == cfg["max_updates"]
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
        "peak_cuda_bytes": peak_memory(), "parameters": counts, "smoke": False, "pilot": False,
        "complete": update == cfg["max_updates"], "representation": cfg["representation"],
        "batch_plan_sha256": cfg["batch_plan_sha256"], "common_ids_sha256": cfg["common_ids_sha256"]}
    write_json(run / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    print(json.dumps(train(read_json(args.config), args.resume, args.stop_after), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
