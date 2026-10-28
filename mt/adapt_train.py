"""Frozen, resumable same-base Marian LoRA/full adaptation and native evaluation.

Run configuration is frozen explicitly before training. The only resumable
training artifact is a complete last.pt belonging to that exact run identity.
"""
import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .adapt_data import load_data
from .adapt_model import (LORA, build_model, inject_lora, learned_state, native_collate,
                          parameter_evidence, restore_learned, state_digest)
from .common import read_json, seed_all, sha256
from .data import read_pairs
from .pretrained_mt import (ROOT, atomic_json, dependency_evidence, digest, exclusive_json,
    generate_batch, model_evidence, now, read_journal, score, summarize_records, verify_result)

SOURCES = ("mt/adapt_model.py", "mt/adapt_train.py", "mt/adapt_data.py",
           "mt/pretrained_mt.py", "mt/common.py", "mt/data.py", "project-pretrained-python.ps1")


def sources():
    return {name: sha256(ROOT / name) for name in SOURCES}


def configure_runtime(cfg):
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state(include_cuda=True):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if include_cuda and torch.cuda.is_available() else None}


def set_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def tree_digest(value):
    """Canonical integrity digest for tensors, optimizer state and RNG arrays."""
    import hashlib
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            h.update(b"tensor:"); h.update(state_digest({"value": item}).encode())
        elif isinstance(item, np.ndarray):
            h.update(b"numpy:"); h.update(str(item.dtype).encode()); h.update(str(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict:")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode()); h.update(str(len(item)).encode())
            for child in item:
                add(child)
        else:
            h.update(type(item).__name__.encode()); h.update(repr(item).encode()); h.update(b";")
    add(value)
    return h.hexdigest()


def resume_digest(state):
    # Learned tensors and the committed log each have their own digest. Include
    # those digests, and bind optimizer/scaler/RNG/scheduler and all metadata.
    return tree_digest({k: v for k, v in state.items() if k not in ("learned", "records", "resume_sha256")})


def learning_rate_factor(update, warmup, total):
    """LR applied at 1-based update, reaching base LR at warmup then zero at total."""
    if not 1 <= update <= total or not 0 < warmup < total:
        raise ValueError("Invalid learning-rate schedule position")
    return update / warmup if update <= warmup else (total - update) / (total - warmup)


def better_checkpoint(candidate, current):
    """Max BLEU, then chrF2, then earliest optimizer update."""
    if current is None:
        return True
    return (candidate["bleu"], candidate["chrf2"], -candidate["update"]) > (
        current["bleu"], current["chrf2"], -current["update"])


def select_learning_rate(candidates, candidate_order):
    """Only validation rows are accepted; test fields are never consumed."""
    if len(candidates) != len(candidate_order) or {r["lr"] for r in candidates} != set(candidate_order):
        raise ValueError("Incomplete or unexpected learning-rate candidates")
    if any(r.get("split") != "validation" or r.get("seed") != 7 or r.get("updates") != 3000 for r in candidates):
        raise ValueError("LR selection requires complete tuning-seed validation evidence")
    return max(candidates, key=lambda r: (r["bleu"], r["chrf2"], -candidate_order.index(r["lr"])))


def validate_config(cfg):
    required = ("model_dir", "data_root", "run_dir", "study_plan", "method", "seed", "lr",
                "updates", "warmup_updates", "train_precision", "micro_batch_size",
                "effective_batch_size", "validation_every", "save_every", "eval_batch_size", "generation")
    if any(key not in cfg for key in required):
        raise ValueError("Incomplete adaptation configuration")
    if cfg["method"] not in ("lora", "full") or cfg["seed"] not in (7, 42, 43, 44):
        raise ValueError("Unexpected adaptation method or seed")
    if cfg["updates"] != 3000 or cfg["warmup_updates"] != 300 or cfg["validation_every"] != 500:
        raise ValueError("Adaptation budget/warmup/validation differs from the approved design")
    allowed = {"lora": (1e-4, 3e-4), "full": (1e-5, 3e-5)}
    if cfg["lr"] not in allowed[cfg["method"]]:
        raise ValueError("Unexpected learning-rate candidate")
    if cfg["train_precision"] not in ("bfloat16", "float16"):
        raise ValueError("Training precision must be common BF16 or FP16")
    if cfg["effective_batch_size"] % cfg["micro_batch_size"] or min(cfg["save_every"], cfg["eval_batch_size"]) < 1:
        raise ValueError("Invalid batch or checkpoint interval")
    if set(cfg["generation"]) != {"greedy", "beam"}:
        raise ValueError("Both native decode configurations are required")
    for method, beams in (("greedy", 1), ("beam", 4)):
        g = cfg["generation"][method]
        if (g.get("num_beams"), g.get("max_length"), g.get("pad_token_id"),
            g.get("decoder_start_token_id"), g.get("eos_token_id"), g.get("forced_eos_token_id")) != (beams, 512, 58100, 58100, 0, 0):
            raise ValueError("Unexpected native generation protocol")
    return cfg


def freeze_run(config_path):
    cfg = validate_config(read_json(config_path))
    run = Path(cfg["run_dir"])
    if (run / "plan.json").exists() or (run / "last.pt").exists() or (run / "train.jsonl").exists():
        raise FileExistsError("A frozen or started adaptation run cannot be overwritten")
    study = Path(cfg["study_plan"])
    if not study.is_file():
        raise ValueError("Freeze the study plan before run plans")
    data = read_json(Path(cfg["data_root"]) / "manifest.json")
    for key in ("updates", "effective_batch_size", "micro_batch_size"):
        if cfg[key] != data[key]:
            raise ValueError("Run differs from the shared native batch plan")
    if data["model_files"] != model_evidence(cfg["model_dir"]):
        raise ValueError("Prepared native model/tokenizer differs")
    run.mkdir(parents=True, exist_ok=True)
    identity = {"created_at": now(), "config_sha256": sha256(config_path), "config": cfg,
        "study_plan_sha256": sha256(study), "source_files": sources(),
        "environment": dependency_evidence(), "model_files": data["model_files"],
        "data_manifest_sha256": sha256(Path(cfg["data_root"]) / "manifest.json"),
        "schedule_sha256": data["schedules"][str(cfg["seed"])]["sha256"],
        "adapter_config": LORA if cfg["method"] == "lora" else None,
        "checkpoint_selection": "complete 888-sentence validation Greedy corpus BLEU, then chrF2, then earliest update",
        "test_use": "final report only; no checkpoint, LR, rank or budget selection"}
    for name, checksum in identity["source_files"].items():
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            f.write((ROOT / name).read_bytes())
        if sha256(destination) != checksum:
            raise ValueError("Execution source changed while freezing")
    config_bytes = Path(config_path).read_bytes()
    if (run / "config.json").exists():
        if (run / "config.json").read_bytes() != config_bytes:
            raise ValueError("Existing run configuration differs from source config bytes")
    else:
        with (run / "config.json").open("xb") as f:
            f.write(config_bytes)
    exclusive_json(run / "plan.json", identity)
    return identity


def verify_run(config_path):
    cfg = validate_config(read_json(config_path))
    run = Path(cfg["run_dir"])
    plan = read_json(run / "plan.json")
    if plan["config_sha256"] != sha256(config_path) or plan["config"] != cfg or read_json(run / "config.json") != cfg:
        raise ValueError("Frozen adaptation configuration changed")
    if sha256(cfg["study_plan"]) != plan["study_plan_sha256"] or plan["source_files"] != sources():
        raise ValueError("Frozen study or adaptation source changed")
    for name, checksum in plan["source_files"].items():
        if sha256(run / "source_snapshot" / name) != checksum:
            raise ValueError("Execution source snapshot changed")
    if model_evidence(cfg["model_dir"]) != plan["model_files"] or dependency_evidence() != plan["environment"]:
        raise ValueError("Base model/tokenizer or dependencies changed")
    if sha256(Path(cfg["data_root"]) / "manifest.json") != plan["data_manifest_sha256"]:
        raise ValueError("Native data manifest changed")
    return cfg, plan


def evaluate_model(model, tokenizer, rows, out, context, warmups=2):
    """FP32 native generation, hashed per-batch journal, strict completed reuse."""
    out = Path(out)
    if (out / "metrics.json").exists():
        return verify_result(out, rows, context)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "started.json").exists():
        if read_json(out / "started.json")["context"] != context:
            raise ValueError("Partial evaluation belongs to a different frozen identity")
    else:
        exclusive_json(out / "started.json", {"created_at": now(), "context": context})
    journal = out / "batches.jsonl"
    records, end = read_journal(journal, rows, context["batch_size"])
    if journal.exists() and journal.stat().st_size != end:
        with journal.open("r+b") as f:
            f.truncate(end)
    completed = sum(len(r["ids"]) for r in records)
    device = next(model.parameters()).device
    was_training, rng = model.training, rng_state(device.type == "cuda")
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Formal native decoding requires FP32 model parameters")
    try:
        if completed < len(rows):
            for _ in range(warmups):
                generate_batch(model, tokenizer, rows[:context["batch_size"]], context["generation"], 512)
            if device.type == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for offset in range(completed, len(rows), context["batch_size"]):
                batch = rows[offset:offset + context["batch_size"]]
                start = time.perf_counter()
                result = generate_batch(model, tokenizer, batch, context["generation"], 512)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                record = {"ids": [r["id"] for r in batch], "elapsed_seconds": time.perf_counter() - start,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                          **result}
                record["record_sha256"] = digest(record)
                with journal.open("ab") as f:
                    f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")); f.flush()
                records.append(record)
                completed += len(batch)
                atomic_json(out / "progress.json", {"updated_at": now(), "completed": completed, "samples": len(rows)})
                if len(records) % 32 == 0 or completed == len(rows):
                    print(json.dumps({"evaluation": str(out), "completed": completed, "samples": len(rows)}), flush=True)
        predictions = [p for r in records for p in r["predictions"]]
        references = [r["de"] for r in rows]
        if len(predictions) != len(rows) or any("\n" in s or "\r" in s for s in predictions + references):
            raise ValueError("Evaluation predictions are incomplete or contain line breaks")
        (out / "predictions.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
        (out / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
        with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
            for row, pred in zip(rows, predictions):
                f.write(json.dumps({"id": row["id"], "source": row["en"], "reference": row["de"],
                                   "prediction": pred}, ensure_ascii=False) + "\n")
        result = {"context": context, "completed_at": now(), "samples": len(rows), **score(predictions, references),
            **summarize_records(records, len(rows), context["device"]),
            "precision": "float32", "warmup_batches_per_attempt": warmups,
            "generation_limit_definition": "generated first EOS or unfinished output plus decoder-start reaches max_length 512; forced EOS at limit is counted",
            "timing": "synchronized batch service includes source tokenization, transfers, generation and detokenization; excludes load/warmups/scoring/writes; latency is amortized per sentence",
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "file_sha256": {name: sha256(out / name) for name in
                            ("predictions.txt", "references.txt", "examples.jsonl", "batches.jsonl")}}
        atomic_json(out / "metrics.json", result)
        return verify_result(out, rows, context)
    finally:
        model.train(was_training)
        set_rng(rng)


def evaluation_context(cfg, run, split, method, model_sha256, device, phase="final_inference"):
    data = read_json(Path(cfg["data_root"]) / "manifest.json")
    return {"plan_sha256": sha256(Path(run) / "plan.json"), "split": split, "method": method,
            "adaptation_method": cfg["method"], "seed": cfg["seed"], "model_sha256": model_sha256,
            "input": data["splits"][split], "batch_size": cfg["eval_batch_size"],
            "precision": "float32", "generation": cfg["generation"][method], "device": device,
            "phase": phase, "optimizer_resident": phase == "checkpoint_selection",
            "runtime": {"cpu_threads": cfg.get("cpu_threads", 4), "matmul_tf32": False, "cudnn_tf32": False,
                        "attention_implementation": "eager"}}


def export_payload(model, cfg, plan, update, evidence):
    state = learned_state(model, cfg["method"])
    return {"schema": 1, "method": cfg["method"], "update": update, "state": state,
            "state_sha256": state_digest(state), "config": cfg,
            "run_plan_sha256": sha256(Path(cfg["run_dir"]) / "plan.json"),
            "base_model_files": plan["model_files"], "parameter_evidence": evidence,
            "adapter_config": LORA if cfg["method"] == "lora" else None}


def verify_export_payload(payload, cfg, plan, expected_update=None, expected_state_sha256=None,
                          expected_evidence=None):
    required = ("schema", "method", "update", "state", "state_sha256", "config", "run_plan_sha256",
                "base_model_files", "parameter_evidence", "adapter_config")
    if any(key not in payload for key in required):
        raise ValueError("Incomplete learned-state export")
    if (payload["schema"] != 1 or payload["method"] != cfg["method"] or payload["config"] != cfg or
        payload["base_model_files"] != plan["model_files"] or
        payload["run_plan_sha256"] != sha256(Path(cfg["run_dir"]) / "plan.json") or
        payload["adapter_config"] != (LORA if cfg["method"] == "lora" else None) or
        payload["state_sha256"] != state_digest(payload["state"])):
        raise ValueError("Export identity or learned tensors changed")
    update = payload["update"]
    if type(update) is not int or not 1 <= update <= cfg["updates"] or update % cfg["validation_every"]:
        raise ValueError("Export update is not a declared validation checkpoint")
    if (expected_update is not None and update != expected_update or
        expected_state_sha256 is not None and payload["state_sha256"] != expected_state_sha256 or
        expected_evidence is not None and payload["parameter_evidence"] != expected_evidence):
        raise ValueError("Export differs from the expected selected checkpoint")
    return payload


def load_export(path, cfg, plan, device="cuda", expected_update=None, expected_state_sha256=None):
    payload = verify_export_payload(torch.load(path, map_location="cpu", weights_only=False), cfg, plan,
                                    expected_update, expected_state_sha256)
    model, tokenizer, evidence = build_model(cfg["model_dir"], cfg["method"], cfg["seed"], device,
                                           payload["parameter_evidence"]["target_modules"] if cfg["method"] == "lora" else None)
    if evidence != payload["parameter_evidence"] or payload["adapter_config"] != (LORA if cfg["method"] == "lora" else None):
        raise ValueError("Export adapter topology or parameters differ")
    restore_learned(model, payload["state"], cfg["method"])
    model.eval()
    return model, tokenizer, payload


def _verify_checkpoint(state, cfg, plan, schedule):
    required = ("config", "plan_sha256", "learned", "learned_sha256", "optimizer", "scheduler", "scaler",
                "rng", "update", "cursor", "records", "records_sha256", "best", "training_seconds",
                "peak_training_cuda_bytes", "parameter_evidence", "source_files")
    if any(k not in state for k in required):
        raise ValueError("Resume requires a complete last.pt")
    if state["config"] != cfg or state["plan_sha256"] != sha256(Path(cfg["run_dir"]) / "plan.json") or state["source_files"] != plan["source_files"]:
        raise ValueError("Resume identity differs from the frozen run")
    if state["learned_sha256"] != state_digest(state["learned"]) or state["records_sha256"] != digest(state["records"]):
        raise ValueError("Resume tensor or log evidence is damaged")
    if state.get("resume_sha256") != resume_digest(state):
        raise ValueError("Resume optimizer, scheduler, scaler or RNG integrity mismatch")
    update = state["update"]
    if type(update) is not int or not 0 <= update <= cfg["updates"] or state["cursor"] != update or len(state["records"]) != update:
        raise ValueError("Resume update/data cursor/log length mismatch")
    for i, record in enumerate(state["records"]):
        if record["update"] != i + 1 or record["ids_sha256"] != digest(schedule[i]):
            raise ValueError("Resume committed log differs from effective batch order")
        micro = [schedule[i][j:j + cfg["micro_batch_size"]] for j in range(0, len(schedule[i]), cfg["micro_batch_size"])]
        if (record["microbatch_sha256"] != digest(micro) or record["sentences"] != len(schedule[i]) or
            record["lr"] != cfg["lr"] * learning_rate_factor(i + 1, cfg["warmup_updates"], cfg["updates"]) or
            any(not math.isfinite(record[k]) for k in ("loss", "gradient_norm", "step_seconds"))):
            raise ValueError("Resume batch/LR/numerical evidence mismatch")
    if state["scheduler"] != {"completed_updates": update, "warmup": cfg["warmup_updates"], "total": cfg["updates"]}:
        raise ValueError("Resume LR scheduler position differs")


def reconcile_log(path, committed):
    """Only incomplete/tail-ahead logs can be repaired; committed corruption fails."""
    path = Path(path)
    if path.exists():
        raw = path.read_bytes(); end = raw.rfind(b"\n") + 1
        lines = raw[:end].splitlines()
        if len(lines) < len(committed):
            raise ValueError("Training log is missing committed records")
        for line, expected in zip(lines, committed):
            try:
                actual = json.loads(line)
            except (ValueError, UnicodeDecodeError) as e:
                raise ValueError("Committed training log is damaged") from e
            if actual != expected:
                raise ValueError("Committed training log differs from last.pt")
    elif committed:
        raise ValueError("Committed training log is missing")
    temporary = path.with_suffix(".repair")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in committed), encoding="utf-8")
    temporary.replace(path)


def verify_completed(cfg, plan, state, manifest, device):
    """Read-only verification of completed checkpoints, validation and selection."""
    run = Path(cfg["run_dir"])
    summary = read_json(run / "summary.json")
    if (summary["last_sha256"] != sha256(run / "last.pt") or summary["best_sha256"] != sha256(run / "best.pt") or
        summary["train_log_sha256"] != sha256(run / "train.jsonl") or summary["plan_sha256"] != sha256(run / "plan.json")):
        raise ValueError("Completed checkpoint/log/plan bytes changed")
    records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if records != state["records"]:
        raise ValueError("Completed training log differs from last.pt")
    rows = read_pairs(Path(manifest["data_dir"]) / "validation.jsonl")
    if len(rows) != 888:
        raise ValueError("Completed validation inputs are incomplete")
    best = None
    found = []
    validation_seconds = 0.0
    for record in records:
        if "validation" not in record:
            continue
        value = record["validation"]
        if value["update"] != record["update"]:
            raise ValueError("Completed validation checkpoint update mismatch")
        export = run / value["export"]
        evaluation = run / "validation" / f"step_{record['update']:05d}"
        if sha256(export) != value["export_sha256"] or sha256(evaluation / "metrics.json") != value["metrics_sha256"]:
            raise ValueError("Completed selection checkpoint or validation metrics changed")
        verify_export_payload(torch.load(export, map_location="cpu", weights_only=False), cfg, plan,
                              record["update"], value["state_sha256"], state["parameter_evidence"])
        context = evaluation_context(cfg, run, "validation", "greedy", value["state_sha256"], device, "checkpoint_selection")
        metrics = verify_result(evaluation, rows, context)
        if metrics["bleu"] != value["bleu"] or metrics["chrf2"] != value["chrf2"]:
            raise ValueError("Completed checkpoint-selection score differs")
        validation_seconds += metrics["elapsed_seconds"]
        found.append(record["update"])
        if better_checkpoint(value, best):
            best = value
    if found != list(range(cfg["validation_every"], cfg["updates"] + 1, cfg["validation_every"])):
        raise ValueError("Completed validation schedule is incomplete")
    if best != state["best"] or best != summary["best"]:
        raise ValueError("Completed best selection differs from the frozen rule")
    verify_export_payload(torch.load(run / "best.pt", map_location="cpu", weights_only=False), cfg, plan,
                          best["update"], best["state_sha256"], state["parameter_evidence"])
    expected = {"method": cfg["method"], "seed": cfg["seed"], "lr": cfg["lr"], "updates": cfg["updates"],
        "training_seconds": state["training_seconds"], "peak_training_cuda_bytes": state["peak_training_cuda_bytes"],
        "parameter_evidence": state["parameter_evidence"], "sentence_exposures": sum(r["sentences"] for r in records),
        "source_token_exposures": sum(r["source_tokens"] for r in records),
        "target_token_exposures": sum(r["target_tokens"] for r in records),
        "overflow_retries": sum(r["overflow_retries"] for r in records), "validation_seconds": validation_seconds,
        "last_bytes": (run / "last.pt").stat().st_size, "best_export_bytes": (run / "best.pt").stat().st_size}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed summary differs from saved evidence")
    return summary


def train(config_path, device="cuda", stop_after=None):
    import portalocker
    cfg, plan = verify_run(config_path)
    run = Path(cfg["run_dir"])
    with portalocker.Lock(run / "execution.lock", mode="a", timeout=0):
        try:
            return _train(cfg, plan, device, stop_after)
        except Exception as error:
            record = {"failed_at": now(), "config_sha256": sha256(config_path), "error_type": type(error).__name__,
                      "error": str(error), "last_exists": (run / "last.pt").exists(),
                      "resume": f".\\project-pretrained-python.ps1 -m mt.adapt_train train --config {run / 'config.json'}"}
            with (run / "failures.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raise


def _train(cfg, plan, device, stop_after):
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("Requested pause must be at a positive optimizer update")
    manifest, indexed, schedule = load_data(cfg["data_root"], cfg["seed"])
    for key in ("updates", "effective_batch_size", "micro_batch_size"):
        if manifest[key] != cfg[key]:
            raise ValueError("Run and data schedule no longer match")
    if cfg["train_precision"] == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("Frozen BF16 precision is unsupported on this device")
    configure_runtime(cfg)
    run = Path(cfg["run_dir"])
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if (run / "last.pt").exists() else None
    if state is not None:
        _verify_checkpoint(state, cfg, plan, schedule)
        if state["update"] == cfg["updates"] and (run / "summary.json").exists():
            # Preserve all completed bytes; strict evidence validation is CPU-only.
            return verify_completed(cfg, plan, state, manifest, device)
    elif (run / "train.jsonl").exists():
        if (run / "train.jsonl").stat().st_size:
            raise ValueError("Training started but no complete last.pt exists")
    model, tokenizer, evidence = build_model(cfg["model_dir"], cfg["method"], cfg["seed"], device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg["train_precision"] == "float16"))
    records, best, elapsed, peak = [], None, 0.0, 0
    if state is not None:
        if state["parameter_evidence"] != evidence:
            raise ValueError("Resume adapter targets/parameters differ")
        restore_learned(model, state["learned"], cfg["method"])
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        records, best = state["records"], state["best"]
        elapsed, peak = state["training_seconds"], state["peak_training_cuda_bytes"]
        set_rng(state["rng"])
    else:
        seed_all(cfg["seed"])  # Shared dropout RNG starts after method-specific adapter initialization.
        exclusive_json(run / "parameter_evidence.json", evidence)
    reconcile_log(run / "train.jsonl", records)
    validation = read_pairs(Path(manifest["data_dir"]) / "validation.jsonl")
    if len(validation) != 888:
        raise ValueError("Adaptation checkpoint selection requires all 888 validation rows")
    dtype = torch.bfloat16 if cfg["train_precision"] == "bfloat16" else torch.float16
    model.train()

    def save_last():
        weights = learned_state(model, cfg["method"])
        saved = {"schema": 1, "config": cfg, "plan_sha256": sha256(run / "plan.json"),
            "learned": weights, "learned_sha256": state_digest(weights), "optimizer": optimizer.state_dict(),
            "scheduler": {"completed_updates": len(records), "warmup": cfg["warmup_updates"], "total": cfg["updates"]},
            "scaler": scaler.state_dict(), "rng": rng_state(device == "cuda"), "update": len(records), "cursor": len(records),
            "records": records, "records_sha256": digest(records), "best": best, "training_seconds": elapsed,
            "peak_training_cuda_bytes": peak, "parameter_evidence": evidence, "source_files": plan["source_files"]}
        saved["resume_sha256"] = resume_digest(saved)
        atomic_torch(run / "last.pt", saved)

    # A complete update-zero checkpoint makes early disconnects safely recoverable.
    if state is None:
        save_last()
    stop = min(stop_after or cfg["updates"], cfg["updates"])
    for cursor in range(len(records), stop):
        ids = schedule[cursor]
        items = [indexed[i] for i in ids]
        total_tokens = sum(len(r["target_ids"]) for r in items)
        source_tokens = sum(len(r["source_ids"]) for r in items)
        lr = cfg["lr"] * learning_rate_factor(cursor + 1, cfg["warmup_updates"], cfg["updates"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        attempts, loss_value = 0, 0.0
        step_rng = rng_state(device == "cuda")
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        while True:
            attempts += 1
            if attempts > 12:
                raise FloatingPointError("Repeated FP16 overflow; last complete checkpoint is preserved")
            optimizer.zero_grad(set_to_none=True); loss_value = 0.0
            for offset in range(0, len(items), cfg["micro_batch_size"]):
                micro = items[offset:offset + cfg["micro_batch_size"]]
                batch = native_collate(micro, device)
                n = sum(len(r["target_ids"]) for r in micro)
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss.float() * (n / total_tokens)
                loss_value += float(loss.detach())
                scaler.scale(loss).backward()
                del outputs, batch, loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            previous_scale = scaler.get_scale()
            if not scaler.is_enabled() and not math.isfinite(float(norm)):
                raise FloatingPointError("Nonfinite BF16/FP32 gradient; refusing an invalid update")
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous_scale:
                break
            set_rng(step_rng)
        if device == "cuda":
            torch.cuda.synchronize(); peak = max(peak, torch.cuda.max_memory_allocated())
        seconds = time.perf_counter() - started; elapsed += seconds
        record = {"update": cursor + 1, "ids_sha256": digest(ids),
            "microbatch_sha256": digest([ids[i:i + cfg["micro_batch_size"]] for i in range(0, len(ids), cfg["micro_batch_size"])]),
            "sentences": len(ids), "source_tokens": source_tokens, "target_tokens": total_tokens,
            "lr": lr, "loss": loss_value, "gradient_norm": float(norm), "overflow_retries": attempts - 1,
            "step_seconds": seconds, "training_seconds_total": elapsed}
        if (cursor + 1) % cfg["validation_every"] == 0:
            payload = export_payload(model, cfg, plan, cursor + 1, evidence)
            export = run / f"checkpoint_{cursor + 1:05d}.pt"
            if export.exists():
                verify_export_payload(torch.load(export, map_location="cpu", weights_only=False), cfg, plan,
                                      cursor + 1, payload["state_sha256"], evidence)
            else:
                atomic_torch(export, payload)
            context = evaluation_context(cfg, run, "validation", "greedy", payload["state_sha256"], device, "checkpoint_selection")
            metrics = evaluate_model(model, tokenizer, validation, run / "validation" / f"step_{cursor + 1:05d}", context)
            candidate = {"update": cursor + 1, "bleu": metrics["bleu"], "chrf2": metrics["chrf2"],
                         "state_sha256": payload["state_sha256"], "export": export.name,
                         "export_sha256": sha256(export), "metrics_sha256": sha256(run / "validation" / f"step_{cursor + 1:05d}" / "metrics.json")}
            record["validation"] = candidate
            if better_checkpoint(candidate, best):
                best = candidate
                atomic_torch(run / "best.pt", payload)
            print(json.dumps({"run": run.name, "update": cursor + 1, "validation_bleu": metrics["bleu"],
                              "validation_chrf2": metrics["chrf2"], "best_update": best["update"]}), flush=True)
        records.append(record)
        with (run / "train.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush()
        if (cursor + 1) % cfg["save_every"] == 0 or cursor + 1 == stop:
            save_last()
        if (cursor + 1) % 50 == 0 or cursor + 1 == stop:
            atomic_json(run / "progress.json", {"updated_at": now(), "stage": "training", "update": cursor + 1,
                "budget": cfg["updates"], "best": best,
                "resume": f".\\project-pretrained-python.ps1 -m mt.adapt_train train --config {run / 'config.json'}"})
            print(json.dumps({"run": run.name, "update": cursor + 1, "loss": loss_value,
                              "training_seconds": elapsed}), flush=True)
    if len(records) < cfg["updates"]:
        return {"stage": "paused_at_requested_update", "update": len(records)}
    summary = {"completed_at": now(), "method": cfg["method"], "seed": cfg["seed"], "lr": cfg["lr"],
        "updates": len(records), "best": best, "parameter_evidence": evidence,
        "training_seconds": elapsed, "peak_training_cuda_bytes": peak,
        "sentence_exposures": sum(r["sentences"] for r in records),
        "source_token_exposures": sum(r["source_tokens"] for r in records),
        "target_token_exposures": sum(r["target_tokens"] for r in records),
        "overflow_retries": sum(r["overflow_retries"] for r in records),
        "validation_seconds": sum(read_json(run / "validation" / f"step_{r['update']:05d}" / "metrics.json")["elapsed_seconds"] for r in records if "validation" in r),
        "last_sha256": sha256(run / "last.pt"), "best_sha256": sha256(run / "best.pt"),
        "best_export_bytes": (run / "best.pt").stat().st_size, "last_bytes": (run / "last.pt").stat().st_size,
        "train_log_sha256": sha256(run / "train.jsonl"), "plan_sha256": sha256(run / "plan.json")}
    exclusive_json(run / "summary.json", summary)
    atomic_json(run / "progress.json", {"stage": "training_complete", "completed_at": now(), "update": len(records)})
    return summary


def evaluate_export(config_path, split, method, device="cuda"):
    import portalocker
    cfg = read_json(config_path)
    with portalocker.Lock(Path(cfg["run_dir"]) / "execution.lock", mode="a", timeout=0):
        return _evaluate_export(config_path, split, method, device)


def _evaluate_export(config_path, split, method, device="cuda"):
    cfg, plan = verify_run(config_path)
    configure_runtime(cfg)
    if split not in ("validation", "test") or method not in ("greedy", "beam"):
        raise ValueError("Unknown final split or decoder")
    run = Path(cfg["run_dir"])
    summary = read_json(run / "summary.json")
    if summary["best_sha256"] != sha256(run / "best.pt") or summary["updates"] != cfg["updates"]:
        raise ValueError("Formal evaluation requires a complete, intact training run")
    manifest = read_json(Path(cfg["data_root"]) / "manifest.json")
    raw_path = Path(manifest["data_dir"]) / f"{split}.jsonl"
    if sha256(raw_path) != manifest["splits"][split]["input_sha256"]:
        raise ValueError("Formal evaluation data differs from frozen native inputs")
    rows = read_pairs(raw_path)
    if len(rows) != manifest["splits"][split]["samples"] or digest([r["id"] for r in rows]) != manifest["splits"][split]["ids_sha256"]:
        raise ValueError("Formal evaluation row count/order differs from frozen native inputs")
    context = evaluation_context(cfg, run, split, method, summary["best"]["state_sha256"], device)
    out = run / "evaluation" / split / method
    if (out / "metrics.json").exists():
        return verify_result(out, rows, context)
    model, tokenizer, _ = load_export(run / "best.pt", cfg, plan, device,
                                      summary["best"]["update"], summary["best"]["state_sha256"])
    return evaluate_model(model, tokenizer, rows, out, context)


def resource_smoke(model_dir, data_root, out, micro_batch_size=8, precision="bfloat16", device="cuda", updates=20):
    """Training-only forward/backward/update and exact zero-adapter checks."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("Resource smoke evidence cannot be overwritten")
    manifest, indexed, schedule = load_data(data_root, 7)
    ids = schedule[0]
    # Include the longest retained example, without changing any formal batch.
    retained = read_json(Path(data_root) / "retained_train_ids.json")
    longest = max(retained, key=lambda i: len(indexed[i]["source_ids"]) + len(indexed[i]["target_ids"]))
    smoke_ids = [longest] + [i for i in ids if i != longest][:manifest["effective_batch_size"] - 1]
    smoke_batches = [smoke_ids] + schedule[1:updates]
    result = {"created_at": now(), "scope": "training-only resource probe, excluded from formal update budget; first batch includes longest retained pair, following batches use frozen seed7 schedule",
              "ids": smoke_batches, "seed": 7, "precision": precision, "micro_batch_size": micro_batch_size,
              "effective_batch_size": len(smoke_ids), "source_files": sources(), "model_files": model_evidence(model_dir),
              "data_manifest_sha256": sha256(Path(data_root) / "manifest.json"), "methods": {},
              "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "bf16_supported": torch.cuda.is_bf16_supported() if device == "cuda" else False}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    for method in ("lora", "full"):
        model = tokenizer = optimizer = None
        batch = output = loss = first = None
        try:
            model, tokenizer, evidence = build_model(model_dir, "full" if method == "full" else "direct", 7, device)
            first = native_collate([indexed[i] for i in smoke_ids[:2]], device)
            if method == "lora":
                model.eval()
                with torch.inference_mode():
                    before = model(**first).logits.detach().cpu()
                    before_gen = model.generate(input_ids=first["input_ids"], attention_mask=first["attention_mask"], max_length=24, num_beams=1).cpu()
                targets = inject_lora(model)
                with torch.inference_mode():
                    after = model(**first).logits.detach().cpu()
                    after_gen = model.generate(input_ids=first["input_ids"], attention_mask=first["attention_mask"], max_length=24, num_beams=1).cpu()
                if not torch.equal(before, after) or not torch.equal(before_gen, after_gen):
                    raise ValueError("Untrained adapter changed native eval output")
                equality = {"logits_bitwise_equal": True, "generation_bitwise_equal": True,
                            "maximum_logit_difference": float((before - after).abs().max())}
                evidence = parameter_evidence(model, method, targets)
                del before, after, before_gen, after_gen
            else:
                equality = None
            del first
            position_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                               if n.endswith("embed_positions.weight")}
            learned_before = state_digest(learned_state(model, method))
            seed_all(7); model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4 if method == "lora" else 1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and precision == "float16")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            step_results = []
            started = time.perf_counter()
            for step, batch_ids in enumerate(smoke_batches, 1):
                step_start, loss_value = time.perf_counter(), 0.0
                total_tokens = sum(len(indexed[i]["target_ids"]) for i in batch_ids)
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(batch_ids), micro_batch_size):
                    items = [indexed[i] for i in batch_ids[offset:offset + micro_batch_size]]
                    batch = native_collate(items, device)
                    with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                        output = model(**batch, use_cache=False)
                        loss = output.loss.float() * sum(len(r["target_ids"]) for r in items) / total_tokens
                    loss_value += float(loss.detach()); scaler.scale(loss).backward()
                    del output, loss, batch
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if not math.isfinite(float(norm)):
                    raise FloatingPointError("Nonfinite resource-test gradient")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError("Resource probe skipped an optimizer update after overflow")
                if device == "cuda":
                    torch.cuda.synchronize()
                step_results.append({"update": step, "seconds": time.perf_counter() - step_start,
                    "loss": loss_value, "gradient_norm": float(norm), "ids_sha256": digest(batch_ids),
                    "source_tokens": sum(len(indexed[i]["source_ids"]) for i in batch_ids), "target_tokens": total_tokens})
                print(json.dumps({"smoke": method, "update": step, "seconds": step_results[-1]["seconds"]}), flush=True)
            duration = time.perf_counter() - started
            peak_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else None
            final_state = learned_state(model, method)
            learned_after = state_digest(final_state)
            position_after = {n: p.detach().cpu() for n, p in model.named_parameters()
                              if n.endswith("embed_positions.weight")}
            positions_unchanged = all(torch.equal(position_before[n], t) for n, t in position_after.items())
            if learned_after == learned_before or not positions_unchanged:
                raise ValueError("Resource updates did not change learned weights or changed native positions")
            # Export/reload check uses actual learned state, without writing over any model.
            model.eval()
            first = native_collate([indexed[smoke_ids[0]]], device)
            with torch.inference_mode():
                original_logits = model(**first).logits.detach().cpu()
            reloaded, _, _ = build_model(model_dir, method, 7, device)
            restore_learned(reloaded, final_state, method)
            reloaded.eval()
            with torch.inference_mode():
                reloaded_logits = reloaded(**first).logits.detach().cpu()
            reload_equal = torch.equal(original_logits, reloaded_logits)
            del reloaded, original_logits, reloaded_logits, final_state, first
            if not reload_equal:
                raise ValueError("Learned state export/reload changed native FP32 logits")
            result["methods"][method] = {"status": "passed", "parameter_evidence": evidence,
                "elapsed_seconds": duration, "loss": loss_value, "gradient_norm": float(norm),
                "peak_cuda_bytes": peak_bytes,
                "step_results": step_results,
                "steady_mean_step_seconds": sum(r["seconds"] for r in step_results[1:]) / max(1, len(step_results) - 1),
                "learned_before_sha256": learned_before, "learned_after_sha256": learned_after,
                "learned_weights_changed": True, "native_positions_unchanged": positions_unchanged,
                "learned_export_reload_logits_bitwise_equal": reload_equal,
                "zero_adapter_equivalence": equality, "optimizer_updates": len(step_results)}
        except Exception as e:
            result["methods"][method] = {"status": "failed", "error_type": type(e).__name__, "error": str(e)}
        finally:
            batch = output = loss = first = None
            del model, tokenizer, optimizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    result["passed"] = all(r["status"] == "passed" for r in result["methods"].values())
    exclusive_json(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "verify", "train", "evaluate", "smoke"))
    p.add_argument("--config"); p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--split", choices=("validation", "test")); p.add_argument("--method", choices=("greedy", "beam"))
    p.add_argument("--stop-after", type=int); p.add_argument("--model-dir"); p.add_argument("--data-root")
    p.add_argument("--out"); p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--precision", default="bfloat16", choices=("bfloat16", "float16"))
    a = p.parse_args()
    if a.action == "freeze": result = freeze_run(a.config)
    elif a.action == "verify": result = verify_run(a.config)
    elif a.action == "train": result = train(a.config, a.device, a.stop_after)
    elif a.action == "evaluate": result = evaluate_export(a.config, a.split, a.method, a.device)
    else: result = resource_smoke(a.model_dir, a.data_root, a.out, a.micro_batch_size, a.precision, a.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if a.action == "smoke" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
