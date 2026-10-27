"""Frozen, resumable direct inference with the native OPUS Marian model.

No training or adaptation takes place.  Formal outputs are independent of all
scratch-model runs.  A metrics.json is a completion marker, never a progress file.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import read_json, sha256, write_json
from .data import read_pairs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/pretrained_mt_2026_10_27"
MODEL_ID = "Helsinki-NLP/opus-mt-en-de"
REVISION = "6183067f769a302e3861815543b9f312c71b0ca4"
FILES = ("README.md", "config.json", "generation_config.json", "tokenizer_config.json",
         "source.spm", "target.spm", "vocab.json", "pytorch_model.bin")
DEPENDENCIES = ("torch", "numpy", "transformers", "huggingface-hub", "tokenizers",
                "safetensors", "sentencepiece", "sacremoses", "sacrebleu", "PyYAML",
                "requests", "urllib3", "certifi", "charset-normalizer", "idna",
                "tqdm", "joblib", "click", "filelock", "fsspec", "packaging", "regex")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def dependency_evidence():
    import torch
    import numpy
    import transformers
    result = {"python": platform.python_version(), "torch_cuda": torch.version.cuda,
              "packages": {}, "torch_module": str(Path(torch.__file__).resolve()),
              "numpy_module": str(Path(numpy.__file__).resolve()),
              "transformers_module": str(Path(transformers.__file__).resolve())}
    for name in DEPENDENCIES:
        dist = importlib.metadata.distribution(name)
        record = next((p for p in (dist.files or []) if p.name == "RECORD"), None)
        result["packages"][name] = {"version": dist.version,
            "record_sha256": sha256(dist.locate_file(record)) if record else None}
    # Bind the implementation involved in generation, tokenization and Marian.
    package = Path(transformers.__file__).parent
    result["transformers_implementation"] = {
        name: sha256(package / name) for name in (
            "generation/utils.py", "generation/beam_search.py",
            "generation/logits_process.py", "models/marian/modeling_marian.py",
            "models/marian/tokenization_marian.py", "tokenization_utils.py")}
    return result


def source_evidence():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in
            (Path(__file__), ROOT / "mt/common.py", ROOT / "mt/data.py",
             ROOT / "project-pretrained-python.ps1")}


def input_evidence(data_dir):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    evidence = {"manifest_sha256": sha256(data_dir / "manifest.json"), "splits": {}}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        checksum = sha256(path)
        rows = read_pairs(path)
        if checksum != manifest["files"][split]["jsonl_sha256"]:
            raise ValueError(f"{split} differs from historical data manifest")
        if len(rows) != manifest["files"][split]["rows_saved"]:
            raise ValueError(f"{split} sample count differs from manifest")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{split} contains duplicate IDs")
        evidence["splits"][split] = {"sha256": checksum, "samples": len(rows),
            "id_order_sha256": digest([r["id"] for r in rows]),
            "source_order_sha256": digest([r["en"] for r in rows]),
            "reference_order_sha256": digest([r["de"] for r in rows])}
    return evidence


def model_evidence(model_dir):
    return {name: {"sha256": sha256(Path(model_dir) / name),
                   "bytes": (Path(model_dir) / name).stat().st_size} for name in FILES}


def download(out):
    """Download a fixed revision, checking the model's published LFS digest."""
    import requests
    out = Path(out)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    remote = response.json()
    if remote["sha"] != REVISION:
        raise ValueError("Hub resolved an unexpected revision")
    listed = {item["rfilename"]: item for item in remote["siblings"]}
    for name in FILES:
        destination = model_dir / name
        expected = listed[name].get("lfs", {}).get("sha256")
        if destination.exists() and (not expected or sha256(destination) == expected):
            continue
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        temporary = destination.with_suffix(destination.suffix + ".download")
        with requests.get(url, stream=True, timeout=(30, 120)) as stream:
            stream.raise_for_status()
            with temporary.open("wb") as f:
                for block in stream.iter_content(1024 * 1024):
                    if block:
                        f.write(block)
        if expected and sha256(temporary) != expected:
            raise ValueError(f"Published LFS digest mismatch: {name}")
        temporary.replace(destination)
        print(json.dumps({"downloaded": name, "bytes": destination.stat().st_size}), flush=True)
    value = {"model_id": MODEL_ID, "revision": REVISION, "downloaded_at": now(),
             "hub_metadata_url": api_url, "files": model_evidence(model_dir),
             "published_lfs": {n: listed[n].get("lfs") for n in FILES}}
    marker = out / "download.json"
    if marker.exists():
        if read_json(marker)["files"] != value["files"]:
            raise ValueError("Previously recorded model download changed")
    else:
        exclusive_json(marker, value)
    return value


def load_native(out, device="cpu"):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    model_dir = Path(out) / "model"
    tokenizer = MarianTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_dir, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager", weights_only=True)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, tokenizer


def check_lengths(rows, tokenizer, maximum=512):
    lengths, oversize = [], []
    for offset in range(0, len(rows), 256):
        batch = rows[offset:offset + 256]
        encoded = tokenizer([r["en"] for r in batch], padding=False, truncation=False)["input_ids"]
        for row, tokens in zip(batch, encoded):
            lengths.append(len(tokens))
            if len(tokens) > maximum:
                oversize.append({"id": row["id"], "length": len(tokens)})
    return {"samples": len(rows), "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths), "oversize": oversize,
            "lengths_sha256": digest(lengths), "includes_special_tokens": True}


def audit_lengths(out=DEFAULT_OUT):
    from transformers import MarianTokenizer
    out = Path(out)
    config = read_json(out / "config.json")
    if (out / "plan.json").exists():
        raise FileExistsError("Length audit must precede plan freeze")
    tokenizer = MarianTokenizer.from_pretrained(out / "model", local_files_only=True)
    value = {"created_at": now(), "truncation": False,
             "config_sha256": sha256(out / "config.json"),
             "inputs_sha256": sha256(out / "inputs.json"),
             "splits": {split: check_lengths(read_pairs(Path(config["data_dir"]) / f"{split}.jsonl"),
                          tokenizer, config["maximum_source_tokens"])
                        for split in ("train", "validation", "test")},
             "training_scope": "training data only supply 16 resource-smoke examples; no training or fitting; oversize training rows are counted, never truncated or inferred"}
    for split in ("validation", "test"):
        if value["splits"][split] != config["length_audit"][split]:
            raise ValueError("Formal input length evidence changed")
    exclusive_json(out / "length_audit.json", value)
    return value


def prepare(out=DEFAULT_OUT, data_dir=ROOT / "artifacts/data"):
    out = Path(out)
    if (out / "plan.json").exists():
        raise FileExistsError("Plan already frozen; use verify or run")
    if (out / "config.json").exists():
        raise FileExistsError("Prepared configuration already exists; do not overwrite")
    download_info = read_json(out / "download.json")
    if download_info["revision"] != REVISION or download_info["files"] != model_evidence(out / "model"):
        raise ValueError("Downloaded model identity differs from manifest")
    model, tokenizer = load_native(out)
    inputs = input_evidence(data_dir)
    lengths = {split: check_lengths(read_pairs(Path(data_dir) / f"{split}.jsonl"), tokenizer,
                                 model.config.max_position_embeddings)
               for split in ("validation", "test")}
    if any(value["oversize"] for value in lengths.values()):
        exclusive_json(out / "length_blocker.json", lengths)
        raise ValueError("Long inputs require a new frozen segmentation protocol before evaluation")
    generation = model.generation_config.to_dict()
    generation.update({"max_length": 512, "do_sample": False, "renormalize_logits": True,
                       "num_return_sequences": 1})
    configs = {method: dict(generation, num_beams=1 if method == "greedy" else 4)
               for method in ("greedy", "beam")}
    params = sum(p.numel() for p in model.parameters())
    value = {"schema": 1, "created_at": now(), "model_id": MODEL_ID, "revision": REVISION,
        "data_dir": str(Path(data_dir).resolve()), "out": str(out.resolve()), "training_updates": 0,
        "batch_size": 16, "warmup_batches": 2, "precision": "float32",
        "attention_implementation": "eager", "truncate": False,
        "maximum_source_tokens": model.config.max_position_embeddings,
        "methods": ["greedy", "beam"], "splits": ["validation", "test"],
        "generation": configs, "parameters": params, "adapted_parameters": 0,
        "parameter_count_definition": "distinct model Parameter objects, tied weights counted once",
        "metrics": {"bleu": {"tokenize": "13a", "lowercase": False},
                    "chrf": {"char_order": 6, "word_order": 0, "beta": 2}},
        "seed": 42, "cpu_threads": 4, "tf32": False,
        "training_smoke_ids": [r["id"] for r in read_pairs(Path(data_dir) / "train.jsonl")[:16]],
        "length_audit": lengths,
        "interpretation": "OPUS pretrained direct inference; pretraining overlap with IWSLT/TED cannot be excluded; no causal attribution to architecture, representation, or LoRA"}
    exclusive_json(out / "environment.json", dependency_evidence())
    exclusive_json(out / "inputs.json", inputs)
    exclusive_json(out / "config.json", value)
    return value


def generate_batch(model, tokenizer, rows, generation, max_source):
    import torch
    from transformers import GenerationConfig
    encoded = tokenizer([r["en"] for r in rows], return_tensors="pt", padding=True,
                        truncation=False)
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if max(lengths) > max_source:
        raise ValueError("Input exceeds frozen position capacity; truncation is forbidden")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        sequences = model.generate(**encoded, generation_config=GenerationConfig.from_dict(generation))
    sequences = sequences.cpu().tolist()
    predictions = tokenizer.batch_decode(sequences, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
    output_lengths = []
    unfinished = 0
    eos, pad = generation["eos_token_id"], generation["pad_token_id"]
    for sequence in sequences:
        # Decoder start is PAD in Marian; exclude it, stop at first generated EOS.
        generated = sequence[1:]
        if eos in generated:
            length = generated.index(eos) + 1
        else:
            length = len([token for token in generated if token != pad])
            unfinished += 1
        output_lengths.append(length)
    return {"predictions": predictions, "source_lengths": lengths,
            "generated_lengths": output_lengths,
            "limit_hits": sum(n + 1 >= generation["max_length"] for n in output_lengths),
            "unfinished": unfinished}


def configure_runtime(config):
    import torch
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]


def smoke(out=DEFAULT_OUT, device="cuda"):
    import torch
    out = Path(out)
    if (out / "plan.json").exists():
        raise FileExistsError("Resource smoke must precede the frozen formal plan")
    config = read_json(out / "config.json")
    configure_runtime(config)
    rows_by_id = {r["id"]: r for r in read_pairs(Path(config["data_dir"]) / "train.jsonl")}
    rows = [rows_by_id[i] for i in config["training_smoke_ids"]]
    model, tokenizer = load_native(out, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = {}
    for method in config["methods"]:
        start = time.perf_counter()
        result = generate_batch(model, tokenizer, rows, config["generation"][method],
                                config["maximum_source_tokens"])
        if device == "cuda":
            torch.cuda.synchronize()
        results[method] = {"elapsed_seconds": time.perf_counter() - start, **result}
    value = {"completed_at": now(), "split": "train", "ids": config["training_smoke_ids"],
             "config_sha256": sha256(out / "config.json"), "device": device,
             "source_sha256": sha256(__file__), "results": results,
             "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    exclusive_json(out / "smoke.json", value)
    return value


def freeze(out=DEFAULT_OUT):
    out = Path(out)
    config = read_json(out / "config.json")
    if any((out / split).exists() for split in config["splits"]):
        raise ValueError("Cannot freeze a plan after formal evaluations started")
    resource = read_json(out / "smoke.json")
    if resource["config_sha256"] != sha256(out / "config.json") or resource["split"] != "train":
        raise ValueError("Resource smoke does not match the frozen configuration")
    if resource["source_sha256"] != sha256(__file__):
        raise ValueError("Code changed after smoke")
    if read_json(out / "environment.json") != dependency_evidence():
        raise ValueError("Dependencies changed after preparation")
    if read_json(out / "inputs.json") != input_evidence(config["data_dir"]):
        raise ValueError("Input evidence changed after preparation")
    length_audit = read_json(out / "length_audit.json")
    if length_audit["config_sha256"] != sha256(out / "config.json") or length_audit["inputs_sha256"] != sha256(out / "inputs.json"):
        raise ValueError("Length audit differs from prepared input/configuration")
    value = {"schema": 1, "created_at": now(), "status": "frozen_before_formal_inference",
        "scope": "new independent pretrained baseline after historical scratch results were known",
        "config_sha256": sha256(out / "config.json"), "input_sha256": sha256(out / "inputs.json"),
        "environment_sha256": sha256(out / "environment.json"),
        "download_sha256": sha256(out / "download.json"), "smoke_sha256": sha256(out / "smoke.json"),
        "length_audit_sha256": sha256(out / "length_audit.json"),
        "source_files": source_evidence(), "model_files": model_evidence(out / "model"),
        "training_updates": 0, "test_use": "final report only; no selection or protocol adjustment",
        "batch_order": "original JSONL order, identical full source/reference sentences across methods",
        "stopping": "one full validation and test pass per method, strict completed-cache reuse",
        "limitations": ["unknown OPUS pretraining overlap", "different size/data/native tokenizer",
            "native Transformers generation differs from project beam implementation",
            "speed comparisons span dates, power and temperature conditions"]}
    exclusive_json(out / "plan.json", value)
    return value


def verify_plan(out=DEFAULT_OUT):
    out = Path(out)
    plan = read_json(out / "plan.json")
    for name, key in (("config", "config_sha256"), ("inputs", "input_sha256"),
                      ("environment", "environment_sha256"), ("download", "download_sha256"),
                      ("smoke", "smoke_sha256"), ("length_audit", "length_audit_sha256")):
        if sha256(out / f"{name}.json") != plan[key]:
            raise ValueError(f"Frozen {name} changed")
    if plan["source_files"] != source_evidence():
        raise ValueError("Frozen inference implementation changed")
    if plan["model_files"] != model_evidence(out / "model"):
        raise ValueError("Frozen model/tokenizer files changed")
    if read_json(out / "environment.json") != dependency_evidence():
        raise ValueError("Frozen environment changed")
    config = read_json(out / "config.json")
    if read_json(out / "inputs.json") != input_evidence(config["data_dir"]):
        raise ValueError("Frozen data changed")
    return config, plan


def score(predictions, references):
    import sacrebleu
    if len(predictions) != len(references) or not references:
        raise ValueError("Predictions and references must be complete and aligned")
    bleu = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    chrf = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2)
    return {"bleu": bleu.corpus_score(predictions, [references]).score,
            "chrf2": chrf.corpus_score(predictions, [references]).score,
            "bleu_signature": str(bleu.get_signature()), "chrf2_signature": str(chrf.get_signature()),
            "sacrebleu_version": sacrebleu.__version__}


def read_journal(path, rows, batch_size):
    """Read complete, hashed batches; an interrupted final line is recoverable."""
    path = Path(path)
    if not path.exists():
        return [], 0
    raw = path.read_bytes()
    completed_end = raw.rfind(b"\n") + 1
    records = []
    offset = 0
    for line in raw[:completed_end].splitlines():
        record = json.loads(line)
        checksum = record.pop("record_sha256", None)
        if checksum != digest(record):
            raise ValueError("Damaged batch journal")
        expected = rows[offset:offset + batch_size]
        if record["ids"] != [r["id"] for r in expected]:
            raise ValueError("Batch journal ID/order mismatch")
        if len(record["predictions"]) != len(expected) or not expected:
            raise ValueError("Batch journal alignment mismatch")
        if not math.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] <= 0:
            raise ValueError("Invalid journal timing")
        if len(record["source_lengths"]) != len(expected) or len(record["generated_lengths"]) != len(expected):
            raise ValueError("Batch journal token counts mismatch")
        record["record_sha256"] = checksum
        records.append(record)
        offset += len(expected)
    return records, completed_end


def summarize_records(records, samples, device):
    elapsed = sum(record["elapsed_seconds"] for record in records)
    hits = sum(record["limit_hits"] for record in records)
    unfinished = sum(record["unfinished"] for record in records)
    if samples < 1 or not records or elapsed <= 0:
        raise ValueError("Cannot summarize an empty evaluation")
    return {"elapsed_seconds": elapsed, "sentences_per_second": samples / elapsed,
        "mean_latency_ms": 1000 * elapsed / samples,
        "mean_batch_latency_ms": 1000 * elapsed / len(records),
        "generation_limit_hits": hits, "generation_limit_fraction": hits / samples,
        "unfinished_generations": unfinished, "unfinished_fraction": unfinished / samples,
        "peak_cuda_bytes": max((r["peak_cuda_bytes"] or 0 for r in records), default=0) if device == "cuda" else None}


def verify_result(directory, rows, context):
    directory = Path(directory)
    metrics = read_json(directory / "metrics.json")
    if metrics["context"] != context or metrics["samples"] != len(rows):
        raise ValueError("Completed result has a different frozen identity")
    for name in ("predictions.txt", "references.txt", "examples.jsonl", "batches.jsonl"):
        if sha256(directory / name) != metrics["file_sha256"][name]:
            raise ValueError(f"Completed result changed: {name}")
    predictions = (directory / "predictions.txt").read_text(encoding="utf-8").splitlines()
    references = (directory / "references.txt").read_text(encoding="utf-8").splitlines()
    examples = [json.loads(line) for line in (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [{"id": row["id"], "source": row["en"], "reference": row["de"], "prediction": pred}
                for row, pred in zip(rows, predictions)]
    if len(predictions) != len(rows) or references != [r["de"] for r in rows] or examples != expected:
        raise ValueError("Completed source/reference/prediction alignment mismatch")
    recomputed = score(predictions, references)
    for name, value in recomputed.items():
        if isinstance(value, float):
            saved = metrics[name]
            if type(saved) not in (int, float) or not math.isfinite(saved) or abs(value - saved) > 1e-10:
                raise ValueError(f"Completed metric mismatch: {name}")
        elif value != metrics[name]:
            raise ValueError(f"Completed metric signature mismatch: {name}")
    records, end = read_journal(directory / "batches.jsonl", rows, context["batch_size"])
    if end != (directory / "batches.jsonl").stat().st_size:
        raise ValueError("Completed journal has an incomplete tail")
    if [p for record in records for p in record["predictions"]] != predictions:
        raise ValueError("Completed batch journal predictions differ")
    for key, expected_value in summarize_records(records, len(rows), context["device"]).items():
        actual = metrics.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isfinite(actual) and abs(expected_value - actual) < 1e-10
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(f"Completed {key} differs from batch journal")
    return metrics


def evaluate(out, split, method, config, model=None, tokenizer=None, device="cuda"):
    import torch
    out = Path(out)
    rows = read_pairs(Path(config["data_dir"]) / f"{split}.jsonl")
    directory = out / split / method
    context = {"plan_sha256": sha256(out / "plan.json"), "config_sha256": sha256(out / "config.json"),
               "split": split, "method": method, "batch_size": config["batch_size"],
               "input": read_json(out / "inputs.json")["splits"][split],
               "generation": config["generation"][method], "precision": config["precision"],
               "device": device}
    if (directory / "metrics.json").exists():
        return verify_result(directory, rows, context)
    directory.mkdir(parents=True, exist_ok=True)
    start_path = directory / "started.json"
    if start_path.exists():
        if read_json(start_path)["context"] != context:
            raise ValueError("Incomplete evaluation has a different identity")
    else:
        exclusive_json(start_path, {"started_at": now(), "context": context})
    journal = directory / "batches.jsonl"
    records, valid_end = read_journal(journal, rows, config["batch_size"])
    if journal.exists() and journal.stat().st_size != valid_end:
        with journal.open("r+b") as f:
            f.truncate(valid_end)
    completed = sum(len(r["ids"]) for r in records)
    if model is None:
        model, tokenizer = load_native(out, device)
    if completed < len(rows):
        for _ in range(config["warmup_batches"]):
            generate_batch(model, tokenizer, rows[:config["batch_size"]],
                           context["generation"], config["maximum_source_tokens"])
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for offset in range(completed, len(rows), config["batch_size"]):
            batch = rows[offset:offset + config["batch_size"]]
            start = time.perf_counter()
            result = generate_batch(model, tokenizer, batch, context["generation"],
                                    config["maximum_source_tokens"])
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            record = {"ids": [r["id"] for r in batch], "elapsed_seconds": elapsed,
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                      **result}
            record["record_sha256"] = digest(record)
            with journal.open("ab") as f:
                f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
            records.append(record)
            completed += len(batch)
            atomic_json(out / "progress.json", {"updated_at": now(), "split": split,
                "method": method, "completed": completed, "samples": len(rows)})
            if len(records) % 32 == 0 or completed == len(rows):
                print(json.dumps({"split": split, "method": method, "completed": completed,
                                  "samples": len(rows)}), flush=True)
    predictions = [p for record in records for p in record["predictions"]]
    references = [row["de"] for row in rows]
    if any("\n" in text or "\r" in text for text in predictions + references):
        raise ValueError("Multiline predictions cannot be written as aligned text")
    (directory / "predictions.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    (directory / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
    with (directory / "examples.jsonl").open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, predictions):
            f.write(json.dumps({"id": row["id"], "source": row["en"], "reference": row["de"],
                                "prediction": pred}, ensure_ascii=False) + "\n")
    metrics = {"context": context, "completed_at": now(), "samples": len(rows),
        **score(predictions, references), **summarize_records(records, len(rows), device),
        "generation_limit_definition": "decoder length includes start token; first generated EOS at length 512 also counts as reaching limit, even though native forced_eos_token_id forces termination",
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "parameters": config["parameters"], "training_updates": 0,
        "warmup_batches_per_attempt": config["warmup_batches"],
        "timing": "sum of synchronized batch service durations, includes native source tokenization, H2D transfer, generation, D2H and target detokenization; excludes model loading, two warmups per attempt, scoring, progress and disk writes; mean_latency_ms is amortized per sentence; final batch may be smaller; resumed batches retain original timings",
        "file_sha256": {name: sha256(directory / name) for name in
                        ("predictions.txt", "references.txt", "examples.jsonl", "batches.jsonl")}}
    atomic_json(directory / "metrics.json", metrics)
    return verify_result(directory, rows, context)


def _run_locked(out=DEFAULT_OUT, device="cuda"):
    out = Path(out)
    config, _ = verify_plan(out)
    configure_runtime(config)
    model, tokenizer = None, None
    results = {}
    for split in config["splits"]:
        results[split] = {}
        for method in config["methods"]:
            if not (out / split / method / "metrics.json").exists() and model is None:
                model, tokenizer = load_native(out, device)
            result = evaluate(out, split, method, config, model, tokenizer, device)
            results[split][method] = {key: result[key] for key in
                ("samples", "bleu", "chrf2", "elapsed_seconds", "sentences_per_second",
                 "mean_latency_ms", "mean_batch_latency_ms", "peak_cuda_bytes", "generation_limit_hits")}
    summary = {"completed_at": now(), "plan_sha256": sha256(out / "plan.json"),
               "model_id": MODEL_ID, "revision": REVISION, "training_updates": 0,
               "parameters": config["parameters"], "results": results}
    summary_path = out / "summary.json"
    if summary_path.exists():
        prior = read_json(summary_path)
        if any(prior[key] != summary[key] for key in summary if key != "completed_at"):
            raise ValueError("Saved summary differs from verified evaluations")
        return prior
    exclusive_json(summary_path, summary)
    atomic_json(out / "progress.json", {"stage": "complete", "completed_at": now()})
    return summary


def run(out=DEFAULT_OUT, device="cuda"):
    import portalocker
    # A second invocation must not share the GPU or append to this journal.
    lock_path = Path(out) / "execution.lock"
    with portalocker.Lock(lock_path, mode="a", timeout=0):
        return _run_locked(out, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare", "audit_lengths", "smoke", "freeze", "verify", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/data")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.out, args.data_dir)
    elif args.action in ("run", "smoke"):
        result = globals()[args.action](args.out, args.device)
    elif args.action == "verify":
        result = verify_plan(args.out)
    else:
        result = globals()[args.action](args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
