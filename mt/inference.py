"""Verified, reusable inference sessions for checkpoints and portable bundles."""
from pathlib import Path
import re

import torch

from .common import read_json, seed_all, sha256
from .decoding import translate_batch
from .model import TranslationTransformer
from .tokenizer import BOS, EOS, PAD, UNK, load


def tokenizer_fingerprint(state):
    fingerprints = state.get("data_fingerprints") or state.get("fingerprints") or {}
    digest = fingerprints.get("tokenizer_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("checkpoint has no valid tokenizer SHA-256; the token-ID mapping cannot be verified")
    return digest


def resolve_data_dir(checkpoint, cfg, expected_hash, override=None):
    """Locate the original tokenizer without relying on the caller's cwd."""
    saved = Path(cfg["data_dir"])
    if override is not None:
        candidates = [Path(override).resolve()]
    elif saved.is_absolute():
        candidates = [saved]
    else:
        candidates = [parent / saved for parent in Path(checkpoint).resolve().parents]
        candidates.append(saved.resolve())
    found = []
    for candidate in dict.fromkeys(candidates):
        model = candidate / "spm.model"
        if model.is_file():
            found.append(str(model))
            if sha256(model) == expected_hash:
                return candidate.resolve()
    if found:
        raise ValueError("tokenizer SHA-256 mismatch; restore the checkpoint's original tokenizer: " + ", ".join(found))
    raise FileNotFoundError("original spm.model not found; provide --data-dir or use a portable --bundle")


def select_device(name="auto"):
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return device


def build_model(cfg, weights, tok, device):
    if [tok.pad_id(), tok.unk_id(), tok.bos_id(), tok.eos_id()] != [PAD, UNK, BOS, EOS]:
        raise ValueError("tokenizer special IDs differ from the model's PAD/UNK/BOS/EOS convention")
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(cfg.get("deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = False
    model = TranslationTransformer(tok.vocab_size(), cfg)
    model.load_state_dict(weights)
    return model.to(device=device, dtype=torch.float32).eval()


def load_checkpoint(checkpoint, device, data_dir=None):
    # Training checkpoints contain optimizer and RNG state. Deserialize once,
    # verify token IDs before allocating the inference model on the device.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    data = resolve_data_dir(checkpoint, cfg, tokenizer_fingerprint(state), data_dir)
    tok = load(data)
    model = build_model(cfg, state["model"], tok, device)
    return model, tok, cfg, state


def validate_protocol(protocol):
    import math
    if (type(protocol.get("beam_size")) is not int or protocol["beam_size"] < 1
            or type(protocol.get("max_generation_tokens")) is not int
            or not 1 <= protocol["max_generation_tokens"] <= 4095
            or not isinstance(protocol.get("length_penalty"), (int, float))
            or not math.isfinite(protocol["length_penalty"])):
        raise ValueError("invalid beam size, length penalty or generation length in decoding protocol")
    return {key: protocol[key] for key in ("beam_size", "length_penalty", "max_generation_tokens")}


class Translator:
    """Load once, then translate many batches with one fixed token mapping."""
    def __init__(self, checkpoint=None, bundle=None, protocol=None, device="auto", data_dir=None):
        if (checkpoint is None) == (bundle is None):
            raise ValueError("choose exactly one checkpoint or bundle")
        self.device = select_device(device)
        if checkpoint is not None:
            self.model, self.tokenizer, self.config, state = load_checkpoint(checkpoint, self.device, data_dir)
            self.data_dir = resolve_data_dir(checkpoint, self.config, tokenizer_fingerprint(state), data_dir)
            default_protocol = self.config
            del state
        else:
            folder = Path(bundle).resolve()
            manifest = read_json(folder / "manifest.json")
            if manifest.get("format") != "mlproject-inference-v1":
                raise ValueError("unsupported inference bundle format")
            for name in ("model.pt", "spm.model", "protocol.json"):
                if sha256(folder / name) != manifest.get("files", {}).get(name):
                    raise ValueError(f"bundle file checksum mismatch: {name}")
            state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            self.config = state["config"]
            self.tokenizer = load(folder)
            self.model = build_model(self.config, state["model"], self.tokenizer, self.device)
            self.data_dir = folder
            default_protocol = read_json(folder / "protocol.json")
            del state
        self.protocol = validate_protocol(read_json(protocol) if protocol else default_protocol)

    def translate(self, texts, method="beam", batch_size=None):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("input must contain nonempty strings")
        if any("\n" in text or "\r" in text for text in texts):
            raise ValueError("each input item must be a single sentence line")
        size = self.config.get("decode_batch_size", 16) if batch_size is None else batch_size
        if type(size) is not int or size < 1:
            raise ValueError("batch size must be a positive integer")
        predictions, hits = [], []
        for start in range(0, len(texts), size):
            values, flags = translate_batch(self.model, self.tokenizer, texts[start:start + size], method,
                self.protocol["beam_size"], self.protocol["length_penalty"], self.protocol["max_generation_tokens"])
            predictions.extend(values)
            hits.extend(flags)
        return predictions, hits
