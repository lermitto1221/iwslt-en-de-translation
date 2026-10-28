"""Native Marian adaptation with an explicit, dependency-free LoRA Linear layer.

This is a local implementation, not PEFT. Only q_proj/v_proj are adapted; the
native tokenizer, label shifting and generation implementation are preserved.
"""
import math
from pathlib import Path

import torch
from torch import nn

from .pretrained_mt import load_native, model_evidence, digest

LORA = {"r": 8, "alpha": 16, "dropout": 0.0, "bias": "none",
        "target_suffixes": ["q_proj", "v_proj"], "implementation": "mt.adapt_model.LoRALinear"}


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or r < 1 or dropout != 0:
            raise ValueError("The frozen study supports Linear, positive rank and zero LoRA dropout")
        self.base = base.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device,
                                              dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device,
                                              dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(model, config=None, expected_modules=None):
    config = dict(LORA if config is None else config)
    if config != LORA:
        raise ValueError("Adapter configuration differs from the declared study")
    model.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in config["target_suffixes"]]
    if not targets or any(not isinstance(m, nn.Linear) for _, m in targets):
        raise ValueError("LoRA targets are missing, wrapped already, or are not Linear")
    names = [name for name, _ in targets]
    if expected_modules is not None and names != expected_modules:
        raise ValueError("LoRA target module list differs from the frozen model")
    for name, module in targets:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(module, config["r"], config["alpha"], config["dropout"]))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable != [f"{name}.{part}" for name in names for part in ("lora_A", "lora_B")]:
        raise ValueError("Unexpected trainable base or missing adapter parameters")
    return names


def parameter_evidence(model, method, targets=()):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"method": method, "total_parameters": total, "trainable_parameters": trainable,
            "trainable_fraction": trainable / total, "target_modules": list(targets),
            "trainable_fraction_of_original_base": trainable / 74410496,
            "target_modules_sha256": digest(list(targets)),
            "adapter_config": LORA if method == "lora" else None,
            "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad]}


def build_model(model_dir, method, seed, device="cpu", expected_modules=None):
    from .common import seed_all
    model_dir = Path(model_dir).resolve()
    if model_dir.name != "model":
        raise ValueError("Native loader requires an explicitly named model directory")
    seed_all(seed)
    model, tokenizer = load_native(model_dir.parent, device)
    if method == "lora":
        targets = inject_lora(model, expected_modules=expected_modules)
    elif method == "full":
        model.requires_grad_(True)
        # Native sinusoidal positions are Parameters for serialization but their
        # forward is explicitly no_grad. They are not learned model weights.
        model.model.encoder.embed_positions.weight.requires_grad_(False)
        model.model.decoder.embed_positions.weight.requires_grad_(False)
        targets = []
    elif method == "direct":
        targets = []
    else:
        raise ValueError("method must be lora, full or direct")
    native = {"pad": tokenizer.pad_token_id, "eos": tokenizer.eos_token_id,
              "decoder_start": model.config.decoder_start_token_id}
    if native != {"pad": 58100, "eos": 0, "decoder_start": 58100}:
        raise ValueError("Unexpected native Marian special IDs")
    evidence = parameter_evidence(model, method, targets)
    evidence["native_fixed_position_parameters"] = 2 * 512 * 512
    evidence["native_fixed_position_names"] = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]
    evidence["native_fixed_position_definition"] = "Marian sinusoidal position tables; native forward is no_grad, retained unchanged for full adaptation"
    evidence["special_tokens"] = native
    if evidence["total_parameters"] - (evidence["trainable_parameters"] if method == "lora" else 0) != 74410496:
        raise ValueError("Unexpected fixed-base parameter count")
    if method == "lora" and (len(targets) != 36 or evidence["trainable_parameters"] != 294912):
        raise ValueError("Unexpected Marian LoRA coverage")
    return model, tokenizer, evidence


def learned_state(model, method):
    if method == "lora":
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    if method == "full":
        # Marian shares encoder/decoder embeddings and lm_head. Preserve that
        # storage aliasing in exports instead of serializing four large copies.
        memo, output = {}, {}
        for name, tensor in model.state_dict().items():
            identity = (tensor.data_ptr(), tensor.storage_offset(), tensor.dtype,
                        tuple(tensor.shape), tuple(tensor.stride()))
            if identity not in memo:
                memo[identity] = tensor.detach().cpu().clone()
            output[name] = memo[identity]
        return output
    raise ValueError("Only trained methods have learned state")


def restore_learned(model, state, method):
    if method == "lora":
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter state has missing or extra tensors")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(n in expected for n in result.missing_keys):
            raise ValueError("Adapter restore failed")
    elif method == "full":
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Unknown learned-state method")


def state_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def native_collate(records, device="cpu", pad=58100):
    """Targets already include native EOS. Native Marian shifts labels itself."""
    if not records or any(not r["source_ids"] or not r["target_ids"] for r in records):
        raise ValueError("Cannot collate empty native sequences")
    src = torch.full((len(records), max(len(r["source_ids"]) for r in records)), pad, dtype=torch.long)
    labels = torch.full((len(records), max(len(r["target_ids"]) for r in records)), -100, dtype=torch.long)
    for i, r in enumerate(records):
        if r["source_ids"][-1] != 0 or r["target_ids"][-1] != 0 or pad in r["target_ids"]:
            raise ValueError("Encoded native sequences must end in EOS and contain no target padding")
        src[i, :len(r["source_ids"])] = torch.tensor(r["source_ids"])
        labels[i, :len(r["target_ids"])] = torch.tensor(r["target_ids"])
    return {"input_ids": src.to(device), "attention_mask": src.ne(pad).long().to(device),
            "labels": labels.to(device)}
