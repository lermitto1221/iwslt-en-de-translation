"""Export inference weights, native BPE and frozen decoding parameters together."""
import argparse
from pathlib import Path
import shutil
import tempfile

import torch

from .common import read_json, sha256, write_json
from .inference import resolve_data_dir, tokenizer_fingerprint, validate_protocol


def export_bundle(checkpoint, protocol, destination, data_dir=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"bundle destination already exists: {destination}; choose a new directory")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = resolve_data_dir(checkpoint, state["config"], tokenizer_fingerprint(state), data_dir)
    selected = read_json(protocol)
    settings = validate_protocol(selected)
    if selected.get("selection_split") != "validation":
        raise ValueError("export requires a decoding protocol selected on validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        folder = Path(temporary) / "bundle"
        folder.mkdir()
        torch.save({"config": state["config"], "model": state["model"]}, folder / "model.pt")
        shutil.copyfile(source / "spm.model", folder / "spm.model")
        write_json(folder / "protocol.json", {**settings, "selection_split": "validation"})
        write_json(folder / "manifest.json", {
            "format": "mlproject-inference-v1", "direction": "en-de",
            "source_checkpoint_sha256": sha256(checkpoint), "checkpoint_update": state.get("update"),
            "source_protocol_sha256": sha256(protocol),
            "smoke": bool(state.get("smoke", False)), "pilot": bool(state.get("pilot", False)),
            "files": {name: sha256(folder / name) for name in ("model.pt", "spm.model", "protocol.json")},
            "note": "Requires this project's model code and Python dependencies; contains no optimizer, RNG states or training corpus. Hashes detect changes, not publisher authenticity."})
        folder.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    print(export_bundle(args.checkpoint, args.protocol, args.out, args.data_dir))


if __name__ == "__main__":
    main()
