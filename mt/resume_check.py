"""Compare uninterrupted training against a saved-and-resumed run on one device."""
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from .common import read_json, sha256, write_json
from .train import train


def _equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def run_check(device="cpu", config_path="smoke.json", output_root="artifacts/checks"):
    cfg = read_json(config_path)
    cfg.update(max_updates=6, validate_every=6, accum_steps=2, device=device,
               amp=device == "cuda", amp_dtype="bfloat16", deterministic=True, save_every=3)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        cfg["run_dir"] = str(Path(folder) / "continuous")
        train(cfg, smoke=True, pilot=True)
        full = torch.load(Path(cfg["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
        full_log = [json.loads(line) for line in (Path(cfg["run_dir"]) / "train.jsonl").read_text().splitlines()]
        cfg["run_dir"] = str(Path(folder) / "resumed")
        train(cfg, smoke=True, pilot=True, stop_after=3)
        path = Path(cfg["run_dir"]) / "last.pt"
        log_path = path.with_name("train.jsonl")
        # Simulate a real checkpoint write failure after updates 4 and 5 have
        # been logged. The last durable checkpoint remains update 3.
        original_save = torch.save

        def fail_after_log(state, *args, **kwargs):
            if isinstance(state, dict) and state.get("update") == 5:
                assert json.loads(log_path.read_text().splitlines()[-1])["update"] == 5
                assert state["log_sha256"] == sha256(log_path)
                raise RuntimeError("injected checkpoint write interruption")
            return original_save(state, *args, **kwargs)

        with patch("torch.save", side_effect=fail_after_log):
            try:
                train(cfg, resume=path, smoke=True, pilot=True, stop_after=2)
            except RuntimeError as error:
                assert str(error) == "injected checkpoint write interruption"
            else:
                raise AssertionError("checkpoint failure was not injected")
        assert torch.load(path, map_location="cpu", weights_only=False)["update"] == 3
        with log_path.open("ab") as handle:
            handle.write(b'{"update": 6, "partial": "\xe4\xb8')
        train(cfg, resume=path, smoke=True, pilot=True)
        resumed = torch.load(path, map_location="cpu", weights_only=False)
        for name in full["model"]:
            assert torch.equal(full["model"][name], resumed["model"][name]), name
        assert full["scheduler"] == resumed["scheduler"]
        assert full["batch_cursor"] == resumed["batch_cursor"]
        assert full["epoch"] == resumed["epoch"]
        assert torch.equal(full["rng"]["torch"], resumed["rng"]["torch"])
        assert _equal_nested(full["optimizer"], resumed["optimizer"])
        assert full["scaler"] == resumed["scaler"]
        final_log = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert [record["update"] for record in final_log] == list(range(1, 7))
        for uninterrupted, restarted in zip(full_log, final_log):
            for key in ("data_order_sha256", "loss", "lr_used", "target_tokens", "sentences"):
                assert uninterrupted[key] == restarted[key], key
        assert list(path.parent.glob("train.jsonl.before-resume.*.bak"))
        assert resumed["log_sha256"] == sha256(log_path)
        # Changed core source provenance must fail without rewriting any of the
        # completed run's artifacts, even though its update budget is exhausted.
        forged = dict(resumed)
        forged["source_hashes"] = dict(resumed["source_hashes"], **{"train.py": "0" * 64})
        rejected = Path(folder) / "changed_source.pt"
        torch.save(forged, rejected)
        before = {str(p.relative_to(path.parent)): sha256(p) for p in path.parent.rglob("*") if p.is_file()}
        try:
            train(cfg, resume=rejected, smoke=True, pilot=True)
        except ValueError as error:
            assert "core training source differs" in str(error)
        else:
            raise AssertionError("changed source provenance was accepted")
        # A valid resume of an already completed run is an artifact-preserving no-op.
        train(cfg, resume=path, smoke=True, pilot=True)
        after = {str(p.relative_to(path.parent)): sha256(p) for p in path.parent.rglob("*") if p.is_file()}
        assert before == after
        result = {"device": device, "updates": 6, "interrupted_after": 3,
                  "failed_checkpoint_update": 5, "torn_log_tail_simulated": True,
                  "model_bitwise_equal": True, "scheduler_equal": True,
                  "optimizer_equal": True, "scaler_equal": True,
                  "batch_cursor_equal": True, "data_order_equal": True,
                  "log_updates_unique_and_contiguous": True,
                  "checkpoint_log_hash_verified": True, "source_change_rejected": True,
                  "completed_run_artifacts_preserved": True, "torch_version": torch.__version__}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--config", default="smoke.json")
    parser.add_argument("--out", help="check result JSON; default artifacts/checks/resume_<device>.json")
    args = parser.parse_args()
    result = run_check(args.device, args.config)
    write_json(args.out or f"artifacts/checks/resume_{args.device}.json", result)
    print(result)


if __name__ == "__main__":
    main()
