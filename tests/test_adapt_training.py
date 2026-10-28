"""Focused adaptation correctness, native labels, shared schedule and resume gates."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from mt.adapt_data import make_schedule, load_data
from mt.adapt_model import (LORA, LoRALinear, inject_lora, learned_state, native_collate, build_model,
                            restore_learned, state_digest)
from mt.adapt_train import (better_checkpoint, learning_rate_factor, reconcile_log,
                            select_learning_rate, _verify_checkpoint, _train, rng_state, tree_digest, configure_runtime,
                            verify_export_payload)
from mt.pretrained_mt import digest
from mt.common import sha256


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ToyAttention()


class TestAdaptModel(unittest.TestCase):
    def test_loader_rejects_ambiguous_model_directory(self):
        with self.assertRaisesRegex(ValueError, "named model directory"):
            build_model("not-the-model-directory", "full", 7)

    def test_zero_adapter_exact_equality(self):
        torch.manual_seed(7)
        base = nn.Linear(6, 5)
        x = torch.randn(2, 3, 6)
        expected = base(x).detach()
        wrapper = LoRALinear(base)
        self.assertTrue(torch.equal(expected, wrapper(x)))

    def test_targets_and_base_freeze(self):
        model = ToyModel()
        names = inject_lora(model)
        self.assertEqual(names, ["attn.q_proj", "attn.v_proj"])
        self.assertFalse(model.attn.k_proj.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 128)
        self.assertTrue(all("lora_" in n for n, p in model.named_parameters() if p.requires_grad))

    def test_target_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "target module"):
            inject_lora(ToyModel(), expected_modules=["wrong"])

    def test_duplicate_injection_rejected(self):
        model = ToyModel(); inject_lora(model)
        with self.assertRaises(ValueError):
            inject_lora(model)

    def test_nonzero_dropout_rejected(self):
        with self.assertRaises(ValueError):
            inject_lora(ToyModel(), dict(LORA, dropout=0.1))

    def test_adapter_export_reload(self):
        torch.manual_seed(7); first = ToyModel(); original = copy.deepcopy(first.state_dict())
        inject_lora(first)
        with torch.no_grad():
            first.attn.q_proj.lora_B.fill_(0.2)
        state = learned_state(first, "lora")
        second = ToyModel(); second.load_state_dict(original); inject_lora(second)
        restore_learned(second, state, "lora")
        x = torch.randn(3, 4)
        self.assertTrue(torch.equal(first.attn.q_proj(x), second.attn.q_proj(x)))
        self.assertEqual(state_digest(state), state_digest(learned_state(second, "lora")))

    def test_adapter_extra_base_tensor_rejected(self):
        model = ToyModel(); inject_lora(model)
        state = learned_state(model, "lora"); state["attn.k_proj.weight"] = model.attn.k_proj.weight
        with self.assertRaises(ValueError):
            restore_learned(model, state, "lora")

    def test_full_export_preserves_tied_storage(self):
        model = nn.Module()
        model.embedding = nn.Embedding(16, 4)
        model.head = nn.Linear(4, 16, bias=False)
        model.head.weight = model.embedding.weight
        state = learned_state(model, "full")
        self.assertEqual(state["embedding.weight"].data_ptr(), state["head.weight"].data_ptr())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "weights.pt"; torch.save(state, path)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(loaded["embedding.weight"].data_ptr(), loaded["head.weight"].data_ptr())
            self.assertEqual(state_digest(state), state_digest(loaded))

    def test_native_label_padding_and_eos(self):
        rows = [{"source_ids": [6, 0], "target_ids": [7, 8, 0]},
                {"source_ids": [3, 4, 0], "target_ids": [9, 0]}]
        batch = native_collate(rows)
        self.assertEqual(batch["input_ids"].tolist(), [[6, 0, 58100], [3, 4, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[7, 8, 0], [9, 0, -100]])

    def test_native_marian_label_shift(self):
        try:
            from transformers.models.marian.modeling_marian import shift_tokens_right
        except ImportError:
            self.skipTest("Native Transformers available through project-pretrained-python.ps1")
        labels = torch.tensor([[7, 0, -100], [8, 9, 0]])
        shifted = shift_tokens_right(labels, 58100, 58100)
        self.assertEqual(shifted.tolist(), [[58100, 7, 0], [58100, 8, 9]])

    def test_train_mode_reaches_frozen_base(self):
        model = ToyModel(); inject_lora(model); model.train()
        self.assertTrue(model.attn.q_proj.base.training)
        self.assertTrue(model.attn.k_proj.training)
        self.assertFalse(model.attn.q_proj.base.weight.requires_grad)

    def test_native_targets_cannot_contain_pad(self):
        with self.assertRaises(ValueError):
            native_collate([{"source_ids": [4, 0], "target_ids": [58100, 0]}])

    def test_native_eos_required(self):
        with self.assertRaises(ValueError):
            native_collate([{"source_ids": [4], "target_ids": [7, 0]}])

    def test_scalar_state_digest(self):
        self.assertNotEqual(state_digest({"x": torch.tensor(2)}), state_digest({"x": torch.tensor(3)}))


class TestAdaptScheduleSelection(unittest.TestCase):
    def test_schedule_seed_reproducibility_and_complete_epochs(self):
        ids = list(range(12))
        a = make_schedule(ids, 7, 4, 6)
        self.assertEqual(a, make_schedule(ids, 7, 4, 6))
        self.assertNotEqual(a, make_schedule(ids, 42, 4, 6))
        self.assertEqual(sorted(a[0] + a[1]), ids)
        self.assertEqual(sorted(a[2] + a[3]), ids)

    def test_duplicate_schedule_ids_rejected(self):
        with self.assertRaises(ValueError):
            make_schedule([1, 1], 7, 4, 2)

    def test_schedule_seed_mislabel_rejected_without_relying_on_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text(json.dumps({"updates": 4, "effective_batch_size": 2,
                "micro_batch_size": 1, "schedules": {"7": {"path": "schedule.json"}}}))
            (root / "retained_train_ids.json").write_text("[0,1,2]")
            (root / "train.jsonl").write_text("\n".join(json.dumps({"id": i}) for i in range(3)))
            schedule = {"seed": 42, "updates": 4, "effective_batch_size": 2, "micro_batch_size": 1,
                        "ids": make_schedule([0, 1, 2], 7, 4, 2)}
            (root / "schedule.json").write_text(json.dumps(schedule))
            with self.assertRaisesRegex(ValueError, "mislabeled"):
                load_data(root, 7, verify=False)

    def test_formal_runtime_disables_tf32(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        configure_runtime({"cpu_threads": 4})
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(torch.get_num_threads(), 4)

    def test_lr_schedule_boundaries(self):
        self.assertEqual(learning_rate_factor(1, 300, 3000), 1 / 300)
        self.assertEqual(learning_rate_factor(300, 300, 3000), 1)
        self.assertEqual(learning_rate_factor(3000, 300, 3000), 0)

    def test_checkpoint_ties_use_chrf_then_earliest(self):
        current = {"bleu": 20, "chrf2": 40, "update": 500}
        self.assertFalse(better_checkpoint(dict(current, update=1000), current))
        self.assertTrue(better_checkpoint(dict(current, chrf2=41, update=1000), current))

    def test_lr_selection_ignores_unrelated_test_scores(self):
        rows = [{"lr": lr, "split": "validation", "seed": 7, "updates": 3000,
                 "bleu": 20, "chrf2": 40, "test_bleu": test} for lr, test in [(1e-4, 0), (3e-4, 100)]]
        self.assertEqual(select_learning_rate(rows, [1e-4, 3e-4])["lr"], 1e-4)

    def test_test_split_cannot_select_lr(self):
        rows = [{"lr": lr, "split": "test", "seed": 7, "updates": 3000,
                 "bleu": 20, "chrf2": 40} for lr in [1e-4, 3e-4]]
        with self.assertRaises(ValueError):
            select_learning_rate(rows, [1e-4, 3e-4])

    def test_missing_lr_candidate_rejected(self):
        with self.assertRaises(ValueError):
            select_learning_rate([], [1e-4, 3e-4])


class TestAdaptRecovery(unittest.TestCase):
    def test_log_tail_is_repaired_only_after_committed_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.jsonl"
            path.write_text('{"update": 1}\n{"update": 2}\n{"upd', encoding="utf-8")
            reconcile_log(path, [{"update": 1}])
            self.assertEqual(path.read_text(), '{"update": 1}\n')

    def test_committed_log_damage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.jsonl"
            path.write_text('{"update": 9}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                reconcile_log(path, [{"update": 1}])

    def test_missing_committed_log_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                reconcile_log(Path(d) / "missing.jsonl", [{"update": 1}])

    def test_complete_state_required(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            _verify_checkpoint({}, {}, {}, [])

    def test_optimizer_and_rng_digest_detect_changes(self):
        state = {"optimizer": {0: {"exp_avg": torch.tensor([1.0, 2.0])}}, "rng": torch.get_rng_state()}
        before = tree_digest(state)
        state["optimizer"][0]["exp_avg"][0] += 1
        self.assertNotEqual(before, tree_digest(state))
        self.assertEqual(tree_digest(state), tree_digest(copy.deepcopy(state)))

    def test_export_tensor_damage_with_unchanged_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "plan.json").write_text("{}")
            cfg = {"run_dir": str(root), "method": "full", "updates": 3000, "validation_every": 500}
            weights = {"test": torch.tensor([1.0])}
            payload = {"schema": 1, "method": "full", "update": 500, "state": weights,
                "state_sha256": state_digest(weights), "config": cfg, "run_plan_sha256": sha256(root / "plan.json"),
                "base_model_files": {}, "parameter_evidence": {}, "adapter_config": None}
            verify_export_payload(payload, cfg, {"model_files": {}}, 500)
            weights["test"][0] = 2
            with self.assertRaisesRegex(ValueError, "tensors changed"):
                verify_export_payload(payload, cfg, {"model_files": {}}, 500)

    def test_cpu_training_resume_matches_continuous_parameters(self):
        class NativeToy(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 4)
                self.dropout = nn.Dropout(0.2)
                self.proj = nn.Linear(4, 16)

            def forward(self, input_ids, attention_mask, labels, use_cache=False):
                from types import SimpleNamespace
                x = self.dropout(self.embedding(input_ids.masked_fill(input_ids == 58100, 0))).mean(1)
                logits = self.proj(x).unsqueeze(1).expand(-1, labels.shape[1], -1)
                loss = nn.functional.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100)
                return SimpleNamespace(loss=loss)

        def build(*args):
            torch.manual_seed(7)
            return NativeToy(), None, {"test": True}

        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "data"; data.mkdir()
            manifest = {"data_dir": str(data), "updates": 3000, "effective_batch_size": 2, "micro_batch_size": 1}
            items = {i: {"id": i, "source_ids": [i + 1, 0], "target_ids": [i + 2, 0]} for i in range(3)}
            schedule = make_schedule(list(items), 7, 3000, 2)
            plan = {"source_files": {}}
            final_states = []
            for name, interrupted in (("continuous", False), ("resumed", True)):
                run = Path(d) / name; run.mkdir(); (run / "plan.json").write_text("{}")
                cfg = {"run_dir": str(run), "model_dir": "unused", "data_root": str(data), "method": "full",
                       "seed": 7, "lr": 1e-5, "updates": 3000, "warmup_updates": 300, "train_precision": "bfloat16",
                       "micro_batch_size": 1, "effective_batch_size": 2, "validation_every": 500, "save_every": 2}
                with patch("mt.adapt_train.load_data", return_value=(manifest, items, schedule)), \
                     patch("mt.adapt_train.build_model", side_effect=build), \
                     patch("mt.adapt_train.read_pairs", return_value=[{}] * 888):
                    if interrupted:
                        _train(cfg, plan, "cpu", 2)
                    _train(cfg, plan, "cpu", 4)
                final_states.append(torch.load(run / "last.pt", map_location="cpu", weights_only=False))
            self.assertEqual(final_states[0]["learned_sha256"], final_states[1]["learned_sha256"])
            self.assertEqual([r["loss"] for r in final_states[0]["records"]], [r["loss"] for r in final_states[1]["records"]])
            self.assertEqual(final_states[1]["cursor"], 4)


if __name__ == "__main__":
    unittest.main()
