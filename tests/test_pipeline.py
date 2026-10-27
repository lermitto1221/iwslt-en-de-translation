"""Pipeline cache regressions; inference and GPU execution are never needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sacrebleu

from mt.common import read_json, sha256, write_json
from mt.evaluate import evaluate_checkpoint
from mt.run_experiments import evaluate_or_reuse, frozen_protocol


class PipelineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "A42"
        self.out = self.run / "validation_beam"
        self.out.mkdir(parents=True)
        self.checkpoint = self.run / "best.pt"
        self.checkpoint.write_bytes(b"synthetic checkpoint: never deserialize")
        self.cfg = {"seed": 42, "beam_size": 4, "length_penalty": 0.6,
                    "max_generation_tokens": 256, "decode_batch_size": 16}
        write_json(self.run / "config.json", self.cfg)
        self.rows = [
            {"id": 0, "en": "The small house is near the river.",
             "de": "Das kleine Haus steht neben dem Fluss."},
            {"id": 1, "en": "The birds fly over the village.",
             "de": "Die Vögel fliegen über das kleine Dorf."},
        ]
        self.source = self.root / "validation.jsonl"
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows),
                               encoding="utf-8")
        references = [row["de"] for row in self.rows]
        for name in ("predictions.txt", "references.txt"):
            (self.out / name).write_text("\n".join(references) + "\n", encoding="utf-8")
        examples = [{"id": row["id"], "source": row["en"],
                     "reference": row["de"], "prediction": row["de"]} for row in self.rows]
        (self.out / "examples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
        metric = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
        score = metric.corpus_score(references, [references]).score
        self.metrics = {"split": "validation", "method": "beam", "config": self.cfg,
                        "checkpoint": str(self.checkpoint), "checkpoint_sha256": sha256(self.checkpoint),
                        "inference_source_sha256": sha256(Path(__file__).resolve().parents[1] / "mt/decoding.py"),
                        "beam_size": 4, "length_penalty": 0.6, "max_generation_tokens": 256,
                        "inference_batch_size": 16, "inference_precision": "float32",
                        "samples": 2, "expected_samples": 2, "bleu": score,
                        "signature": str(metric.get_signature()), "sacrebleu_version": sacrebleu.__version__,
                        "reference_source": str(self.source), "reference_sha256": sha256(self.source),
                        "predictions_sha256": sha256(self.out / "predictions.txt"),
                        "references_sha256": sha256(self.out / "references.txt")}
        write_json(self.out / "metrics.json", self.metrics)

    def cache(self, **kwargs):
        return evaluate_or_reuse(self.checkpoint, kwargs.pop("split", "validation"),
                                 kwargs.pop("method", "beam"), self.out, **kwargs)

    def output_hashes(self):
        return {path.name: sha256(path) for path in self.out.iterdir() if path.is_file()}

    def test_valid_cache_is_verified_and_reused_without_inference_or_writes(self):
        before = self.output_hashes()
        with patch("mt.run_experiments.evaluate_checkpoint", side_effect=AssertionError("must not infer")) as infer:
            actual = self.cache()
        infer.assert_not_called()
        self.assertEqual(actual, self.metrics)
        self.assertEqual(self.output_hashes(), before)

    def test_changed_request_cannot_overwrite_or_reuse_a_completed_result(self):
        before = self.output_hashes()
        requests = ({"split": "test"}, {"method": "greedy"}, {"beam_size": 2},
                    {"alpha": 0.0}, {"max_tokens": 512})
        with patch("mt.run_experiments.evaluate_checkpoint", side_effect=AssertionError("must not infer")) as infer:
            for request in requests:
                with self.subTest(request=request), self.assertRaisesRegex(ValueError, "differs from this evaluation"):
                    self.cache(**request)
        infer.assert_not_called()
        self.assertEqual(self.output_hashes(), before)

    def test_changed_checkpoint_or_decoder_version_rejects_cached_result(self):
        with patch("mt.run_experiments.evaluate_checkpoint", side_effect=AssertionError("must not infer")) as infer:
            self.checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "saved checkpoint_sha256 differs"):
                self.cache()
            self.checkpoint.write_bytes(b"synthetic checkpoint: never deserialize")
            original_hash = sha256

            def changed_decoder(path):
                return "0" * 64 if Path(path).name == "decoding.py" else original_hash(path)

            with patch("mt.run_experiments.sha256", side_effect=changed_decoder):
                with self.assertRaisesRegex(ValueError, "saved inference_source_sha256 differs"):
                    self.cache()
        infer.assert_not_called()

    def test_changed_batch_config_rejects_cached_result(self):
        write_json(self.run / "config.json", {**self.cfg, "decode_batch_size": 8})
        with patch("mt.run_experiments.evaluate_checkpoint") as infer:
            with self.assertRaisesRegex(ValueError, "saved config differs"):
                self.cache()
        infer.assert_not_called()

    def test_invalid_saved_outputs_are_not_replaced_with_new_inference(self):
        (self.out / "predictions.txt").write_text("corrupted prediction\n", encoding="utf-8")
        before = self.output_hashes()
        with patch("mt.run_experiments.evaluate_checkpoint") as infer:
            with self.assertRaisesRegex(ValueError, "predictions: SHA-256 mismatch"):
                self.cache()
        infer.assert_not_called()
        self.assertEqual(self.output_hashes(), before)

    def test_incomplete_directory_without_completion_marker_is_retried(self):
        (self.out / "metrics.json").unlink()
        (self.out / "metrics.json.tmp").write_text("{incomplete", encoding="utf-8")
        sentinel = {"completed_by_mock": True}
        with patch("mt.run_experiments.evaluate_checkpoint", return_value=sentinel) as infer:
            actual = self.cache(alpha=0.0)
        self.assertIs(actual, sentinel)
        infer.assert_called_once_with(self.checkpoint, "validation", "beam", self.out,
                                      beam_size=None, alpha=0.0, max_tokens=None)

    def test_direct_evaluation_refuses_completed_directory_before_loading_model(self):
        before = self.output_hashes()
        with patch("mt.evaluate.load_checkpoint", side_effect=AssertionError("must not load model")) as loader:
            with patch("mt.evaluate.torch.cuda.is_available", side_effect=AssertionError("must not query GPU")):
                with self.assertRaisesRegex(FileExistsError, "completed evaluation"):
                    evaluate_checkpoint(self.checkpoint, "validation", "beam", self.out)
        loader.assert_not_called()
        self.assertEqual(self.output_hashes(), before)

    def test_even_malformed_completion_marker_is_not_overwritten(self):
        marker = self.out / "metrics.json"
        marker.write_text("{unfinished", encoding="utf-8")
        with patch("mt.evaluate.load_checkpoint") as loader:
            with self.assertRaises(FileExistsError):
                evaluate_checkpoint(self.checkpoint, "validation", "beam", self.out)
        loader.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "{unfinished")

    def test_frozen_protocol_rejects_test_selection_and_wrong_core_beam(self):
        path = self.root / "protocol.json"
        valid = {"selection_split": "validation", "beam_size": 4,
                 "length_penalty": 0.6, "max_generation_tokens": 256}
        write_json(path, valid)
        self.assertEqual(frozen_protocol(path), valid)
        for changes in ({"selection_split": "test"}, {"beam_size": 2}):
            with self.subTest(changes=changes):
                write_json(path, {**valid, **changes})
                with self.assertRaisesRegex(ValueError, "validation-selected beam=4"):
                    frozen_protocol(path)


if __name__ == "__main__":
    unittest.main()
