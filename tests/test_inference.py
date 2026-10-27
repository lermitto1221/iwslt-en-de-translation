import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sentencepiece as spm
import torch

from mt.common import sha256, write_json
from mt.export import export_bundle
from mt.inference import Translator, validate_protocol
from mt.model import TranslationTransformer


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = cls.root / "data"
        data.mkdir()
        corpus = data / "corpus.txt"
        corpus.write_text("Hello world\nHallo Welt\nGood morning\nGuten Morgen\n" * 20, encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(data / "spm"),
            vocab_size=40, model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False, num_threads=1, minloglevel=2)
        tok = spm.SentencePieceProcessor(model_file=str(data / "spm.model"))
        cfg = dict(seed=42, data_dir="data", d_model=16, nhead=2, dim_ff=32,
                   dropout=0.0, num_layers=1, beam_size=2, length_penalty=0.6,
                   max_generation_tokens=4, cpu_threads=1, decode_batch_size=2)
        torch.manual_seed(42)
        model = TranslationTransformer(tok.vocab_size(), cfg)
        cls.checkpoint = cls.root / "runs" / "best.pt"
        cls.checkpoint.parent.mkdir()
        torch.save({"config": cfg, "model": model.state_dict(), "update": 1,
                    "fingerprints": {"tokenizer_sha256": sha256(data / "spm.model")}}, cls.checkpoint)
        cls.protocol = cls.root / "protocol.json"
        write_json(cls.protocol, {"selection_split": "validation", "beam_size": 2,
                                  "length_penalty": 0.6, "max_generation_tokens": 4})
        cls.bundle = export_bundle(cls.checkpoint, cls.protocol, cls.root / "bundle")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_checkpoint_loaded_once_outside_project_cwd(self):
        current = Path.cwd()
        try:
            os.chdir(self.root.parent)
            with patch("mt.inference.torch.load", wraps=torch.load) as loader:
                session = Translator(checkpoint=self.checkpoint, device="cpu")
                loader.assert_called_once()
                self.assertEqual(session.data_dir, self.root / "data")
        finally:
            os.chdir(current)

    def test_export_preserves_both_search_outputs_without_original_data(self):
        checkpoint_session = Translator(checkpoint=self.checkpoint, protocol=self.protocol, device="cpu")
        expected = {m: checkpoint_session.translate(["Hello world", "Good morning"], m)
                    for m in ("greedy", "beam")}
        # The bundle is usable without any source data directory or checkpoint.
        data, hidden = self.root / "data", self.root / "hidden_data"
        data.rename(hidden)
        try:
            exported = Translator(bundle=self.bundle, device="cpu")
            for method in expected:
                self.assertEqual(expected[method], exported.translate(["Hello world", "Good morning"], method))
        finally:
            hidden.rename(data)

    def test_bundle_rejects_modified_protocol(self):
        path = self.bundle / "protocol.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch: protocol.json"):
                Translator(bundle=self.bundle, device="cpu")
        finally:
            path.write_bytes(original)

    def test_export_does_not_overwrite_existing_bundle(self):
        with self.assertRaises(FileExistsError):
            export_bundle(self.checkpoint, self.protocol, self.bundle)

    def test_reject_wrong_tokenizer_and_missing_fingerprint(self):
        wrong = self.root / "wrong"
        wrong.mkdir(exist_ok=True)
        (wrong / "spm.model").write_bytes(b"wrong tokenizer")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            Translator(checkpoint=self.checkpoint, data_dir=wrong, device="cpu")
        state = torch.load(self.checkpoint, weights_only=False)
        del state["fingerprints"]
        legacy = self.root / "legacy.pt"
        torch.save(state, legacy)
        with self.assertRaisesRegex(ValueError, "no valid tokenizer"):
            Translator(checkpoint=legacy, device="cpu")

    def test_reject_invalid_inputs(self):
        session = Translator(bundle=self.bundle, device="cpu")
        for texts in ([], [""], [None], ["one\ntwo"]):
            with self.assertRaises(ValueError):
                session.translate(texts)
        with self.assertRaises(ValueError):
            session.translate(["hello"], batch_size=0)
        for value in (0, True, 5000):
            with self.assertRaises(ValueError):
                validate_protocol({"beam_size": 2, "length_penalty": 0.6, "max_generation_tokens": value})


if __name__ == "__main__":
    unittest.main()
