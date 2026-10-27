import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import torch

from mt.common import read_json, sha256, write_json
from mt.explore_vectors import (FEATURES, _load_checked_npz, array_hash,
    cluster_diagnostics, feature_associations, fit_kmeans, fit_standardized_pca,
    masked_mean_pool, predict_clusters, run_vectors, sample_training_rows, transform_pca)
from mt.model import TranslationTransformer


class RepresentationMathTests(unittest.TestCase):
    def test_pooling_excludes_special_tokens_keeps_unk(self):
        tokens = torch.tensor([[2, 4, 1, 3, 0]])
        hidden = torch.tensor([[[99., 99.], [2., 4.], [6., 8.], [99., 99.], [float("nan"), 99.]]])
        self.assertTrue(torch.equal(masked_mean_pool(hidden, tokens), torch.tensor([[4., 6.]])))
        with self.assertRaisesRegex(ValueError, "empty effective"):
            masked_mean_pool(torch.ones(1, 3, 2), torch.tensor([[0, 2, 3]]))

    def test_real_encoder_pooling_is_padding_invariant(self):
        cfg = dict(d_model=8, nhead=2, dim_ff=16, dropout=.2, num_layers=1)
        torch.manual_seed(7)
        model = TranslationTransformer(20, cfg).eval()
        tokens = torch.tensor([[2, 4, 5, 3]])
        padded = torch.tensor([[2, 4, 5, 3, 0, 0, 0]])
        with torch.inference_mode():
            first = masked_mean_pool(model.encode(tokens)[0], tokens)
            second = masked_mean_pool(model.encode(padded)[0], padded)
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_sampling_is_reproducible_and_original_ids_survive(self):
        rows = [{"id": i * 3, "en": str(i)} for i in range(50)]
        first = sample_training_rows(rows, 10, 123)
        self.assertEqual(first, sample_training_rows(rows, 10, 123))
        expected = sorted(rows[i]["id"] for i in random.Random(123).sample(range(50), 10))
        self.assertEqual([row["id"] for row in first], expected)
        self.assertNotEqual(first, sample_training_rows(rows, 10, 124))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sample_training_rows(rows + rows[:1], 10, 123)

    def test_pca_train_only_fit_and_reconstruction(self):
        train = np.array([[0., 0., 7.], [2., 1., 7.], [4., 3., 7.], [6., 6., 7.]])
        fitted = fit_standardized_pca(train, 3)
        np.testing.assert_allclose(fitted["standard_mean"], train.mean(0))
        np.testing.assert_allclose(fitted["standard_scale"][:2], train.std(0)[:2])
        self.assertEqual(fitted["standard_scale"][2], 1.)
        before = {key: value.copy() for key, value in fitted.items()}
        valid = np.array([[10000., -10000., 7.]])
        projected = transform_pca(valid, fitted)
        self.assertEqual(projected.shape, (1, 3))
        for key in fitted:
            np.testing.assert_array_equal(before[key], fitted[key])
        scores = transform_pca(train, fitted)
        reconstructed = ((scores @ fitted["pca_components"] + fitted["pca_mean"])
                         * fitted["standard_scale"] + fitted["standard_mean"])
        np.testing.assert_allclose(reconstructed, train, atol=1e-12)
        self.assertAlmostEqual(fitted["explained_variance_ratio"].sum(), 1.)
        np.testing.assert_allclose(fitted["pca_components"] @ fitted["pca_components"].T,
                                   np.eye(3), atol=1e-12)

    def test_kmeans_reproducible_matches_nearest_and_means(self):
        rng = np.random.default_rng(19)
        train = np.concatenate([rng.normal(center, .05, (20, 3)) for center in (-5., 0., 5.)])
        first = fit_kmeans(train, 3, 12, 4, 100, 1e-8)
        second = fit_kmeans(train, 3, 12, 4, 100, 1e-8)
        np.testing.assert_array_equal(first["centers"], second["centers"])
        np.testing.assert_array_equal(first["labels"], predict_clusters(train, first["centers"]))
        for cluster in range(3):
            np.testing.assert_allclose(first["centers"][cluster], train[first["labels"] == cluster].mean(0))
        exact = np.square(train - first["centers"][first["labels"]]).sum()
        self.assertAlmostEqual(first["inertia"], exact, places=9)
        self.assertEqual(first["inertia"], min(run["inertia"] for run in first["restarts"]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            fit_kmeans(np.ones((10, 2)), 3)

    def test_diagnostics_match_small_hand_computation(self):
        values = np.array([[0.], [1.], [10.], [11.]])
        labels = np.array([0, 0, 1, 1])
        result = cluster_diagnostics(values, labels, np.array([3, 6, 9, 12]), 4, 7)
        expected = np.mean([(10.5 - 1) / 10.5, (9.5 - 1) / 9.5,
                            (9.5 - 1) / 9.5, (10.5 - 1) / 10.5])
        self.assertAlmostEqual(result["silhouette"]["score"], expected)
        self.assertAlmostEqual(result["calinski_harabasz"], 200.)
        self.assertAlmostEqual(result["davies_bouldin"], .1)
        self.assertEqual(result["silhouette"]["sample_ids"], [3, 6, 9, 12])
        self.assertEqual(cluster_diagnostics(values, labels, np.array([3, 6, 9, 12]), 3, 7),
                         cluster_diagnostics(values, labels, np.array([3, 6, 9, 12]), 3, 7))

    def test_feature_associations_handle_constants(self):
        rows = [{key: float(i // 2) for key in FEATURES} for i in range(4)]
        for row in rows:
            row["src_unseen_fraction"] = 0.
        labels = np.array([0, 0, 1, 1])
        scores = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        result = feature_associations(rows, labels, scores)
        self.assertAlmostEqual(result["eta_squared"]["src_words"], 1.)
        self.assertAlmostEqual(result["pearson_pc1"]["src_words"], 1.)
        self.assertIsNone(result["eta_squared"]["src_unseen_fraction"])
        self.assertIsNone(result["pearson_pc1"]["src_unseen_fraction"])

    def test_array_hash_includes_dtype_shape_and_order(self):
        values = np.arange(6, dtype=np.int64).reshape(2, 3)
        self.assertNotEqual(array_hash(values), array_hash(values.reshape(3, 2)))
        self.assertNotEqual(array_hash(values), array_hash(values.astype(np.float64)))
        self.assertEqual(array_hash(values), array_hash(np.asfortranarray(values)))


class RepresentationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / "source"
        data.mkdir()
        corpus = data / "corpus.txt"
        corpus.write_text("one two three four five six seven eight nine ten\nHallo Welt\n" * 20, encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(data / "spm"),
            vocab_size=50, model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False, num_threads=1, minloglevel=2)
        tok = spm.SentencePieceProcessor(model_file=str(data / "spm.model"))
        cfg = dict(seed=42, data_dir=str(data), d_model=8, nhead=2, dim_ff=16,
                   dropout=.1, num_layers=1, deterministic=True, cpu_threads=1)
        torch.manual_seed(42)
        model = TranslationTransformer(tok.vocab_size(), cfg)
        checkpoint = self.root / "best.pt"
        torch.save({"config": cfg, "model": model.state_dict(), "update": 12,
                    "fingerprints": {"tokenizer_sha256": sha256(data / "spm.model")}}, checkpoint)
        words = "one two three four five six seven eight nine ten".split()
        self.rows = {}
        for split, size in (("train", 30), ("validation", 6)):
            self.rows[split] = [{"id": i * 2, "en": " ".join(words[i % 10:] + words[:i % 10]) + (" one" * (i // 10)),
                                 "de": "Hallo Welt"} for i in range(size)]
            (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows[split]), encoding="utf-8")
        self.output = self.root / "analysis"
        (self.output / "data").mkdir(parents=True)
        for split, rows in self.rows.items():
            # Reverse feature rows to ensure matching uses original IDs, not position.
            feature_rows = [{"id": row["id"], **{key: float(row["id"] + 1) for key in FEATURES}} for row in reversed(rows)]
            (self.output / "data" / f"{split}_features.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")
        self.config = {"data_dir": str(data), "checkpoint": str(checkpoint),
                       "sampling": {"seed": 20261027, "train_size": 20},
                       "vectors": {"batch_size": 4, "pca_components": 4, "kmeans_clusters": 3,
                                   "kmeans_seed": 42, "kmeans_n_init": 2, "kmeans_max_iter": 50,
                                   "kmeans_tol": .0001, "silhouette_size": 12, "silhouette_seed": 42},
                       "small_cluster_size": 3, "representatives_per_cluster": 2}

    def tearDown(self):
        self.temp.cleanup()

    def _run(self):
        with patch("mt.explore_vectors._plot"):
            return run_vectors(self.config, self.output, "cpu")

    def test_pipeline_train_only_fit_complete_validation_and_verified_reuse(self):
        with patch("mt.explore_vectors.fit_standardized_pca", wraps=fit_standardized_pca) as pca:
            summary = self._run()
        self.assertEqual(summary["validation_size"], 6)
        self.assertEqual(sum(cluster["validation"]["n"] for cluster in summary["clusters"]), 6)
        selection = read_json(self.output / "vectors" / "selection.json")
        with np.load(self.output / "vectors" / "embeddings.npz") as embeddings:
            np.testing.assert_array_equal(embeddings["train_ids"], selection["train_ids"])
            np.testing.assert_array_equal(embeddings["validation_ids"], [row["id"] for row in self.rows["validation"]])
            np.testing.assert_array_equal(pca.call_args.args[0], embeddings["train_embeddings"])
        with np.load(self.output / "vectors" / "fit.npz") as fitted:
            np.testing.assert_array_equal(fitted["fitted_train_ids"], selection["train_ids"])
        assignments = [json.loads(line) for line in (self.output / "vectors" / "assignments.jsonl").read_text().splitlines()]
        valid_ids = [row["id"] for row in assignments if row["split"] == "validation"]
        self.assertEqual(valid_ids, selection["validation_ids"])
        for cluster in summary["clusters"]:
            expected = sorted(row["id"] for row in assignments if row["split"] == "validation" and row["cluster"] == cluster["cluster"])[:2]
            self.assertEqual(cluster["validation"]["representative_ids"], expected)
        with patch("mt.explore_vectors.load_checkpoint", side_effect=AssertionError("unexpected extraction")), \
                patch("mt.explore_vectors.fit_standardized_pca", side_effect=AssertionError("unexpected fit")):
            reused = self._run()
        self.assertEqual(summary["clusters"], reused["clusters"])
        self.assertEqual(summary["fit_sha256"], reused["fit_sha256"])

    def test_saved_selection_and_embeddings_corruption_are_rejected(self):
        self._run()
        path = self.output / "vectors" / "selection.json"
        original = path.read_bytes()
        changed = read_json(path)
        changed["train_ids"][0] = 99999
        write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self._run()
        path.write_bytes(original)
        vectors = self.output / "vectors" / "embeddings.npz"
        vectors.write_bytes(vectors.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._run()

    def test_parameter_corruption_and_validation_feature_mismatch_are_rejected(self):
        self._run()
        path = self.output / "vectors" / "fit_meta.json"
        original = path.read_bytes()
        metadata = read_json(path)
        metadata["inputs"]["fitted_split"] = "validation"
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "stage inputs differ"):
            self._run()
        path.write_bytes(original)
        features = self.output / "data" / "validation_features.jsonl"
        rows = features.read_text(encoding="utf-8").splitlines()
        features.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing feature IDs"):
            self._run()


if __name__ == "__main__":
    unittest.main()
