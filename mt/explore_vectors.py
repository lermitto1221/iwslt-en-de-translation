"""Exploratory encoder representations, with transformations fitted on train only.

NumPy implements population standardization, full-SVD PCA, k-means++ and Lloyd
iterations explicitly. No extra ML runtime or change to the training stack is
needed. Test/reference translations are never inputs to representation fitting.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import read_json, sha256
from .data import read_pairs
from .inference import load_checkpoint, select_device
from .tokenizer import BOS, EOS, PAD, encode


FEATURES = ("src_words", "tgt_words", "src_bpe", "tgt_bpe",
            "src_fragmentation", "tgt_fragmentation", "src_rare_fraction",
            "tgt_rare_fraction", "src_unseen_fraction", "tgt_unseen_fraction",
            "length_ratio")
SOURCE_FILES = ("explore_vectors.py", "inference.py", "model.py", "tokenizer.py", "common.py", "data.py")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def array_hash(array):
    """Hash dtype, shape and C-order bytes, not just the ZIP container."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def sample_training_rows(rows, size, seed):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original training IDs")
    if type(size) is not int or not 1 <= size <= len(rows):
        raise ValueError("invalid training sample size")
    positions = random.Random(seed).sample(range(len(rows)), size)
    return sorted((rows[index] for index in positions), key=lambda row: row["id"])


def masked_mean_pool(hidden, token_ids):
    """Mean of last encoder layer over non-PAD/BOS/EOS positions; UNK stays."""
    if hidden.ndim != 3 or token_ids.ndim != 2 or hidden.shape[:2] != token_ids.shape:
        raise ValueError("encoder output and token IDs have incompatible shapes")
    mask = (token_ids != PAD) & (token_ids != BOS) & (token_ids != EOS)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("empty effective source sequence after removing PAD/BOS/EOS")
    # where avoids padded NaNs entering the sum (NaN * 0 would still be NaN).
    pooled = torch.where(mask.unsqueeze(-1), hidden, 0).sum(dim=1) / counts.unsqueeze(-1)
    if not torch.isfinite(pooled).all():
        raise ValueError("non-finite encoder sentence vector")
    return pooled


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite feature matrix")
    return values


def fit_standardized_pca(train, components):
    """Fit only supplied train rows; variance uses n-1, scaling uses n."""
    train = _matrix(train)
    if len(train) < 2 or not 1 <= components <= min(len(train), train.shape[1]):
        raise ValueError("PCA component count exceeds the training matrix")
    mean, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    constant = scale == 0
    scale[constant] = 1.0
    standardized = (train - mean) / scale
    pca_mean = standardized.mean(axis=0)
    _, singular, axes = np.linalg.svd(standardized - pca_mean, full_matrices=False)
    # Resolve arbitrary SVD sign by requiring each largest loading positive.
    signs = np.sign(axes[np.arange(len(axes)), np.abs(axes).argmax(axis=1)])
    axes *= np.where(signs == 0, 1, signs)[:, None]
    variance = singular ** 2 / (len(train) - 1)
    if not variance.sum() > 0:
        raise ValueError("PCA training vectors have zero total variance")
    return {"standard_mean": mean, "standard_scale": scale,
            "constant_features": constant, "pca_mean": pca_mean,
            "pca_components": axes[:components],
            "explained_variance": variance[:components],
            "explained_variance_ratio": variance[:components] / variance.sum(),
            "total_standardized_variance": np.asarray(variance.sum())}


def transform_pca(values, fitted):
    values = _matrix(values)
    if values.shape[1] != len(fitted["standard_mean"]):
        raise ValueError("embedding dimension differs from fitted standardization")
    return ((values - fitted["standard_mean"]) / fitted["standard_scale"]
            - fitted["pca_mean"]) @ fitted["pca_components"].T


def squared_distances(values, centers):
    distances = (np.sum(values * values, axis=1)[:, None]
                 + np.sum(centers * centers, axis=1)[None, :] - 2 * values @ centers.T)
    return np.maximum(distances, 0.0)


def predict_clusters(values, centers):
    return squared_distances(_matrix(values), _matrix(centers)).argmin(axis=1)


def fit_kmeans(train, clusters=5, seed=20261027, n_init=10, max_iter=300, tol=1e-4):
    """k-means++ / Lloyd; stop when max center displacement <= absolute tol.

    Each restart draws a seed from one declared NumPy Generator. Empty centers
    are relocated to the furthest currently represented training points. Ties
    in nearest-center assignment and best restart retain the first occurrence.
    """
    train = _matrix(train)
    if (type(clusters) is not int or not 1 < clusters < len(train)
            or np.unique(train, axis=0).shape[0] < clusters
            or type(n_init) is not int or n_init < 1 or max_iter < 1 or tol < 0):
        raise ValueError("invalid KMeans settings or insufficient distinct training vectors")
    generator = np.random.default_rng(seed)
    best, restarts = None, []
    for restart in range(n_init):
        restart_seed = int(generator.integers(0, 2**32))
        rng = np.random.default_rng(restart_seed)
        chosen = [int(rng.integers(len(train)))]
        nearest = squared_distances(train, train[chosen]).ravel()
        nearest[chosen] = 0.0
        while len(chosen) < clusters:
            if not nearest.sum() > 0:
                raise ValueError("k-means++ exhausted distinct positive-distance training points")
            index = int(rng.choice(len(train), p=nearest / nearest.sum()))
            chosen.append(index)
            nearest = np.minimum(nearest, squared_distances(train, train[index:index + 1]).ravel())
            nearest[chosen] = 0.0
        centers = train[chosen].copy()
        converged, relocations = False, 0
        for iteration in range(1, max_iter + 1):
            distances = squared_distances(train, centers)
            labels = distances.argmin(axis=1)
            updated = centers.copy()
            furthest = np.argsort(-distances[np.arange(len(train)), labels], kind="stable")
            used = set()
            for cluster in range(clusters):
                members = train[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    index = next(int(i) for i in furthest if int(i) not in used)
                    updated[cluster] = train[index]
                    used.add(index)
                    relocations += 1
            displacement = float(np.linalg.norm(updated - centers, axis=1).max())
            centers = updated
            if displacement <= tol:
                converged = True
                break
        final_distances = squared_distances(train, centers)
        labels = final_distances.argmin(axis=1)
        inertia = float(final_distances[np.arange(len(train)), labels].sum())
        restart_info = {"restart": restart, "seed": restart_seed, "inertia": inertia,
                        "iterations": iteration, "converged": converged,
                        "empty_center_relocations": relocations}
        restarts.append(restart_info)
        if best is None or inertia < best["inertia"]:
            best = {**restart_info, "centers": centers.copy(), "labels": labels.copy()}
    best["restarts"] = restarts
    return best


def cluster_diagnostics(train, labels, ids, sample_size, seed):
    """Train-only Euclidean diagnostics, all in the retained PCA space.

    Silhouette is computed in a fixed random subset and distances refer only
    to other members of that same subset; singleton silhouettes are zero.
    """
    train = _matrix(train)
    labels, ids = np.asarray(labels), np.asarray(ids)
    groups = np.unique(labels)
    if len(train) != len(labels) or len(ids) != len(train):
        raise ValueError("diagnostic IDs/labels do not align with training vectors")
    if not 1 < len(groups) < len(train):
        raise ValueError("cluster diagnostics require 2 through n-1 occupied clusters")
    positions = sorted(random.Random(seed).sample(range(len(train)), min(sample_size, len(train))))
    sampled, sampled_labels = train[positions], labels[positions]
    distances = np.sqrt(squared_distances(sampled, sampled))
    np.fill_diagonal(distances, 0.0)
    silhouettes = []
    subset_groups = np.unique(sampled_labels)
    for index, group in enumerate(sampled_labels):
        own = sampled_labels == group
        if own.sum() <= 1 or len(subset_groups) < 2:
            silhouettes.append(0.0)
            continue
        a = distances[index, own].sum() / (own.sum() - 1)
        b = min(distances[index, sampled_labels == other].mean()
                for other in subset_groups if other != group)
        silhouettes.append(float((b - a) / max(a, b)) if max(a, b) > 0 else 0.0)
    centers = np.vstack([train[labels == group].mean(axis=0) for group in groups])
    within = sum(float(np.square(train[labels == group] - center).sum())
                 for group, center in zip(groups, centers))
    between = sum(int(np.sum(labels == group)) * float(np.square(center - train.mean(axis=0)).sum())
                  for group, center in zip(groups, centers))
    ch = (between / (len(groups) - 1)) / (within / (len(train) - len(groups))) if within else None
    scatter = np.asarray([np.linalg.norm(train[labels == group] - center, axis=1).mean()
                          for group, center in zip(groups, centers)])
    center_distances = np.sqrt(squared_distances(centers, centers))
    np.fill_diagonal(center_distances, np.inf)
    if np.any(center_distances == 0):
        db = None
    else:
        db = float(((scatter[:, None] + scatter[None, :]) / center_distances).max(axis=1).mean())
    return {"space": f"standardized encoder PCA-{train.shape[1]} (Euclidean)",
            "silhouette": {"score": float(np.mean(silhouettes)), "sample_size": len(positions),
                           "seed": seed, "sample_ids": ids[positions].tolist(),
                           "definition": "fixed training subset; all pair distances within subset; singleton=0"},
            "calinski_harabasz": ch, "davies_bouldin": db,
            "full_training_sample_size": len(train), "occupied_clusters": len(groups)}


def _describe(values):
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75))}


def feature_associations(features, labels, scores):
    output = {"eta_squared": {}, "pearson_pc1": {}, "pearson_pc2": {}}
    for key in FEATURES:
        values = np.asarray([row[key] for row in features], dtype=float)
        total = float(np.square(values - values.mean()).sum())
        between = sum(np.sum(labels == group) * float((values[labels == group].mean() - values.mean()) ** 2)
                      for group in np.unique(labels))
        output["eta_squared"][key] = float(between / total) if total > 0 else None
        for column in range(min(2, scores.shape[1])):
            value = None
            if total > 0 and np.std(scores[:, column]) > 0:
                value = float(np.corrcoef(values, scores[:, column])[0, 1])
            output[f"pearson_pc{column + 1}"][key] = value
    return output


def _aligned_features(path, ids):
    rows = read_pairs(path)
    index = {row["id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"duplicate feature IDs: {path}")
    if any(int(i) not in index for i in ids):
        raise ValueError(f"missing feature IDs: {path}")
    selected = [index[int(i)] for i in ids]
    if any(any(key not in row or not np.isfinite(row[key]) for key in FEATURES) for row in selected):
        raise ValueError(f"missing/nonfinite feature values: {path}")
    return selected


def _input_hashes(config):
    data = Path(config["data_dir"])
    paths = {"checkpoint": Path(config["checkpoint"]), "train": data / "train.jsonl",
             "validation": data / "validation.jsonl", "tokenizer": data / "spm.model"}
    result = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    result["source_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    return result


def _extract(model, tok, rows, batch_size, device, split):
    vectors, counts, sequence_lengths = [], [], []
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            sequences = [encode(tok, row["en"]) for row in batch]
            if any(len(sequence) > model.src_pos.pe.shape[1] for sequence in sequences):
                raise ValueError("source exceeds model positional capacity; no truncation is permitted")
            tokens = torch.full((len(batch), max(map(len, sequences))), PAD, dtype=torch.long, device=device)
            for index, sequence in enumerate(sequences):
                tokens[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            hidden, _ = model.encode(tokens)
            vectors.append(masked_mean_pool(hidden, tokens).cpu().numpy())
            counts.extend(((tokens != PAD) & (tokens != BOS) & (tokens != EOS)).sum(dim=1).cpu().tolist())
            sequence_lengths.extend(map(len, sequences))
            if offset == 0 or (offset // batch_size + 1) % 100 == 0 or offset + batch_size >= len(rows):
                print(json.dumps({"vector_progress": min(offset + batch_size, len(rows)), "split": split,
                                  "total": len(rows)}, ensure_ascii=False), flush=True)
    return (np.concatenate(vectors), np.asarray(counts, dtype=np.int64),
            np.asarray(sequence_lengths, dtype=np.int64), time.perf_counter() - start)


def _load_checked_npz(path, metadata, expected):
    if metadata.get("inputs") != expected:
        raise ValueError(f"saved stage inputs differ: {path}")
    if sha256(path) != metadata.get("file_sha256"):
        raise ValueError(f"saved stage file checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if {key: array_hash(value) for key, value in arrays.items()} != metadata.get("array_sha256"):
        raise ValueError(f"saved stage vector/ID/parameter checksum mismatch: {path}")
    return arrays


def _plot(summary, arrays, features, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = Path(out) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ratios = summary["pca"]["explained_variance_ratio"]
    for axis, split in zip(axes, ("train", "validation")):
        values, labels = arrays[split + "_scores"], arrays[split + "_labels"]
        axis.scatter(values[:, 0], values[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9,
                     s=4 if split == "train" else 12, alpha=.45 if split == "train" else .7)
        axis.set(title=f"{split.title()} (n={len(values):,})",
                 xlabel=f"PC1 ({ratios[0]:.1%} variance)", ylabel=f"PC2 ({ratios[1]:.1%} variance)")
        axis.grid(alpha=.15)
        axis.legend(handles=[Line2D([], [], marker="o", linestyle="", color=plt.get_cmap("tab10")(i),
                                    label=f"C{i}") for i in range(summary["cluster_count"])],
                    title="Cluster", fontsize=8)
    fig.suptitle(f"KMeans fitted in PCA-{len(ratios)}; two-dimensional display only")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"pca_clusters.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    selected = ("src_words", "src_rare_fraction", "src_fragmentation")
    for row, split in enumerate(("train", "validation")):
        labels = arrays[split + "_labels"]
        for axis, feature in zip(axes[row], selected):
            groups = [np.asarray([record[feature] for record, label in zip(features[split], labels) if label == cluster])
                      for cluster in range(summary["cluster_count"])]
            # Empty boxes are explicit placeholders; sample counts remain on x labels.
            axis.boxplot([group if len(group) else [np.nan] for group in groups], showfliers=False,
                         tick_labels=[f"C{i}\nn={len(group)}" for i, group in enumerate(groups)])
            axis.set(title=f"{split.title()}: {feature}", xlabel="Cluster (arbitrary label)")
            axis.grid(axis="y", alpha=.2)
    fig.suptitle("Feature distributions by cluster; boxes show median / IQR, outliers hidden")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"cluster_features.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_vectors(config, out, device="cuda"):
    """Extract frozen-checkpoint vectors, then fit only selected training rows.

    Completed vector and fitted-parameter stages are reused only after exact
    input, file, array, and ID verification. Plots/summaries can be regenerated.
    """
    output = Path(out) / "vectors"
    output.mkdir(parents=True, exist_ok=True)
    settings, sampling = config["vectors"], config["sampling"]
    if (type(settings["batch_size"]) is not int or settings["batch_size"] < 1
            or type(settings["pca_components"]) is not int or settings["pca_components"] < 2
            or type(settings["silhouette_size"]) is not int or settings["silhouette_size"] < 2):
        raise ValueError("positive batch size and at least two PCA components / silhouette samples are required")
    inputs = _input_hashes(config)
    data = Path(config["data_dir"])
    train = sample_training_rows(read_pairs(data / "train.jsonl"), sampling["train_size"], sampling["seed"])
    validation = read_pairs(data / "validation.jsonl")
    valid_ids = [row["id"] for row in validation]
    if len(set(valid_ids)) != len(valid_ids) or not validation:
        raise ValueError("validation must have nonempty unique original IDs")
    selection = {"schema": 1, "sampling": sampling,
                 "rule": "random.Random(seed).sample(cleaned training row positions, size); sort by original ID",
                 "train_ids": [row["id"] for row in train], "validation_ids": valid_ids,
                 "train_sha256": inputs["train"]["sha256"], "validation_sha256": inputs["validation"]["sha256"]}
    selection_path = output / "selection.json"
    if selection_path.exists():
        if read_json(selection_path) != selection:
            raise ValueError("saved selection differs from fixed original IDs or data hashes")
    else:
        _atomic_json(selection_path, selection)
    embedding_inputs = {"files": inputs, "selection_sha256": sha256(selection_path),
                        "batch_size": settings["batch_size"], "precision": "float32", "device": str(device),
                        "pooling": "encode complete BOS/source/EOS; mean of final outputs excluding PAD/BOS/EOS; retain UNK; no truncation"}
    vector_path, vector_meta_path = output / "embeddings.npz", output / "embeddings_meta.json"
    if vector_meta_path.exists():
        vector_meta = read_json(vector_meta_path)
        vectors = _load_checked_npz(vector_path, vector_meta, embedding_inputs)
    else:
        if vector_path.exists():
            raise ValueError("uncommitted embeddings.npz exists without metadata; inspect incomplete stage before recovery")
        selected_device = select_device(device)
        model, tok, model_config, checkpoint = load_checkpoint(config["checkpoint"], selected_device, data)
        checkpoint_update = checkpoint.get("update")
        del checkpoint
        vectors, timings = {}, {}
        for split, rows in (("train", train), ("validation", validation)):
            embedded, counts, lengths, elapsed = _extract(model, tok, rows, settings["batch_size"], selected_device, split)
            vectors[split + "_embeddings"] = embedded
            vectors[split + "_ids"] = np.asarray([row["id"] for row in rows], dtype=np.int64)
            vectors[split + "_token_counts"] = counts
            vectors[split + "_sequence_lengths"] = lengths
            timings[split] = elapsed
        vector_meta = {"schema": 1, "created_at": _now(), "inputs": embedding_inputs,
                       "dimension": int(vectors["train_embeddings"].shape[1]), "model_config": model_config,
                       "checkpoint_update": checkpoint_update, "elapsed_seconds": timings,
                       "torch_version": torch.__version__, "numpy_version": np.__version__,
                       "eval_mode": not model.training, "gradient_enabled": False,
                       "empty_effective_sequences": "raise ValueError", "long_sequences": "no truncation; capacity/OOM errors propagate"}
        _atomic_npz(vector_path, **vectors)
        vector_meta.update(file_sha256=sha256(vector_path), array_sha256={k: array_hash(v) for k, v in vectors.items()})
        _atomic_json(vector_meta_path, vector_meta)
        del model
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    for split, rows in (("train", train), ("validation", validation)):
        ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
        if (not np.array_equal(vectors[split + "_ids"], ids)
                or vectors[split + "_embeddings"].shape != (len(ids), vector_meta["dimension"])
                or not np.isfinite(vectors[split + "_embeddings"]).all()
                or np.any(vectors[split + "_token_counts"] <= 0)):
            raise ValueError(f"saved {split} embedding IDs/shapes/content are inconsistent")
    fitted_inputs = {"embeddings_file_sha256": sha256(vector_path), "selection_sha256": sha256(selection_path),
                     "settings": settings, "source_sha256": inputs["source_sha256"],
                     "fitted_split": "train", "fitted_ids_sha256": array_hash(vectors["train_ids"])}
    fit_path, fit_meta_path = output / "fit.npz", output / "fit_meta.json"
    if fit_meta_path.exists():
        fit_meta = read_json(fit_meta_path)
        fitted = _load_checked_npz(fit_path, fit_meta, fitted_inputs)
    else:
        if fit_path.exists():
            raise ValueError("uncommitted fit.npz exists without metadata; inspect incomplete stage before recovery")
        started = time.perf_counter()
        fitted = fit_standardized_pca(vectors["train_embeddings"], settings["pca_components"])
        train_scores = transform_pca(vectors["train_embeddings"], fitted)
        km = fit_kmeans(train_scores, settings["kmeans_clusters"], settings["kmeans_seed"],
                        settings["kmeans_n_init"], settings["kmeans_max_iter"], settings["kmeans_tol"])
        fitted.update(centers=km["centers"], fitted_train_ids=vectors["train_ids"])
        fit_meta = {"schema": 1, "created_at": _now(), "inputs": fitted_inputs,
                    "implementation": "NumPy population standardization + full SVD PCA + k-means++/Lloyd",
                    "standardization_ddof": 0, "pca_variance_ddof": 1,
                    "constant_feature_rule": "scale=1", "svd_sign_rule": "largest absolute loading positive",
                    "convergence": "maximum Euclidean center displacement <= absolute tol",
                    "empty_cluster_rule": "relocate to farthest currently represented training point",
                    "selected_restart": {k: v for k, v in km.items() if k not in ("centers", "labels", "restarts")},
                    "restarts": km["restarts"], "fit_elapsed_seconds": time.perf_counter() - started,
                    "numpy_version": np.__version__}
        _atomic_npz(fit_path, **fitted)
        fit_meta.update(file_sha256=sha256(fit_path), array_sha256={k: array_hash(v) for k, v in fitted.items()})
        _atomic_json(fit_meta_path, fit_meta)
    if not np.array_equal(fitted["fitted_train_ids"], vectors["train_ids"]):
        raise ValueError("fitted transform training IDs differ from fixed training selection")
    arrays, features = {}, {}
    assignment_lines = []
    for split in ("train", "validation"):
        scores = transform_pca(vectors[split + "_embeddings"], fitted)
        labels = predict_clusters(scores, fitted["centers"])
        arrays.update({split + "_scores": scores, split + "_labels": labels})
        features[split] = _aligned_features(Path(out) / "data" / f"{split}_features.jsonl", vectors[split + "_ids"])
        for identity, label, score in zip(vectors[split + "_ids"], labels, scores):
            assignment_lines.append(json.dumps({"split": split, "id": int(identity), "cluster": int(label),
                                               "pc1": float(score[0]), "pc2": float(score[1])}))
    assignments = output / "assignments.jsonl"
    assignment_text = "\n".join(assignment_lines) + "\n"
    if assignments.exists() and assignments.read_text(encoding="utf-8") != assignment_text:
        raise ValueError("saved assignments disagree with verified transforms and source IDs")
    if not assignments.exists():
        assignments.write_text(assignment_text, encoding="utf-8")
    clusters = []
    for cluster in range(settings["kmeans_clusters"]):
        record = {"cluster": cluster}
        for split in ("train", "validation"):
            mask = arrays[split + "_labels"] == cluster
            members = [row for row, include in zip(features[split], mask) if include]
            record[split] = {"n": len(members), "empty": len(members) == 0,
                "small": len(members) < config.get("small_cluster_size", 30),
                "features": {key: _describe([row[key] for row in members]) for key in FEATURES}}
            if split == "validation":
                record[split]["representative_ids"] = sorted(int(i) for i in vectors[split + "_ids"][mask])[:config.get("representatives_per_cluster", 3)]
        clusters.append(record)
    ratios = fitted["explained_variance_ratio"].tolist()
    summary = {"schema": 1, "created_at": _now(), "purpose": "exploratory analysis after completed translation experiments",
               "cluster_count": settings["kmeans_clusters"], "clusters": clusters,
               "pca": {"explained_variance_ratio": ratios, "pc1": ratios[0], "pc2": ratios[1],
                       "first_two": sum(ratios[:2]), "total_retained": sum(ratios)},
               "diagnostics": cluster_diagnostics(arrays["train_scores"], arrays["train_labels"], vectors["train_ids"],
                                                  settings["silhouette_size"], settings["silhouette_seed"]),
               "feature_associations": {split: feature_associations(features[split], arrays[split + "_labels"], arrays[split + "_scores"])
                                        for split in ("train", "validation")},
               "fitted_split": "train", "fit_sample_size": len(train), "validation_size": len(validation),
               "representative_rule": "smallest original validation IDs within each cluster",
               "interpretation_limits": ["cluster labels are arbitrary", "two-dimensional separation is not semantic validation",
                                         "associations are descriptive, not causal", "no selection by validation/test translation quality"],
               "inputs": fitted_inputs, "embeddings_sha256": sha256(vector_path), "fit_sha256": sha256(fit_path),
               "assignments_sha256": sha256(assignments),
               "feature_files_sha256": {split: sha256(Path(out) / "data" / f"{split}_features.jsonl") for split in ("train", "validation")}}
    _atomic_json(output / "summary.json", summary)
    _plot(summary, arrays, features, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_vectors(read_json(args.config), args.out, args.device)
    print(json.dumps({"cluster_count": summary["cluster_count"], "train": summary["fit_sample_size"],
                      "validation": summary["validation_size"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
