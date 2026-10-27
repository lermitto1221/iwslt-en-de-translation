"""Audit saved exploration: train-only fit, exact joins, metric recomputation, history."""
import argparse
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import sacrebleu

from .common import read_json, sha256
from .data import read_pairs
from .explore import DEFAULT_CONFIG, atomic_json, check_hashes, run, utc_now, validate_plan, verify_history
from .explore_vectors import array_hash


def verify(config_path=DEFAULT_CONFIG):
    cfg, out = validate_plan(config_path)
    for stage in ("data", "vectors", "quality"):
        check_hashes(read_json(out / "stages" / f"{stage}.json")["files_sha256"])
    data = Path(cfg["data_dir"])
    rows = {s: read_pairs(data / f"{s}.jsonl") for s in ("train", "validation")}
    features = {s: read_pairs(out / "data" / f"{s}_features.jsonl") for s in rows}
    for split in rows:
        assert [r["id"] for r in rows[split]] == [r["id"] for r in features[split]], f"{split} feature alignment"
    selected = sorted(random.Random(cfg["sampling"]["seed"]).sample(rows["train"], cfg["sampling"]["train_size"]), key=lambda r:r["id"])
    selection = read_json(out / "vectors/selection.json")
    assert selection["train_ids"] == [r["id"] for r in selected]
    assert selection["validation_ids"] == [r["id"] for r in rows["validation"]]
    arrays = {}
    for filename in ("embeddings", "fit"):
        path = out / "vectors" / f"{filename}.npz"
        meta = read_json(out / "vectors" / f"{filename}_meta.json")
        assert meta["file_sha256"] == sha256(path)
        with np.load(path, allow_pickle=False) as f:
            arrays[filename] = {k:f[k] for k in f.files}
        assert {k:array_hash(v) for k,v in arrays[filename].items()} == meta["array_sha256"]
    emb, fit = arrays["embeddings"], arrays["fit"]
    assert fit["fitted_train_ids"].tolist() == selection["train_ids"] == emb["train_ids"].tolist()
    assert emb["validation_ids"].tolist() == selection["validation_ids"]
    train = emb["train_embeddings"].astype(np.float64)
    expected_scale = train.std(axis=0, ddof=0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(fit["standard_mean"], train.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit["standard_scale"], expected_scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit["pca_components"] @ fit["pca_components"].T,
                               np.eye(cfg["vectors"]["pca_components"]), atol=1e-12)
    assignments = read_pairs(out / "vectors/assignments.jsonl")
    for split in rows:
        sample = [a for a in assignments if a["split"] == split]
        assert [a["id"] for a in sample] == emb[split + "_ids"].tolist()
        matrix = emb[split + "_embeddings"].astype(np.float64)
        projected = ((matrix-fit["standard_mean"])/fit["standard_scale"]-fit["pca_mean"]) @ fit["pca_components"].T
        # Independent direct Euclidean formula; no call to fitted implementation.
        distances = np.sum((projected[:,None,:]-fit["centers"][None,:,:])**2, axis=2)
        assert np.array_equal(np.argmin(distances,axis=1), [a["cluster"] for a in sample])
        np.testing.assert_allclose(projected[:,:2], [[a["pc1"],a["pc2"]] for a in sample], atol=1e-11)
    joined = read_pairs(out / "quality/predictions_joined.jsonl")
    assert [r["id"] for r in joined] == selection["validation_ids"]
    joined_by_id = {r["id"]:r for r in joined}
    labels = {r["id"]:r["cluster"] for r in assignments if r["split"] == "validation"}
    for actual, original in zip(joined,rows["validation"]):
        assert actual["source"] == original["en"] and actual["reference"] == original["de"] and actual["cluster"] == labels[actual["id"]]
    groups = read_json(out / "quality/group_scores.json")
    recalculated = 0
    for group in groups:
        category = group["category"]
        if category == "overall":
            expected_ids = [r["id"] for r in joined]
        elif category == "cluster":
            expected_ids = [r["id"] for r in joined if r["cluster"] == int(group["group"])]
        else:
            from .explore_quality import bin_label
            bounds=cfg["quality_bins"][category]
            idx=[bin_label(i,bounds) for i in range(len(bounds)+1)].index(group["group"])
            low=float('-inf') if idx==0 else bounds[idx-1]
            high=float('inf') if idx==len(bounds) else bounds[idx]
            expected_ids=[r["id"] for r in joined if low < r["features"][category] <= high]
        assert group["ids"] == expected_ids and group["n"] == len(expected_ids)
        values = [joined_by_id[i] for i in expected_ids]
        refs = [r["reference"] for r in values]
        for key, saved in group["scores"].items():
            preds=[r["predictions"][key] for r in values]
            if not refs:
                assert saved["bleu"] is None and saved["chrf"] is None
                continue
            bleu=sacrebleu.metrics.BLEU(tokenize="13a",lowercase=False)
            chrf=sacrebleu.metrics.CHRF(char_order=6,word_order=0,beta=2,lowercase=False)
            for name, metric in (("bleu",bleu),("chrf",chrf)):
                score=metric.corpus_score(preds,[refs]).score
                assert math.isclose(score,saved[name],abs_tol=1e-10)
                assert str(metric.get_signature())==saved[name+"_signature"]
            recalculated += 1
        for method in cfg["methods"]:
            for name in ("bleu","chrf"):
                expected=group["scores"][f"long_{method}"][name]-group["scores"][f"baseline_{method}"][name] if values else None
                assert group["long_minus_baseline"][method][name] == expected
    examples=read_pairs(out / "quality/cluster_examples.jsonl")
    expected=[]
    for cluster in range(cfg["vectors"]["kmeans_clusters"]):
        expected += sorted((r for r in joined if r["cluster"]==cluster),key=lambda r:r["id"])[:cfg["representatives_per_cluster"]]
    assert examples==expected
    before={str(p): (sha256(p),p.stat().st_mtime_ns) for directory in ("data","vectors","quality","figures") for p in (out/directory).rglob('*') if p.is_file()}
    with patch('mt.explore_data.run_data_analysis',side_effect=AssertionError('must reuse data')), \
         patch('mt.explore_vectors.run_vectors',side_effect=AssertionError('must reuse vectors')), \
         patch('mt.explore_quality.run_quality_analysis',side_effect=AssertionError('must reuse quality')), redirect_stdout(io.StringIO()):
        run(config_path)
    after={str(p): (sha256(p),p.stat().st_mtime_ns) for directory in ("data","vectors","quality","figures") for p in (out/directory).rglob('*') if p.is_file()}
    assert before==after
    history=verify_history(out)
    result={"verified_at_utc":utc_now(),"status":"verified","train_features":len(features['train']),"validation_features":len(features['validation']),
            "sample_ids_reproduced":len(selected),"all_assignments_recomputed":len(assignments),"training_only_standardization":True,"pca_orthogonality":True,
            "prediction_reference_cluster_order_verified":True,"groups_checked":len(groups),"corpus_score_sets_recomputed":recalculated,
            "fixed_examples":len(examples),"completed_stage_reuse_no_compute":True,"unchanged_analysis_files_hash_and_mtime":len(before),
            "history":history,"no_model_training_or_decoding":True}
    atomic_json(out / "verification.json",result)
    print(result,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=DEFAULT_CONFIG)
    a=p.parse_args()
    verify(a.config)


if __name__ == '__main__':
    main()
