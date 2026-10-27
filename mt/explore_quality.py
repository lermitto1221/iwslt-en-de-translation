"""Paired validation diagnostics with saved corpus scores and exact ID joins."""
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np
import sacrebleu

from .common import read_json, sha256, write_json
from .data import read_pairs
from .run_experiments import evaluate_or_reuse, frozen_protocol
from .summary import validate_saved_outputs


def jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def corpus_scores(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("prediction/reference size mismatch")
    if not references:
        return {"n": 0, "bleu": None, "chrf": None, "bleu_signature": None, "chrf_signature": None}
    bleu = sacrebleu.metrics.BLEU(tokenize="13a", lowercase=False)
    chrf = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2, lowercase=False)
    a, b = bleu.corpus_score(predictions, [references]), chrf.corpus_score(predictions, [references])
    return {"n": len(references), "bleu": a.score, "chrf": b.score,
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


def unique_index(rows, label):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate IDs")
    return {r["id"]: r for r in rows}


def join_validation(rows, features, assignments, predictions, clusters):
    reference = unique_index(rows, "validation")
    feature_map = unique_index(features, "features")
    assignment_map = unique_index([r for r in assignments if r["split"] == "validation"], "assignments")
    if set(reference) != set(feature_map) or set(reference) != set(assignment_map):
        raise ValueError("validation/feature/cluster IDs differ")
    mapped = {}
    for name, values in predictions.items():
        # Saved example rows are checked in their original order, not silently sorted.
        if [r["id"] for r in values] != list(reference):
            raise ValueError(f"{name}: prediction order or IDs differ")
        mapped[name] = unique_index(values, name)
        for sample_id, row in reference.items():
            pred = mapped[name][sample_id]
            if pred["source"] != row["en"] or pred["reference"] != row["de"]:
                raise ValueError(f"{name}: source/reference misalignment")
    joined = []
    for sample_id, row in reference.items():
        cluster = assignment_map[sample_id]["cluster"]
        if type(cluster) is not int or not 0 <= cluster < clusters:
            raise ValueError("invalid validation cluster label")
        joined.append({"id": sample_id, "source": row["en"], "reference": row["de"],
                       "cluster": cluster, "features": feature_map[sample_id],
                       "predictions": {name: values[sample_id]["prediction"] for name, values in mapped.items()}})
    return joined


def bin_index(value, boundaries):
    return sum(value > upper for upper in boundaries)


def bin_label(index, boundaries):
    if index == 0:
        return f"<= {boundaries[0]:g}"
    if index == len(boundaries):
        return f"> {boundaries[-1]:g}"
    return f"({boundaries[index - 1]:g}, {boundaries[index]:g}]"


def describe_group(rows):
    result = {}
    for key in ("src_words", "tgt_words", "src_bpe", "src_rare_fraction", "src_unseen_fraction", "src_fragmentation"):
        values = [r["features"][key] for r in rows]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "median": float(np.median(values)) if values else None}
    return result


def group_scores(joined, config):
    groups = [("overall", "overall", joined)]
    for cluster in range(config["vectors"]["kmeans_clusters"]):
        groups.append(("cluster", str(cluster), [r for r in joined if r["cluster"] == cluster]))
    for feature, boundaries in config["quality_bins"].items():
        for index in range(len(boundaries) + 1):
            groups.append((feature, bin_label(index, boundaries),
                           [r for r in joined if bin_index(r["features"][feature], boundaries) == index]))
    output = []
    for category, name, rows in groups:
        scores = {}
        for budget in ("baseline", "long"):
            for method in config["methods"]:
                key = f"{budget}_{method}"
                scores[key] = corpus_scores([r["predictions"][key] for r in rows], [r["reference"] for r in rows])
        differences = {method: {measure: scores[f"long_{method}"][measure] - scores[f"baseline_{method}"][measure]
                                if rows else None for measure in ("bleu", "chrf")} for method in config["methods"]}
        output.append({"category": category, "group": name, "n": len(rows), "ids": [r["id"] for r in rows],
                       "small": 0 < len(rows) < config["small_cluster_size"], "empty": not rows,
                       "features": describe_group(rows), "scores": scores, "long_minus_baseline": differences})
    return output


def lexical_profiles(config, assignments):
    """Training-only descriptive document prevalence, never automatic topic labels."""
    training = unique_index(read_pairs(Path(config["data_dir"]) / "train.jsonl"), "train")
    labels = [r for r in assignments if r["split"] == "train"]
    unique_index(labels, "train assignments")
    k = config["vectors"]["kmeans_clusters"]
    counts, sizes, overall = [Counter() for _ in range(k)], [0] * k, Counter()
    stop = set(config["lexical_profile"]["stopwords"].split())
    for row in labels:
        words = {word.casefold() for word in training[row["id"]]["en"].split()}
        words = {word for word in words if word.isalpha() and word not in stop}
        counts[row["cluster"]].update(words)
        sizes[row["cluster"]] += 1
        overall.update(words)
    result = []
    for cluster, count in enumerate(counts):
        terms = []
        for word, support in count.items():
            if support < config["lexical_profile"]["minimum_cluster_document_count"]:
                continue
            local = (support + 1) / (sizes[cluster] + 2)
            global_rate = (overall[word] + 1) / (len(labels) + 2)
            terms.append({"term": word, "cluster_documents": support, "all_documents": overall[word],
                          "prevalence_ratio": local / global_rate})
        terms.sort(key=lambda row: (-row["prevalence_ratio"], row["term"]))
        result.append({"cluster": cluster, "train_n": sizes[cluster], "terms": terms[:config["lexical_profile"]["top_terms"]]})
    return {"fit_split": "train", "sample_n": len(labels), "definition": "add-one smoothed document prevalence ratio; alphabetic whitespace-casefold tokens, fixed stopwords, support>=10; hints, not semantic labels", "clusters": result}


def plot_quality(groups, config, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(out) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    selected = [r for r in groups if r["category"] == "cluster"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, measure in zip(axes, ("bleu", "chrf")):
        for budget, color in (("baseline", "#4278a4"), ("long", "#be652f")):
            ax.plot(range(len(selected)), [r["scores"][budget + "_beam"][measure] for r in selected],
                    marker="o", color=color, label=budget + " beam")
        ax.set_xticks(range(len(selected)), [f"C{r['group']}\nn={r['n']}" for r in selected])
        ax.set(title="Validation by frozen encoder cluster", ylabel="Corpus " + measure.upper())
        ax.legend(); ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"quality_clusters.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, feature in zip(axes, config["quality_bins"]):
        rows = [r for r in groups if r["category"] == feature]
        for budget, color in (("baseline", "#4278a4"), ("long", "#be652f")):
            values = [r["scores"][budget + "_beam"]["bleu"] if r["n"] else np.nan for r in rows]
            ax.plot(range(len(rows)), values, marker="o", color=color, label=budget + " beam")
        ax.set_xticks(range(len(rows)), [f"{r['group']}\nn={r['n']}" for r in rows], rotation=15)
        ax.set(title=feature, ylabel="Validation corpus BLEU (13a)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"quality_features.{suffix}", dpi=180)
    plt.close(fig)


def run_quality_analysis(config, out):
    out = Path(out)
    folder = out / "quality"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = frozen_protocol(config["protocol"])
    checkpoints = {"baseline": Path(config["baseline_checkpoint"]), "long": Path(config["checkpoint"])}
    configs = [read_json(p.parent / "config.json") for p in checkpoints.values()]
    normalized = [{k: v for k, v in cfg.items() if k not in ("run_dir", "max_updates")} for cfg in configs]
    if normalized[0] != normalized[1] or configs[0]["seed"] != 42:
        raise ValueError("validation budget contrast differs beyond run directory and update budget")
    sources, predictions = {}, {}
    for budget, checkpoint in checkpoints.items():
        for method in config["methods"]:
            key = f"{budget}_{method}"
            dest = folder / key
            original = config["validation_reuse"].get(key)
            if original:
                metrics = evaluate_or_reuse(checkpoint, "validation", method, original,
                    beam_size=protocol["beam_size"], alpha=protocol["length_penalty"], max_tokens=protocol["max_generation_tokens"])
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("metrics.json", "predictions.txt", "references.txt", "examples.jsonl"):
                    if (dest / name).exists() and sha256(dest / name) != sha256(Path(original) / name):
                        raise ValueError("local copy differs from historical validation output")
                    if not (dest / name).exists():
                        shutil.copyfile(Path(original) / name, dest / name)
            else:
                metrics = evaluate_or_reuse(checkpoint, "validation", method, dest,
                    beam_size=protocol["beam_size"], alpha=protocol["length_penalty"], max_tokens=protocol["max_generation_tokens"])
            validate_saved_outputs(metrics, dest, key)
            if metrics["samples"] != 888 or metrics["split"] != "validation":
                raise ValueError("complete validation required")
            sources[key] = {"origin": str(Path(original).resolve()) if original else "generated in this analysis",
                            "historical_timing_reused": bool(original), "metrics": metrics,
                            "files_sha256": {name: sha256(dest / name) for name in ("metrics.json", "predictions.txt", "references.txt", "examples.jsonl")}}
            predictions[key] = read_pairs(dest / "examples.jsonl")
            print({"validation": key, "bleu": metrics["bleu"], "samples": metrics["samples"], "historical_reuse": bool(original)}, flush=True)
    if sources["baseline_beam"]["metrics"]["initial_model_sha256"] != sources["long_beam"]["metrics"]["initial_model_sha256"]:
        raise ValueError("same-seed initialization mismatch")
    rows = read_pairs(Path(config["data_dir"]) / "validation.jsonl")
    features = read_pairs(out / "data/validation_features.jsonl")
    assignments = read_pairs(out / "vectors/assignments.jsonl")
    joined = join_validation(rows, features, assignments, predictions, config["vectors"]["kmeans_clusters"])
    groups = group_scores(joined, config)
    for key, source in sources.items():
        if not math.isclose(groups[0]["scores"][key]["bleu"], source["metrics"]["bleu"], abs_tol=1e-9):
            raise ValueError("independent corpus BLEU differs from saved metric")
    examples = []
    for cluster in range(config["vectors"]["kmeans_clusters"]):
        examples += sorted((r for r in joined if r["cluster"] == cluster), key=lambda r: r["id"])[:config["representatives_per_cluster"]]
    jsonl(folder / "predictions_joined.jsonl", joined)
    jsonl(folder / "cluster_examples.jsonl", examples)
    write_json(folder / "group_scores.json", groups)
    write_json(folder / "evaluation_sources.json", sources)
    write_json(folder / "lexical_profiles.json", lexical_profiles(config, assignments))
    plot_quality(groups, config, out)
    result = {"status": "complete", "split": "validation", "samples": len(joined),
              "overall": groups[0], "groups": groups[1:], "protocol": protocol,
              "protocol_sha256": sha256(config["protocol"]), "sources": sources,
              "score_definition": "SacreBLEU case-sensitive 13a corpus BLEU; chrF2 char_order6 word_order0 beta2. Scores separately computed from each group's saved texts, never sentence BLEU averages.",
              "selection_bias": "Validation selected both checkpoints and the historic decoder; these scores and clusters are exploratory diagnostics, not unbiased held-out confirmation.",
              "example_ids": [r["id"] for r in examples], "example_rule": "lowest3 original validation IDs per cluster; agent review only",
              "small_group_threshold": config["small_cluster_size"],
              "no_training_performed": True, "no_test_based_selection": True}
    write_json(folder / "summary.json", result)
    return result
