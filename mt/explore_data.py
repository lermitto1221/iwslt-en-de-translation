"""Training-only vocabulary statistics and reproducible corpus exploration.

This module reads historical data but never updates it.  Test text is deliberately
outside its input set; test counts, when shown, are labelled manifest references.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import read_json, sha256, write_json
from .data import read_pairs
from .tokenizer import BOS, EOS, PAD, UNK, load


REMOVAL_STEPS = ("empty", "duplicate_pair", "long_words", "length_ratio")
FEATURES = ("src_words", "tgt_words", "src_bpe", "tgt_bpe", "src_fragmentation",
            "tgt_fragmentation", "src_rare_fraction", "tgt_rare_fraction",
            "src_unseen_fraction", "tgt_unseen_fraction", "length_ratio")


def word_tokens(text):
    """Whitespace tokens, casefolded; punctuation is retained exactly as attached."""
    return [word.casefold() for word in text.split()]


def replay_cleaning(raw_rows, split, *, max_words=300, max_ratio=3.0):
    """Replay historical first-match rules, retaining original Parquet row IDs.

    A pair enters ``seen`` before length checks, matching ``mt.data.prepare``.
    Validation is stripped but is neither deduplicated nor length-filtered.
    """
    if split not in ("train", "validation"):
        raise ValueError("exploration reads train and validation only")
    if max_words < 1 or max_ratio < 1:
        raise ValueError("invalid cleaning thresholds")
    pairs, raw_lengths, counts, seen = [], [], Counter(), set()
    for row_id, row in enumerate(raw_rows):
        if isinstance(row, str):
            row = json.loads(row)
        if not isinstance(row, dict) or not isinstance(row.get("en"), str) or not isinstance(row.get("de"), str):
            raise ValueError(f"unaligned/non-string {split} raw row {row_id}")
        en, de = row["en"].strip(), row["de"].strip()
        a, b = len(en.split()), len(de.split())
        raw_lengths.append((a, b))
        if not en or not de:
            counts["empty"] += 1
            if split != "train":
                raise ValueError(f"empty validation pair at {row_id}; no silent filtering")
            continue
        if split == "train":
            key = (en, de)
            if key in seen:
                counts["duplicate_pair"] += 1
                continue
            seen.add(key)
            if max(a, b) > max_words:
                counts["long_words"] += 1
                continue
            if max(a, b) / max(1, min(a, b)) > max_ratio:
                counts["length_ratio"] += 1
                continue
        pairs.append({"id": row_id, "en": en, "de": de})
    return pairs, {name: counts[name] for name in REMOVAL_STEPS}, raw_lengths


def assert_pair_identity(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(f"{label}: cleaned row count mismatch {len(actual)} != {len(expected)}")
    ids = [row["id"] for row in actual]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate original IDs")
    for position, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            raise ValueError(f"{label}: original ID/text/order mismatch at position {position}")


def training_word_frequencies(train_rows):
    return {lang: Counter(token for row in train_rows for token in word_tokens(row[lang]))
            for lang in ("en", "de")}


def sentence_features(rows, tokenizer, frequencies, rare_max_count=5, max_train_tokens=128):
    if not isinstance(rare_max_count, int) or rare_max_count < 1:
        raise ValueError("rare_max_count must be a positive integer")
    if not isinstance(max_train_tokens, int) or max_train_tokens < 3:
        raise ValueError("max_train_tokens must include BOS and EOS")
    result, seen = [], set()
    for row in rows:
        row_id = row["id"]
        if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id in seen:
            raise ValueError("features require unique integer original IDs")
        seen.add(row_id)
        item = {"id": row_id}
        for lang, prefix in (("en", "src"), ("de", "tgt")):
            words = word_tokens(row[lang])
            if not words:
                raise ValueError(f"empty {lang} sequence at original ID {row_id}")
            # Direct SentencePiece encoding adds no BOS/EOS.  UNK remains a piece.
            pieces = tokenizer.encode(row[lang], out_type=int)
            if not pieces or any(piece in (PAD, BOS, EOS) for piece in pieces):
                raise ValueError(f"empty/special-only or unexpected structural token at ID {row_id}")
            counts = [frequencies[lang].get(word, 0) for word in words]
            item.update({f"{prefix}_words": len(words), f"{prefix}_bpe": len(pieces),
                         f"{prefix}_fragmentation": len(pieces) / len(words),
                         f"{prefix}_rare_fraction": sum(c <= rare_max_count for c in counts) / len(words),
                         f"{prefix}_unseen_fraction": sum(c == 0 for c in counts) / len(words)})
        item["length_ratio"] = item["tgt_words"] / item["src_words"]
        item["train_length_eligible"] = max(item["src_bpe"], item["tgt_bpe"]) + 2 <= max_train_tokens
        result.append(item)
    return result


def describe(values):
    """Full-population descriptive statistics; spread is population SD (ddof=0)."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "population_std": None, "min": None, "max": None,
                "quantiles": {name: None for name in ("p05", "p25", "p50", "p75", "p95", "p99")},
                "histogram": {"counts": [], "bin_edges": []}}
    if not np.isfinite(values).all():
        raise ValueError("non-finite corpus features")
    quantiles = np.quantile(values, [.05, .25, .5, .75, .95, .99], method="linear")
    counts, edges = np.histogram(values, bins=20)
    return {"n": len(values), "mean": float(values.mean()), "population_std": float(values.std(ddof=0)),
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": dict(zip(("p05", "p25", "p50", "p75", "p95", "p99"), map(float, quantiles))),
            "histogram": {"counts": counts.tolist(), "bin_edges": edges.tolist(),
                          "definition": "20 equal-width bins over the full observed range; left-closed/right-open except final bin includes its right edge."}}


def feature_summary(rows):
    return {name: describe([row[name] for row in rows]) for name in FEATURES}


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _input_evidence(data_dir):
    inputs = {name: {"path": str((data_dir / name).resolve()), "sha256": sha256(data_dir / name)}
              for name in ("manifest.json", "tokenizer.json", "spm.model", "spm.vocab", "train.jsonl", "validation.jsonl")}
    for split in ("train", "validation"):
        path = data_dir / "raw" / f"iwslt2017-{split}.parquet"
        inputs[f"raw/{path.name}"] = {"path": str(path.resolve()), "sha256": sha256(path) if path.exists() else None}
    return inputs


def audit_cleaned_data(data_dir, manifest, inputs):
    """Rebuild available raw splits; missing raw data is explicitly unverified."""
    import pyarrow.parquet as pq
    if manifest.get("train_limit") is not None:
        raise ValueError("analysis requires the full cleaned train corpus, not a train_limit subset")
    rules = manifest.get("cleaning_rules", {}).get("train", {})
    if rules.get("order") != list(REMOVAL_STEPS):
        raise ValueError("unknown cleaning order; cannot claim exact replay")
    if manifest.get("cleaning_rules", {}).get("all_splits") != {
            "strip_outer_whitespace": True, "require_aligned_string_pairs": True}:
        raise ValueError("unknown general cleaning rules")
    if manifest.get("cleaning_rules", {}).get("validation_and_test") != {
            "empty_pair": "error", "filter_length": False, "deduplicate": False, "truncate": False}:
        raise ValueError("unknown validation cleaning rules")
    rows, audit = {}, {}
    for split in ("train", "validation"):
        evidence = manifest["files"][split]
        if inputs[f"{split}.jsonl"]["sha256"] != evidence["jsonl_sha256"]:
            raise ValueError(f"{split} cleaned input hash mismatch")
        saved = read_pairs(data_dir / f"{split}.jsonl")
        if len(saved) != evidence["rows_saved"]:
            raise ValueError(f"{split} saved count mismatch")
        assert_pair_identity(saved, saved, split)
        raw_path = data_dir / "raw" / f"iwslt2017-{split}.parquet"
        if not raw_path.exists():
            audit[split] = {"status": "raw_unavailable", "rows_saved": len(saved),
                            "rows_raw_manifest_only": evidence["rows_raw"],
                            "exclusive_removals_manifest_only": manifest["cleaning"][split],
                            "limitation": "Raw Parquet unavailable; no independent cleaning replay claimed."}
            rows[split] = saved
            continue
        if inputs[f"raw/{raw_path.name}"]["sha256"] != evidence["parquet_sha256"]:
            raise ValueError(f"{split} raw input hash mismatch")
        table = pq.read_table(raw_path, columns=["translation"])
        if table.num_rows != evidence["rows_raw"]:
            raise ValueError(f"{split} raw count mismatch")
        rebuilt, removed, raw_lengths = replay_cleaning(table.column("translation").to_pylist(), split,
            max_words=rules["max_words_per_side"], max_ratio=rules["max_length_ratio"])
        before_sampling = len(rebuilt)
        assert_pair_identity(rebuilt, saved, split)
        if removed != manifest["cleaning"][split]:
            raise ValueError(f"{split} exclusive cleaning removals mismatch")
        audit[split] = {"status": "verified", "rows_raw": len(raw_lengths),
                        "rows_after_cleaning_before_sampling": before_sampling, "rows_saved": len(saved),
                        "exclusive_removals": removed,
                        "raw_src_words": describe([a for a, _ in raw_lengths]),
                        "raw_tgt_words": describe([b for _, b in raw_lengths]),
                        "identity_check": "All original IDs, texts, and order equal saved JSONL exactly."}
        rows[split] = saved
    return rows, audit


def vocabulary_summary(frequencies, validation_rows, rare_max_count):
    output = {}
    for lang in ("en", "de"):
        counts = frequencies[lang]
        total = sum(counts.values())
        validation_counts = Counter(word for row in validation_rows for word in word_tokens(row[lang]))
        valid_total = sum(validation_counts.values())
        output[lang] = {"training_token_count": total, "training_type_count": len(counts),
            "rare_training_types": sum(count <= rare_max_count for count in counts.values()),
            "rare_training_type_fraction": sum(count <= rare_max_count for count in counts.values()) / len(counts),
            "rare_training_token_count": sum(count for count in counts.values() if count <= rare_max_count),
            "rare_training_token_fraction": sum(count for count in counts.values() if count <= rare_max_count) / total,
            "validation_token_count": valid_total, "validation_type_count": len(validation_counts),
            "validation_unseen_type_count": sum(word not in counts for word in validation_counts),
            "validation_unseen_type_fraction": sum(word not in counts for word in validation_counts) / len(validation_counts),
            "validation_unseen_token_count": sum(count for word, count in validation_counts.items() if word not in counts),
            "validation_unseen_token_fraction": sum(count for word, count in validation_counts.items() if word not in counts) / valid_total,
            "validation_rare_token_fraction": sum(count for word, count in validation_counts.items()
                                                  if counts.get(word, 0) <= rare_max_count) / valid_total}
    return output


def _figures(out, rows, frequencies, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"train": "#276FBF", "validation": "#D1495B"}

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = figure_dir / f"data_{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight",
                        metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
            paths.append(str(path.relative_to(out)).replace("\\", "/"))
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, field, title in zip(axes.flat, ("src_words", "tgt_words", "src_bpe", "length_ratio"),
                                  ("English whitespace length", "German whitespace length", "English BPE length (no BOS/EOS)", "German / English word-count ratio")):
        for split in ("train", "validation"):
            values = np.sort([row[field] for row in rows[split]])
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), color=colors[split],
                      label=f"{split} (n={len(values):,})")
        axis.set(title=title, xlabel=field, ylabel="Empirical cumulative fraction")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Complete cleaned training and validation distributions", y=1.02)
    save(fig, "lengths")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for lang, color in (("en", "#276FBF"), ("de", "#D1495B")):
        counts = np.array(sorted(frequencies[lang].values(), reverse=True))
        axes[0].loglog(np.arange(1, len(counts) + 1), counts, label=lang, color=color)
    axes[0].set(title="Training word frequency", xlabel="Frequency rank", ylabel="Occurrences")
    axes[0].legend()
    groups = ["en", "de"]
    positions = np.arange(2)
    for offset, field, label in ((-.25, "rare_training_token_fraction", "Training rare tokens"),
                                 (0, "validation_rare_token_fraction", "Validation rare tokens"),
                                 (.25, "validation_unseen_token_fraction", "Validation unseen tokens")):
        axes[1].bar(positions + offset, [100 * summary["vocabulary"][lang][field] for lang in groups],
                    width=.24, label=label)
    axes[1].set(xticks=positions, xticklabels=groups, ylabel="Percent of whitespace tokens",
                title=f"Rare: training count <= {summary['rare_max_count']}")
    axes[1].legend(fontsize=8)
    for split, linestyle in (("train", "-"), ("validation", "--")):
        for prefix, color in (("src", "#276FBF"), ("tgt", "#D1495B")):
            values = np.sort([row[f"{prefix}_fragmentation"] for row in rows[split]])
            axes[2].plot(values, np.arange(1, len(values) + 1) / len(values), linestyle=linestyle,
                         color=color, label=f"{split} {'en' if prefix == 'src' else 'de'}")
    axes[2].set(title="BPE fragmentation", xlabel="BPE pieces / whitespace token", ylabel="Cumulative fraction")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    save(fig, "vocabulary")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    audit = summary["cleaning_audit"]["train"]
    if audit["status"] == "verified":
        labels = ["Raw", "After empty", "After duplicates", "After words", "After ratio", "BPE eligible"]
        values = [audit["rows_raw"]]
        for step in REMOVAL_STEPS:
            values.append(values[-1] - audit["exclusive_removals"][step])
        values.append(summary["training_length_filter"]["retained"])
        axes[0].barh(labels[::-1], values[::-1], color="#276FBF")
        for index, value in enumerate(values[::-1]):
            axes[0].text(value, index, f" {value:,}", va="center", fontsize=8)
        axes[0].set_xlim(0, max(values) * 1.22)
        axes[0].set(title="Retained training sentence pairs", xlabel="Pairs (axis starts at zero)")
    else:
        axes[0].text(.1, .5, "Raw corpus unavailable;\ncleaning counts not independently replayed", transform=axes[0].transAxes)
        axes[0].axis("off")
    for axis, field, title in ((axes[1], "src_words", "Length-filter effect: English words"),
                               (axes[2], "src_rare_fraction", "Length-filter effect: rare English words")):
        for eligible, label, color in ((True, "Retained", "#276FBF"), (False, "Excluded", "#D1495B")):
            values = np.sort([row[field] for row in rows["train"] if row["train_length_eligible"] == eligible])
            if len(values):
                axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=f"{label} (n={len(values):,})", color=color)
        axis.set(title=title, xlabel=field, ylabel="Cumulative fraction")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    save(fig, "filtering")
    return paths


def run_data_analysis(config, out):
    """Write independently located full-data features, evidence, and six figures."""
    data_dir, out = Path(config["data_dir"]), Path(out)
    rare_max_count = config.get("rare_max_count", 5)
    max_train_tokens = config.get("max_train_tokens", 128)
    data_target = (out / "data").resolve()
    if (out.resolve() == data_dir.resolve() or data_dir.resolve() in out.resolve().parents or
            data_target == data_dir.resolve() or data_target in data_dir.resolve().parents):
        raise ValueError("analysis output must be independent of historical data")
    manifest = read_json(data_dir / "manifest.json")
    inputs = _input_evidence(data_dir)
    effective_config = {"data_dir": str(data_dir.resolve()), "rare_max_count": rare_max_count,
                        "max_train_tokens": max_train_tokens}
    config_hash = hashlib.sha256(json.dumps(effective_config, sort_keys=True).encode()).hexdigest()
    summary_path = out / "data" / "summary.json"
    if summary_path.exists():
        saved = read_json(summary_path)
        if saved["config_sha256"] != config_hash or saved["inputs"] != inputs:
            raise ValueError("existing data analysis does not match current inputs/config")
        for path, digest in saved["output_sha256"].items():
            if not (out / path).exists() or sha256(out / path) != digest:
                raise ValueError(f"saved analysis artifact missing/corrupt: {path}")
        return saved
    rows, audit = audit_cleaned_data(data_dir, manifest, inputs)
    token_meta = read_json(data_dir / "tokenizer.json")
    if (token_meta["model_sha256"] != inputs["spm.model"]["sha256"] or
            token_meta["vocab_sha256"] != inputs["spm.vocab"]["sha256"] or
            token_meta["training_file_sha256"] != inputs["train.jsonl"]["sha256"]):
        raise ValueError("tokenizer provenance mismatch")
    tokenizer = load(data_dir)
    if [tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()] != [PAD, UNK, BOS, EOS]:
        raise ValueError("tokenizer special token mismatch")
    frequencies = training_word_frequencies(rows["train"])
    features = {split: sentence_features(values, tokenizer, frequencies, rare_max_count, max_train_tokens)
                for split, values in rows.items()}
    retained = [row for row in features["train"] if row["train_length_eligible"]]
    excluded = [row for row in features["train"] if not row["train_length_eligible"]]
    definitions = {
        "words": "Python str.split() followed by str.casefold(); punctuation retained attached or as standalone tokens; no additional normalization.",
        "frequency_reference": "Complete cleaned training corpus, before BPE training-length filtering; validation never contributes counts.",
        "rare": f"Training token count <= {rare_max_count}; count zero in validation is rare and unseen.",
        "fractions": "Sentence rare/unseen fractions use token occurrences, not distinct types; aggregate vocabulary fractions are token-weighted unless labelled type.",
        "bpe": "SentencePiece model's ordinary encoding only; PAD/BOS/EOS excluded, UNK counted as a piece; no truncation.",
        "fragmentation": "Ordinary BPE piece count divided by whitespace word count of the whole sentence; not per-word re-encoding.",
        "length_ratio": "target whitespace word count / source whitespace word count (directional, unlike cleaning's symmetric ratio).",
        "training_length_filter": f"Eligible iff both BPE lengths + BOS + EOS <= {max_train_tokens}; applied to training only; validation features report counterfactual eligibility without dropping rows.",
        "statistics": "Full-population means and population SD ddof=0; linear-interpolated quantiles. No random sampling for corpus statistics.",
        "figures": "ECDF curves include every row and the full value range; counts and frequency-rank curves use complete data.",
        "test": "Test count only quoted from historical manifest. Test text, predictions and distributions are not explored."}
    summary = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": effective_config, "config_sha256": config_hash, "inputs": inputs,
        "definitions": definitions, "rare_max_count": rare_max_count, "cleaning_audit": audit,
        "counts": {split: len(values) for split, values in features.items()},
        "test_count_manifest_reference_only": manifest["files"].get("test", {}).get("rows_saved"),
        "vocabulary": vocabulary_summary(frequencies, rows["validation"], rare_max_count),
        "distributions": {split: feature_summary(values) for split, values in features.items()},
        "training_length_filter": {"threshold_including_bos_eos": max_train_tokens,
            "retained": len(retained), "excluded": len(excluded), "excluded_fraction": len(excluded) / len(features["train"]),
            "retained_distributions": feature_summary(retained), "excluded_distributions": feature_summary(excluded)},
        "validation_counterfactual_length_exclusions": sum(not row["train_length_eligible"] for row in features["validation"])}
    data_out = out / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    paths = []
    for split, values in features.items():
        path = data_out / f"{split}_features.jsonl"
        _write_jsonl(path, values)
        paths.append(path.relative_to(out).as_posix())
    for lang, counts in frequencies.items():
        path = data_out / f"word_frequency_{lang}.jsonl"
        _write_jsonl(path, ({"word": word, "count": count} for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))))
        paths.append(path.relative_to(out).as_posix())
    write_json(data_out / "cleaning_audit.json", audit)
    paths.append("data/cleaning_audit.json")
    paths.extend(_figures(out, features, frequencies, summary))
    summary["output_sha256"] = {path: sha256(out / path) for path in paths}
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_data_analysis(read_json(args.config), args.out)
    print(json.dumps({"counts": result["counts"], "out": args.out,
                      "training_length_filter_excluded": result["training_length_filter"]["excluded"]}, indent=2))


if __name__ == "__main__":
    main()
