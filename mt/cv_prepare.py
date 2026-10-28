"""Auditable grouped CV preparation, separate from historical data artifacts.

The paired-schedule serializer is reused with duplicate length aliases. Both
aliases refer to the same fold BPE tokenizer; this is not a unigram experiment.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
import uuid
import zipfile

from .common import read_json, sha256
from .data import EXPECTED, PARQUET_SHA256, REVISION, read_pairs
from .representation_batches import common_rows, schedule_records
from .representation_prepare import _immutable_bytes, _immutable_json, _jsonl_bytes, trainer_options
from .tokenizer import encode

BUILDER_REVISION = "49aad53811d8a9b3dd4a19ca24490f379d59f037"
BUILDER_URL = f"https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/{BUILDER_REVISION}/iwslt2017.py"
ARCHIVE_URL = f"https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/{BUILDER_REVISION}/data/2017-01-trnted/texts/en/de/en-de.zip"
GROUP_SPEC = {
    "schema_version": 1, "split_seed": 20261028, "folds": 3,
    "source_key": "Unicode NFKC, then casefold, then Python split/join whitespace collapse",
    "translation_text": "unchanged; grouping normalization never rewrites training/evaluation text",
    "primary_group": "connected components of same TED talk ID OR identical normalized English source",
    "fallback_group": "identical normalized English source only; does not guarantee talk isolation",
    "feasible": {"largest_component_fraction_max": 0.45, "fold_max_min_ratio_max": 1.2},
    "assignment": "components shuffled with split_seed, stable sort by descending size, assign to smallest fold; fold ties use fixed seeded fold order",
    "fallback_trigger": "primary exceeds either feasibility bound; never split a connected component",
    "source": "official train only; no validation/test text or metrics used to group or choose folds",
    "tokenizer": {"model_type": "bpe", "vocab_size": 8000, "fit_scope": "fold training IDs only, before token length filtering"},
    "train_length_limit": 128, "length_definition": "encoded length includes BOS and EOS on each side",
    "evaluation_filter": False, "evaluation_truncation": False,
    "batching": {"model_seed": 42, "max_updates": 3000, "token_budget": 4096, "accum_steps": 4, "pool_size": 512},
    "candidate_order": [0.0, 0.1], "checkpoint": "exactly update 3000; no fold-based checkpoint selection",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def normalize_source(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_digest(text):
    return hashlib.sha256(normalize_source(text).encode("utf-8")).hexdigest()


def tagged_rows(source_lines, target_lines):
    """Follow the pinned HF builder's line parser, retaining explicit talk IDs.

    Metadata is not inferred from adjacency. A missing/discordant talk tag or
    line count is fatal, and caller subsequently checks every Parquet pair.
    """
    talk_id = None
    row_id = 0
    for line_number, pair in enumerate(itertools.zip_longest(source_lines, target_lines), 1):
        left, right = pair
        if left is None or right is None:
            raise ValueError("source/target tag files have unequal line counts")
        left, right = left.strip(), right.strip()
        if left.startswith("<doc"):
            talk_id = None
        if left.startswith("<talkid>"):
            a, b = re.fullmatch(r"<talkid>([^<>]+)</talkid>", left), re.fullmatch(r"<talkid>([^<>]+)</talkid>", right)
            if not a or not b or a.group(1) != b.group(1):
                raise ValueError(f"unaligned talk IDs at tag-file line {line_number}")
            talk_id = a.group(1)
        if left.startswith("<"):
            if left.startswith("<seg"):
                # Deliberately identical extraction to the pinned builder.
                left = left.split(">")[1].split("<")[0].strip()
                right = right.split(">")[1].split("<")[0].strip()
            else:
                continue
        if talk_id is None:
            raise ValueError(f"text has no explicit talk ID at line {line_number}")
        yield {"id": row_id, "en": left, "de": right, "talk_id": talk_id,
               "tag_file_line": line_number}
        row_id += 1


def verify_raw_alignment(tagged, parquet_rows):
    if len(tagged) != len(parquet_rows):
        raise ValueError("tagged archive and Parquet row counts differ")
    for i, (tag, pair) in enumerate(zip(tagged, parquet_rows)):
        if isinstance(pair, str):
            pair = json.loads(pair)
        if tag["id"] != i or any(tag[lang] != pair[lang] for lang in ("en", "de")):
            raise ValueError(f"tagged archive differs from pinned Parquet at raw ID {i}")
    return len(tagged)


def components(rows, use_talk=True):
    """Return indivisible components as sorted original-ID lists."""
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate original training IDs")
    parent = list(range(len(rows)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first_source, first_talk = {}, {}
    for i, row in enumerate(rows):
        key = normalize_source(row["en"])
        if not key:
            raise ValueError("empty normalized source cannot form a group")
        union(i, first_source.setdefault(key, i))
        if use_talk:
            if not row.get("talk_id"):
                raise ValueError("talk grouping requires verified talk IDs")
            union(i, first_talk.setdefault(row["talk_id"], i))
    result = defaultdict(list)
    for i, original_id in enumerate(ids):
        result[find(i)].append(original_id)
    return sorted((sorted(group) for group in result.values()), key=lambda group: group[0])


def assign_components(groups, seed=20261028, folds=3):
    if not groups or folds < 2:
        raise ValueError("nonempty groups and at least two folds required")
    flattened = [i for group in groups for i in group]
    if any(not group for group in groups) or len(flattened) != len(set(flattened)):
        raise ValueError("groups must be nonempty and disjoint")
    order = [list(group) for group in groups]
    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda group: -len(group))
    fold_order = list(range(folds))
    rng.shuffle(fold_order)
    tie_rank = {fold: i for i, fold in enumerate(fold_order)}
    sizes, assigned = [0] * folds, {}
    for group in order:
        fold = min(range(folds), key=lambda f: (sizes[f], tie_rank[f]))
        sizes[fold] += len(group)
        for original_id in group:
            assigned[original_id] = fold
    return assigned, sizes


def group_diagnostics(groups, sizes):
    total = sum(sizes)
    largest = max(map(len, groups))
    ratio = max(sizes) / min(sizes) if min(sizes) else None
    return {"component_count": len(groups), "largest_component": largest,
            "largest_component_fraction": largest / total,
            "top20_component_sizes": sorted(map(len, groups), reverse=True)[:20],
            "fold_sizes": sizes, "fold_max_min_ratio": ratio,
            "feasible": largest / total <= .45 and ratio is not None and ratio <= 1.2}


def choose_grouping(rows):
    joint = components(rows, use_talk=True)
    joint_assignment, joint_sizes = assign_components(joint)
    joint_diagnostic = group_diagnostics(joint, joint_sizes)
    if joint_diagnostic["feasible"]:
        selected, assignment, diagnostic = joint, joint_assignment, joint_diagnostic
        strategy, reason = "talk_and_normalized_source", None
    else:
        selected = components(rows, use_talk=False)
        assignment, sizes = assign_components(selected)
        diagnostic = group_diagnostics(selected, sizes)
        strategy = "normalized_source_only"
        reason = "joint connected components exceed predeclared size/balance feasibility bounds"
    if not diagnostic["feasible"]:
        raise ValueError("even normalized-source components cannot satisfy the predeclared 3-fold feasibility bounds")
    return selected, assignment, {"strategy": strategy, "fallback_reason": reason,
        "joint_grouping": joint_diagnostic, "selected_grouping": diagnostic}, joint


def partition_rows(rows, assignment, fold):
    if len(rows) != len(assignment) or {r["id"] for r in rows} != set(assignment):
        raise ValueError("fold assignment must cover every training row exactly once")
    if any(type(value) is not int or value not in (0, 1, 2) for value in assignment.values()):
        raise ValueError("invalid fold ID")
    train = [r for r in rows if assignment[r["id"]] != fold]
    holdout = [r for r in rows if assignment[r["id"]] == fold]
    if not train or not holdout:
        raise ValueError("empty fold training or holdout set")
    if {normalize_source(r["en"]) for r in train} & {normalize_source(r["en"]) for r in holdout}:
        raise ValueError("normalized source leakage across fold boundary")
    return train, holdout


def _verify_files(directory, manifest):
    for name, digest in manifest["artifact_sha256"].items():
        path = Path(directory) / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"missing or damaged CV preparation evidence: {path}")


def _source_inputs(source):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    if manifest.get("train_limit") is not None:
        raise ValueError("CV requires the complete official cleaned training set")
    if sha256(source / "train.jsonl") != manifest["files"]["train"]["jsonl_sha256"]:
        raise ValueError("historical cleaned training file fails its manifest")
    raw = source / "raw" / "iwslt2017-train.parquet"
    if sha256(raw) != PARQUET_SHA256["train"]:
        raise ValueError("historical raw training Parquet does not match pinned revision")
    return {"source_train_sha256": sha256(source / "train.jsonl"),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "source_parquet_sha256": sha256(raw), "parquet_revision": REVISION}


def _download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    temporary.rename(path)


def prepare_groups(root, source="artifacts/data"):
    import pyarrow.parquet as pq
    root, source = Path(root), Path(source)
    design = read_json(root / "design.json")
    cv = design["cv"]
    expected = {"folds": 3, "fold_seed": 20261028, "model_seed": 42,
                "candidate_order": [0.0, 0.1], "max_train_tokens_including_special": 128,
                "batch_tokens": 4096, "accum_steps": 4, "updates": 3000}
    if any(cv.get(key) != value for key, value in expected.items()):
        raise ValueError("CV design does not match this frozen preparation implementation")
    out = root / "data" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    specification = out / "grouping_spec.json"
    if specification.exists():
        if read_json(specification)["specification"] != GROUP_SPEC:
            raise ValueError("saved grouping specification differs; refuse post-hoc replacement")
    else:
        _immutable_json(specification, {"created_at_utc": timestamp(), "specification": GROUP_SPEC,
            "context": "new research designed after historical test results; saved before grouping calculations and formal runs"})
    request = {**_source_inputs(source), "design_sha256": sha256(root / "design.json"), "grouping_spec_sha256": sha256(specification),
               "implementation_sha256": sha256(Path(__file__)), "builder_url": BUILDER_URL,
               "archive_url": ARCHIVE_URL, "builder_revision": BUILDER_REVISION}
    _immutable_json(out / "grouping_request.json", request)
    marker = out / "grouping_complete.json"
    if marker.exists():
        result = read_json(marker)
        if result["request"] != request:
            raise ValueError("grouping request differs from completed evidence")
        _verify_files(out, result)
        return result
    origin = out / "origin"
    _download(BUILDER_URL, origin / "iwslt2017.py")
    _download(ARCHIVE_URL, origin / "en-de.zip")
    with zipfile.ZipFile(origin / "en-de.zip") as archive:
        for lang in ("en", "de"):
            name = f"en-de/train.tags.en-de.{lang}"
            _immutable_bytes(origin / f"train.tags.en-de.{lang}", archive.read(name))
    with (origin / "train.tags.en-de.en").open(encoding="utf-8") as en, (origin / "train.tags.en-de.de").open(encoding="utf-8") as de:
        tagged = list(tagged_rows(en, de))
    parquet = pq.read_table(source / "raw" / "iwslt2017-train.parquet", columns=["translation"]).column("translation").to_pylist()
    if verify_raw_alignment(tagged, parquet) != EXPECTED["train"]:
        raise ValueError("wrong official raw training count")
    rows = read_pairs(source / "train.jsonl")
    mapped = []
    for row in rows:
        original = tagged[row["id"]]
        if any(row[lang] != original[lang].strip() for lang in ("en", "de")):
            raise ValueError(f"cleaned training pair does not match recovered source ID {row['id']}")
        mapped.append({**row, "talk_id": original["talk_id"]})
    _immutable_bytes(out / "raw_talk_map.jsonl", _jsonl_bytes(
        {"id": r["id"], "talk_id": r["talk_id"], "tag_file_line": r["tag_file_line"],
         "source_sha256": hashlib.sha256(r["en"].encode()).hexdigest(),
         "target_sha256": hashlib.sha256(r["de"].encode()).hexdigest()} for r in tagged))
    selected, assignments, audit, joint = choose_grouping(mapped)
    groups_by_id = {original_id: group[0] for group in selected for original_id in group}
    joints_by_id = {original_id: group[0] for group in joint for original_id in group}
    records = [{"id": r["id"], "talk_id": r["talk_id"], "normalized_source_sha256": source_digest(r["en"]),
                "joint_component": joints_by_id[r["id"]], "group_id": groups_by_id[r["id"]],
                "fold": assignments[r["id"]]} for r in mapped]
    _immutable_bytes(out / "groups.jsonl", _jsonl_bytes(records))
    _immutable_bytes(out / "components.jsonl", _jsonl_bytes(
        {"group_id": g[0], "fold": assignments[g[0]], "ids": g} for g in selected))
    _immutable_bytes(out / "joint_components.jsonl", _jsonl_bytes(
        {"group_id": g[0], "ids": g} for g in joint))
    source_folds, talk_folds = defaultdict(set), defaultdict(set)
    for r in records:
        source_folds[r["normalized_source_sha256"]].add(r["fold"])
        talk_folds[r["talk_id"]].add(r["fold"])
    folds = []
    for fold in range(3):
        train, holdout = partition_rows(rows, assignments, fold)
        fold_records = [r for r in records if r["fold"] == fold]
        folds.append({"fold": fold, "train_pairs_before_length_filter": len(train), "holdout_pairs": len(holdout),
            "components": len({r["group_id"] for r in fold_records}),
            "talks_present": len({r["talk_id"] for r in fold_records}),
            "mean_source_whitespace_words": sum(len(r["en"].split()) for r in holdout) / len(holdout),
            "mean_target_whitespace_words": sum(len(r["de"].split()) for r in holdout) / len(holdout)})
    audit.update({"created_at_utc": timestamp(), "cleaned_pairs": len(rows), "raw_pairs_verified": len(tagged),
        "raw_alignment": "every source and target string exactly matches pinned Parquet in original order",
        "talks": len(talk_folds), "folds": folds,
        "cross_fold_normalized_source_keys": sum(len(f) > 1 for f in source_folds.values()),
        "cross_fold_talk_ids": sum(len(f) > 1 for f in talk_folds.values()),
        "talk_isolation_guaranteed": audit["strategy"] == "talk_and_normalized_source",
        "original_source": {"builder_url": BUILDER_URL, "archive_url": ARCHIVE_URL,
            "builder_sha256": sha256(origin / "iwslt2017.py"), "archive_sha256": sha256(origin / "en-de.zip")}})
    if audit["cross_fold_normalized_source_keys"] != 0:
        raise ValueError("cross-fold normalized duplicates found")
    _immutable_json(out / "group_audit.json", audit)
    artifacts = [p for p in out.rglob("*") if p.is_file() and p.name != marker.name and ".part" not in p.name]
    result = {"schema_version": 1, "created_at_utc": timestamp(), "request": request,
              "artifact_sha256": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(artifacts)},
              "strategy": audit["strategy"], "fold_sizes": audit["selected_grouping"]["fold_sizes"]}
    _immutable_json(marker, result)
    return result


def fit_scope(train_rows, holdout_rows):
    train_ids, holdout_ids = [r["id"] for r in train_rows], [r["id"] for r in holdout_rows]
    if len(train_ids) != len(set(train_ids)) or len(holdout_ids) != len(set(holdout_ids)):
        raise ValueError("duplicate IDs in tokenizer fit/evaluation scope")
    if set(train_ids) & set(holdout_ids):
        raise ValueError("tokenizer fit IDs include fold holdout IDs")
    return {"train_ids": train_ids, "excluded_holdout_ids": holdout_ids,
            "fit_policy": "all and only fold training rows; English then German; before token-length filtering"}


def _prepare_dataset(root, target, train_rows, evaluations, seeds, provenance):
    import sentencepiece as spm
    if target.exists():
        marker = target / "preparation_complete.json"
        if not marker.exists():
            raise ValueError(f"incomplete published CV data directory: {target}")
        complete = read_json(marker)
        if complete["provenance"] != provenance:
            raise ValueError("CV dataset provenance changed")
        _verify_files(target, complete)
        return complete
    stage = root / "partial" / (target.name + "-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    all_rows = {"train": train_rows, **evaluations}
    for split, rows in all_rows.items():
        _immutable_bytes(stage / (split + ".jsonl"), _jsonl_bytes(rows))
    scope = fit_scope(train_rows, evaluations.get("holdout", []))
    if "holdout" not in evaluations:
        scope["fit_policy"] = "all and only full official cleaned training rows; English then German; before token-length filtering"
        scope["evaluation_ids_are_split_local"] = True
    _immutable_json(stage / "tokenizer_fit_ids.json", scope)
    corpus = stage / "spm_train.txt"
    longest = 0
    with corpus.open("x", encoding="utf-8", newline="\n") as handle:
        for row in train_rows:
            for lang in ("en", "de"):
                line = row[lang].replace("\n", " ")
                longest = max(longest, len(line.encode("utf-8")))
                handle.write(line + "\n")
    options = trainer_options("bpe", corpus.resolve(), (stage / "spm").resolve())
    if longest > options["max_sentence_length"]:
        raise ValueError("SentencePiece maximum would silently exclude fit text")
    print(json.dumps({"stage": "fit_fold_bpe", "target": str(target), "pairs": len(train_rows)}), flush=True)
    with (stage / "sentencepiece_training.log").open("wb") as log:
        result = subprocess.run([sys.executable, "-S", "-c",
            "import json,sys,sentencepiece as s;s.SentencePieceTrainer.train(**json.loads(sys.argv[1]))", json.dumps(options)],
            stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"fold tokenizer failed; evidence retained: {stage}")
    tok = spm.SentencePieceProcessor(model_file=str(stage / "spm.model"))
    if tok.vocab_size() != 8000 or [tok.pad_id(), tok.unk_id(), tok.bos_id(), tok.eos_id()] != [0, 1, 2, 3]:
        raise ValueError("fold tokenizer vocabulary/special IDs differ from frozen specification")
    _immutable_json(stage / "tokenizer.json", {"model_type": "bpe", "requested_vocab_size": 8000,
        "actual_vocab_size": tok.vocab_size(), "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "training_pairs": len(train_rows), "training_file_sha256": sha256(stage / "train.jsonl"),
        "model_sha256": sha256(stage / "spm.model"), "vocab_sha256": sha256(stage / "spm.vocab"),
        "fit_ids_sha256": sha256(stage / "tokenizer_fit_ids.json"),
        "sentencepiece_version": importlib.metadata.version("sentencepiece"), "trainer_arguments": options,
        "corpus_sha256": sha256(corpus), "corpus_lines": 2 * len(train_rows), "longest_utf8_bytes": longest,
        "path_note": "trainer paths identify original staging location; published files are in this directory"})
    lengths = []
    for row in train_rows:
        a, b = len(encode(tok, row["en"])), len(encode(tok, row["de"]))
        lengths.append({"id": row["id"], "bpe_src": a, "bpe_tgt": b, "unigram_src": a, "unigram_tgt": b})
    _immutable_bytes(stage / "lengths.jsonl", _jsonl_bytes(lengths))
    retained = common_rows(lengths, 128)
    keep = {r["id"] for r in retained}
    _immutable_json(stage / "common_ids.json", {"schema_version": 1, "ids": [r["id"] for r in retained],
        "excluded_ids": [r["id"] for r in lengths if r["id"] not in keep],
        "max_tokens": 128, "pool_size": 512, "lengths_file": "lengths.jsonl", "lengths_sha256": sha256(stage / "lengths.jsonl"),
        "source_train_sha256": sha256(stage / "train.jsonl"),
        "tokenizer_sha256": {alias: sha256(stage / "spm.model") for alias in ("bpe", "unigram")},
        "alias_note": "both legacy schedule fields refer to the same BPE tokenizer; no unigram model exists in this comparison",
        "filter": "train only: both source and target length including BOS/EOS <=128; full holdout retained"})
    for seed in seeds:
        _immutable_bytes(stage / f"batch_plan_seed{seed}.jsonl", _jsonl_bytes(schedule_records(retained, seed)))
    _immutable_json(stage / "manifest.json", {"schema_version": 1, "dataset": "IWSLT/iwslt2017", "provenance": provenance,
        "files": {split: {"jsonl_sha256": sha256(stage / (split + ".jsonl")), "rows_saved": len(rows)} for split, rows in all_rows.items()},
        "train_retained": len(retained), "train_excluded": len(train_rows) - len(retained),
        "holdout_filtered": False, "holdout_truncated": False, "tokenizer_fit_ids_sha256": sha256(stage / "tokenizer_fit_ids.json")})
    complete = {"schema_version": 1, "created_at_utc": timestamp(), "provenance": provenance,
        "artifact_sha256": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()},
        "training_rows": len(train_rows), "retained_rows": len(retained), "evaluation_counts": {k: len(v) for k, v in evaluations.items()}}
    _immutable_json(stage / "preparation_complete.json", complete)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    return complete


def prepare_folds(root, source="artifacts/data"):
    root, source = Path(root), Path(source)
    groups = prepare_groups(root, source)
    out = root / "data" / "cv"
    assignment = {r["id"]: r["fold"] for r in read_pairs(out / "groups.jsonl")}
    rows = read_pairs(source / "train.jsonl")
    results = []
    for fold in range(3):
        train, holdout = partition_rows(rows, assignment, fold)
        provenance = {"kind": "grouped_cv_fold", "fold": fold, "grouping_complete_sha256": sha256(out / "grouping_complete.json"),
            "source_train_sha256": groups["request"]["source_train_sha256"], "implementation_sha256": sha256(Path(__file__)),
            "scope": "official training set only; complete holdout is neither checkpoint selector nor tokenizer fit input"}
        results.append(_prepare_dataset(root, out / f"fold{fold}", train, {"holdout": holdout}, [42], provenance))
    return results


def prepare_final(root, source="artifacts/data", selection=None):
    root, source = Path(root), Path(source)
    selection = Path(selection) if selection else root / "cv" / "selection.json"
    decision = read_json(selection)
    if decision.get("selected_label_smoothing") not in (0, .1):
        raise ValueError("final preparation requires an actual completed CV selection")
    inputs = _source_inputs(source)
    original = read_json(source / "manifest.json")
    for split in ("validation", "test"):
        if sha256(source / (split + ".jsonl")) != original["files"][split]["jsonl_sha256"]:
            raise ValueError("official evaluation data changed")
    provenance = {"kind": "selected_cv_full_training_refit", "selection_path": str(selection.resolve()),
        "selection_sha256": sha256(selection), "selected_label_smoothing": decision["selected_label_smoothing"],
        "inputs": inputs, "implementation_sha256": sha256(Path(__file__)),
        "scope": "fresh BPE fit on full official cleaned train only; official evaluation text never used to fit"}
    return _prepare_dataset(root, root / "data" / "cv" / "final", read_pairs(source / "train.jsonl"),
        {split: read_pairs(source / (split + ".jsonl")) for split in ("validation", "test")}, [42, 43, 44], provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", default="artifacts/data")
    parser.add_argument("--stage", choices=("groups", "folds", "final"), default="groups")
    parser.add_argument("--selection")
    args = parser.parse_args()
    result = {"groups": prepare_groups, "folds": prepare_folds}.get(args.stage)
    result = result(args.root, args.source) if result else prepare_final(args.root, args.source, args.selection)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
