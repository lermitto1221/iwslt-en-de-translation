import argparse
import json
from pathlib import Path

from .summary import (artifact_path, read_object, require, validate_comparability,
                      validate_result, validate_training_log)


def select_examples(groups, modulus):
    """Select the fixed IDs only after all four complete groups are aligned."""
    require(type(modulus) is int and modulus > 0, "example modulus must be a positive integer")
    require(set(groups) == {"A1", "A2", "B1", "B2"}, "all four example groups A1/A2/B1/B2 are required")
    by_group = {}
    for name, rows in groups.items():
        require(bool(rows), f"{name}: examples are empty")
        identities = [row.get("id") for row in rows]
        require(all(type(identity) is int and identity >= 0 for identity in identities),
                f"{name}: invalid example id")
        require(len(identities) == len(set(identities)), f"{name}: duplicate example ids")
        by_group[name] = {row["id"]: row for row in rows}
    expected_ids = list(by_group["A1"])
    for name, rows in by_group.items():
        require(list(rows) == expected_ids, f"{name}: example IDs/order differ from A1; refusing to discard samples")
        for identity, row in rows.items():
            anchor = by_group["A1"][identity]
            require(all(isinstance(row.get(key), str) for key in ("source", "reference", "prediction")),
                    f"{name}: example {identity} has missing/non-text fields")
            require(row["source"] == anchor.get("source") and row["reference"] == anchor.get("reference"),
                    f"{name}: source/reference mismatch for example {identity}")
    return [{"id": identity, "source": by_group["A1"][identity]["source"],
             "reference": by_group["A1"][identity]["reference"],
             "predictions": {name: rows[identity]["prediction"] for name, rows in by_group.items()}}
            for identity in sorted(expected_ids) if identity % modulus == 0]


def export_examples(root, seed, modulus, out):
    require(type(modulus) is int and modulus > 0, "example modulus must be a positive integer")
    root = Path(root)
    groups, metrics, orders = {}, {}, {}
    for condition in ("A", "B"):
        run = root / f"{condition}{seed}"
        for number, method in ((1, "greedy"), (2, "beam")):
            name = f"{condition}{number}"
            path = artifact_path(run / method / "metrics.json", name)
            item = read_object(path, name)
            groups[name] = validate_result(item, run, condition, seed, method)
            metrics[f"{name}_seed{seed}"] = item
            if condition not in orders:
                orders[condition] = validate_training_log(run, item["config"], f"{condition}{seed}")
    validate_comparability(metrics)
    require(orders["A"] == orders["B"], f"seed {seed}: training example order or token/sentence counts differ between A/B")
    records = select_examples(groups, modulus)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="artifacts/runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modulus", type=int, default=400)
    p.add_argument("--out", default="artifacts/fixed_examples.jsonl")
    args = p.parse_args()
    records = export_examples(args.runs, args.seed, args.modulus, args.out)
    print(f"saved {len(records)} fixed-rule examples to {args.out}")


if __name__ == "__main__":
    main()
