import argparse
from pathlib import Path

from .common import read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="config.json")
    p.add_argument("--out", default="experiments")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()
    base = read_json(args.base)
    for seed in args.seeds:
        for name, smoothing in (("A", 0.1), ("B", 0.0)):
            cfg = dict(base, seed=seed, label_smoothing=smoothing,
                       run_dir=str(Path("artifacts/runs") / f"{name}{seed}"))
            write_json(Path(args.out) / f"{name}{seed}.json", cfg)


if __name__ == "__main__":
    main()
