"""Initialize fixed reproduction recipes; never replace an existing run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def prepare(project=PROJECT, output="artifacts/representation_2026_10_27"):
    project = Path(project).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = project / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output already contains files; preserve its design and resume the existing run.")
    protocol_path = project / "artifacts/decode_protocol_seed42.json"
    if not protocol_path.exists():
        exclusive(protocol_path, load(project / "recipes/decode_protocol_seed42.json"))
    protocol = load(protocol_path)
    expected = {"selection_split": "validation", "beam_size": 4,
                "length_penalty": 0.6, "max_generation_tokens": 256}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing protocol differs from the published fixed recipe; it was not overwritten.")
    design = load(project / "recipes/representation_design.json")
    design.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  status="new_reproduction_design_before_preparation",
                  source_plan="docs/REPRODUCING.md",
                  source_plan_sha256=sha(project / "docs/REPRODUCING.md"))
    design["decode"].update(protocol="artifacts/decode_protocol_seed42.json",
                            protocol_sha256=sha(protocol_path))
    exclusive(output / "design.json", design)
    return output / "design.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/representation_2026_10_27")
    args = parser.parse_args()
    print("Created new design:", prepare(output=args.root))
