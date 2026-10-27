"""Check links, public result arithmetic and the curated file allowlist."""
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


root = Path(__file__).resolve().parents[1]
forbidden_dirs = {".venv", "venv", ".analysis-deps", "node_modules", "artifacts", "__pycache__"}
forbidden_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".parquet", ".model", ".vocab", ".jsonl", ".pem", ".key"}
issues = []
files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
for path in files:
    relative = path.relative_to(root)
    if set(relative.parts) & forbidden_dirs or path.suffix.lower() in forbidden_suffixes:
        issues.append(f"Unexpected generated/private file: {relative}")
    if path.stat().st_size >= 25 * 1024 * 1024:
        issues.append(f"File exceeds browser upload size: {relative}")
    if path.suffix.lower() in {".md", ".json", ".txt", ".yml", ".yaml", ".toml"}:
        content = path.read_text(encoding="utf-8")
        if re.search(r"(?i)(?:[a-z]:[\\/](?:users|develop)|/users/|/home/[^/]+/)", content):
            issues.append(f"Local absolute path in public document: {relative}")
    if path.suffix.lower() == ".md":
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            href = match.group(1).strip().strip("<>")
            if urlparse(href).scheme or href.startswith("#"):
                continue
            target = unquote(href.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                issues.append(f"Broken link: {relative} -> {target}")
summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
budget = summary["training_budget_comparison"]
assert budget["n"] == 3
expected = (23.13723375311204, 0.32055083606193596)
actual = budget["groups"]["long_beam"]["bleu"]
assert abs(actual["mean"] - expected[0]) < 1e-10
assert abs(actual["sample_std"] - expected[1]) < 1e-10
if issues:
    raise SystemExit("\n".join(issues))
print(f"PASS: {len(files)} public files; links, size, generated-file exclusions and result summary verified.")
