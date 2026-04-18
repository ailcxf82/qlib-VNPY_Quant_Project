"""Peek a few factor.py scripts to understand RD-Agent factor function signature."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
ws_root = _ROOT / "git_ignore_folder" / "RD-Agent_workspace"

ws_dirs = [d for d in ws_root.iterdir() if d.is_dir() and (d / "factor.py").exists()]
print(f"total with factor.py: {len(ws_dirs)}")
print(f"sample 3 heads:\n")

import_lines: list[str] = []
func_sigs: list[str] = []
for d in ws_dirs[:3]:
    code = (d / "factor.py").read_text(encoding="utf-8", errors="replace")
    print("#" * 80)
    print(f"# {d.name}/factor.py ({len(code)} bytes)")
    print("#" * 80)
    print(code[:1400])
    print("...\n")

# Collect imports + def signatures across all 195
from collections import Counter

sigs = Counter()
imports = Counter()
for d in ws_dirs:
    try:
        lines = (d / "factor.py").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        continue
    for ln in lines:
        s = ln.strip()
        if s.startswith("def calculate_"):
            head = s.split("(", 1)[0]
            sigs[head[: min(60, len(head))]] += 1
        if s.startswith("import ") or s.startswith("from "):
            imports[s[:80]] += 1

print("=" * 80)
print(f"unique def calculate_* signatures: {len(sigs)}")
print("top 10 imports across 195 factor.py files:")
for k, v in imports.most_common(12):
    print(f"  {v:4d}  {k}")
