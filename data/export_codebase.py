"""
export_codebase.py

Walks the CellFlux repo and dumps every text-based source file
(Python, YAML, shell scripts, markdown) into a single .txt file.

Usage:
    python data/export_codebase.py
    # Output: codebase_export.txt (in the repo root)
"""

import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent   # one level up from data/
OUTPUT_FILE = REPO_ROOT / "codebase_export.txt"

INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".cfg", ".toml"}

EXCLUDE_DIRS = {".git", "__pycache__", ".idea", ".vscode", "node_modules"}
EXCLUDE_FILES = {"codebase_export.txt", "emb_fp.csv", "bbbc021_df_all.csv"}
# ──────────────────────────────────────────────────────────────────


def should_include(path: Path) -> bool:
    # Skip excluded directories
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    # Skip excluded filenames
    if path.name in EXCLUDE_FILES:
        return False
    # Only include whitelisted extensions
    return path.suffix.lower() in INCLUDE_EXTENSIONS


def main():
    lines = []
    total_files = 0

    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if not should_include(path):
            continue

        rel = path.relative_to(REPO_ROOT)
        separator = "=" * 72
        lines.append(f"\n{separator}")
        lines.append(f"FILE: {rel}")
        lines.append(f"{separator}\n")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines.append(content)
        except Exception as e:
            lines.append(f"[ERROR reading file: {e}]")

        total_files += 1

    output = "\n".join(lines)
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print(f"Exported {total_files} files → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
