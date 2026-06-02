from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Iterable

from src.qwen_font_pipeline.svg_utils import extract_svg_path

CODE_RE = re.compile(r"(?:^|[_-])(?:U\+|u)?([0-9A-Fa-f]{2,6})(?:$|[_-])")


def infer_metadata(stem: str) -> tuple[str, int | None]:
    """Infer font and glyph code from common `<font>_<code>` pair filenames."""
    match = CODE_RE.search(stem)
    if not match:
        return stem, None
    raw = match.group(1)
    code = int(raw, 16) if any(c in "abcdefABCDEF" for c in raw) or raw.startswith("0") else int(raw)
    font = stem[: match.start()].rstrip("_-") or "unknown"
    return font, code


def build_pairs(outline_dir: Path, skeleton_dir: Path, strict: bool = False) -> list[dict]:
    rows = []
    for outline_file in sorted(outline_dir.glob("*.svg")):
        target_file = skeleton_dir / outline_file.name
        if not target_file.exists():
            if strict:
                raise FileNotFoundError(f"Missing skeleton pair for {outline_file.name}")
            continue
        try:
            outline, target = extract_svg_path(outline_file), extract_svg_path(target_file)
        except ValueError:
            if strict:
                raise
            continue
        font, code = infer_metadata(outline_file.stem)
        rows.append({
            "id": outline_file.stem, "font": font, "code": code,
            "input_svg": str(outline_file), "target_svg": str(target_file),
            "input_path": outline.path, "target_path": target.path,
            "view_box": outline.view_box, "width": outline.width, "height": outline.height,
        })
    return rows


def _partition(values: Iterable, ratio: float, rng: random.Random) -> set:
    values = sorted(set(values), key=lambda item: str(item))
    rng.shuffle(values)
    count = min(len(values), max(1, round(len(values) * ratio))) if values and ratio > 0 else 0
    return set(values[:count])


def split_rows(rows: list[dict], mode: str, val_size: float, test_size: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if mode == "random":
        n_test, n_val = round(len(rows) * test_size), round(len(rows) * val_size)
        test, val, train = shuffled[:n_test], shuffled[n_test:n_test + n_val], shuffled[n_test + n_val:]
    else:
        fonts = _partition((r["font"] for r in rows), test_size, rng) if mode in {"font", "font_glyph"} else set()
        codes = _partition((r["code"] for r in rows), test_size, rng) if mode in {"glyph", "font_glyph"} else set()
        if mode == "font":
            is_test = lambda r: r["font"] in fonts
        elif mode == "glyph":
            is_test = lambda r: r["code"] in codes
        elif mode == "font_glyph":
            is_test = lambda r: r["font"] in fonts or r["code"] in codes
        else:
            raise ValueError(f"Unknown split mode: {mode}")
        test, remaining = [r for r in rows if is_test(r)], [r for r in rows if not is_test(r)]
        rng.shuffle(remaining)
        n_val = round(len(remaining) * val_size)
        val, train = remaining[:n_val], remaining[n_val:]
    for split, items in (("train", train), ("val", val), ("test", test)):
        for row in items:
            row["split"] = split
            row["generalization_split"] = "unseen" if split == "test" and mode != "random" else "seen"
            row["split_mode"] = mode
    return train, val, test


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired outline-to-skeleton JSONL datasets.")
    parser.add_argument("--outline_dir", required=True, type=Path)
    parser.add_argument("--skeleton_dir", required=True, type=Path)
    parser.add_argument("--out_dir", default=Path("dataset/processed"), type=Path)
    parser.add_argument("--split_mode", choices=["random", "font", "glyph", "font_glyph"], default="font_glyph")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = build_pairs(args.outline_dir, args.skeleton_dir, args.strict)
    train, val, test = split_rows(rows, args.split_mode, args.val_size, args.test_size, args.seed)
    for name, items in (("train", train), ("val", val), ("test", test)):
        write_jsonl(args.out_dir / f"{name}.jsonl", items)
    print(f"Saved {len(rows)} pairs: train={len(train)}, val={len(val)}, test={len(test)} ({args.split_mode})")


if __name__ == "__main__":
    main()
