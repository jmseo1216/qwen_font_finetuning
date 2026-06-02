from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.metrics import bbox_errors, chamfer_distance, command_accuracy, pixel_iou, pixel_l1, rasterize_path, valid_path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    keys = ["valid", "command_accuracy", "pixel_l1", "pixel_iou", "chamfer_distance", "bbox_center_error", "bbox_scale_error"]
    return {"count": len(rows), **{key: float(np.mean([r[key] for r in rows])) if rows else None for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated skeleton SVG paths.")
    parser.add_argument("--targets", required=True, type=Path, help="test JSONL produced by preprocessing")
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL rows containing id and predicted_path")
    parser.add_argument("--output", default=Path("evaluation/results.json"), type=Path)
    parser.add_argument("--raster_size", type=int, default=256)
    args = parser.parse_args()
    targets = {row["id"]: row for row in read_jsonl(args.targets)}
    results = []
    for prediction in read_jsonl(args.predictions):
        row, pred = targets[prediction["id"]], prediction["predicted_path"]
        result = {"id": row["id"], "generalization_split": row.get("generalization_split", "seen"), "valid": float(valid_path(pred))}
        result["command_accuracy"] = command_accuracy(pred, row["target_path"])
        if result["valid"]:
            a, b = rasterize_path(pred, row.get("view_box", "0 0 1024 1024"), args.raster_size), rasterize_path(row["target_path"], row.get("view_box", "0 0 1024 1024"), args.raster_size)
            result.update(pixel_l1=pixel_l1(a, b), pixel_iou=pixel_iou(a, b), chamfer_distance=chamfer_distance(a, b))
            result["bbox_center_error"], result["bbox_scale_error"] = bbox_errors(a, b)
        else:
            result.update(pixel_l1=1.0, pixel_iou=0.0, chamfer_distance=float("inf"), bbox_center_error=float("inf"), bbox_scale_error=float("inf"))
        results.append(result)
    groups = defaultdict(list)
    for row in results:
        groups[row["generalization_split"]].append(row)
    report = {"overall": summarize(results), "by_generalization_split": {name: summarize(items) for name, items in groups.items()}, "samples": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
