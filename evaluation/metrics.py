from __future__ import annotations

import io
import math
from collections import Counter

import numpy as np

from src.qwen_font_pipeline.svg_utils import command_sequence, path_to_svg, validate_svg_path


def valid_path(path_d: str) -> bool:
    try:
        validate_svg_path(path_d)
        return True
    except ValueError:
        return False


def command_accuracy(pred: str, target: str) -> float:
    a, b = command_sequence(pred), command_sequence(target)
    if not b:
        return float(not a)
    matches = sum(x == y for x, y in zip(a, b))
    return matches / max(len(a), len(b))


def rasterize_path(path_d: str, view_box: str, size: int = 256) -> np.ndarray:
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(bytestring=path_to_svg(path_d, view_box).encode(), output_width=size, output_height=size)
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=np.float32) / 255.0


def foreground(image: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    return np.argwhere(image < threshold)


def pixel_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pixel_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.95) -> float:
    a, b = pred < threshold, target < threshold
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def chamfer_distance(pred: np.ndarray, target: np.ndarray) -> float:
    a, b = foreground(pred), foreground(target)
    if not len(a) or not len(b):
        return float("inf")
    def directed(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean([np.min(np.sum((y - point) ** 2, axis=1)) for point in x]))
    return math.sqrt(directed(a, b)) + math.sqrt(directed(b, a))


def bbox_errors(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    a, b = foreground(pred), foreground(target)
    if not len(a) or not len(b):
        return float("inf"), float("inf")
    amin, amax, bmin, bmax = a.min(0), a.max(0), b.min(0), b.max(0)
    center = float(np.linalg.norm((amin + amax) / 2 - (bmin + bmax) / 2))
    scale = float(np.mean(np.abs((amax - amin) - (bmax - bmin))))
    return center, scale
