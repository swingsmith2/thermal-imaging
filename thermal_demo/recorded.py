"""读取外部 16 bit 原始帧、坏点表和标定文件。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_raw_u16(path: Path, width: int, height: int) -> np.ndarray:
    blob = path.read_bytes()
    expected = width * height * 2
    if len(blob) != expected:
        raise ValueError(f"{path} has {len(blob)} bytes, expected {expected}")
    return np.frombuffer(blob, dtype="<u2").reshape(height, width).copy()


def read_bad_pixels(path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or text.lower().startswith("x"):
            continue
        x_text, y_text = text.split(",")[:2]
        x, y = int(x_text), int(y_text)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"bad pixel {(x, y)} is outside {width}x{height}")
        mask[y, x] = True
    return mask


def read_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
