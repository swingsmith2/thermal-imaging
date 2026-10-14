"""铁红伪彩。固定温度窗，方便对比校正前后。"""

from __future__ import annotations

import numpy as np

STOPS = (
    (0.00, (0, 0, 32)),
    (0.20, (90, 0, 140)),
    (0.45, (210, 30, 20)),
    (0.70, (250, 170, 20)),
    (0.88, (255, 240, 160)),
    (1.00, (255, 255, 255)),
)


def _lut() -> np.ndarray:
    table = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        t = index / 255.0
        for (left, color_l), (right, color_r) in zip(STOPS, STOPS[1:]):
            if left <= t <= right:
                mix = 0.0 if right == left else (t - left) / (right - left)
                rgb = [round(a + (b - a) * mix) for a, b in zip(color_l, color_r)]
                table[index] = rgb
                break
    return table


LUT = _lut()


def colorize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")
    norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    index = np.rint(norm * 255).astype(np.uint8)
    return LUT[index]


def gray_agc(raw: np.ndarray) -> np.ndarray:
    """原始计数用百分位拉伸，只为了看见固定图案噪声。"""
    low, high = np.percentile(raw, (1, 99))
    if high <= low:
        high = low + 1
    norm = np.clip((raw.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    return np.repeat(np.rint(norm * 255).astype(np.uint8)[..., None], 3, axis=2)
