"""合成一幅可解释的测温场景：背景、电机、线缆、热点，以及一块低发射率金属。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scene:
    temperature_c: np.ndarray
    emissivity: np.ndarray
    reflected_c: float
    boxes: dict[str, tuple[int, int, int, int]]


def _fill_rect(image: np.ndarray, box: tuple[int, int, int, int], value: float) -> None:
    x, y, w, h = box
    image[y : y + h, x : x + w] = value


def build_scene(height: int = 120, width: int = 160) -> Scene:
    temperature = np.full((height, width), 28.0, dtype=np.float64)
    emissivity = np.full((height, width), 0.95, dtype=np.float64)
    boxes = {
        "motor": (18, 30, 36, 28),
        "cable": (70, 18, 70, 14),
        "hotspot": (108, 62, 16, 16),
        "metal": (28, 78, 40, 26),
    }
    _fill_rect(temperature, boxes["motor"], 62.0)
    _fill_rect(temperature, boxes["cable"], 44.0)
    _fill_rect(temperature, boxes["hotspot"], 110.0)
    _fill_rect(temperature, boxes["metal"], 70.0)
    _fill_rect(emissivity, boxes["metal"], 0.25)
    return Scene(temperature, emissivity, reflected_c=25.0, boxes=boxes)
