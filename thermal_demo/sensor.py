"""把场景辐射量变成 16 bit 原始计数值，并带上像素不一致、漂移和坏点。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_demo.radiance import radiance, scene_radiance

COUNT_SCALE = 8000.0


@dataclass(frozen=True)
class SensorModel:
    gain: np.ndarray
    offset: np.ndarray
    bad_pixels: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.gain.shape


def make_sensor(height: int, width: int, rng: np.random.Generator, bad_count: int = 8) -> SensorModel:
    gain = np.clip(rng.normal(1.0, 0.08, size=(height, width)), 0.7, 1.3)
    offset = rng.normal(2000.0, 350.0, size=(height, width))
    flat = rng.choice(height * width, size=bad_count, replace=False)
    bad = np.zeros((height, width), dtype=bool)
    bad.flat[flat] = True
    return SensorModel(gain.astype(np.float64), offset.astype(np.float64), bad)


def render_counts(
    sensor: SensorModel,
    radiance_map: np.ndarray,
    drift_counts: float = 0.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """raw = gain·scale·L + offset + drift + noise。坏点固定成饱和或死黑。"""
    counts = sensor.gain * COUNT_SCALE * radiance_map + sensor.offset + drift_counts
    if noise_std > 0:
        if rng is None:
            raise ValueError("rng is required when noise_std > 0")
        counts = counts + rng.normal(0.0, noise_std, size=counts.shape)
    image = np.clip(np.rint(counts), 0, 65535).astype(np.uint16)
    image[sensor.bad_pixels] = 0
    # 一半坏点打成饱和，避免校正时被当成正常暗像素。
    saturated = np.flatnonzero(sensor.bad_pixels.ravel())[::2]
    image.ravel()[saturated] = 65535
    return image


def blackbody_frame(sensor: SensorModel, temp_c: float) -> np.ndarray:
    """均匀黑体，发射率按 1，没有漂移。用于两点非均匀校正。"""
    level = np.full(sensor.shape, radiance(temp_c), dtype=np.float64)
    return render_counts(sensor, level, drift_counts=0.0, noise_std=0.0)


def shutter_frame(sensor: SensorModel, shutter_c: float, drift_counts: float) -> np.ndarray:
    """快门合上后的均匀场景，带上当前探测器漂移。"""
    level = np.full(sensor.shape, radiance(shutter_c), dtype=np.float64)
    return render_counts(sensor, level, drift_counts=drift_counts, noise_std=0.0)


def capture_scene(
    sensor: SensorModel,
    temp_c: np.ndarray,
    emissivity: np.ndarray,
    reflected_c: float,
    drift_counts: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    level = scene_radiance(temp_c, emissivity, reflected_c)
    return render_counts(sensor, level, drift_counts=drift_counts, noise_std=noise_std, rng=rng)
