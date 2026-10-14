"""热成像处理链：坏点 → 两点 NUC → 快门 FFC → 发射率测温 → ROI。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_demo.radiance import object_temperature, radiance
from thermal_demo.sensor import COUNT_SCALE


@dataclass(frozen=True)
class NucTables:
    """raw = response·L + offset。response 已含计数缩放。"""

    response: np.ndarray
    offset: np.ndarray
    cold_c: float
    hot_c: float

    def with_offset(self, offset: np.ndarray) -> "NucTables":
        return NucTables(self.response, offset, self.cold_c, self.hot_c)


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    calibration_version: int
    state: str
    detector_drift_counts: float


@dataclass(frozen=True)
class RoiStats:
    name: str
    min_c: float
    max_c: float
    mean_c: float
    alarm: bool


def replace_bad_pixels(raw: np.ndarray, bad_pixels: np.ndarray) -> np.ndarray:
    """用有效邻域均值替换坏点。坏点不参与后续增益计算。"""
    out = raw.astype(np.float64)
    ys, xs = np.nonzero(bad_pixels)
    height, width = bad_pixels.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        samples: list[float] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and not bad_pixels[yy, xx]:
                    samples.append(float(raw[yy, xx]))
        if samples:
            out[y, x] = float(np.mean(samples))
    return out


def calibrate_nuc(raw_cold: np.ndarray, raw_hot: np.ndarray, cold_c: float, hot_c: float) -> NucTables:
    """两点标定：response = (raw_hot - raw_cold) / (L_hot - L_cold)。"""
    if cold_c >= hot_c:
        raise ValueError("cold blackbody must be cooler than hot blackbody")
    cold_l = float(radiance(cold_c))
    hot_l = float(radiance(hot_c))
    response = (raw_hot.astype(np.float64) - raw_cold.astype(np.float64)) / (hot_l - cold_l)
    response = np.where(np.abs(response) < 1.0, np.nan, response)
    offset = raw_cold.astype(np.float64) - response * cold_l
    return NucTables(response, offset, cold_c, hot_c)


def apply_nuc(raw: np.ndarray, tables: NucTables) -> np.ndarray:
    return (raw.astype(np.float64) - tables.offset) / tables.response


def ffc_offset(raw_shutter: np.ndarray, tables: NucTables, shutter_c: float) -> np.ndarray:
    """快门提供一块已知温度的均匀场景，重估偏移。增益表不动。"""
    expected = float(radiance(shutter_c))
    return raw_shutter.astype(np.float64) - tables.response * expected


@dataclass(frozen=True)
class ProcessedFrame:
    radiance_map: np.ndarray
    temperature_c: np.ndarray
    meta: FrameMeta
    calibrating: bool


def process_frame(
    raw: np.ndarray,
    bad_pixels: np.ndarray,
    tables: NucTables,
    emissivity: np.ndarray,
    reflected_c: float,
    meta: FrameMeta,
    held_temperature: np.ndarray | None = None,
) -> ProcessedFrame:
    """校正中的帧不对外发布新温度，沿用上一帧，并打上 calibrating。"""
    cleaned = replace_bad_pixels(raw, bad_pixels)
    if meta.state == "ffc":
        if held_temperature is None:
            raise ValueError("ffc requires the previous published temperature")
        return ProcessedFrame(np.full(raw.shape, np.nan), held_temperature, meta, True)
    measured = apply_nuc(cleaned, tables)
    temperature = object_temperature(measured, emissivity, reflected_c)
    return ProcessedFrame(measured, temperature, meta, False)


def roi_stats(temperature_c: np.ndarray, box: tuple[int, int, int, int], name: str, alarm_c: float) -> RoiStats:
    x, y, w, h = box
    view = temperature_c[y : y + h, x : x + w]
    maximum = float(np.max(view))
    return RoiStats(name, float(np.min(view)), maximum, float(np.mean(view)), maximum >= alarm_c)


def isotherm(temperature_c: np.ndarray, low_c: float, high_c: float) -> np.ndarray:
    return (temperature_c >= low_c) & (temperature_c <= high_c)


def bandwidth_mb_s(width: int, height: int, fps: float, bytes_per_pixel: int = 2) -> float:
    return width * height * bytes_per_pixel * fps / (1024 * 1024)


def counts_per_radiance() -> float:
    return COUNT_SCALE
