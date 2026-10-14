"""把温度映射成相对辐射量，并反解回来。

用斯特藩-玻尔兹曼形式做教学近似：R(T) = (T_kelvin / 273.15) ** 4。
真实长波红外是光谱积分，工厂用黑体分段标定，不使用这个闭式。
"""

from __future__ import annotations

import numpy as np

ZERO_C = 273.15


def radiance(temp_c: np.ndarray | float) -> np.ndarray:
    kelvin = np.asarray(temp_c, dtype=np.float64) + ZERO_C
    if np.any(kelvin <= 0):
        raise ValueError("temperature must be above absolute zero")
    return (kelvin / ZERO_C) ** 4


def temperature_c(radiance_value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(radiance_value, dtype=np.float64)
    if np.any(value <= 0):
        raise ValueError("radiance must be positive")
    kelvin = np.power(value, 0.25) * ZERO_C
    return kelvin - ZERO_C


def scene_radiance(temp_c: np.ndarray, emissivity: np.ndarray, reflected_c: float) -> np.ndarray:
    """L = ε·R(Tobj) + (1-ε)·R(Tref)。"""
    eps = np.asarray(emissivity, dtype=np.float64)
    if np.any((eps <= 0.05) | (eps > 1.0)):
        raise ValueError("emissivity must be in (0.05, 1]")
    reflected = radiance(reflected_c)
    return eps * radiance(temp_c) + (1.0 - eps) * reflected


def object_temperature(measured_radiance: np.ndarray, emissivity: np.ndarray, reflected_c: float) -> np.ndarray:
    """由测得辐射量和发射率反解物体温度。发射率设错时，温度会偏。"""
    eps = np.asarray(emissivity, dtype=np.float64)
    reflected = radiance(reflected_c)
    object_radiance = (measured_radiance - (1.0 - eps) * reflected) / eps
    object_radiance = np.maximum(object_radiance, 1e-6)
    return temperature_c(object_radiance)
