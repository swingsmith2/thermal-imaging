"""FLIR 辐射测温：用标定文件里的普朗克常数把 16 bit 原始值解成温度。

公式是公开的原始值-温度关系：先按发射率、大气透过率和窗口透过率
扣掉非目标辐射，再反解普朗克函数。水汽压用与 FLIR 大气系数配套的经验式。
"""

from __future__ import annotations

import numpy as np


def signal_from_temperature(temp_c: float, planck: dict) -> float:
    kelvin = temp_c + 273.15
    return planck["r1"] / (planck["r2"] * (np.exp(planck["b"] / kelvin) - planck["f"])) - planck["o"]


def temperature_from_signal(signal: np.ndarray, planck: dict) -> np.ndarray:
    kelvin = planck["b"] / np.log(planck["r1"] / planck["r2"] / (signal + planck["o"]) + planck["f"])
    return kelvin - 273.15


def _vapor_pressure_mmhg(temp_c: float) -> float:
    return float(
        np.exp(1.5587 + 0.06939 * temp_c - 0.00027816 * temp_c**2 + 0.00000068455 * temp_c**3)
    )


def atmosphere_transmission(distance_m: float, temp_c: float, humidity: float, atmosphere: dict) -> float:
    if distance_m <= 0:
        return 1.0
    partial = humidity * _vapor_pressure_mmhg(temp_c)
    root = np.sqrt(partial)
    span = np.sqrt(distance_m)

    def factor(alpha: float, beta: float) -> float:
        return float(np.exp(-span * (alpha + beta * root)))

    mix = atmosphere["x"]
    return mix * factor(atmosphere["alpha1"], atmosphere["beta1"]) + (1.0 - mix) * factor(
        atmosphere["alpha2"], atmosphere["beta2"]
    )


def raw_to_temperature(raw: np.ndarray, calibration: dict) -> np.ndarray:
    """按标定文件把原始计数解成摄氏温度。坏点应先替换。"""
    planck = calibration["planck"]
    atmosphere = calibration["atmosphere"]
    emissivity = float(calibration["emissivity"])
    if not 0.05 < emissivity <= 1.0:
        raise ValueError("emissivity must be in (0.05, 1]")
    distance = float(calibration["distance_m"])
    humidity = float(calibration["humidity"])
    if not 0.0 <= humidity <= 1.0:
        raise ValueError("humidity must be a fraction from 0 to 1")
    reflected_c = float(calibration["reflected_c"])
    atmospheric_c = float(calibration["atmospheric_c"])
    window_c = float(calibration["window_c"])
    window_tau = float(calibration["window_transmission"])

    to_window = distance / 2.0
    from_window = distance - to_window
    tau_before = atmosphere_transmission(to_window, atmospheric_c, humidity, atmosphere)
    tau_after = atmosphere_transmission(from_window, atmospheric_c, humidity, atmosphere)
    window_emissivity = 1.0 - window_tau

    divisor = emissivity
    reflected = (1.0 - emissivity) / divisor * signal_from_temperature(reflected_c, planck)
    divisor *= tau_before
    air_before = (1.0 - tau_before) / divisor * signal_from_temperature(atmospheric_c, planck)
    divisor *= window_tau
    window = window_emissivity / divisor * signal_from_temperature(window_c, planck)
    divisor *= tau_after
    air_after = (1.0 - tau_after) / divisor * signal_from_temperature(atmospheric_c, planck)
    object_signal = raw.astype(np.float64) / divisor - (reflected + air_before + window + air_after)
    return temperature_from_signal(object_signal, planck)
