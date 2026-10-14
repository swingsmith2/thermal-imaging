from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thermal_demo.pipeline import (
    FrameMeta,
    apply_nuc,
    calibrate_nuc,
    ffc_offset,
    process_frame,
    replace_bad_pixels,
    roi_stats,
)
from thermal_demo.radiance import object_temperature, radiance, temperature_c
from thermal_demo.sensor import blackbody_frame, make_sensor, render_counts


def test_radiance_roundtrip() -> None:
    temps = np.array([0.0, 25.0, 70.0, 110.0])
    assert np.allclose(temperature_c(radiance(temps)), temps)


def test_nuc_removes_pixel_pattern() -> None:
    rng = np.random.default_rng(1)
    sensor = make_sensor(24, 32, rng, bad_count=0)
    cold = blackbody_frame(sensor, 20.0)
    hot = blackbody_frame(sensor, 50.0)
    tables = calibrate_nuc(cold, hot, 20.0, 50.0)
    raw = blackbody_frame(sensor, 40.0)
    measured = apply_nuc(raw, tables)
    # 16 bit 量化会留下约百分之几度的残差。
    assert np.nanstd(measured) < 0.002
    assert np.allclose(temperature_c(measured), 40.0, atol=0.15)


def test_ffc_removes_drift() -> None:
    rng = np.random.default_rng(2)
    sensor = make_sensor(20, 20, rng, bad_count=0)
    tables = calibrate_nuc(blackbody_frame(sensor, 20.0), blackbody_frame(sensor, 50.0), 20.0, 50.0)
    drift = 4000.0
    raw = render_counts(sensor, np.full((20, 20), radiance(36.0)), drift_counts=drift)
    before = apply_nuc(raw, tables)
    shutter = render_counts(sensor, np.full((20, 20), radiance(35.0)), drift_counts=drift)
    updated = tables.with_offset(ffc_offset(shutter, tables, 35.0))
    after = apply_nuc(raw, updated)
    assert abs(float(np.mean(temperature_c(before))) - 36.0) > 8.0
    assert np.allclose(temperature_c(after), 36.0, atol=0.2)


def test_wrong_emissivity_cools_metal() -> None:
    reflected = 25.0
    true_t = np.array([[70.0]])
    metal_eps = np.array([[0.25]])
    from thermal_demo.radiance import scene_radiance

    measured = scene_radiance(true_t, metal_eps, reflected)
    wrong = object_temperature(measured, np.array([[0.95]]), reflected)
    right = object_temperature(measured, metal_eps, reflected)
    assert float(wrong[0, 0]) < 45.0
    assert abs(float(right[0, 0]) - 70.0) < 1e-6


def test_bad_pixel_and_ffc_hold() -> None:
    raw = np.array([[100, 110, 120], [130, 65535, 140], [150, 160, 170]], dtype=np.uint16)
    bad = np.zeros((3, 3), dtype=bool)
    bad[1, 1] = True
    cleaned = replace_bad_pixels(raw, bad)
    assert cleaned[1, 1] == np.mean([100, 110, 120, 130, 140, 150, 160, 170])

    rng = np.random.default_rng(3)
    sensor = make_sensor(16, 16, rng, bad_count=0)
    tables = calibrate_nuc(blackbody_frame(sensor, 20.0), blackbody_frame(sensor, 50.0), 20.0, 50.0)
    live = render_counts(sensor, np.full(sensor.shape, radiance(30.0)))
    previous = process_frame(live, sensor.bad_pixels, tables, np.ones(sensor.shape), 25.0, FrameMeta(1, 1, "live", 0))
    held = process_frame(
        live,
        sensor.bad_pixels,
        tables,
        np.ones(sensor.shape),
        25.0,
        FrameMeta(2, 1, "ffc", 0),
        held_temperature=previous.temperature_c,
    )
    assert held.calibrating
    assert np.allclose(held.temperature_c, previous.temperature_c)
    stats = roi_stats(previous.temperature_c, (0, 0, 16, 16), "full", 90.0)
    assert stats.alarm is False


if __name__ == "__main__":
    test_radiance_roundtrip()
    test_nuc_removes_pixel_pattern()
    test_ffc_removes_drift()
    test_wrong_emissivity_cools_metal()
    test_bad_pixel_and_ffc_hold()
    print("5 passed")
