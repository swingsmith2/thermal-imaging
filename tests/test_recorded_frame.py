from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thermal_demo.pipeline import replace_bad_pixels
from thermal_demo.planck import raw_to_temperature
from thermal_demo.recorded import read_bad_pixels, read_calibration, read_raw_u16

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_recorded_frame_matches_reference() -> None:
    calibration = read_calibration(DATA / "calibration.json")
    raw = read_raw_u16(DATA / "frame_u16.raw", calibration["width"], calibration["height"])
    bad = read_bad_pixels(DATA / "bad_pixels.csv", raw.shape[0], raw.shape[1])
    assert int(np.count_nonzero(bad)) == 0
    assert int(raw[0, 0]) == 18090
    temperature = raw_to_temperature(replace_bad_pixels(raw, bad), calibration)
    reference = calibration["reference"]
    measured = float(temperature[reference["y"], reference["x"]])
    assert abs(measured - reference["temperature_c"]) < 1e-4


def test_bad_pixel_table_is_applied_before_temperature() -> None:
    calibration = read_calibration(DATA / "calibration.json")
    raw = np.array([[18090, 18100, 18110], [18200, 50, 18220], [18300, 18310, 18320]], dtype=np.uint16)
    mask = np.zeros(raw.shape, dtype=bool)
    mask[1, 1] = True
    cleaned = replace_bad_pixels(raw, mask)
    assert cleaned[1, 1] != 50
    temperature = raw_to_temperature(cleaned, calibration)
    assert temperature.shape == raw.shape
    assert np.isfinite(temperature).all()


if __name__ == "__main__":
    test_recorded_frame_matches_reference()
    test_bad_pixel_table_is_applied_before_temperature()
    print("2 passed")
