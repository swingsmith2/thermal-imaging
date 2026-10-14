"""跑通一帧热成像：标定、漂移、快门校正、发射率、ROI 告警。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from thermal_demo.palette import colorize
from thermal_demo.pipeline import (
    FrameMeta,
    bandwidth_mb_s,
    calibrate_nuc,
    ffc_offset,
    process_frame,
    replace_bad_pixels,
    roi_stats,
)
from thermal_demo.report import write_report
from thermal_demo.scene import build_scene
from thermal_demo.sensor import blackbody_frame, capture_scene, make_sensor, shutter_frame

WIDTH = 160
HEIGHT = 120
FPS = 50
ALARM_C = 90.0
DRIFT = 4000.0
VMIN = 15.0
VMAX = 120.0


def main() -> None:
    rng = np.random.default_rng(7)
    scene = build_scene(HEIGHT, WIDTH)
    sensor = make_sensor(HEIGHT, WIDTH, rng)

    cold = replace_bad_pixels(blackbody_frame(sensor, 20.0), sensor.bad_pixels)
    hot = replace_bad_pixels(blackbody_frame(sensor, 50.0), sensor.bad_pixels)
    tables = calibrate_nuc(cold, hot, 20.0, 50.0)

    raw = capture_scene(
        sensor,
        scene.temperature_c,
        scene.emissivity,
        scene.reflected_c,
        drift_counts=DRIFT,
        noise_std=25.0,
        rng=rng,
    )
    before = process_frame(
        raw,
        sensor.bad_pixels,
        tables,
        scene.emissivity,
        scene.reflected_c,
        FrameMeta(1, 1, "live", DRIFT),
    )
    shutter = replace_bad_pixels(shutter_frame(sensor, 35.0, DRIFT), sensor.bad_pixels)
    updated = tables.with_offset(ffc_offset(shutter, tables, 35.0))
    held = process_frame(
        shutter,
        sensor.bad_pixels,
        updated,
        scene.emissivity,
        scene.reflected_c,
        FrameMeta(2, 1, "ffc", DRIFT),
        held_temperature=before.temperature_c,
    )
    after = process_frame(
        raw,
        sensor.bad_pixels,
        updated,
        scene.emissivity,
        scene.reflected_c,
        FrameMeta(3, 2, "live", DRIFT),
    )
    wrong_eps = np.full(scene.emissivity.shape, 0.95)
    mistaken = process_frame(
        raw,
        sensor.bad_pixels,
        updated,
        wrong_eps,
        scene.reflected_c,
        FrameMeta(3, 2, "live", DRIFT),
    )

    rois = [
        roi_stats(after.temperature_c, scene.boxes[name], name, ALARM_C)
        for name in ("hotspot", "motor", "metal", "cable")
    ]
    high_eps = scene.emissivity > 0.9
    mae_before = float(np.mean(np.abs(before.temperature_c - scene.temperature_c)[high_eps]))
    mae_after = float(np.mean(np.abs(after.temperature_c - scene.temperature_c)[high_eps]))
    metal = scene.boxes["metal"]
    mx, my, mw, mh = metal
    metal_true = float(np.mean(scene.temperature_c[my : my + mh, mx : mx + mw]))
    metal_wrong = float(np.mean(mistaken.temperature_c[my : my + mh, mx : mx + mw]))
    metal_right = float(np.mean(after.temperature_c[my : my + mh, mx : mx + mw]))

    metrics = {
        "分辨率": f"{WIDTH}×{HEIGHT}",
        "位深 / 帧率": f"16 bit / {FPS} fps",
        "原始带宽": f"{bandwidth_mb_s(WIDTH, HEIGHT, FPS):.2f} MiB/s",
        "640×512@50fps 对照": f"{bandwidth_mb_s(640, 512, 50):.1f} MiB/s",
        "漂移": f"{DRIFT:.0f} counts",
        "高发射率平均绝对误差，校正前": f"{mae_before:.2f} °C",
        "高发射率平均绝对误差，快门后": f"{mae_after:.2f} °C",
        "金属真实温度": f"{metal_true:.1f} °C",
        "金属按发射率 0.95 读数": f"{metal_wrong:.1f} °C",
        "金属按发射率 0.25 读数": f"{metal_right:.1f} °C",
        "快门帧是否发布新温度": "否" if held.calibrating else "是",
        "发布帧标定版本": after.meta.calibration_version,
        "告警阈值": f"{ALARM_C:.0f} °C",
    }
    images = {
        "raw": raw,
        "before_ffc": colorize(before.temperature_c, VMIN, VMAX),
        "wrong_eps": colorize(mistaken.temperature_c, VMIN, VMAX),
        "temperature": colorize(after.temperature_c, VMIN, VMAX),
        "display": colorize(after.temperature_c, VMIN, VMAX),
        "isotherm": colorize(after.temperature_c, VMIN, VMAX),
    }
    iso = (after.temperature_c >= 60.0) & (after.temperature_c <= 80.0)
    images["isotherm"][iso] = (255, 255, 255)

    report = write_report(ROOT / "output", images, metrics, rois, scene.boxes)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    for item in rois:
        state = "告警" if item.alarm else "正常"
        print(f"{item.name}: min {item.min_c:.1f}  max {item.max_c:.1f}  mean {item.mean_c:.1f}  {state}")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
