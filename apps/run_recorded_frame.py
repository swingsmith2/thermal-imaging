"""用外部原始帧、坏点表和标定文件测温，并和参考温度比较。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from PIL import Image

from thermal_demo.palette import colorize
from thermal_demo.pipeline import replace_bad_pixels
from thermal_demo.planck import raw_to_temperature
from thermal_demo.recorded import read_bad_pixels, read_calibration, read_raw_u16

DATA = ROOT / "data"


def main() -> None:
    calibration = read_calibration(DATA / "calibration.json")
    width = int(calibration["width"])
    height = int(calibration["height"])
    raw = read_raw_u16(DATA / calibration["raw_file"], width, height)
    bad_pixels = read_bad_pixels(DATA / calibration["bad_pixel_file"], height, width)
    cleaned = replace_bad_pixels(raw, bad_pixels)
    temperature = raw_to_temperature(cleaned, calibration)

    reference = calibration["reference"]
    x, y = int(reference["x"]), int(reference["y"])
    measured = float(temperature[y, x])
    expected = float(reference["temperature_c"])
    error = measured - expected

    wrong = dict(calibration)
    wrong["emissivity"] = 0.50
    wrong_temperature = float(raw_to_temperature(cleaned, wrong)[y, x])

    metrics = {
        "来源": calibration["source_note"],
        "相机": calibration["camera"],
        "原始帧": f"{width}×{height} uint16",
        "像素 (0,0) 原始值": int(raw[y, x]),
        "坏点数": int(np.count_nonzero(bad_pixels)),
        "NUC/FFC": "已在机芯内完成，本帧不再重做两点校正",
        "参考温度": f"{expected:.4f} °C",
        "解算温度": f"{measured:.4f} °C",
        "误差": f"{error:+.5f} °C",
        "发射率按 0.50 的读数": f"{wrong_temperature:.2f} °C",
        "发射率设错后的误差": f"{wrong_temperature - expected:+.2f} °C",
        "全图最低": f"{float(temperature.min()):.2f} °C",
        "全图最高": f"{float(temperature.max()):.2f} °C",
        "全图平均": f"{float(temperature.mean()):.2f} °C",
    }
    span = float(np.percentile(temperature, 99) - np.percentile(temperature, 1))
    vmin = float(np.percentile(temperature, 1))
    vmax = vmin + max(span, 1.0)
    rgb = colorize(temperature, vmin, vmax)
    out_dir = ROOT / "output" / "recorded"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_dir / "temperature.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"报告: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
