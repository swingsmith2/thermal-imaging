"""把 FLIR 样张拆成原始帧、坏点表和标定文件。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermal_demo.flir_frame import read_flir_jpeg

DATA = ROOT / "data"


def main() -> None:
    raw, calibration = read_flir_jpeg(DATA / "IR_2412.jpg")
    height, width = raw.shape
    calibration["width"] = width
    calibration["height"] = height
    calibration["raw_file"] = "frame_u16.raw"
    calibration["bad_pixel_file"] = "bad_pixels.csv"
    (DATA / "frame_u16.raw").write_bytes(raw.astype("<u2").tobytes())
    # 这张样张没有 0 或 65535 的饱和坏点，表头保留，行数为空。
    (DATA / "bad_pixels.csv").write_text("x,y\n", encoding="utf-8")
    (DATA / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"raw {width}x{height} pixel0={int(raw[0, 0])} range={int(raw.min())}..{int(raw.max())}")
    print(f"wrote {DATA / 'frame_u16.raw'}")


if __name__ == "__main__":
    main()
