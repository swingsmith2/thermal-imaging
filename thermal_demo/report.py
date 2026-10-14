"""把各阶段图像和测温数字写成单文件 HTML。"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from thermal_demo.palette import colorize, gray_agc
from thermal_demo.pipeline import RoiStats


def _png(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb, mode="RGB").resize((rgb.shape[1] * 3, rgb.shape[0] * 3), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _draw_rois(rgb: np.ndarray, boxes: dict[str, tuple[int, int, int, int]], alarm_names: set[str]) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for name, (x, y, w, h) in boxes.items():
        color = (255, 64, 64) if name in alarm_names else (80, 220, 120)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color)
        draw.text((x + 2, y + 2), name, fill=color, font=font)
    return np.asarray(image)


def write_report(
    out_dir: Path,
    images: dict[str, np.ndarray],
    metrics: dict,
    rois: list[RoiStats],
    boxes: dict[str, tuple[int, int, int, int]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    alarm_names = {item.name for item in rois if item.alarm}
    display = _draw_rois(images["display"], boxes, alarm_names)
    isotherm = images["isotherm"]
    payload = {
        "原始计数（百分位拉伸）": gray_agc(images["raw"]),
        "只做 NUC，未做快门校正": images["before_ffc"],
        "快门校正后，发射率全按 0.95": images["wrong_eps"],
        "快门校正后，使用真实发射率": images["temperature"],
        "发布帧：伪彩 + ROI": display,
        "等温线 60–80°C": isotherm,
    }
    cards = []
    for title, rgb in payload.items():
        cards.append(
            f"<figure><img alt='{html.escape(title)}' src='data:image/png;base64,{_png(rgb)}'/>"
            f"<figcaption>{html.escape(title)}</figcaption></figure>"
        )
    rows = []
    for item in rois:
        flag = "告警" if item.alarm else "正常"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.name)}</td>"
            f"<td>{item.min_c:.1f}</td><td>{item.max_c:.1f}</td><td>{item.mean_c:.1f}</td>"
            f"<td>{flag}</td></tr>"
        )
    metric_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>热成像流水线演示</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
h1 {{ font-size: 22px; }}
p {{ max-width: 820px; line-height: 1.5; color: #ccc; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
figure {{ margin: 0; background: #1b1b1b; padding: 8px; }}
img {{ width: 100%; height: auto; image-rendering: pixelated; }}
figcaption {{ margin-top: 6px; font-size: 13px; }}
table {{ border-collapse: collapse; margin: 16px 0 28px; }}
td, th {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
</style>
</head>
<body>
<h1>热成像流水线演示</h1>
<p>合成的 160×120、16 bit 机芯。两点黑体做非均匀校正，快门帧更新偏移，再按发射率把辐射量解成温度。金属块真实 70°C、发射率 0.25；若全场按 0.95，读数会明显偏低。热点超过 90°C 告警。校正中的帧不发布新温度。</p>
<h2>测温</h2>
<table>
<tr><th>区域</th><th>最低 °C</th><th>最高 °C</th><th>平均 °C</th><th>状态</th></tr>
{''.join(rows)}
</table>
<h2>流水线数字</h2>
<table><tr><th>项</th><th>值</th></tr>{metric_rows}</table>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    report = out_dir / "report.html"
    report.write_text(document, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
