"""从 FLIR 辐射 JPEG 里取出 16 bit 原始帧和标定常数。"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

KELVIN = 273.15


def _flir_payload(blob: bytes) -> bytes:
    parts: list[bytes] = []
    index = 0
    while index < len(blob) - 4:
        if blob[index] == 0xFF and blob[index + 1] == 0xE1:
            segment_length = int.from_bytes(blob[index + 2 : index + 4], "big")
            body = blob[index + 4 : index + 2 + segment_length]
            if body.startswith(b"FLIR\x00"):
                parts.append(body[8:])
            index += 2 + segment_length
            continue
        if blob[index] == 0xFF and blob[index + 1] == 0xDA:
            break
        index += 1
    if not parts:
        raise ValueError("JPEG has no FLIR raw segment")
    return b"".join(parts)


def _fff_entries(payload: bytes) -> list[tuple[int, int, int, int]]:
    entries = []
    cursor = 0x40
    while cursor + 32 <= len(payload):
        record = payload[cursor : cursor + 32]
        if record[:2] == b"\x00\x00":
            break
        kind, subtype, _version, _index, offset, length, _a, _b, _checksum = struct.unpack(
            ">HHIIIIIII", record
        )
        entries.append((kind, subtype, offset, length))
        cursor += 32
    return entries


def _f32(block: bytes, offset: int) -> float:
    return struct.unpack_from("<f", block, offset)[0]


def read_flir_jpeg(path: Path) -> tuple[np.ndarray, dict]:
    payload = _flir_payload(path.read_bytes())
    entries = _fff_entries(payload)
    camera = next(item for item in entries if item[0] == 0x20)
    raw_entry = next(item for item in entries if item[0] == 0x01)
    info = payload[camera[2] : camera[2] + camera[3]]
    raw_record = payload[raw_entry[2] : raw_entry[2] + raw_entry[3]]
    width, height = struct.unpack_from("<HH", raw_record, 2)
    pixels = np.frombuffer(raw_record[32:], dtype="<u2").reshape(height, width).copy()
    planck_o = struct.unpack_from("<i", info, 776)[0]
    calibration = {
        "camera": "FLIR SC660",
        "source_image": path.name,
        "source_note": "Thermimage inst/extdata/IR_2412.jpg, Glenn J. Tattersall",
        "width": int(width),
        "height": int(height),
        "endian": "little",
        "nuc_ffc": "already applied in the camera; raw values are radiometric ADC counts",
        "emissivity": _f32(info, 32),
        "distance_m": _f32(info, 36),
        "reflected_c": _f32(info, 40) - KELVIN,
        "atmospheric_c": _f32(info, 44) - KELVIN,
        "window_c": _f32(info, 48) - KELVIN,
        "window_transmission": _f32(info, 52),
        "humidity": _f32(info, 60),
        "planck": {
            "r1": _f32(info, 88),
            "b": _f32(info, 92),
            "f": _f32(info, 96),
            "o": float(planck_o),
            "r2": _f32(info, 780),
        },
        "atmosphere": {
            "alpha1": _f32(info, 112),
            "alpha2": _f32(info, 116),
            "beta1": _f32(info, 120),
            "beta2": _f32(info, 124),
            "x": _f32(info, 128),
        },
        "reference": {
            "x": 0,
            "y": 0,
            "raw": 18090,
            "temperature_c": 23.73440586671677,
            "source": "Published conversion of IR_2412.jpg pixel (0, 0)",
        },
    }
    return pixels, calibration
