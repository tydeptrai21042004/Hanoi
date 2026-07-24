#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.baselines import (
    ALL_BASELINE_IDS,
    SPECS,
    embed_baseline,
    extract_baseline,
)


def load_binary(path: Path, size: int) -> np.ndarray:
    arr = np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean round-trip smoke test for all integrated baselines")
    parser.add_argument("--host", type=Path, default=ROOT / "data/host/lenna.bmp")
    parser.add_argument("--watermark", type=Path, default=ROOT / "data/watermark/wm.png")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline_integration_smoke.json")
    args = parser.parse_args()

    host = np.asarray(Image.open(args.host).convert("RGB"), dtype=np.uint8)
    wm64 = load_binary(args.watermark, 64)
    wm16 = load_binary(args.watermark, 16)
    rows: list[dict] = []

    for method_id in ALL_BASELINE_IDS:
        watermark = wm64 if SPECS[method_id].common_4096_payload else wm16
        started = time.perf_counter()
        try:
            watermarked, key = embed_baseline(method_id, host, watermark, seed=2026, repeat=1)
            embedded = time.perf_counter()
            kwargs = {"original_host": host} if SPECS[method_id].requires_original_host else {}
            recovered = extract_baseline(watermarked, key, **kwargs)
            finished = time.perf_counter()
            accuracy = float(np.mean((recovered > 127) == (watermark > 127)))
            mse = float(np.mean((watermarked.astype(np.float64) - host.astype(np.float64)) ** 2))
            psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))
            row = {
                "method_id": method_id,
                "status": "ok",
                "watermark_shape": list(watermark.shape),
                "clean_accuracy": accuracy,
                "psnr_db": psnr,
                "embed_seconds": embedded - started,
                "extract_seconds": finished - embedded,
            }
        except Exception as exc:
            row = {
                "method_id": method_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
