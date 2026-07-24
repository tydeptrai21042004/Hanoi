#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified import embed_proposal, extract_proposal, list_supported_methods
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr


def main() -> None:
    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png", size=64)
    rows = []
    for method in list_supported_methods():
        method_id = method["id"]
        watermarked, key = embed_proposal(method_id, host, watermark)
        recovered, metadata = extract_proposal(watermarked, key, return_metadata=True)
        row = {
            "method_id": method_id,
            "psnr": psnr(host, watermarked),
            "clean_nc": nc(watermark, recovered),
            "det_nonzero": metadata["det_nonzero"],
            "fully_blind": bool(getattr(key, "fully_blind", False)),
        }
        rows.append(row)
        if row["psnr"] < 45.0 or row["clean_nc"] != 1.0 or not row["det_nonzero"]:
            raise SystemExit(f"Smoke test failed: {row}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
