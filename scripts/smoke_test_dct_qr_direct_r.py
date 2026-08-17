#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified import DCT_QR_DIRECT_R, embed_proposal, extract_proposal
from qr64_certified.attacks import apply_attack, get_attack_suite
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import ber, nc, ncc, psnr, ssim


def main() -> None:
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png", size=64)
    watermarked, key, embed_meta = embed_proposal(
        DCT_QR_DIRECT_R, host, watermark, return_metadata=True
    )

    rows = []
    for attack in get_attack_suite("sanity"):
        attacked = apply_attack(watermarked, attack)
        recovered, metadata = extract_proposal(attacked, key, return_metadata=True)
        rows.append(
            {
                "attack": attack.attack_id,
                "nc": float(nc(watermark, recovered)),
                "ncc": float(ncc(watermark, recovered)),
                "ber": float(ber(watermark, recovered)),
                "det_nonzero": bool(metadata["det_nonzero"]),
            }
        )

    result = {
        "method_id": DCT_QR_DIRECT_R,
        "host": "lenna.bmp",
        "watermark": "wm.png",
        "configuration": key.config,
        "embedding_psnr_db": float(psnr(host, watermarked)),
        "embedding_ssim": float(ssim(host, watermarked)),
        "embed_metadata": embed_meta,
        "attacks": rows,
    }
    output = ROOT / "results" / "dct_qr_direct_r_small_attack.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
