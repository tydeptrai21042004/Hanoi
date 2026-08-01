#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import (  # noqa: E402
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    embed_proposal,
    extract_proposal,
)
from qr64_certified.direct_schur_rescue import extract_components  # noqa: E402
from three_method_utils import evaluate_method, read_config  # noqa: E402
from qr64_certified.common.io import load_host_rgb, load_watermark_binary  # noqa: E402
from qr64_certified.common.metrics import nc, psnr  # noqa: E402


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image).rotate(
            degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        ),
        dtype=np.uint8,
    )


def shear(image: np.ndarray, amount: float) -> np.ndarray:
    h, w = image.shape[:2]
    return np.asarray(
        Image.fromarray(image).transform(
            (w, h),
            Image.Transform.AFFINE,
            (1.0, -amount, 0.0, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        ),
        dtype=np.uint8,
    )


def main() -> None:
    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)

    dct_cfg = read_config(ROOT / "configs/dct_qr_after_abc.json", DCT_QR)
    dct_summary, *_ = evaluate_method(DCT_QR, dct_cfg, host, watermark)

    spatial_cfg = read_config(
        ROOT / "configs/spatial_cd_detqr_after_abc.json", SPATIAL_CD_DETQR
    )
    spatial_watermarked, spatial_key = embed_proposal(
        SPATIAL_CD_DETQR, host, watermark, config=spatial_cfg
    )
    spatial_results: dict[str, object] = {
        "psnr": float(psnr(host, spatial_watermarked)),
    }
    for name, image in {
        "clean": spatial_watermarked,
        "rotation_2": rotate(spatial_watermarked, 2.0),
        "shear_0_08": shear(spatial_watermarked, 0.08),
        "unwatermarked_host": host,
    }.items():
        recovered, metadata = extract_proposal(image, spatial_key, return_metadata=True)
        spatial_results[name] = {
            "nc": float(nc(watermark, recovered)),
            "pilot_detected": bool(metadata.get("pilot_detected", False)),
            "sync": metadata.get("sync", {}),
        }

    schur_cfg = read_config(
        ROOT / "configs/dct_schur_rescue_after_abc.json", DCT_SCHUR_RESCUE
    )
    schur_watermarked, schur_key = embed_proposal(
        DCT_SCHUR_RESCUE, host, watermark, config=schur_cfg
    )
    components = extract_components(schur_watermarked, schur_key)
    schur_clean_accuracy = float(
        np.mean((np.asarray(components["schur_map"]) > 0) == (watermark > 0))
    )
    schur_recovered = extract_proposal(schur_watermarked, schur_key)

    result = {
        "dct_qr": dct_summary,
        "spatial_cd_detqr": spatial_results,
        "dct_schur_rescue": {
            "psnr": float(psnr(host, schur_watermarked)),
            "fused_clean_nc": float(nc(watermark, schur_recovered)),
            "independent_schur_clean_accuracy": schur_clean_accuracy,
            "acceptance_target": 0.99,
            "accepted": bool(schur_clean_accuracy >= 0.99),
        },
    }
    output = ROOT / "results/scientific_validation/integrated_component_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
