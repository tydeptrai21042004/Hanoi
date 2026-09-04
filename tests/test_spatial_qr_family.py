from __future__ import annotations

from pathlib import Path

import numpy as np

from qr64_certified import embed_proposal, extract_proposal
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("spatial_qr", "spatial_qr_direct_r", "spatial_qr_r11_qim")


def _inputs():
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png", size=64)
    return host, watermark


def test_spatial_qr_family_clean_roundtrip_and_no_dct_metadata() -> None:
    host, watermark = _inputs()
    for method in METHODS:
        watermarked, key, metadata = embed_proposal(
            method, host, watermark, return_metadata=True
        )
        recovered, extract_metadata = extract_proposal(
            watermarked, key, return_metadata=True
        )
        assert nc(watermark, recovered) == 1.0
        assert metadata["dct_used"] is False
        assert metadata["transform_used"] is False
        assert extract_metadata["dct_used"] is False
        assert extract_metadata["transform_used"] is False
        assert psnr(host, watermarked) > 45.0


def test_spatial_qr_sources_do_not_import_fft_transforms() -> None:
    proposal_dir = ROOT / "src" / "qr64_certified" / "proposals"
    for name in ("spatial_qr.py", "spatial_qr_direct_r.py", "spatial_qr_r11_qim.py", "spatial_qr_common.py"):
        text = (proposal_dir / name).read_text(encoding="utf-8").lower()
        assert "scipy.fft" not in text
        assert "dctn(" not in text
        assert "idctn(" not in text
