from pathlib import Path

import numpy as np

from qr64_certified import SPATIAL_CD_DETQR, embed_proposal, extract_proposal
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr

ROOT = Path(__file__).resolve().parents[1]


def test_cd_detqr_psnr54_blind_actual_embedding_and_clean_exact():
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    wm = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    watermarked, key, embed_metadata = embed_proposal(
        SPATIAL_CD_DETQR, host, wm, return_metadata=True
    )
    recovered, metadata = extract_proposal(watermarked, key, return_metadata=True)
    original_guess, original_metadata = extract_proposal(
        host, key, return_metadata=True
    )

    assert 53.5 <= psnr(host, watermarked) <= 55.0
    assert nc(wm, recovered) == 1.0
    assert nc(wm, original_guess) < 0.60
    assert metadata["fully_blind"]
    assert not metadata["original_host_used"]
    assert not metadata["original_watermark_used"]
    assert metadata["pilot_detected"]
    assert not original_metadata["pilot_detected"]
    assert embed_metadata["domain"] == "single_spatial_domain"
    assert embed_metadata["det_nonzero"]
    assert metadata["det_nonzero"]
    assert metadata["min_abs_det"] >= 0.49
    assert not np.array_equal(host, watermarked)


def test_cd_detqr_activates_singular_blocks_without_leaving_spatial_qr_domain():
    host = load_host_rgb(ROOT / "data/host/athens.bmp")
    wm = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    watermarked, key, metadata = embed_proposal(
        SPATIAL_CD_DETQR, host, wm, return_metadata=True
    )
    recovered = extract_proposal(watermarked, key)

    assert key.report["activated_singular_payload_blocks"] > 0
    assert metadata["det_nonzero"]
    assert nc(wm, recovered) == 1.0
