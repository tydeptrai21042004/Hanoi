from __future__ import annotations

from pathlib import Path

from qr64_certified import DCT_QR_DIRECT_R, embed_proposal, extract_proposal
from qr64_certified.attacks import AttackConfig, apply_attack
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import ber, nc, psnr

ROOT = Path(__file__).resolve().parents[1]


def test_direct_r_is_transform_domain_and_exact_clean() -> None:
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png", size=64)
    watermarked, key, metadata = embed_proposal(
        DCT_QR_DIRECT_R, host, watermark, return_metadata=True
    )
    recovered, extract_meta = extract_proposal(watermarked, key, return_metadata=True)

    assert metadata["domain"] == "transform_domain_dct_then_qr_direct_r"
    assert metadata["qr_input"] == "regularized_4x4_low_frequency_DCT_matrix"
    assert metadata["r11_modified"] is False
    assert extract_meta["det_nonzero"] is True
    assert psnr(host, watermarked) > 45.0
    assert nc(watermark, recovered) == 1.0
    assert ber(watermark, recovered) == 0.0


def test_direct_r_survives_small_jpeg_attack() -> None:
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png", size=64)
    watermarked, key = embed_proposal(DCT_QR_DIRECT_R, host, watermark)
    attacked = apply_attack(
        watermarked,
        AttackConfig(
            "jpeg_q90_test", "jpeg", {"quality": 90}, "compression", "mild"
        ),
    )
    recovered = extract_proposal(attacked, key)
    assert nc(watermark, recovered) >= 0.99
