from __future__ import annotations

from pathlib import Path

import numpy as np

from qr64_certified import (
    DIRECT_SCHUR_RESCUE,
    DirectSchurRescueConfig,
    embed_proposal,
    extract_proposal,
    list_supported_methods,
)
from qr64_certified.direct_schur_rescue import _embed_schur_coefficients
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import psnr

ROOT = Path(__file__).resolve().parents[1]


def _assets():
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    return host, watermark


def test_registry_lists_three_nonreplacing_proposals():
    methods = {row["id"] for row in list_supported_methods()}
    assert methods == {
        "dct_qr",
        "dct_schur_rescue",
        "spatial_cd_detqr",
    }


def test_schur_float_update_preserves_determinant_and_spectrum():
    rng = np.random.default_rng(2026)
    coeff = rng.normal(size=(8, 8, 8)).astype(np.float64)
    bits = rng.integers(0, 2, size=8, dtype=np.uint8)
    stats = _embed_schur_coefficients(
        coeff,
        bits,
        step=0.04,
        lift=1.0,
        max_log_scale=1.0,
    )
    assert stats["max_relative_det_error_float"] < 1e-8
    assert stats["max_spectrum_error_float"] < 1e-8


def test_direct_schur_rescue_clean_roundtrip(monkeypatch):
    monkeypatch.setenv("JILP_NUM_THREADS", "1")
    host, watermark = _assets()
    watermarked, key, embed_metadata = embed_proposal(
        DIRECT_SCHUR_RESCUE,
        host,
        watermark,
        config=DirectSchurRescueConfig(),
        return_metadata=True,
    )
    recovered, metadata = extract_proposal(watermarked, key, return_metadata=True)
    assert psnr(host, watermarked) > 50.0
    assert np.array_equal(recovered, watermark)
    assert metadata["method_id"] == DIRECT_SCHUR_RESCUE
    assert metadata["certificate_mode"] == "schur"
    assert metadata["det_nonzero"]
    assert embed_metadata["direct_schur_all_det_nonzero"]
    assert key.fully_blind
