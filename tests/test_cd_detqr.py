from pathlib import Path

import numpy as np

from qr64_certified import SPATIAL_CD_DETQR, embed_proposal, extract_proposal
from qr64_certified.cd_detqr import minimum_integer_amplitude, qr_residual_features
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

    assert psnr(host, watermarked) > 55.0
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
    assert metadata["min_abs_det"] >= 0.099
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


def test_closed_form_qr_update_is_minimal_and_enforces_det_nonzero():
    u = np.array([[120.0, 80.0]])
    v = np.array([[119.95, 82.0]])
    sign = np.array([1])
    mu = 0.005
    delta = 0.1
    amplitude = minimum_integer_amplitude(u, v, sign, mu, delta)
    d = u - v
    k = u + v
    y = sign[:, None] * d + 2.0 * amplitude
    z = np.sqrt(2.0) * y / np.sqrt(k * k + y * y)
    assert np.all(z >= mu - 1e-12)
    assert np.all(y >= delta - 1e-12)
    smaller = np.maximum(amplitude - 1, 0)
    y_smaller = sign[:, None] * d + 2.0 * smaller
    z_smaller = np.sqrt(2.0) * y_smaller / np.sqrt(k * k + y_smaller * y_smaller)
    needs_update = amplitude > 0
    assert np.all((z_smaller < mu) | (y_smaller < delta) | (~needs_update))


def test_qr_residual_is_det_over_r11():
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    u, v, determinant, residual = qr_residual_features(host, 3)
    expected = determinant / np.sqrt(u * u + v * v)
    assert np.allclose(determinant, u - v, atol=1e-10)
    assert np.allclose(residual, expected, atol=1e-10)
