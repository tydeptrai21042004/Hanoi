from __future__ import annotations

from pathlib import Path

import numpy as np

from qr64_certified import DCT_QR
from qr64_certified.common.io import load_host_rgb
from qr64_certified.proposals.certificate import analysis_matrices, qr_gain_scale
from qr64_certified.proposals.proposal_registry import default_config_for_method

ROOT = Path(__file__).resolve().parents[1]


def test_public_dct_qr_default_uses_carrier_specific_qr_gain() -> None:
    cfg = default_config_for_method(DCT_QR)
    assert cfg.qr_gain_mode == "carrier_r11"
    assert np.allclose(cfg.adaptive_step_levels(), (18.25, 13.25, 10.25))
    assert cfg.gain_gamma == 0.90


def test_carrier_r11_is_exact_first_column_norm() -> None:
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    matrices = analysis_matrices(host, eta=0.07, lift=0.0)
    expected = np.linalg.norm(matrices[:, :, 0], axis=1)
    measured = qr_gain_scale(host, eta=0.07, mode="carrier_r11")
    assert np.allclose(measured, expected, rtol=1e-11, atol=1e-11)


def test_carrier_r11_obeys_multiplicative_homogeneity() -> None:
    rng = np.random.default_rng(2026)
    columns = rng.normal(size=(128, 4))
    alpha = 0.73
    original = np.linalg.norm(columns, axis=1)
    attacked = np.linalg.norm(alpha * columns, axis=1)
    assert np.allclose(attacked / original, alpha, rtol=1e-12, atol=1e-12)


def test_carrier_r11_perturbation_bound() -> None:
    rng = np.random.default_rng(2104)
    columns = rng.normal(size=(256, 4))
    errors = 0.03 * rng.normal(size=(256, 4))
    alpha = 0.82
    reference = np.linalg.norm(columns, axis=1)
    observed = np.linalg.norm(alpha * columns + errors, axis=1)
    lhs = np.abs(observed / reference - alpha)
    rhs = np.linalg.norm(errors, axis=1) / reference
    assert np.all(lhs <= rhs + 1e-12)
