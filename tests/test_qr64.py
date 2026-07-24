from pathlib import Path

import numpy as np

from qr64_certified import QR64Config, embed, extract
from qr64_certified.certificate import analysis_matrices, canonical_schur, compute_certificate
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import psnr

ROOT = Path(__file__).resolve().parents[1]


def _assets():
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    wm = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    return host, wm


def test_qr_det_nonzero_and_blind_clean_roundtrip():
    host, wm = _assets()
    cert = compute_certificate(host, mode="qr")
    assert cert.mode == "qr"
    assert np.all(cert.determinant > 0)
    watermarked, key = embed(host, wm, config=QR64Config(certificate_mode="qr"))
    recovered, metadata = extract(watermarked, key, return_metadata=True)
    assert recovered.shape == (64, 64)
    assert psnr(host, watermarked) > 50.0
    assert np.array_equal(recovered, wm)
    assert metadata["certificate_mode"] == "qr"
    assert key.fully_blind


def test_schur_clean_roundtrip_and_exact_factor_validation(monkeypatch):
    # One JILP worker avoids thread-pool contention with small eigenvalue tasks.
    monkeypatch.setenv("JILP_NUM_THREADS", "1")
    host, wm = _assets()
    cert = compute_certificate(host, mode="schur")
    assert cert.mode == "schur"
    assert np.all(cert.determinant > 0)

    a = analysis_matrices(host)[17]
    t, z = canonical_schur(a)
    assert np.allclose(a, z @ t @ z.conj().T, atol=1e-9)
    expected_balance = np.min(np.abs(np.diag(t))) / (np.max(np.abs(np.diag(t))) + 1e-12)
    assert np.isclose(expected_balance, cert.balance[17], atol=1e-9)

    watermarked, key = embed(host, wm, config=QR64Config(certificate_mode="schur"))
    recovered = extract(watermarked, key)
    assert psnr(host, watermarked) > 50.0
    assert np.array_equal(recovered, wm)


def test_default_configuration_is_corrected_qr():
    cfg = QR64Config().validated()
    assert cfg.certificate_mode == "qr"
    assert cfg.step == 8.25
    assert cfg.evidence_conf_power == 2.0
    assert cfg.sync_improvement_threshold == 0.05
