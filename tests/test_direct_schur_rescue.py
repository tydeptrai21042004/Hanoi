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
    assert psnr(host, watermarked) > 47.0
    assert np.array_equal(recovered, watermark)
    assert metadata["method_id"] == DIRECT_SCHUR_RESCUE
    assert metadata["certificate_mode"] == "schur"
    assert metadata["det_nonzero"]
    assert embed_metadata["direct_schur_all_det_nonzero"]
    assert not embed_metadata["legacy_secondary_embedded"]
    assert metadata["gain_normalization_enabled"]
    assert metadata["inference_path"] == "independent_schur_coupling_evidence"
    assert not metadata["dct_qr_engine_used"]
    assert embed_metadata["minimum_frobenius_projection"]
    assert embed_metadata["spectrum_preserved_float"]
    assert key.fully_blind


def test_schur_coupling_basis_is_orthonormal():
    from qr64_certified.proposals.schur_coupling_qim import SCHUR_COUPLING_BASIS
    gram = SCHUR_COUPLING_BASIS @ SCHUR_COUPLING_BASIS.T
    assert np.allclose(gram, np.eye(3), atol=1e-12)


def test_public_schur_module_has_one_embed_and_extract_definition():
    import ast
    source = (ROOT / "src/qr64_certified/proposals/schur_coupling_qim.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert names.count("embed") == 1
    assert names.count("extract") == 1


def test_schur_is_independent_of_dct_qr_public_engine(monkeypatch):
    import qr64_certified.proposals.method as qr_method

    def forbidden(*args, **kwargs):
        raise AssertionError("DCT-QR engine must not be called by SP-SCQIM")

    monkeypatch.setattr(qr_method, "embed", forbidden)
    monkeypatch.setattr(qr_method, "extract", forbidden)
    host, watermark = _assets()
    watermarked, key = embed_proposal(
        DIRECT_SCHUR_RESCUE,
        host,
        watermark,
        config=DirectSchurRescueConfig(candidate_search_enabled=False),
    )
    recovered = extract_proposal(watermarked, key)
    assert np.array_equal(recovered, watermark)


def test_step_parameter_is_active():
    host, watermark = _assets()
    low, _ = embed_proposal(
        DIRECT_SCHUR_RESCUE,
        host,
        watermark,
        config=DirectSchurRescueConfig(step=8.0, candidate_search_enabled=False),
    )
    high, _ = embed_proposal(
        DIRECT_SCHUR_RESCUE,
        host,
        watermark,
        config=DirectSchurRescueConfig(step=9.0, candidate_search_enabled=False),
    )
    assert not np.array_equal(low, high)
