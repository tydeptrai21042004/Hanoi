from __future__ import annotations

from pathlib import Path

import numpy as np

from qr64_certified import (
    ABLATION_FLAGS,
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    apply_proposal_flags,
    embed_proposal,
    extract_proposal,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.proposals.proposal_registry import default_config_for_method


ROOT = Path(__file__).resolve().parents[1]


def test_every_ablation_builds_a_valid_config() -> None:
    for method, names in ABLATION_FLAGS.items():
        base = default_config_for_method(method)
        for name in names:
            config, report = apply_proposal_flags(
                method, base, {"ablation": [name]}
            )
            assert report["ablations"] == [name]
            if hasattr(config, "validated"):
                config.validated()
            else:
                config.validate()


def test_hyperparameter_overrides_reach_independent_schur_config() -> None:
    base = default_config_for_method(DCT_SCHUR_RESCUE)
    config, report = apply_proposal_flags(
        DCT_SCHUR_RESCUE,
        base,
        {
            "ablation": [],
            "step": 9.0,
            "gain_gamma": 0.6,
            "closure_rounds": 3,
        },
    )
    assert config.step == 9.0
    assert config.gain_gamma == 0.6
    assert config.closure_rounds == 3
    assert report["hyperparameter_overrides"]["step"] == 9.0


def test_spatial_fixed_pattern_and_no_mask_clean_roundtrip() -> None:
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    watermark = load_watermark_binary(
        ROOT / "data" / "watermark" / "wm.png", size=64
    )
    base = default_config_for_method(SPATIAL_CD_DETQR)
    config, _ = apply_proposal_flags(
        SPATIAL_CD_DETQR,
        base,
        {
            "ablation": ["fixed_payload_pattern", "no_payload_mask"],
            "fixed_payload_pattern": 0,
        },
    )
    watermarked, key = embed_proposal(
        SPATIAL_CD_DETQR, host, watermark, config=config
    )
    recovered = extract_proposal(watermarked, key)
    assert np.array_equal(recovered > 0, watermark > 0)
    assert key.payload["payload_pattern_search_enabled"] is False
    assert key.payload["payload_mask_enabled"] is False


def test_dct_qr_direct_flags_override_ablation_base() -> None:
    base = default_config_for_method(DCT_QR)
    config, _ = apply_proposal_flags(
        DCT_QR,
        base,
        {
            "ablation": ["no_spatial_map"],
            "step": 12.0,
            "adaptive_step_ratios": (1.2, 1.0, 0.8),
            "gain_gamma": 0.8,
        },
    )
    assert config.step == 12.0
    assert config.adaptive_step_ratios == (1.2, 1.0, 0.8)
    assert config.gain_gamma == 0.8
    assert config.qr_map_lambda == 0.0
    assert config.qr_map_iters == 0
