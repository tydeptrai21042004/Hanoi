from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from qr64_certified import DCT_QR
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr
from qr64_certified.proposals.method import (
    _coset_flip_mask_from_key,
    _pack_binary_mask,
    _pairwise_coset_optimize,
    embed,
    extract,
)
from qr64_certified.proposals.proposal_registry import default_config_for_method

ROOT = Path(__file__).resolve().parents[1]


def test_public_dct_qr_enables_pairwise_coset_rule() -> None:
    cfg = default_config_for_method(DCT_QR)
    assert cfg.coset_optimization_enabled is True
    assert cfg.coset_group_size == 2


def test_packed_flip_mask_round_trip() -> None:
    rng = np.random.default_rng(2026)
    flips = rng.integers(0, 2, size=4096, dtype=np.uint8)
    params = {
        "coset_flip_mask_b64": _pack_binary_mask(flips),
        "coset_flip_count": int(flips.size),
    }
    recovered = _coset_flip_mask_from_key(params, flips.size)
    assert np.array_equal(recovered, flips)


def test_pairwise_coset_projection_energy_never_increases() -> None:
    rng = np.random.default_rng(2104)
    carrier = rng.normal(0.0, 20.0, size=256)
    bits = rng.integers(0, 2, size=256, dtype=np.uint8)
    steps = rng.choice(np.array([10.25, 13.25, 18.25]), size=256)
    reliability = rng.random(256)
    encoded, flips, stats = _pairwise_coset_optimize(
        carrier,
        bits,
        steps,
        reliability,
        rho_frac=0.45,
        group_size=2,
    )
    assert encoded.shape == bits.shape
    assert flips.shape == bits.shape
    assert float(stats["optimized_projection_energy"]) <= float(
        stats["original_projection_energy"]
    ) + 1e-10
    assert np.array_equal(np.bitwise_xor(encoded, flips), bits)


def test_new_dct_qr_clean_round_trip_and_psnr_gain_on_lenna() -> None:
    host = load_host_rgb(ROOT / "data/host/lenna.bmp")
    watermark = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    new_cfg = default_config_for_method(DCT_QR)
    old_cfg = replace(new_cfg, coset_optimization_enabled=False)

    old_marked, _ = embed(host, watermark, config=old_cfg)
    new_marked, new_key = embed(host, watermark, config=new_cfg)
    recovered, metadata = extract(new_marked, new_key, return_metadata=True)

    assert nc(watermark, recovered) == 1.0
    assert metadata["coset_optimization_enabled"] is True
    assert metadata["coset_group_size"] == 2
    assert psnr(host, new_marked) > psnr(host, old_marked) + 2.0
    assert psnr(host, new_marked) > 50.0
