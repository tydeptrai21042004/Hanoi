from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from qr64_certified.attacks import (
    AttackConfig,
    apply_attack,
    available_attack_groups,
    get_attack_suite,
    list_attack_suites,
)
from qr64_certified.baselines import evaluate_baseline_attacks

ROOT = Path(__file__).resolve().parents[1]


def _host() -> np.ndarray:
    return np.asarray(Image.open(ROOT / "data/host/lenna.bmp").convert("RGB"), dtype=np.uint8)


def _wm() -> np.ndarray:
    arr = np.asarray(Image.open(ROOT / "data/watermark/wm.png").convert("L").resize((64, 64)), dtype=np.uint8)
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def test_attack_library_has_broad_unified_coverage():
    groups = set(available_attack_groups())
    assert len(groups) >= 45
    assert {"jpeg", "webp", "perspective", "affine", "bilateral_filter", "combined"} <= groups
    suites = list_attack_suites()
    assert suites["sanity"] >= 6
    assert suites["common"] >= 40
    assert suites["stress"] >= 70


def test_new_attack_types_are_deterministic_and_shape_preserving():
    image = _host()
    configs = [
        AttackConfig("test_webp", "webp", {"quality": 70}),
        AttackConfig("test_hue", "hue_shift", {"degrees": 15.0}),
        AttackConfig("test_affine", "affine", {"rotation": 1.0, "scale": 0.99}),
        AttackConfig("test_perspective", "perspective", {"strength": 0.02, "seed": 123}),
        AttackConfig("test_bilateral", "bilateral_filter", {"diameter": 5}),
    ]
    for cfg in configs:
        first = apply_attack(image, cfg)
        second = apply_attack(image, cfg)
        assert first.shape == image.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)


def test_baseline_attack_evaluation_emits_canonical_metadata():
    rows = evaluate_baseline_attacks(
        "dm_qim",
        _host(),
        _wm(),
        get_attack_suite("sanity")[:2],
        repeat=1,
    )
    assert len(rows) == 2
    assert all(row["baseline_id"] == "dct_dm_qim_chen2001_blind" for row in rows)
    assert all(row["blindness_tier"] == "blind" for row in rows)
    assert all(row["status"] == "ok" for row in rows)
