from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest

from qr64_certified.baselines import (
    ALL_BASELINE_IDS,
    PRIMARY_BLIND_BASELINE_IDS,
    SPECS,
    embed_baseline,
    extract_baseline,
    list_baselines,
    normalize_baseline_id,
)

ROOT = Path(__file__).resolve().parents[1]


def _host() -> np.ndarray:
    return np.asarray(Image.open(ROOT / "data/host/lenna.bmp").convert("RGB"), dtype=np.uint8)


def _wm(size: int = 64) -> np.ndarray:
    return np.asarray(
        Image.open(ROOT / "data/watermark/wm.png").convert("L").resize((size, size)),
        dtype=np.uint8,
    )


def test_exactly_sixteen_baselines_under_one_package_namespace():
    assert len(ALL_BASELINE_IDS) == 16
    assert not (ROOT / "src/realtime_watermark").exists()
    assert not (ROOT / "src/watermarklab").exists()
    assert (ROOT / "src/qr64_certified/baselines/implementations/dct").is_dir()
    assert (ROOT / "src/qr64_certified/baselines/implementations/transform").is_dir()


def test_all_public_ids_follow_one_rule_and_legacy_ids_are_aliases():
    pattern = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+_(?:blind|keyassisted|semiblind|nonblind)$")
    assert all(pattern.fullmatch(method_id) for method_id in ALL_BASELINE_IDS)
    assert normalize_baseline_id("dct_dm_qim_chen2001") == "dct_dm_qim_chen2001_blind"
    assert normalize_baseline_id("kumar2021") == "dwt_entropy_kumar2021_nonblind"
    assert normalize_baseline_id("ca-qim") == "dct_ca_qim_mao2024_semiblind"


def test_registry_scientific_tiers_are_explicit():
    rows = list_baselines()
    assert len(rows) == 16
    assert {row["blindness_tier"] for row in rows} == {
        "blind", "key_assisted_blind", "semi_blind", "non_blind"
    }
    assert set(PRIMARY_BLIND_BASELINE_IDS) == {
        "dct_dm_qim_chen2001_blind",
        "dct_stdm_qim_chen2001_blind",
        "dct_iss_malvar2003_blind",
        "hessenberg_nha2023_blind",
        "qwt_qsvd_zhang2022_blind",
    }
    assert not SPECS["dct_ca_qim_mao2024_semiblind"].primary_blind_eligible
    assert not SPECS["dct_spread_spectrum_cox1997_nonblind"].primary_blind_eligible
    assert not SPECS["dct_dew_langelaar2001_blind"].primary_blind_eligible


@pytest.mark.parametrize("method_id", [
    "dct_dm_qim_chen2001_blind",
    "dct_stdm_qim_chen2001_blind",
    "dct_iss_malvar2003_blind",
    "dct_ca_qim_mao2024_semiblind",
    "dct_spread_spectrum_cox1997_nonblind",
])
def test_dct_family_common_image_baselines_roundtrip(method_id: str):
    host, wm = _host(), _wm()
    watermarked, key = embed_baseline(method_id, host, wm, seed=2026, repeat=1)
    recovered = extract_baseline(watermarked, key)
    assert key.canonical_id == method_id
    assert watermarked.shape == host.shape
    assert recovered.shape == wm.shape
    assert np.mean((recovered > 127) == (wm > 127)) >= 0.95


@pytest.mark.parametrize("method_id", [
    "hessenberg_nha2023_blind",
    "qwt_qsvd_zhang2022_blind",
    "iwt_svd_qim_zhu2021_blind",
])
def test_transform_family_blind_baselines_roundtrip(method_id: str):
    host, wm = _host(), _wm()
    watermarked, key = embed_baseline(method_id, host, wm)
    recovered = extract_baseline(watermarked, key)
    assert watermarked.shape == host.shape
    assert recovered.shape == wm.shape
    assert np.mean((recovered > 127) == (wm > 127)) >= 0.90


def test_non_blind_original_host_requirement_is_enforced():
    host, wm = _host(), _wm()
    method_id = "dwt_entropy_kumar2021_nonblind"
    watermarked, key = embed_baseline(method_id, host, wm)
    with pytest.raises(ValueError, match="requires original_host"):
        extract_baseline(watermarked, key)
    recovered = extract_baseline(watermarked, key, original_host=host)
    assert recovered.shape == wm.shape
