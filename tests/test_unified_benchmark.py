from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from qr64_certified.attacks import get_attack_suite
from qr64_certified.benchmark import BenchmarkAdapter, evaluate_trial, resolve_methods
from qr64_certified.common.io import load_watermark_binary

ROOT = Path(__file__).resolve().parents[1]


def _host() -> np.ndarray:
    return np.asarray(Image.open(ROOT / "data/host/lenna.bmp").convert("RGB"), dtype=np.uint8)


def test_method_selectors_include_proposals_and_primary_baselines():
    methods = resolve_methods("paper_comparison")
    assert len(methods) == 14
    assert sum(spec.method_kind == "proposal" for spec in methods) == 9
    assert sum(spec.method_kind == "baseline" for spec in methods) == 5


def test_unified_evaluator_emits_same_schema_for_proposal_and_baseline():
    attacks = get_attack_suite("sanity")[:1]
    rows = []
    for method_id in ("dct_qr", "dct_dm_qim_chen2001_blind"):
        spec = resolve_methods((method_id,))[0]
        watermark = load_watermark_binary(ROOT / "data/watermark/wm.png", size=spec.payload_size)
        trial = evaluate_trial(
            BenchmarkAdapter(spec, ROOT),
            _host(),
            watermark,
            attacks,
            protocol_id="test_unified",
            host_id="lenna.bmp",
            watermark_id="wm.png",
            seed=2026,
            continue_on_error=False,
            strict_clean_proposal=True,
        )
        assert trial["status"] == "ok"
        assert len(trial["rows"]) == 1
        rows.append(trial["rows"][0])
    assert set(rows[0]) == set(rows[1])
    for required in (
        "embedding_psnr_db", "embedding_ssim", "attacked_nc", "attacked_ncc",
        "attacked_ber", "attacked_bit_accuracy", "extract_seconds", "key_size_bytes",
    ):
        assert required in rows[0]
