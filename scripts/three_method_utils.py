from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qr64_certified import (
    CDDetQRConfig,
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    DirectSchurRescueConfig,
    QR64Config,
    embed_proposal,
    extract_proposal,
)
from qr64_certified.attacks.presets import moderate_attacks
from qr64_certified.attacks import apply_attack
from qr64_certified.common.metrics import ber, nc, ncc, psnr, ssim

METHODS = (DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR)


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return asdict(config)


def config_from_dict(method: str, values: Mapping[str, Any] | None):
    raw = dict(values or {})
    if method == DCT_QR:
        raw["certificate_mode"] = "qr"
        return QR64Config.from_mapping(raw)
    if method == DCT_SCHUR_RESCUE:
        return DirectSchurRescueConfig.from_mapping(raw)
    allowed = set(CDDetQRConfig.__dataclass_fields__)
    cfg = CDDetQRConfig(**{k: v for k, v in raw.items() if k in allowed})
    cfg.validate()
    return cfg


def read_config(path: Path, method: str):
    return config_from_dict(method, json.loads(path.read_text(encoding="utf-8")))


def write_config(path: Path, config: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2), encoding="utf-8")


def evaluate_method(
    method: str,
    config: Any,
    host: np.ndarray,
    watermark: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, Any, np.ndarray]:
    watermarked, key, embed_meta = embed_proposal(
        method, host, watermark, config=config, return_metadata=True
    )
    recovered_clean, clean_meta = extract_proposal(watermarked, key, return_metadata=True)
    host_psnr = float(psnr(host, watermarked))
    host_ssim = float(ssim(host, watermarked))
    clean_nc = float(nc(watermark, recovered_clean))
    rows: list[dict[str, Any]] = []
    for attack in moderate_attacks():
        attacked = apply_attack(watermarked, attack)
        recovered, metadata = extract_proposal(attacked, key, return_metadata=True)
        rows.append(
            {
                "method_id": method,
                "attack": attack.name,
                "psnr": host_psnr,
                "ssim": host_ssim,
                "embedding_psnr": host_psnr,
                "embedding_ssim": host_ssim,
                "clean_nc": clean_nc,
                "nc": float(nc(watermark, recovered)),
                "ncc": float(ncc(watermark, recovered)),
                "ber": float(ber(watermark, recovered)),
                "det_nonzero": bool(metadata.get("det_nonzero", True)),
                "min_abs_det": metadata.get("min_abs_det", ""),
                "inference_path": metadata.get("inference_path", ""),
            }
        )
    summary = {
        "method_id": method,
        "configuration": config_to_dict(config),
        "host_psnr": host_psnr,
        "host_ssim": host_ssim,
        "clean_nc": clean_nc,
        "mean_nc": float(np.mean([r["nc"] for r in rows])),
        "q10_nc": float(np.quantile([r["nc"] for r in rows], 0.10)),
        "min_nc": float(np.min([r["nc"] for r in rows])),
        "mean_ncc": float(np.mean([r["ncc"] for r in rows])),
        "mean_ber": float(np.mean([r["ber"] for r in rows])),
        "embedding_det_nonzero": bool(embed_meta.get("det_nonzero", True)),
        "embedding_min_abs_det": embed_meta.get("min_abs_det", ""),
        "all_attacked_det_nonzero": bool(all(r["det_nonzero"] for r in rows)),
        # Backward-compatible field; unlike earlier versions, extraction now
        # recomputes determinants on each attacked image instead of reusing
        # stale embedding metadata.
        "all_det_nonzero": bool(all(r["det_nonzero"] for r in rows)),
        "attack_count": len(rows),
        "fully_blind": True,
    }
    return summary, rows, watermarked, key, recovered_clean


def optimization_score(summary: Mapping[str, Any]) -> float:
    """Robust scientific objective rather than a mean-only engineering score.

    Average NC measures overall behavior, the lower decile and minimum prevent
    a few attacks from being hidden by the mean, and normalized PSNR represents
    imperceptibility.  Clean inexactness and PSNR below 45 dB are hard penalties.
    """
    psnr_value = float(summary["host_psnr"])
    clean = float(summary["clean_nc"])
    mean_nc = float(summary["mean_nc"])
    q10_nc = float(summary.get("q10_nc", mean_nc))
    min_nc = float(summary.get("min_nc", q10_nc))
    quality = min(max((psnr_value - 45.0) / 10.0, 0.0), 1.0)
    clean_penalty = 100.0 * max(0.0, 1.0 - clean)
    psnr_penalty = 5.0 * max(0.0, 45.0 - psnr_value)
    return (
        0.35 * mean_nc
        + 0.30 * q10_nc
        + 0.20 * min_nc
        + 0.15 * quality
        - clean_penalty
        - psnr_penalty
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
