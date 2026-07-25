from __future__ import annotations

"""Public registry exposing exactly three proposal methods."""

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .cd_detqr import (
    CDDetQRConfig,
    CDDetQRKey,
    embed_cd_detqr,
    extract_cd_detqr,
)
from .config import QR64Config
from .direct_schur_rescue import (
    DirectSchurRescueConfig,
    DirectSchurRescueKey,
    embed as embed_direct_schur_rescue,
    extract as extract_direct_schur_rescue,
)
from .method import QR64Key, embed as embed_certified, extract as extract_certified

DCT_QR = "dct_qr"
DCT_SCHUR_RESCUE = "dct_schur_rescue"
SPATIAL_CD_DETQR = "spatial_cd_detqr"

# Compatibility aliases retained for old commands, but only three methods are listed.
QR_CERTIFIED = DCT_QR
DIRECT_SCHUR_RESCUE = DCT_SCHUR_RESCUE

SUPPORTED_PROPOSAL_METHODS: dict[str, dict[str, Any]] = {
    DCT_QR: {
        "id": DCT_QR,
        "display_name": "DCT-QR Carrier-Subspace Gain-Normalized QIM",
        "domain": "DCT-QIM with QR reliability allocation and carrier-specific QR gain compensation",
        "scientific_status": "validated proposal",
        "description": (
            "QR reliability selects local QIM spacing, while the canonical first QR "
            "diagonal r11 of the carrier-bearing DCT column estimates local attenuation "
            "for regularized blind gain compensation."
        ),
    },
    DCT_SCHUR_RESCUE: {
        "id": DCT_SCHUR_RESCUE,
        "display_name": "DCT-Schur Spectral-Gain QIM",
        "domain": "DCT-QIM with Schur reliability and spectral-departure gain compensation",
        "scientific_status": "validated proposal",
        "description": (
            "Schur spectral balance allocates local QIM spacing, while eigenvalue energy and "
            "departure from normality estimate attack-induced local attenuation."
        ),
    },
    SPATIAL_CD_DETQR: {
        "id": SPATIAL_CD_DETQR,
        "display_name": "Spatial Normalized-Residual DetQR",
        "domain": "Spatial QR residual with the hard constraint det(A) != 0",
        "scientific_status": "validated proposal",
        "description": (
            "A closed-form minimum integer update enforces a signed normalized-QR margin and "
            "strict determinant nonsingularity; the same QR domain supplies affine pilots."
        ),
    },
}

METHOD_ALIASES = {
    "qr": DCT_QR,
    "qr64": DCT_QR,
    "qr_certified": DCT_QR,
    "dct_qr": DCT_QR,
    "schur": DCT_SCHUR_RESCUE,
    "schur_rescue": DCT_SCHUR_RESCUE,
    "direct_schur": DCT_SCHUR_RESCUE,
    "direct_schur_rescue": DCT_SCHUR_RESCUE,
    "dct_schur_rescue": DCT_SCHUR_RESCUE,
    "cd_detqr": SPATIAL_CD_DETQR,
    "detqr": SPATIAL_CD_DETQR,
    "spatial_cd_detqr": SPATIAL_CD_DETQR,
}


def normalize_proposal_method_id(method_id: str) -> str:
    normalized = METHOD_ALIASES.get(str(method_id).strip().lower())
    if normalized is None:
        valid = ", ".join(SUPPORTED_PROPOSAL_METHODS)
        raise KeyError(f"Unknown proposal method '{method_id}'. Valid methods: {valid}")
    return normalized


def list_supported_methods() -> list[dict[str, Any]]:
    return [dict(SUPPORTED_PROPOSAL_METHODS[key]) for key in SUPPORTED_PROPOSAL_METHODS]


def default_config_for_method(method_id: str):
    method = normalize_proposal_method_id(method_id)
    if method == DCT_QR:
        return QR64Config(
            certificate_mode="qr",
            step=13.25,
            pilot_step=6.0,
            adaptive_step_enabled=True,
            adaptive_step_ratios=(18.25 / 13.25, 1.0, 10.25 / 13.25),
            adaptive_step_fractions=(0.20, 0.60),
            evidence_conf_power=2.1,
            gain_normalization_enabled=True,
            qr_gain_mode="carrier_r11",
            gain_gamma=0.90,
        ).validated()
    if method == DCT_SCHUR_RESCUE:
        return DirectSchurRescueConfig().validated()
    return CDDetQRConfig()


def embed_proposal(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: QR64Config | DirectSchurRescueConfig | CDDetQRConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    method = normalize_proposal_method_id(method_id)
    if method == DCT_SCHUR_RESCUE:
        result = embed_direct_schur_rescue(
            host_rgb,
            watermark_binary,
            config=DirectSchurRescueConfig.from_mapping(config),
            return_metadata=return_metadata,
        )
        if return_metadata:
            watermarked, key, metadata = result
            metadata = dict(metadata)
            metadata["method_id"] = DCT_SCHUR_RESCUE
            return watermarked, key, metadata
        return result
    if method == SPATIAL_CD_DETQR:
        return embed_cd_detqr(
            host_rgb,
            watermark_binary,
            config=config,
            return_metadata=return_metadata,
        )

    if config is None:
        cfg = default_config_for_method(DCT_QR)
    elif isinstance(config, QR64Config):
        cfg = config
    else:
        cfg = QR64Config.from_mapping(config)
    if cfg.certificate_mode != "qr":
        cfg = replace(cfg, certificate_mode="qr")
    cfg = cfg.validated()
    watermarked, key = embed_certified(host_rgb, watermark_binary, config=cfg)
    if return_metadata:
        metadata = {
            "method_id": DCT_QR,
            "certificate_mode": "qr",
            "direct_schur_carrier": False,
            "configuration": cfg.to_dict(),
        }
        return watermarked, key, metadata
    return watermarked, key


def extract_proposal(
    possibly_attacked_rgb: np.ndarray,
    key: QR64Key | DirectSchurRescueKey | CDDetQRKey,
    *,
    return_metadata: bool = False,
):
    if isinstance(key, DirectSchurRescueKey):
        result = extract_direct_schur_rescue(
            possibly_attacked_rgb, key, return_metadata=return_metadata
        )
        if return_metadata:
            recovered, metadata = result
            metadata = dict(metadata)
            metadata["method_id"] = DCT_SCHUR_RESCUE
            return recovered, metadata
        return result
    if isinstance(key, CDDetQRKey):
        return extract_cd_detqr(
            possibly_attacked_rgb, key, return_metadata=return_metadata
        )
    return extract_certified(
        possibly_attacked_rgb, key, return_metadata=return_metadata
    )


def method_id_from_key(key: QR64Key | DirectSchurRescueKey | CDDetQRKey) -> str:
    if isinstance(key, DirectSchurRescueKey):
        return DCT_SCHUR_RESCUE
    if isinstance(key, CDDetQRKey):
        return SPATIAL_CD_DETQR
    return DCT_QR


__all__ = [
    "DCT_QR",
    "DCT_SCHUR_RESCUE",
    "SPATIAL_CD_DETQR",
    "QR_CERTIFIED",
    "DIRECT_SCHUR_RESCUE",
    "SUPPORTED_PROPOSAL_METHODS",
    "METHOD_ALIASES",
    "normalize_proposal_method_id",
    "list_supported_methods",
    "default_config_for_method",
    "embed_proposal",
    "extract_proposal",
    "method_id_from_key",
]
