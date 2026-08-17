from __future__ import annotations

"""Public registry exposing the proposal methods."""

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
from .dct_qr_direct_r import (
    DCTQRDirectRConfig,
    DCTQRDirectRKey,
    embed as embed_dct_qr_direct_r,
    extract as extract_dct_qr_direct_r,
)
from .dct_qr_r11_qim import (
    DCTQRR11QIMConfig,
    DCTQRR11QIMKey,
    embed as embed_dct_qr_r11_qim,
    extract as extract_dct_qr_r11_qim,
)
from .method import QR64Key, embed as embed_certified, extract as extract_certified

DCT_QR = "dct_qr"
DCT_QR_DIRECT_R = "dct_qr_direct_r"
DCT_QR_R11_QIM = "dct_qr_r11_qim"
DCT_SCHUR_RESCUE = "dct_schur_rescue"
SPATIAL_CD_DETQR = "spatial_cd_detqr"

# Compatibility aliases retained for old commands.
QR_CERTIFIED = DCT_QR
DIRECT_SCHUR_RESCUE = DCT_SCHUR_RESCUE

SUPPORTED_PROPOSAL_METHODS: dict[str, dict[str, Any]] = {
    DCT_QR: {
        "id": DCT_QR,
        "display_name": "DCT-QR Pairwise Coset-Optimized Gain-Normalized QIM",
        "domain": "DCT-QIM with QR reliability grouping, pairwise coset optimization, and carrier-specific QR gain compensation",
        "scientific_status": "validated proposal",
        "description": (
            "QR reliability selects local QIM spacing and forms homogeneous block pairs. "
            "An exact binary coset choice minimizes pairwise projection distortion at unchanged "
            "QIM step and margin, while carrier r11 supplies blind gain compensation."
        ),
    },
    DCT_QR_DIRECT_R: {
        "id": DCT_QR_DIRECT_R,
        "display_name": "Transform-Domain DCT-QR Direct-R Differential QIM",
        "domain": "8x8 DCT followed by direct QR of a regularized 4x4 low-frequency coefficient matrix",
        "scientific_status": "new proposal; smoke-validated",
        "description": (
            "The payload is embedded after the QR decomposition itself. A parity-QIM rule "
            "modifies the first-row differential (R12-R13)/2 while preserving R11 as the "
            "dominant low-frequency energy anchor. QR is therefore performed in the transform "
            "domain, not on spatial image blocks."
        ),
    },
    DCT_QR_R11_QIM: {
        "id": DCT_QR_R11_QIM,
        "display_name": "Transform-Domain DCT-QR Direct-R11 QIM",
        "domain": "8x8 DCT followed by QR of a regularized 4x4 low-frequency coefficient matrix; direct R11 embedding",
        "scientific_status": "new proposal; smoke-validated",
        "description": (
            "The watermark bit is embedded directly in R11 after transform-domain QR. "
            "Canonical positive-diagonal QR makes R11 the norm of the first low-frequency "
            "DCT column, and parity-QIM changes only R11 before QR reconstruction and IDCT."
        ),
    },
    DCT_SCHUR_RESCUE: {
        "id": DCT_SCHUR_RESCUE,
        "display_name": "DCT-Schur Spectrum-Preserving Orthogonal Coupling QIM",
        "domain": "DCT strict-upper Schur couplings with three interleaved parity projections",
        "scientific_status": "independent proposal; performance gate under validation",
        "description": (
            "Three orthogonal strict-upper Schur coupling projections carry interleaved payload "
            "copies. The closed-form minimum-Frobenius update preserves the constructed Schur "
            "spectrum, trace, and determinant while using an independent decoder and key."
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
    "dct_qr_direct_r": DCT_QR_DIRECT_R,
    "direct_r_qr": DCT_QR_DIRECT_R,
    "transform_qr": DCT_QR_DIRECT_R,
    "dct_qr_r11_qim": DCT_QR_R11_QIM,
    "r11_qim": DCT_QR_R11_QIM,
    "direct_r11_qr": DCT_QR_R11_QIM,
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
        return QR64Config.public_dct_qr()
    if method == DCT_QR_DIRECT_R:
        return DCTQRDirectRConfig().validated()
    if method == DCT_QR_R11_QIM:
        return DCTQRR11QIMConfig().validated()
    if method == DCT_SCHUR_RESCUE:
        return DirectSchurRescueConfig().validated()
    return CDDetQRConfig()


def embed_proposal(
    method_id: str,
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: QR64Config | DCTQRDirectRConfig | DCTQRR11QIMConfig | DirectSchurRescueConfig | CDDetQRConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    method = normalize_proposal_method_id(method_id)
    if method == DCT_QR_DIRECT_R:
        return embed_dct_qr_direct_r(
            host_rgb,
            watermark_binary,
            config=DCTQRDirectRConfig.from_mapping(config),
            return_metadata=return_metadata,
        )
    if method == DCT_QR_R11_QIM:
        return embed_dct_qr_r11_qim(
            host_rgb,
            watermark_binary,
            config=DCTQRR11QIMConfig.from_mapping(config),
            return_metadata=return_metadata,
        )
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
    key: QR64Key | DCTQRDirectRKey | DCTQRR11QIMKey | DirectSchurRescueKey | CDDetQRKey,
    *,
    return_metadata: bool = False,
):
    if isinstance(key, DCTQRDirectRKey):
        return extract_dct_qr_direct_r(
            possibly_attacked_rgb, key, return_metadata=return_metadata
        )
    if isinstance(key, DCTQRR11QIMKey):
        return extract_dct_qr_r11_qim(
            possibly_attacked_rgb, key, return_metadata=return_metadata
        )
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


def method_id_from_key(key: QR64Key | DCTQRDirectRKey | DCTQRR11QIMKey | DirectSchurRescueKey | CDDetQRKey) -> str:
    if isinstance(key, DCTQRDirectRKey):
        return DCT_QR_DIRECT_R
    if isinstance(key, DCTQRR11QIMKey):
        return DCT_QR_R11_QIM
    if isinstance(key, DirectSchurRescueKey):
        return DCT_SCHUR_RESCUE
    if isinstance(key, CDDetQRKey):
        return SPATIAL_CD_DETQR
    return DCT_QR


__all__ = [
    "DCT_QR",
    "DCT_QR_DIRECT_R",
    "DCT_QR_R11_QIM",
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
