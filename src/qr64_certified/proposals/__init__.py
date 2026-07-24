"""The three original proposal methods.

The numerical implementations in this package are the original proposal code,
relocated into an explicit namespace so they are visible beside ``baselines``
and ``attacks``. Compatibility modules remain at ``qr64_certified.<module>``.
"""

from .config import QR64Config
from .method import QR64Key, embed, extract
from .direct_schur_rescue import DirectSchurRescueConfig, DirectSchurRescueKey
from .cd_detqr import CDDetQRConfig, CDDetQRKey, BlindCDDetQR
from .proposal_registry import (
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    QR_CERTIFIED,
    DIRECT_SCHUR_RESCUE,
    SUPPORTED_PROPOSAL_METHODS,
    embed_proposal,
    extract_proposal,
    list_supported_methods,
    normalize_proposal_method_id,
    default_config_for_method,
    method_id_from_key,
)

__all__ = [
    "QR64Config",
    "QR64Key",
    "DirectSchurRescueConfig",
    "DirectSchurRescueKey",
    "CDDetQRConfig",
    "CDDetQRKey",
    "BlindCDDetQR",
    "DCT_QR",
    "DCT_SCHUR_RESCUE",
    "SPATIAL_CD_DETQR",
    "QR_CERTIFIED",
    "DIRECT_SCHUR_RESCUE",
    "SUPPORTED_PROPOSAL_METHODS",
    "embed",
    "extract",
    "embed_proposal",
    "extract_proposal",
    "list_supported_methods",
    "normalize_proposal_method_id",
    "default_config_for_method",
    "method_id_from_key",
]
