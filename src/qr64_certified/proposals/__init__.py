"""Public proposal methods.

The DCT-QR path uses the active pairwise-coset embedding law. DCT-Schur and
Spatial DetQR retain their validated implementations. ``dct_qr_direct_r`` is
the direct transform-domain DCT->QR->R proposal. Compatibility modules remain
at ``qr64_certified.<module>``.
"""

from .config import QR64Config
from .method import QR64Key, embed, extract
from .direct_schur_rescue import DirectSchurRescueConfig, DirectSchurRescueKey
from .cd_detqr import CDDetQRConfig, CDDetQRKey, BlindCDDetQR
from .dct_qr_direct_r import DCTQRDirectRConfig, DCTQRDirectRKey
from .dct_qr_r11_qim import DCTQRR11QIMConfig, DCTQRR11QIMKey
from .dct_qr_theory import DCTQRTheoryConfig, DCTQRTheoryKey
from .spatial_qr import SpatialQRConfig, SpatialQRKey
from .spatial_qr_direct_r import SpatialQRDirectRConfig, SpatialQRDirectRKey
from .spatial_qr_r11_qim import SpatialQRR11QIMConfig, SpatialQRR11QIMKey
from .proposal_registry import (
    DCT_QR,
    DCT_QR_DIRECT_R,
    DCT_QR_R11_QIM,
    DCT_QR_THEORY,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    SPATIAL_QR,
    SPATIAL_QR_DIRECT_R,
    SPATIAL_QR_R11_QIM,
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
from .flags import ABLATION_FLAGS, HYPERPARAMETER_FLAGS, apply_proposal_flags

__all__ = [
    "QR64Config",
    "QR64Key",
    "DirectSchurRescueConfig",
    "DirectSchurRescueKey",
    "CDDetQRConfig",
    "CDDetQRKey",
    "BlindCDDetQR",
    "DCTQRDirectRConfig",
    "DCTQRDirectRKey",
    "DCTQRR11QIMConfig",
    "DCTQRR11QIMKey",
    "DCTQRTheoryConfig",
    "DCTQRTheoryKey",
    "SpatialQRConfig",
    "SpatialQRKey",
    "SpatialQRDirectRConfig",
    "SpatialQRDirectRKey",
    "SpatialQRR11QIMConfig",
    "SpatialQRR11QIMKey",
    "ABLATION_FLAGS",
    "HYPERPARAMETER_FLAGS",
    "apply_proposal_flags",
    "DCT_QR",
    "DCT_QR_DIRECT_R",
    "DCT_QR_R11_QIM",
    "DCT_QR_THEORY",
    "DCT_SCHUR_RESCUE",
    "SPATIAL_CD_DETQR",
    "SPATIAL_QR",
    "SPATIAL_QR_DIRECT_R",
    "SPATIAL_QR_R11_QIM",
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
