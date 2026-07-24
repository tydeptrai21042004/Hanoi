from __future__ import annotations

from qr64_certified import (
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    list_supported_methods,
)
from qr64_certified.proposals import (
    embed_proposal,
    extract_proposal,
)
from qr64_certified.proposals import method as explicit_dct_qr
from qr64_certified.proposals import direct_schur_rescue as explicit_schur
from qr64_certified.proposals import cd_detqr as explicit_spatial
from qr64_certified import method as compatibility_dct_qr
from qr64_certified import direct_schur_rescue as compatibility_schur
from qr64_certified import cd_detqr as compatibility_spatial


def test_exactly_three_proposals_are_visible() -> None:
    assert [row["id"] for row in list_supported_methods()] == [
        DCT_QR,
        DCT_SCHUR_RESCUE,
        SPATIAL_CD_DETQR,
    ]


def test_explicit_and_compatibility_imports_share_implementations() -> None:
    assert compatibility_dct_qr.embed is explicit_dct_qr.embed
    assert compatibility_dct_qr.extract is explicit_dct_qr.extract
    assert compatibility_schur.embed is explicit_schur.embed
    assert compatibility_schur.extract is explicit_schur.extract
    assert compatibility_spatial.embed_cd_detqr is explicit_spatial.embed_cd_detqr
    assert compatibility_spatial.extract_cd_detqr is explicit_spatial.extract_cd_detqr


def test_public_proposal_api_is_callable() -> None:
    assert callable(embed_proposal)
    assert callable(extract_proposal)
