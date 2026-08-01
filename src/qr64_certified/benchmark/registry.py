from __future__ import annotations

from qr64_certified.baselines.registry import (
    ALL_BASELINE_IDS,
    PRIMARY_BLIND_BASELINE_IDS,
    get_baseline_spec,
    normalize_baseline_id,
)
from qr64_certified.proposals.proposal_registry import (
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    SUPPORTED_PROPOSAL_METHODS,
    normalize_proposal_method_id,
)

from .types import BenchmarkMethodSpec

PROPOSAL_IDS = (DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR)


def proposal_spec(method_id: str) -> BenchmarkMethodSpec:
    canonical = normalize_proposal_method_id(method_id)
    metadata = SUPPORTED_PROPOSAL_METHODS[canonical]
    return BenchmarkMethodSpec(
        method_id=canonical,
        method_kind="proposal",
        display_name=str(metadata["display_name"]),
        blindness_tier="key_assisted_blind",
        fidelity_tier="proposal",
        requires_original_host=False,
        common_4096_payload=True,
        payload_size=64,
        cover_dependent_key=True,
        comparison_group="proposal_key_assisted_4096",
        config_path=f"configs/{canonical}_after_abc.json",
    )


def baseline_spec(method_id: str) -> BenchmarkMethodSpec:
    native = get_baseline_spec(normalize_baseline_id(method_id))
    payload_size = 64 if native.common_4096_payload else 16
    return BenchmarkMethodSpec(
        method_id=native.canonical_id,
        method_kind="baseline",
        display_name=native.display_name,
        blindness_tier=native.blindness_tier,
        fidelity_tier=native.fidelity_tier,
        requires_original_host=native.requires_original_host,
        common_4096_payload=native.common_4096_payload,
        payload_size=payload_size,
        cover_dependent_key=native.blindness_tier != "blind",
        comparison_group=f"baseline_{native.blindness_tier}_{payload_size}x{payload_size}",
    )


def get_method_spec(method_id: str) -> BenchmarkMethodSpec:
    normalized = str(method_id).strip().lower()
    try:
        return proposal_spec(normalized)
    except KeyError:
        return baseline_spec(normalized)


def all_method_specs() -> tuple[BenchmarkMethodSpec, ...]:
    return tuple(proposal_spec(v) for v in PROPOSAL_IDS) + tuple(baseline_spec(v) for v in ALL_BASELINE_IDS)


def _selector_ids(selector: str) -> tuple[str, ...]:
    key = selector.strip().lower()
    if key == "all":
        return PROPOSAL_IDS + tuple(ALL_BASELINE_IDS)
    if key in {"proposal", "proposals"}:
        return PROPOSAL_IDS
    if key in {"baseline", "baselines"}:
        return tuple(ALL_BASELINE_IDS)
    if key in {"primary_blind", "primary_blind_baselines"}:
        return tuple(PRIMARY_BLIND_BASELINE_IDS)
    if key in {"paper", "paper_comparison", "final"}:
        return PROPOSAL_IDS + tuple(PRIMARY_BLIND_BASELINE_IDS)
    specs = all_method_specs()
    if key in {"strict_blind", "blind"}:
        return tuple(s.method_id for s in specs if s.blindness_tier == "blind")
    if key in {"key_assisted", "key_assisted_blind"}:
        return tuple(s.method_id for s in specs if s.blindness_tier == "key_assisted_blind")
    if key in {"semi_blind", "semiblind"}:
        return tuple(s.method_id for s in specs if s.blindness_tier == "semi_blind")
    if key in {"non_blind", "nonblind"}:
        return tuple(s.method_id for s in specs if s.blindness_tier == "non_blind")
    if key in {"common_4096", "payload_4096"}:
        return tuple(s.method_id for s in specs if s.common_4096_payload)
    return tuple(part.strip() for part in selector.split(",") if part.strip())


def resolve_methods(value: str | tuple[str, ...] | list[str]) -> tuple[BenchmarkMethodSpec, ...]:
    raw_ids: tuple[str, ...]
    if isinstance(value, str):
        raw_ids = _selector_ids(value)
    else:
        raw_ids = tuple(str(v) for v in value)
    output: list[BenchmarkMethodSpec] = []
    seen: set[str] = set()
    for method_id in raw_ids:
        spec = get_method_spec(method_id)
        if spec.method_id not in seen:
            output.append(spec)
            seen.add(spec.method_id)
    if not output:
        raise ValueError("The method selection is empty")
    return tuple(output)


__all__ = [
    "PROPOSAL_IDS", "proposal_spec", "baseline_spec", "get_method_spec",
    "all_method_specs", "resolve_methods",
]
