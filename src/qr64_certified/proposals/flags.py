from __future__ import annotations

"""CLI/config support for proposal ablations and hyperparameter overrides.

The flags in this module are deliberately mapped to explicit scientific
components.  They are usable from command-line scripts and from Python tests,
so an ablation is reproducible without editing source code.
"""

import argparse
from dataclasses import replace
from typing import Any, Mapping

from .cd_detqr import CDDetQRConfig
from .config import QR64Config
from .direct_schur_rescue import DirectSchurRescueConfig
from .proposal_registry import DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR


ABLATION_FLAGS: dict[str, tuple[str, ...]] = {
    DCT_QR: (
        "no_coset_optimization",
        "uniform_step",
        "no_gain_normalization",
        "global_qr_gain",
        "no_spatial_map",
        "no_certificate_evidence",
        "no_sync_certificate",
    ),
    DCT_SCHUR_RESCUE: (
        "uniform_step",
        "no_gain_normalization",
        "no_schur_departure",
        "no_spatial_map",
        "no_certificate_evidence",
        "no_sync_certificate",
    ),
    SPATIAL_CD_DETQR: (
        "fixed_payload_pattern",
        "no_boundary_penalty",
        "minimal_det_safety",
        "no_payload_mask",
        "no_affine_sync",
        "single_closure",
    ),
}

HYPERPARAMETER_FLAGS: dict[str, tuple[str, ...]] = {
    DCT_QR: (
        "step", "rho_frac", "pilot_count", "pilot_step", "pilot_rho_frac",
        "eta", "qr_lift", "adaptive_step_enabled", "adaptive_step_ratios",
        "adaptive_step_fractions", "exact_confidence_gate",
        "coset_optimization_enabled", "coset_group_size",
        "gain_normalization_enabled", "gain_gamma", "gain_clip",
        "qr_gain_mode", "qr_map_lambda", "qr_map_iters",
        "evidence_conf_power", "sync_certificate_weight",
        "sync_improvement_threshold",
    ),
    DCT_SCHUR_RESCUE: (
        "step", "rho_frac", "pilot_count", "pilot_step", "pilot_rho_frac",
        "eta", "qr_lift", "adaptive_step_enabled", "adaptive_step_ratios",
        "adaptive_step_fractions", "exact_confidence_gate",
        "gain_normalization_enabled", "gain_gamma", "gain_clip",
        "schur_departure_weight", "qr_map_lambda", "qr_map_iters",
        "evidence_conf_power", "sync_certificate_weight",
        "sync_improvement_threshold", "schur_step", "schur_lift",
        "schur_max_log_scale", "schur_closure_iters", "fusion_weight",
        "gate_power", "schur_conf_floor", "schur_conf_scale",
        "agreement_bonus", "direct_det_epsilon",
    ),
    SPATIAL_CD_DETQR: (
        "target_margin", "boundary_penalty", "max_patterns",
        "payload_pattern_search_enabled", "fixed_payload_pattern",
        "determinant_floor", "embedded_determinant_margin",
        "determinant_safety_enabled", "mask_seed", "payload_mask_enabled",
        "pilot_seed", "pilot_count", "pilot_margin",
        "affine_sync_enabled", "sync_acceptance_gain",
        "sync_min_pilot_score", "rotation_grid", "shear_grid",
        "closure_rounds",
    ),
}


def _csv_floats(expected: int):
    def parse(value: str) -> tuple[float, ...]:
        try:
            result = tuple(float(part.strip()) for part in value.split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected {expected} comma-separated real values."
            ) from exc
        if len(result) != expected:
            raise argparse.ArgumentTypeError(
                f"Expected {expected} comma-separated values; received {len(result)}."
            )
        return result

    return parse


def _csv_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated real values.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return result


def add_proposal_flag_arguments(parser: argparse.ArgumentParser) -> None:
    """Add repeatable ablations and explicit hyperparameter overrides."""

    ablation = parser.add_argument_group("scientific ablation flags")
    ablation.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Repeatable component ablation. Run scripts/list_proposal_flags.py "
            "to see method-specific names."
        ),
    )

    common = parser.add_argument_group("DCT-QIM / decomposition hyperparameters")
    common.add_argument("--step", type=float, default=None)
    common.add_argument("--rho-frac", type=float, default=None)
    common.add_argument("--pilot-count", type=int, default=None)
    common.add_argument("--pilot-step", type=float, default=None)
    common.add_argument("--pilot-rho-frac", type=float, default=None)
    common.add_argument("--eta", type=float, default=None)
    common.add_argument("--qr-lift", type=float, default=None)
    common.add_argument(
        "--adaptive-step",
        dest="adaptive_step_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    common.add_argument(
        "--adaptive-step-ratios",
        type=_csv_floats(3),
        default=None,
        metavar="WEAK,MID,STRONG",
    )
    common.add_argument(
        "--adaptive-step-fractions",
        type=_csv_floats(2),
        default=None,
        metavar="Q1,Q2",
    )
    common.add_argument("--exact-confidence-gate", type=float, default=None)
    common.add_argument(
        "--coset-optimization",
        dest="coset_optimization_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    common.add_argument("--coset-group-size", type=int, default=None)
    common.add_argument(
        "--gain-normalization",
        dest="gain_normalization_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    common.add_argument("--gain-gamma", type=float, default=None)
    common.add_argument(
        "--gain-clip",
        type=_csv_floats(2),
        default=None,
        metavar="LOW,HIGH",
    )
    common.add_argument("--qr-map-lambda", type=float, default=None)
    common.add_argument("--qr-map-iters", type=int, default=None)
    common.add_argument("--evidence-conf-power", type=float, default=None)
    common.add_argument("--sync-certificate-weight", type=float, default=None)
    common.add_argument("--sync-improvement-threshold", type=float, default=None)

    qr = parser.add_argument_group("DCT-QR hyperparameters")
    qr.add_argument(
        "--qr-gain-mode",
        choices=("carrier_r11", "diag_l2"),
        default=None,
    )

    schur = parser.add_argument_group("DCT-Schur hyperparameters")
    schur.add_argument("--schur-departure-weight", type=float, default=None)
    schur.add_argument("--schur-step", type=float, default=None)
    schur.add_argument("--schur-lift", type=float, default=None)
    schur.add_argument("--schur-max-log-scale", type=float, default=None)
    schur.add_argument("--schur-closure-iters", type=int, default=None)
    schur.add_argument("--fusion-weight", type=float, default=None)
    schur.add_argument("--gate-power", type=float, default=None)
    schur.add_argument("--schur-conf-floor", type=float, default=None)
    schur.add_argument("--schur-conf-scale", type=float, default=None)
    schur.add_argument("--agreement-bonus", type=float, default=None)
    schur.add_argument("--direct-det-epsilon", type=float, default=None)

    spatial = parser.add_argument_group("Spatial CD-DetQR hyperparameters")
    spatial.add_argument("--target-margin", type=float, default=None)
    spatial.add_argument("--boundary-penalty", type=float, default=None)
    spatial.add_argument("--max-patterns", type=int, default=None)
    spatial.add_argument("--fixed-payload-pattern", type=int, default=None)
    spatial.add_argument("--determinant-floor", type=float, default=None)
    spatial.add_argument("--embedded-determinant-margin", type=float, default=None)
    spatial.add_argument("--mask-seed", type=int, default=None)
    spatial.add_argument("--pilot-seed", type=int, default=None)
    spatial.add_argument("--pilot-margin", type=float, default=None)
    spatial.add_argument("--sync-acceptance-gain", type=float, default=None)
    spatial.add_argument("--sync-min-pilot-score", type=float, default=None)
    spatial.add_argument("--closure-rounds", type=int, default=None)
    spatial.add_argument("--rotation-grid", type=_csv_float_list, default=None)
    spatial.add_argument("--shear-grid", type=_csv_float_list, default=None)
    spatial.add_argument(
        "--payload-pattern-search",
        dest="payload_pattern_search_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    spatial.add_argument(
        "--determinant-safety",
        dest="determinant_safety_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    spatial.add_argument(
        "--payload-mask",
        dest="payload_mask_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    spatial.add_argument(
        "--affine-sync",
        dest="affine_sync_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _present(values: Mapping[str, Any], name: str) -> bool:
    return name in values and values[name] is not None


def _apply_qr_fields(config: QR64Config, values: Mapping[str, Any]) -> tuple[QR64Config, dict[str, Any]]:
    names = (
        "step",
        "rho_frac",
        "pilot_count",
        "pilot_step",
        "pilot_rho_frac",
        "eta",
        "qr_lift",
        "adaptive_step_enabled",
        "adaptive_step_ratios",
        "adaptive_step_fractions",
        "exact_confidence_gate",
        "coset_optimization_enabled",
        "coset_group_size",
        "gain_normalization_enabled",
        "gain_gamma",
        "gain_clip",
        "qr_map_lambda",
        "qr_map_iters",
        "evidence_conf_power",
        "sync_certificate_weight",
        "sync_improvement_threshold",
        "qr_gain_mode",
        "schur_departure_weight",
    )
    updates = {name: values[name] for name in names if _present(values, name)}
    if updates:
        config = replace(config, **updates)
    return config.validated(), updates


def apply_proposal_flags(
    method: str,
    config: QR64Config | DirectSchurRescueConfig | CDDetQRConfig,
    values: Mapping[str, Any] | argparse.Namespace,
) -> tuple[QR64Config | DirectSchurRescueConfig | CDDetQRConfig, dict[str, Any]]:
    """Apply method-specific ablations and hyperparameters to a config.

    Returns the validated config and a serializable report describing every
    activated ablation and direct override.
    """

    raw = vars(values) if isinstance(values, argparse.Namespace) else dict(values)
    requested = tuple(dict.fromkeys(str(x) for x in raw.get("ablation", []) if x))
    valid = set(ABLATION_FLAGS[method])
    invalid = [name for name in requested if name not in valid]
    if invalid:
        raise ValueError(
            f"Invalid ablation(s) for {method}: {', '.join(invalid)}. "
            f"Valid: {', '.join(ABLATION_FLAGS[method])}"
        )

    overrides: dict[str, Any] = {}

    if method == DCT_QR:
        if not isinstance(config, QR64Config):
            config = QR64Config.from_mapping(config)  # type: ignore[arg-type]
        config, direct = _apply_qr_fields(config, raw)
        overrides.update(direct)
        for name in requested:
            if name == "no_coset_optimization":
                config = replace(config, coset_optimization_enabled=False)
            elif name == "uniform_step":
                config = replace(
                    config,
                    adaptive_step_enabled=False,
                    coset_optimization_enabled=False,
                )
            elif name == "no_gain_normalization":
                config = replace(config, gain_normalization_enabled=False)
            elif name == "global_qr_gain":
                config = replace(config, qr_gain_mode="diag_l2")
            elif name == "no_spatial_map":
                config = replace(config, qr_map_lambda=0.0, qr_map_iters=0)
            elif name == "no_certificate_evidence":
                config = replace(
                    config,
                    evidence_certificate_floor=1.0,
                    evidence_certificate_scale=0.0,
                )
            elif name == "no_sync_certificate":
                config = replace(config, sync_certificate_weight=0.0)
        config = config.validated()

    elif method == DCT_SCHUR_RESCUE:
        if not isinstance(config, DirectSchurRescueConfig):
            config = DirectSchurRescueConfig.from_mapping(config)  # type: ignore[arg-type]
        base, direct_base = _apply_qr_fields(config.base_config, raw)
        overrides.update({f"base_config.{k}": v for k, v in direct_base.items()})
        outer_names = (
            "schur_step",
            "schur_lift",
            "schur_max_log_scale",
            "schur_closure_iters",
            "fusion_weight",
            "gate_power",
            "schur_conf_floor",
            "schur_conf_scale",
            "agreement_bonus",
            "direct_det_epsilon",
        )
        outer_updates = {
            name: raw[name] for name in outer_names if _present(raw, name)
        }
        overrides.update(outer_updates)
        config = replace(config, base_config=replace(base, certificate_mode="schur"), **outer_updates)
        for name in requested:
            if name == "uniform_step":
                config = replace(
                    config,
                    base_config=replace(config.base_config, adaptive_step_enabled=False),
                )
            elif name == "no_gain_normalization":
                config = replace(
                    config,
                    base_config=replace(
                        config.base_config, gain_normalization_enabled=False
                    ),
                )
            elif name == "no_schur_departure":
                config = replace(
                    config,
                    base_config=replace(
                        config.base_config, schur_departure_weight=0.0
                    ),
                )
            elif name == "no_spatial_map":
                config = replace(
                    config,
                    base_config=replace(
                        config.base_config, qr_map_lambda=0.0, qr_map_iters=0
                    ),
                )
            elif name == "no_certificate_evidence":
                config = replace(
                    config,
                    base_config=replace(
                        config.base_config,
                        evidence_certificate_floor=1.0,
                        evidence_certificate_scale=0.0,
                    ),
                )
            elif name == "no_sync_certificate":
                config = replace(
                    config,
                    base_config=replace(
                        config.base_config, sync_certificate_weight=0.0
                    ),
                )
        config = config.validated()

    else:
        if not isinstance(config, CDDetQRConfig):
            allowed = set(CDDetQRConfig.__dataclass_fields__)
            config = CDDetQRConfig(
                **{k: v for k, v in dict(config).items() if k in allowed}  # type: ignore[arg-type]
            )
        names = (
            "target_margin",
            "boundary_penalty",
            "max_patterns",
            "fixed_payload_pattern",
            "determinant_floor",
            "embedded_determinant_margin",
            "mask_seed",
            "pilot_seed",
            "pilot_count",
            "pilot_margin",
            "sync_acceptance_gain",
            "sync_min_pilot_score",
            "closure_rounds",
            "rotation_grid",
            "shear_grid",
            "payload_pattern_search_enabled",
            "determinant_safety_enabled",
            "payload_mask_enabled",
            "affine_sync_enabled",
        )
        updates = {name: raw[name] for name in names if _present(raw, name)}
        overrides.update(updates)
        if updates:
            config = replace(config, **updates)
        for name in requested:
            if name == "fixed_payload_pattern":
                config = replace(config, payload_pattern_search_enabled=False)
            elif name == "no_boundary_penalty":
                config = replace(config, boundary_penalty=0.0)
            elif name == "minimal_det_safety":
                config = replace(config, determinant_safety_enabled=False)
            elif name == "no_payload_mask":
                config = replace(config, payload_mask_enabled=False)
            elif name == "no_affine_sync":
                config = replace(config, affine_sync_enabled=False)
            elif name == "single_closure":
                config = replace(config, closure_rounds=1)
        config.validate()

    return config, {
        "method_id": method,
        "ablations": list(requested),
        "hyperparameter_overrides": overrides,
    }


__all__ = [
    "ABLATION_FLAGS",
    "HYPERPARAMETER_FLAGS",
    "add_proposal_flag_arguments",
    "apply_proposal_flags",
]
