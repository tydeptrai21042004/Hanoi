from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class QR64Config:
    """Parameters of the QR-certified blind watermarking model.

    The model separates three mathematical roles:

    1. ``step`` and ``rho_frac`` define the QIM decision regions;
    2. the QR certificate estimates local carrier stability;
    3. the spatial prior regularizes uncertain extracted bits.

    When ``adaptive_step_enabled`` is true, QR reliability controls the local
    QIM step.  The ratios are relative to ``step`` so that the same scientific
    rule remains meaningful when the global strength is changed.
    """

    seed: int = 2026
    step: float = 8.25
    rho_frac: float = 0.45
    pilot_count: int = 128
    pilot_step: float = 10.0
    pilot_rho_frac: float = 0.49
    eta: float = 0.07

    certificate_mode: str = "qr"
    qr_lift: float = 1.0
    qr_det_epsilon: float = 1e-12

    # Reliability-conditioned strength allocation.  For step=16 these ratios
    # give the validated levels (20.5, 16.0, 13.0).
    adaptive_step_enabled: bool = True
    adaptive_step_ratios: tuple[float, float, float] = (1.28125, 1.0, 0.8125)
    adaptive_step_fractions: tuple[float, float] = (0.20, 0.60)
    exact_confidence_gate: float = 0.79

    qr_map_lambda: float = 0.33
    qr_map_iters: int = 12
    evidence_conf_floor: float = 0.15
    evidence_conf_scale: float = 1.40
    evidence_conf_power: float = 2.0
    evidence_certificate_floor: float = 0.70
    evidence_certificate_scale: float = 0.30

    sync_quick_accept: float = 0.72
    sync_improvement_threshold: float = 0.05
    sync_certificate_weight: float = 0.08
    sync_geometry_penalty: float = 0.04
    translation_radius: int = 8
    translation_subset: int = 96

    watermark_size: int = 64

    def validated(self) -> "QR64Config":
        if self.watermark_size != 64:
            raise ValueError("The protocol keeps the input watermark at 64x64.")
        if self.step <= 0 or self.pilot_step <= 0:
            raise ValueError("QIM steps must be positive.")
        if not 0 <= self.rho_frac < 0.5:
            raise ValueError("rho_frac must be in [0, 0.5).")
        if str(self.certificate_mode).lower() not in {"qr", "schur"}:
            raise ValueError("certificate_mode must be 'qr' or 'schur'.")
        if self.qr_lift <= 0:
            raise ValueError("qr_lift must be positive.")
        if self.qr_det_epsilon <= 0:
            raise ValueError("qr_det_epsilon must be positive.")

        ratios = tuple(float(x) for x in self.adaptive_step_ratios)
        fractions = tuple(float(x) for x in self.adaptive_step_fractions)
        if len(ratios) != 3 or any(x <= 0 for x in ratios):
            raise ValueError("adaptive_step_ratios must contain three positive values.")
        if len(fractions) != 2 or not 0 < fractions[0] < fractions[1] < 1:
            raise ValueError(
                "adaptive_step_fractions must contain two increasing values in (0,1)."
            )
        if not 0 <= self.exact_confidence_gate <= 1:
            raise ValueError("exact_confidence_gate must be in [0,1].")

        if self.qr_map_iters < 0:
            raise ValueError("qr_map_iters cannot be negative.")
        if self.evidence_conf_power <= 0:
            raise ValueError("evidence_conf_power must be positive.")
        if self.sync_improvement_threshold < 0:
            raise ValueError("sync_improvement_threshold cannot be negative.")
        if self.sync_certificate_weight < 0 or self.sync_geometry_penalty < 0:
            raise ValueError("Synchronization weights cannot be negative.")
        return self

    def adaptive_step_levels(self) -> tuple[float, float, float]:
        """Return the three local QIM steps implied by the global reference step."""
        return tuple(float(self.step) * float(r) for r in self.adaptive_step_ratios)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "QR64Config":
        if value is None:
            return cls().validated()
        allowed = set(cls.__dataclass_fields__)
        raw = {k: v for k, v in dict(value).items() if k in allowed}
        if "adaptive_step_ratios" in raw:
            raw["adaptive_step_ratios"] = tuple(raw["adaptive_step_ratios"])
        if "adaptive_step_fractions" in raw:
            raw["adaptive_step_fractions"] = tuple(raw["adaptive_step_fractions"])
        return cls(**raw).validated()
