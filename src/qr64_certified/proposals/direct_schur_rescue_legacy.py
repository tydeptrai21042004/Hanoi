from __future__ import annotations

"""Direct-Schur rescue carrier for the certified blind 64x64 proposal.

This module studies an exploratory secondary-channel hypothesis.  The primary
DCT-QIM channel is kept as a controlled reference, while the Schur component
asks whether departure from normality can supply independent evidence for weak
primary decisions.  The repository does not claim that this rescue channel is
validated: its scientific acceptance criterion is independent clean accuracy
of at least 0.99 before fusion.

The direct carrier quantizes a normalized Henrici departure-from-normality
quantity.  Embedding scales only the strict block-upper part of a real Schur
form while keeping its 1x1/2x2 diagonal blocks fixed.  Consequently, the
floating-point update preserves the spectrum, trace, and determinant up to
numerical precision.  Pixel rounding is checked for nonsingularity rather than
claimed to preserve the determinant exactly.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg import schur

from qr64_certified.common.core import arnold_transform
from qr64_certified._jilp_core import engine as _eng
from qr64_certified._jilp_core.method import (
    PAY10_BA,
    PAY10_BB,
    _call_with_realtime_thread_budget,
    _extract_payload_bits_confidence,
    _normalize_jilp_key,
    _schedules_from_key,
)

from .certificate import compute_schur_certificate, opponent_field
from .config import QR64Config
from .method import (
    QR64Key,
    _certified_alignment,
    _icm_map,
    _inverse_arnold_array,
    embed as _certified_embed,
)

METHOD_ID = "direct_schur_rescue"
DISPLAY_NAME = "Direct-Schur Rescue Carrier (DCT-QIM primary + Schur secondary)"

# Sixteen low/mid-frequency AC coefficients.  The primary/pilot pair
# (0,1)/(1,0) is excluded, so the direct Schur carrier is disjoint from the
# primary carrier in the floating-point DCT model.
SCHUR_POSITIONS = np.array(
    [
        (0, 2), (2, 0), (1, 1), (0, 3),
        (3, 0), (1, 2), (2, 1), (0, 4),
        (4, 0), (1, 3), (3, 1), (2, 2),
        (0, 5), (5, 0), (1, 4), (4, 1),
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class DirectSchurRescueConfig:
    """Configuration of the exploratory invariant-preserving Schur hypothesis."""

    # Adaptive QR strength is disabled here to isolate the effect of the Schur
    # secondary channel from changes in the primary DCT reference model.
    base_config: QR64Config = field(
        default_factory=lambda: QR64Config(
            certificate_mode="schur", adaptive_step_enabled=False
        )
    )
    schur_step: float = 0.04
    schur_lift: float = 1.0
    schur_max_log_scale: float = 1.0
    schur_closure_iters: int = 1
    fusion_weight: float = 0.20
    gate_power: float = 1.0
    schur_conf_floor: float = 0.10
    schur_conf_scale: float = 0.90
    agreement_bonus: float = 0.0
    direct_det_epsilon: float = 1e-12

    def validated(self) -> "DirectSchurRescueConfig":
        base = self.base_config.validated()
        if str(base.certificate_mode).lower() != "schur":
            raise ValueError(
                "Direct-Schur Rescue requires base_config.certificate_mode='schur'."
            )
        if self.schur_step <= 0:
            raise ValueError("schur_step must be positive.")
        if self.schur_lift <= 0:
            raise ValueError("schur_lift must be positive.")
        if self.schur_max_log_scale <= 0:
            raise ValueError("schur_max_log_scale must be positive.")
        if self.schur_closure_iters < 1:
            raise ValueError("schur_closure_iters must be at least 1.")
        if self.fusion_weight < 0 or self.gate_power <= 0:
            raise ValueError("fusion_weight must be nonnegative and gate_power positive.")
        if self.direct_det_epsilon <= 0:
            raise ValueError("direct_det_epsilon must be positive.")
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_config"] = self.base_config.to_dict()
        data["method_id"] = METHOD_ID
        return data

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "DirectSchurRescueConfig" | None
    ) -> "DirectSchurRescueConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        if isinstance(value, QR64Config):
            from dataclasses import replace
            return cls(base_config=replace(value, certificate_mode="schur")).validated()
        raw = dict(value)
        base_raw = raw.pop("base_config", None)
        if base_raw is None:
            # Accept flat QR64 fields for convenience while keeping rescue fields.
            qr_fields = set(QR64Config.__dataclass_fields__)
            qr_values = {k: raw.pop(k) for k in list(raw) if k in qr_fields}
            qr_values["certificate_mode"] = "schur"
            base = QR64Config.from_mapping(qr_values)
        elif isinstance(base_raw, QR64Config):
            base = base_raw
        else:
            base_values = dict(base_raw)
            base_values["certificate_mode"] = "schur"
            base = QR64Config.from_mapping(base_values)
        allowed = set(cls.__dataclass_fields__) - {"base_config"}
        return cls(base_config=base, **{k: v for k, v in raw.items() if k in allowed}).validated()


@dataclass
class DirectSchurRescueKey:
    """Blind extraction key for the hybrid method.

    It contains the existing certified key and compact method parameters.  It
    does not contain the original image or original watermark.
    """

    base: QR64Key
    config: dict[str, Any]
    schur_params: dict[str, Any]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def method_id(self) -> str:
        return METHOD_ID

    @property
    def certificate_mode(self) -> str:
        return "schur"

    @property
    def qr_summary(self) -> dict[str, Any]:
        """Compatibility alias used by existing benchmark readers."""
        return self.base.qr_summary

    @property
    def certificate_summary(self) -> dict[str, Any]:
        return self.base.qr_summary


# ---------------------------------------------------------------------------
# Direct Schur carrier construction
# ---------------------------------------------------------------------------


def _blocks_from_field(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    if h % 8 or w % 8:
        raise ValueError("Image dimensions must be divisible by 8.")
    return field.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def _field_from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    return blocks.reshape(h // 8, w // 8, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)


def _matrices_from_coeff(coeff: np.ndarray, lift: float) -> np.ndarray:
    values = coeff[:, SCHUR_POSITIONS[:, 0], SCHUR_POSITIONS[:, 1]]
    matrices = values.reshape(-1, 4, 4).copy()
    matrices += float(lift) * np.eye(4, dtype=np.float64)[None, :, :]
    return matrices


def _write_matrices_to_coeff(
    coeff: np.ndarray,
    matrices: np.ndarray,
    lift: float,
    indices: np.ndarray | None = None,
) -> None:
    if indices is None:
        indices = np.arange(matrices.shape[0], dtype=np.int32)
        selected = matrices
    else:
        indices = np.asarray(indices, dtype=np.int32)
        selected = matrices[indices]
    values = selected.copy()
    values -= float(lift) * np.eye(4, dtype=np.float64)[None, :, :]
    flat = values.reshape(-1, 16)
    for j, (u, v) in enumerate(SCHUR_POSITIONS):
        coeff[indices, int(u), int(v)] = flat[:, j]


def _real_schur_split(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Return real Schur T, Z and its strict block-upper component N.

    The real Schur form contains 1x1 and 2x2 diagonal blocks.  ``N`` excludes
    these blocks.  Scaling N therefore leaves all diagonal blocks unchanged.
    """
    t, z = schur(np.asarray(matrix, dtype=np.float64), output="real", check_finite=False)
    diagonal_blocks = np.zeros_like(t)
    tolerance = 1e-10 * max(float(np.linalg.norm(t)), 1.0)
    i = 0
    n = t.shape[0]
    while i < n:
        if i + 1 < n and abs(float(t[i + 1, i])) > tolerance:
            diagonal_blocks[i : i + 2, i : i + 2] = t[i : i + 2, i : i + 2]
            i += 2
        else:
            diagonal_blocks[i, i] = t[i, i]
            i += 1

    strict_block_upper = t - diagonal_blocks
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalue_energy = float(np.sum(np.abs(eigenvalues) ** 2))
    diagonal_energy = float(np.sum(diagonal_blocks * diagonal_blocks))
    intra_block_departure_sq = max(diagonal_energy - eigenvalue_energy, 0.0)
    upper_energy_sq = float(np.sum(strict_block_upper * strict_block_upper))
    scale = math.sqrt(max(eigenvalue_energy, 0.0)) + 1.0
    return (
        t,
        z,
        strict_block_upper,
        intra_block_departure_sq,
        upper_energy_sq,
        scale,
    )


def _nearest_parity_level(
    value: float, step: float, bit: int, minimum: float = 0.0
) -> float:
    """Nearest nonnegative QIM lattice point with the requested parity."""
    unit = max(float(value) / float(step), 0.0)
    center = int(np.rint(unit))
    candidates: list[int] = []
    for q in range(max(0, center - 4), center + 6):
        if (q & 1) == int(bit) and q * step >= minimum - 1e-15:
            candidates.append(q)
    if not candidates:
        q = max(0, int(math.ceil(minimum / step)))
        if (q & 1) != int(bit):
            q += 1
        return float(q * step)
    return float(min(candidates, key=lambda q: abs(q - unit)) * step)


def _embed_schur_coefficients(
    coeff: np.ndarray,
    bits_by_block: np.ndarray,
    *,
    step: float,
    lift: float,
    max_log_scale: float,
    only_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    matrices = _matrices_from_coeff(coeff, lift)
    indices = (
        np.arange(matrices.shape[0], dtype=np.int32)
        if only_indices is None
        else np.asarray(only_indices, dtype=np.int32)
    )
    output = matrices.copy()
    det_before = np.abs(np.linalg.det(matrices[indices]))

    relative_det_errors: list[float] = []
    spectrum_errors: list[float] = []
    infeasible = 0
    clipped = 0
    alphas: list[float] = []
    carrier_shifts: list[float] = []

    for local_index, block_index in enumerate(indices):
        block_index = int(block_index)
        matrix = matrices[block_index]
        t, z, nmat, intra_sq, n_sq, scale = _real_schur_split(matrix)
        carrier = math.sqrt(max(intra_sq + n_sq, 0.0)) / scale
        minimum_carrier = math.sqrt(max(intra_sq, 0.0)) / scale
        target = _nearest_parity_level(
            carrier, step, int(bits_by_block[block_index]), minimum_carrier
        )
        target_departure_sq = (target * scale) ** 2

        if n_sq <= 1e-18:
            # (0,3) lies outside every possible 1x1/2x2 diagonal block in 4x4.
            nmat = np.zeros_like(t)
            nmat[0, 3] = 1.0
            n_sq = 1.0

        numerator = target_departure_sq - intra_sq
        if numerator < -1e-10:
            infeasible += 1
            target = _nearest_parity_level(
                carrier,
                step,
                int(bits_by_block[block_index]),
                minimum_carrier + step,
            )
            numerator = max((target * scale) ** 2 - intra_sq, 0.0)

        alpha = math.sqrt(max(numerator, 0.0) / max(n_sq, 1e-18))
        lower = math.exp(-float(max_log_scale))
        upper = math.exp(float(max_log_scale))
        clipped_alpha = min(max(alpha, lower), upper)
        if abs(clipped_alpha - alpha) > 1e-12:
            clipped += 1
        alpha = clipped_alpha

        diagonal_blocks = t - nmat
        updated_t = diagonal_blocks + alpha * nmat
        updated_matrix = np.real_if_close(z @ updated_t @ z.T, tol=1000).real
        output[block_index] = updated_matrix

        det_after = abs(float(np.linalg.det(updated_matrix)))
        relative_det_errors.append(
            abs(det_after - float(det_before[local_index]))
            / (float(det_before[local_index]) + 1e-18)
        )
        eig_before = np.sort_complex(np.linalg.eigvals(matrix))
        eig_after = np.sort_complex(np.linalg.eigvals(updated_matrix))
        spectrum_errors.append(float(np.max(np.abs(eig_after - eig_before))))

        eig_energy_after = float(np.sum(np.abs(eig_after) ** 2))
        departure_after = math.sqrt(
            max(float(np.sum(updated_matrix * updated_matrix)) - eig_energy_after, 0.0)
        )
        carrier_after = departure_after / (math.sqrt(max(eig_energy_after, 0.0)) + 1.0)
        carrier_shifts.append(abs(carrier_after - carrier))
        alphas.append(alpha)

    _write_matrices_to_coeff(coeff, output, lift, indices)
    return {
        "max_relative_det_error_float": float(max(relative_det_errors, default=0.0)),
        "mean_relative_det_error_float": float(np.mean(relative_det_errors) if relative_det_errors else 0.0),
        "max_spectrum_error_float": float(max(spectrum_errors, default=0.0)),
        "infeasible_count": int(infeasible),
        "scale_clipped_count": int(clipped),
        "mean_alpha": float(np.mean(alphas) if alphas else 1.0),
        "min_alpha": float(np.min(alphas) if alphas else 1.0),
        "max_alpha": float(np.max(alphas) if alphas else 1.0),
        "mean_carrier_shift": float(np.mean(carrier_shifts) if carrier_shifts else 0.0),
    }


def _schur_carrier_from_image(
    image: np.ndarray, *, eta: float, lift: float
) -> tuple[np.ndarray, np.ndarray]:
    field = opponent_field(image, eta)
    blocks = _blocks_from_field(field)
    coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    matrices = _matrices_from_coeff(coeff, lift)
    eigenvalues = np.linalg.eigvals(matrices)
    eigenvalue_energy = np.sum(np.abs(eigenvalues) ** 2, axis=1)
    frobenius_energy = np.sum(matrices * matrices, axis=(1, 2))
    departure = np.sqrt(np.maximum(frobenius_energy - eigenvalue_energy, 0.0))
    scale = np.sqrt(np.maximum(eigenvalue_energy, 0.0)) + 1.0
    carrier = departure / scale
    return carrier.astype(np.float64), matrices


def _extract_schur_bits_confidence(
    image: np.ndarray, *, eta: float, lift: float, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    carrier, _matrices = _schur_carrier_from_image(image, eta=eta, lift=lift)
    unit = carrier / float(step)
    lattice_index = np.rint(unit).astype(np.int64)
    bits = (lattice_index & 1).astype(np.uint8)
    confidence = np.clip(1.0 - 2.0 * np.abs(unit - lattice_index), 0.0, 1.0)
    return bits, confidence.astype(np.float64), carrier


def _apply_field_delta(
    base_rgb: np.ndarray, delta_field: np.ndarray, eta: float
) -> np.ndarray:
    projection = np.array(
        [0.299 + eta, 0.587 - 0.5 * eta, 0.114 - 0.5 * eta],
        dtype=np.float64,
    )
    weights = projection / float(np.dot(projection, projection))
    output = np.asarray(base_rgb, dtype=np.float64) + delta_field[..., None] * weights[None, None, :]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _validate_direct_matrices(
    image: np.ndarray, *, eta: float, lift: float, epsilon: float
) -> dict[str, Any]:
    _carrier, matrices = _schur_carrier_from_image(image, eta=eta, lift=lift)
    determinant = np.abs(np.linalg.det(matrices))
    return {
        "direct_schur_all_det_nonzero": bool(np.all(determinant > float(epsilon))),
        "direct_schur_min_abs_det": float(np.min(determinant)),
        "direct_schur_median_abs_det": float(np.median(determinant)),
    }


def _payload_bits_by_block(
    watermark_binary: np.ndarray, key_params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wm = (np.asarray(watermark_binary, dtype=np.uint8) > 0).astype(np.uint8)
    payload_indices, *_ = _schedules_from_key(key_params)
    bits_scrambled = arnold_transform(wm, int(key_params["arnold_iter"])).ravel().astype(np.uint8)
    bits_by_block = np.zeros(payload_indices.size, dtype=np.uint8)
    bits_by_block[payload_indices] = bits_scrambled
    return payload_indices.astype(np.int32), bits_scrambled, bits_by_block


def _add_direct_schur_carrier(
    primary_watermarked: np.ndarray,
    watermark_binary: np.ndarray,
    base_key: QR64Key,
    cfg: DirectSchurRescueConfig,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    params = _normalize_jilp_key(base_key.base_key)
    payload_indices, bits_scrambled, bits_by_block = _payload_bits_by_block(
        watermark_binary, params
    )
    eta = float(cfg.base_config.eta)
    current = np.asarray(primary_watermarked, dtype=np.uint8).copy()
    embedding_stats: list[dict[str, Any]] = []

    for iteration in range(int(cfg.schur_closure_iters)):
        field = opponent_field(current, eta)
        blocks = _blocks_from_field(field)
        coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
        selected: np.ndarray | None = None
        if iteration > 0:
            observed, _confidence, _carrier = _extract_schur_bits_confidence(
                current,
                eta=eta,
                lift=cfg.schur_lift,
                step=cfg.schur_step,
            )
            selected = np.flatnonzero(observed != bits_by_block)
            if selected.size == 0:
                break
        stats = _embed_schur_coefficients(
            coeff,
            bits_by_block,
            step=cfg.schur_step,
            lift=cfg.schur_lift,
            max_log_scale=cfg.schur_max_log_scale,
            only_indices=selected,
        )
        target_blocks = idctn(coeff, type=2, norm="ortho", axes=(-2, -1))
        target_field = _field_from_blocks(target_blocks, field.shape[0], field.shape[1])
        current = _apply_field_delta(current, target_field - field, eta)
        stats["iteration"] = int(iteration)
        stats["updated_blocks"] = int(
            coeff.shape[0] if selected is None else selected.size
        )
        embedding_stats.append(stats)

    # Adding the secondary carrier can cause a few uint8-rounding flips in the
    # primary carrier.  Re-close only the proven DCT lattice.  This is the same
    # deterministic integer-domain closure used by the existing proposal.
    execution_order = np.argsort(payload_indices, kind="stable")
    indices_execution = payload_indices[execution_order].astype(np.int32)
    bits_execution = bits_scrambled[execution_order].astype(np.uint8)
    primary_step = float(params.get("step_carrier1", params.get("step", 8.25)))
    primary_rho = primary_step * float(params.get("rho_frac", 0.45))
    updates, unresolved = _call_with_realtime_thread_budget(
        _eng._embed_one_carrier_jilp,
        current,
        indices_execution,
        bits_execution,
        PAY10_BA,
        PAY10_BB,
        primary_step,
        primary_rho,
        int(current.shape[1]),
        8,
        eta,
    )
    if int(unresolved) != 0:
        raise RuntimeError(
            f"Primary DCT closure failed for {int(unresolved)} payload blocks."
        )

    _bits, _positions, reference_confidence = _extract_payload_bits_confidence(
        current, params, 0, 0
    )
    direct_validation = _validate_direct_matrices(
        current,
        eta=eta,
        lift=cfg.schur_lift,
        epsilon=cfg.direct_det_epsilon,
    )
    if not direct_validation["direct_schur_all_det_nonzero"]:
        raise RuntimeError(
            "Direct Schur nonsingularity check failed after uint8 closure."
        )

    schur_params = {
        "method_id": METHOD_ID,
        "method": "spectrum-preserving Schur departure rescue QIM",
        "step": float(cfg.schur_step),
        "lift": float(cfg.schur_lift),
        "eta": eta,
        "max_log_scale": float(cfg.schur_max_log_scale),
        "closure_iters": int(cfg.schur_closure_iters),
        "fusion_weight": float(cfg.fusion_weight),
        "gate_power": float(cfg.gate_power),
        "schur_conf_floor": float(cfg.schur_conf_floor),
        "schur_conf_scale": float(cfg.schur_conf_scale),
        "agreement_bonus": float(cfg.agreement_bonus),
        "positions": SCHUR_POSITIONS.astype(int).tolist(),
        "payload_indices": payload_indices.astype(int).tolist(),
        "primary_conf_reference": float(np.mean(reference_confidence)),
        "fusion": "confidence-gated secondary Schur rescue evidence",
        "invariants_float": "Schur diagonal blocks, spectrum, trace, determinant",
        "scientific_status": "exploratory",
        "acceptance_criterion": "independent clean Schur accuracy >= 0.99 before fusion",
    }
    metadata = {
        "method_id": METHOD_ID,
        "embed_stats": embedding_stats,
        "primary_closure_updates": int(updates),
        "primary_closure_unresolved": int(unresolved),
        **direct_validation,
    }
    return current, schur_params, metadata


# ---------------------------------------------------------------------------
# Public embedding and extraction
# ---------------------------------------------------------------------------


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DirectSchurRescueConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    """Embed with the existing DCT-QIM carrier plus direct Schur rescue.

    Returns ``(watermarked, key)`` by default for compatibility.  Set
    ``return_metadata=True`` to also obtain determinant/spectrum diagnostics.
    """
    cfg = DirectSchurRescueConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm_input = np.asarray(watermark_binary, dtype=np.uint8)
    wm = (wm_input > 0).astype(np.uint8)
    # The existing JILP implementation thresholds at 127, so pass an explicit
    # {0,255} watermark to preserve the original primary DCT-QIM payload.
    wm_for_primary = wm * np.uint8(255)
    if host.shape != (512, 512, 3):
        raise ValueError(
            f"Direct-Schur Rescue currently requires a 512x512 RGB host; got {host.shape}."
        )
    if wm.shape != (64, 64):
        raise ValueError(
            f"Direct-Schur Rescue keeps the watermark at 64x64; got {wm.shape}."
        )

    primary_watermarked, base_key = _certified_embed(
        host, wm_for_primary, config=cfg.base_config
    )
    watermarked, schur_params, metadata = _add_direct_schur_carrier(
        primary_watermarked, wm, base_key, cfg
    )
    base_key.base_key.params["direct_schur_rescue"] = schur_params
    key = DirectSchurRescueKey(
        base=base_key,
        config=cfg.to_dict(),
        schur_params=schur_params,
    )
    return (watermarked, key, metadata) if return_metadata else (watermarked, key)


def extract_components(
    possibly_attacked_rgb: np.ndarray,
    key: DirectSchurRescueKey,
) -> dict[str, Any]:
    """Extract primary, rescue, and certificate evidence before MAP fusion."""
    cfg = DirectSchurRescueConfig.from_mapping(key.config)
    base_key = key.base
    params = _normalize_jilp_key(base_key.base_key)
    aligned, certificate, sync_metadata = _certified_alignment(
        np.asarray(possibly_attacked_rgb, dtype=np.uint8),
        params,
        base_key.qr_summary,
        cfg.base_config,
    )

    watermark_shape = tuple(int(x) for x in params["watermark_shape"])
    payload_len = int(np.prod(watermark_shape))

    primary_bits, positions, primary_confidence = _extract_payload_bits_confidence(
        aligned, params, 0, 0
    )
    primary_array = np.zeros(payload_len, dtype=np.uint8)
    primary_conf_array = np.zeros(payload_len, dtype=np.float64)
    primary_array[positions] = primary_bits
    primary_conf_array[positions] = primary_confidence
    primary_map = _inverse_arnold_array(
        primary_array.reshape(watermark_shape), params
    )
    primary_conf_map = _inverse_arnold_array(
        primary_conf_array.reshape(watermark_shape), params
    )

    schur_bits_all, schur_conf_all, schur_carrier = _extract_schur_bits_confidence(
        aligned,
        eta=float(key.schur_params["eta"]),
        lift=float(key.schur_params["lift"]),
        step=float(key.schur_params["step"]),
    )
    payload_indices = np.asarray(key.schur_params["payload_indices"], dtype=np.int32)
    schur_map = _inverse_arnold_array(
        schur_bits_all[payload_indices].reshape(watermark_shape), params
    )
    schur_conf_map = _inverse_arnold_array(
        schur_conf_all[payload_indices].reshape(watermark_shape), params
    )

    certificate_scrambled = certificate.reliability[payload_indices].reshape(
        watermark_shape
    )
    certificate_map = _inverse_arnold_array(certificate_scrambled, params)
    return {
        "primary_map": primary_map,
        "primary_confidence": primary_conf_map,
        "schur_map": schur_map,
        "schur_confidence": schur_conf_map,
        "certificate_map": certificate_map,
        "schur_carrier": schur_carrier,
        "certificate": certificate,
        "sync": sync_metadata,
        "primary_conf_reference": float(
            key.schur_params.get("primary_conf_reference", 0.80)
        ),
    }


def recover_from_components(
    components: Mapping[str, Any],
    *,
    config: DirectSchurRescueConfig | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse DCT and direct-Schur evidence with a low-confidence rescue gate."""
    cfg = DirectSchurRescueConfig.from_mapping(config)
    primary_map = np.asarray(components["primary_map"])
    primary_confidence = np.asarray(components["primary_confidence"], dtype=np.float64)
    schur_map = np.asarray(components["schur_map"])
    schur_confidence = np.asarray(components["schur_confidence"], dtype=np.float64)
    certificate_map = np.asarray(components["certificate_map"], dtype=np.float64)
    sync_metadata = dict(components["sync"])

    primary_sign = np.where(primary_map > 0, 1.0, -1.0)
    schur_sign = np.where(schur_map > 0, 1.0, -1.0)
    base = cfg.base_config
    primary_evidence = (
        primary_sign
        * (
            base.evidence_conf_floor
            + base.evidence_conf_scale
            * np.power(primary_confidence, base.evidence_conf_power)
        )
        * (
            base.evidence_certificate_floor
            + base.evidence_certificate_scale * certificate_map
        )
    )

    rescue_gate = np.power(
        np.clip(1.0 - primary_confidence, 0.0, 1.0), cfg.gate_power
    )
    rescue_evidence = (
        cfg.fusion_weight
        * rescue_gate
        * schur_sign
        * (
            cfg.schur_conf_floor
            + cfg.schur_conf_scale * np.square(schur_confidence)
        )
    )
    if cfg.agreement_bonus:
        rescue_evidence += (
            cfg.agreement_bonus
            * (primary_sign == schur_sign)
            * schur_sign
            * schur_confidence
            * rescue_gate
        )

    fused_evidence = primary_evidence + rescue_evidence
    reference = float(components.get("primary_conf_reference", 0.80))
    if (
        float(sync_metadata.get("score", 0.0)) >= 0.999
        and float(np.mean(primary_confidence)) >= 0.98 * reference
    ):
        recovered = (primary_map > 0).astype(np.uint8) * 255
        path = "exact_primary_high_confidence"
    else:
        recovered = _icm_map(
            fused_evidence,
            base.qr_map_lambda,
            base.qr_map_iters,
        )
        path = "confidence_gated_direct_schur_rescue_map"

    metadata = {
        "inference_path": path,
        "mean_qim_confidence": float(np.mean(primary_confidence)),
        "mean_schur_confidence": float(np.mean(schur_confidence)),
        "mean_rescue_gate": float(np.mean(rescue_gate)),
        "mean_rescue_abs_evidence": float(np.mean(np.abs(rescue_evidence))),
        "mean_certificate_reliability": float(np.mean(certificate_map)),
        "mean_qr_reliability": float(np.mean(certificate_map)),
        "fusion_weight": float(cfg.fusion_weight),
        "gate_power": float(cfg.gate_power),
        "scientific_status": "exploratory_secondary_channel",
        "independent_clean_acceptance_target": 0.99,
    }
    return recovered, metadata


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: DirectSchurRescueKey,
    *,
    return_metadata: bool = False,
):
    """Fully blind extraction for the direct-Schur rescue proposal."""
    cfg = DirectSchurRescueConfig.from_mapping(key.config)
    components = extract_components(possibly_attacked_rgb, key)
    recovered, fusion_metadata = recover_from_components(components, config=cfg)
    certificate = components["certificate"]
    metadata = {
        "method_id": METHOD_ID,
        "fully_blind": True,
        "original_host_used": False,
        "original_watermark_used": False,
        "certificate_mode": "schur",
        "sync": components["sync"],
        "det_nonzero": bool(certificate.all_nonsingular),
        "min_abs_det": float(certificate.determinant.min()),
        "median_abs_det": float(np.median(certificate.determinant)),
        **fusion_metadata,
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "METHOD_ID",
    "DISPLAY_NAME",
    "SCHUR_POSITIONS",
    "DirectSchurRescueConfig",
    "DirectSchurRescueKey",
    "embed",
    "extract",
    "extract_components",
    "recover_from_components",
]
