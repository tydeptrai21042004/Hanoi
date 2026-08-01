from __future__ import annotations

"""Spectrum-preserving Schur Coupling QIM (SP-SCQIM).

This module is intentionally independent of the DCT-QR proposal engine.  A
4x4 upper-triangular Schur-coordinate matrix is constructed in every 8x8 DCT
block.  The diagonal is left unchanged and the six strict-upper entries form
three orthogonal coupling carriers.  Three interleaved copies of the 64x64
payload are embedded by minimum-Frobenius parity projections.

For a block coupling vector n and orthonormal rows h_k, the constraints are

    h_k^T n* = q_k Delta,   q_k mod 2 = bit_k.

The closed-form update

    n* = n + H^T(t - H n)

is the unique minimum-Euclidean/Frobenius update satisfying all three carrier
constraints.  Because only the strict-upper Schur coordinates change, the
constructed triangular matrix preserves its spectrum, trace, and determinant
exactly in floating point (up to round-off in verification).
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.fft import dctn, idctn
from scipy.ndimage import median_filter

from qr64_certified.common.core import arnold_period, arnold_transform
from .certificate import opponent_field

METHOD_ID = "dct_schur_rescue"
DISPLAY_NAME = "DCT-Schur Spectrum-Preserving Orthogonal Coupling QIM"

BLOCK_SIZE = 8
PAYLOAD_SIDE = 64
PAYLOAD_BITS = PAYLOAD_SIDE * PAYLOAD_SIDE

# Six low/mid-frequency DCT samples are interpreted as the strict-upper
# coordinates (t12,t13,t14,t23,t24,t34) of a virtual 4x4 Schur form.
SCHUR_COUPLING_POSITIONS = np.asarray(
    [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (1, 2)], dtype=np.int32
)

# Disjoint coefficients define the unchanged Schur diagonal / local gain.
SCHUR_DIAGONAL_POSITIONS = np.asarray(
    [(2, 1), (0, 3), (3, 0), (2, 2)], dtype=np.int32
)
SCHUR_SCALE_POSITIONS = np.asarray(
    [(2, 1), (0, 3), (3, 0), (2, 2), (1, 3), (3, 1)], dtype=np.int32
)

# Three mutually orthonormal coupling directions.  They are not the QR
# proposal's one-dimensional (C01-C10)/2 carrier.
SCHUR_COUPLING_BASIS = np.asarray(
    [
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, -2.0, -2.0],
    ],
    dtype=np.float64,
)
SCHUR_COUPLING_BASIS /= np.linalg.norm(
    SCHUR_COUPLING_BASIS, axis=1, keepdims=True
)


@dataclass(frozen=True)
class DirectSchurRescueConfig:
    """Configuration for the independent Schur-coupling proposal."""

    step: float = 9.0
    eta: float = 0.07
    seed: int = 2026
    arnold_iterations: int = 17
    closure_rounds: int = 2
    gain_normalization_enabled: bool = True
    gain_gamma: float = 0.75
    gain_clip: tuple[float, float] = (0.55, 1.45)
    confidence_floor: float = 0.10
    confidence_scale: float = 0.90
    confidence_power: float = 1.50
    map_lambda: float = 0.55
    map_iters: int = 20
    candidate_search_enabled: bool = True
    sharpness_factors: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0)
    unsharp_candidates: tuple[tuple[float, int], ...] = (
        (0.5, 100),
        (1.0, 100),
        (1.0, 150),
        (1.0, 250),
        (1.5, 150),
        (2.0, 200),
    )
    candidate_agreement_weight: float = 0.55
    candidate_evidence_weight: float = 0.35
    candidate_confidence_weight: float = 0.10
    schur_lift: float = 128.0
    determinant_epsilon: float = 1e-12

    def validated(self) -> "DirectSchurRescueConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError("eta must be in [0,1]")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if self.gain_gamma < 0:
            raise ValueError("gain_gamma must be nonnegative")
        if len(self.gain_clip) != 2 or not (0 < self.gain_clip[0] <= self.gain_clip[1]):
            raise ValueError("gain_clip must be an ordered positive pair")
        if self.confidence_floor < 0 or self.confidence_scale < 0:
            raise ValueError("confidence weights must be nonnegative")
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be positive")
        if self.map_lambda < 0 or self.map_iters < 0:
            raise ValueError("invalid MAP parameters")
        total = (
            self.candidate_agreement_weight
            + self.candidate_evidence_weight
            + self.candidate_confidence_weight
        )
        if total <= 0:
            raise ValueError("candidate score weights must have positive sum")
        if self.schur_lift <= 0 or self.determinant_epsilon <= 0:
            raise ValueError("Schur lift and determinant epsilon must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["method_id"] = METHOD_ID
        data["scientific_name"] = "SP-SCQIM"
        return data

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "DirectSchurRescueConfig" | None
    ) -> "DirectSchurRescueConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        raw = dict(value)
        # Old repository configurations contained a nested QR64 configuration
        # and inactive rescue fields.  They are deliberately not imported into
        # the new independent method.  Only unambiguous shared values migrate.
        legacy_base = raw.pop("base_config", None)
        if isinstance(legacy_base, Mapping):
            for name in ("eta", "seed", "gain_gamma", "gain_clip"):
                if name not in raw and name in legacy_base:
                    raw[name] = legacy_base[name]
        aliases = {
            "schur_closure_iters": "closure_rounds",
            "direct_det_epsilon": "determinant_epsilon",
            "qr_map_lambda": "map_lambda",
            "qr_map_iters": "map_iters",
            "evidence_conf_power": "confidence_power",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        # Legacy schur_step was a dimensionless departure step around 0.04;
        # mapping it to the new DCT coupling step would be scientifically wrong.
        raw.pop("schur_step", None)
        ignored = {
            "schur_max_log_scale",
            "fusion_weight",
            "gate_power",
            "schur_conf_floor",
            "schur_conf_scale",
            "agreement_bonus",
            "schur_departure_weight",
            "rho_frac",
            "pilot_count",
            "pilot_step",
            "pilot_rho_frac",
            "qr_lift",
            "adaptive_step_enabled",
            "adaptive_step_ratios",
            "adaptive_step_fractions",
            "exact_confidence_gate",
            "sync_certificate_weight",
            "sync_improvement_threshold",
        }
        for name in ignored:
            raw.pop(name, None)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed}).validated()


@dataclass
class DirectSchurRescueKey:
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]
    config: dict[str, Any]
    spectral_reference: list[float]
    arnold_period: int
    key_version: int = 2

    @property
    def fully_blind(self) -> bool:
        # Extraction never receives the original host.  The content-dependent
        # spectral reference is an embedding key, so metadata also exposes the
        # stricter "key_assisted_blind" classification.
        return True

    @property
    def method_id(self) -> str:
        return METHOD_ID

    @property
    def certificate_mode(self) -> str:
        return "schur"

    @property
    def qr_summary(self) -> dict[str, Any]:
        return {
            "certificate_mode": "schur",
            "carrier": "orthogonal_strict_upper_schur_couplings",
            "blindness_class": "key_assisted_blind",
        }


# ---------------------------------------------------------------------------
# Transform and permutation utilities
# ---------------------------------------------------------------------------


def _blocks(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    if h % BLOCK_SIZE or w % BLOCK_SIZE:
        raise ValueError("Image dimensions must be divisible by 8")
    return (
        field.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def _unblocks(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        blocks.reshape(height // 8, width // 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )


def _coefficients(image: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    field = opponent_field(np.asarray(image, dtype=np.uint8), eta)
    coeff = dctn(_blocks(field), type=2, norm="ortho", axes=(-2, -1))
    return field, np.asarray(coeff, dtype=np.float64)


def _apply_field_delta(base: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
    projection = np.asarray(
        [0.299 + eta, 0.587 - 0.5 * eta, 0.114 - 0.5 * eta],
        dtype=np.float64,
    )
    rgb_direction = projection / float(np.dot(projection, projection))
    updated = np.asarray(base, dtype=np.float64) + delta[..., None] * rgb_direction
    return np.clip(np.rint(updated), 0, 255).astype(np.uint8)


def _arnold_permute_preserve_dtype(array: np.ndarray, iterations: int) -> np.ndarray:
    src = np.asarray(array).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError("Arnold transform requires a square array")
    n = src.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        dst = np.empty_like(src)
        dst[rr, cc] = src
        src = dst
    return src


def _inverse_arnold(array: np.ndarray, iterations: int, period: int) -> np.ndarray:
    inv = (int(period) - (int(iterations) % int(period))) % int(period)
    return _arnold_permute_preserve_dtype(np.asarray(array), inv)


def _payload_permutations(seed: int, count: int = PAYLOAD_BITS) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(seed))
    return tuple(rng.permutation(count).astype(np.int32) for _ in range(3))


# ---------------------------------------------------------------------------
# Schur carrier and exact projection
# ---------------------------------------------------------------------------


def _couplings(coeff: np.ndarray) -> np.ndarray:
    return coeff[:, SCHUR_COUPLING_POSITIONS[:, 0], SCHUR_COUPLING_POSITIONS[:, 1]]


def _write_couplings(coeff: np.ndarray, values: np.ndarray) -> None:
    for j, (u, v) in enumerate(SCHUR_COUPLING_POSITIONS):
        coeff[:, int(u), int(v)] = values[:, j]


def _spectral_scale(coeff: np.ndarray) -> np.ndarray:
    values = coeff[:, SCHUR_SCALE_POSITIONS[:, 0], SCHUR_SCALE_POSITIONS[:, 1]]
    return np.sqrt(np.mean(values * values, axis=1) + 1e-6)


def _nearest_parity_target(
    carrier: np.ndarray, step: float, bits: np.ndarray
) -> np.ndarray:
    normalized = np.asarray(carrier, dtype=np.float64) / float(step)
    q = np.rint(normalized).astype(np.int64)
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    wrong = (q & 1) != bits
    left = q - 1
    right = q + 1
    alternate = np.where(
        np.abs(normalized - left) <= np.abs(normalized - right), left, right
    )
    q = np.where(wrong, alternate, q)
    return q.astype(np.float64) * float(step)


def _embed_coupling_constraints(
    coeff: np.ndarray,
    bits_by_carrier: np.ndarray,
    *,
    step: float,
) -> dict[str, Any]:
    """Apply the closed-form minimum-Frobenius Schur coupling update."""
    n0 = _couplings(coeff).copy()
    carriers0 = n0 @ SCHUR_COUPLING_BASIS.T
    targets = np.empty_like(carriers0)
    for k in range(3):
        targets[:, k] = _nearest_parity_target(
            carriers0[:, k], step, bits_by_carrier[k]
        )
    # H has orthonormal rows, hence H^T(t-Hn) is the orthogonal projection.
    n1 = n0 + (targets - carriers0) @ SCHUR_COUPLING_BASIS
    _write_couplings(coeff, n1)
    achieved = n1 @ SCHUR_COUPLING_BASIS.T
    residual = achieved - targets
    return {
        "max_constraint_residual": float(np.max(np.abs(residual))),
        "mean_constraint_residual": float(np.mean(np.abs(residual))),
        "projection_energy": float(np.sum((n1 - n0) ** 2)),
        "mean_projection_energy_per_block": float(np.mean(np.sum((n1 - n0) ** 2, axis=1))),
    }


def _constructed_schur_matrices(coeff: np.ndarray, lift: float) -> np.ndarray:
    count = coeff.shape[0]
    matrices = np.zeros((count, 4, 4), dtype=np.float64)
    diagonal = coeff[:, SCHUR_DIAGONAL_POSITIONS[:, 0], SCHUR_DIAGONAL_POSITIONS[:, 1]]
    diagonal = diagonal + float(lift)
    idx = np.arange(4)
    matrices[:, idx, idx] = diagonal
    n = _couplings(coeff)
    upper = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for j, (r, c) in enumerate(upper):
        matrices[:, r, c] = n[:, j]
    return matrices


def _schur_invariant_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    lift: float,
    determinant_epsilon: float,
) -> dict[str, Any]:
    a = _constructed_schur_matrices(before, lift)
    b = _constructed_schur_matrices(after, lift)
    eig_a = np.sort_complex(np.linalg.eigvals(a))
    eig_b = np.sort_complex(np.linalg.eigvals(b))
    det_a = np.linalg.det(a)
    det_b = np.linalg.det(b)
    trace_a = np.trace(a, axis1=1, axis2=2)
    trace_b = np.trace(b, axis1=1, axis2=2)
    rel_det = np.abs(det_b - det_a) / np.maximum(np.abs(det_a), determinant_epsilon)
    return {
        "max_spectrum_error_float": float(np.max(np.abs(eig_b - eig_a))),
        "max_trace_error_float": float(np.max(np.abs(trace_b - trace_a))),
        "max_relative_det_error_float": float(np.max(rel_det)),
        "min_abs_det_float": float(np.min(np.abs(det_b))),
        "all_det_nonzero_float": bool(np.all(np.abs(det_b) > determinant_epsilon)),
    }


# Compatibility helper retained for older theorem tests.  It now verifies the
# active coupling law rather than the shadowed legacy implementation.
def _embed_schur_coefficients(
    coefficients: np.ndarray,
    bits: np.ndarray,
    *,
    step: float,
    lift: float,
    max_log_scale: float | None = None,
) -> dict[str, Any]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.ndim != 3 or coeff.shape[1:] != (8, 8):
        raise ValueError("coefficients must have shape (N,8,8)")
    before = coeff.copy()
    payload = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if payload.size != coeff.shape[0]:
        raise ValueError("one bit is required per coefficient block")
    repeated = np.vstack([payload, payload, payload])
    stats = _embed_coupling_constraints(coeff, repeated, step=float(step))
    stats.update(
        _schur_invariant_summary(
            before,
            coeff,
            lift=float(lift),
            determinant_epsilon=1e-12,
        )
    )
    return stats


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _raw_payload_evidence(
    image: np.ndarray,
    cfg: DirectSchurRescueConfig,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, coeff = _coefficients(image, cfg.eta)
    n = _couplings(coeff)
    current_scale = _spectral_scale(coeff)
    ratio: np.ndarray | None = None
    if cfg.gain_normalization_enabled:
        side = int(round(np.sqrt(current_scale.size)))
        raw_ratio = current_scale / np.maximum(reference, 1e-6)
        ratio = median_filter(
            raw_ratio.reshape(side, side), size=3, mode="nearest"
        ).reshape(-1)
        ratio = np.clip(ratio, cfg.gain_clip[0], cfg.gain_clip[1])

    permutations = _payload_permutations(cfg.seed, n.shape[0])
    votes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for k in range(3):
        carrier = n @ SCHUR_COUPLING_BASIS[k]
        if ratio is not None:
            carrier = carrier / np.power(ratio, cfg.gain_gamma)
        normalized = carrier / cfg.step
        q = np.rint(normalized).astype(np.int64)
        bits = (q & 1).astype(np.uint8)
        confidence = np.clip(1.0 - 2.0 * np.abs(normalized - q), 0.0, 1.0)
        signed = np.where(bits > 0, 1.0, -1.0) * (
            cfg.confidence_floor
            + cfg.confidence_scale * np.power(confidence, cfg.confidence_power)
        )
        payload_vote = np.zeros(n.shape[0], dtype=np.float64)
        payload_conf = np.zeros(n.shape[0], dtype=np.float64)
        payload_vote[permutations[k]] = signed
        payload_conf[permutations[k]] = confidence
        votes.append(payload_vote)
        confidences.append(payload_conf)

    vote_matrix = np.asarray(votes, dtype=np.float64)
    conf_matrix = np.asarray(confidences, dtype=np.float64)
    evidence = np.sum(vote_matrix, axis=0)
    confidence = np.mean(conf_matrix, axis=0)
    return evidence, confidence, vote_matrix, current_scale


def _candidate_images(
    image: np.ndarray, cfg: DirectSchurRescueConfig
) -> list[tuple[str, np.ndarray]]:
    base = np.asarray(image, dtype=np.uint8)
    if not cfg.candidate_search_enabled:
        return [("identity", base)]
    pil = Image.fromarray(base)
    output: list[tuple[str, np.ndarray]] = [("identity", base)]
    for factor in cfg.sharpness_factors:
        candidate = ImageEnhance.Sharpness(pil).enhance(float(factor))
        output.append((f"sharpness_{factor:g}", np.asarray(candidate, dtype=np.uint8)))
    for radius, percent in cfg.unsharp_candidates:
        candidate = pil.filter(
            ImageFilter.UnsharpMask(
                radius=float(radius), percent=int(percent), threshold=0
            )
        )
        output.append(
            (
                f"unsharp_r{float(radius):g}_p{int(percent)}",
                np.asarray(candidate, dtype=np.uint8),
            )
        )
    return output


def _select_candidate(
    image: np.ndarray,
    cfg: DirectSchurRescueConfig,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    best: tuple[float, str, np.ndarray, np.ndarray, float] | None = None
    details: list[dict[str, Any]] = []
    for name, candidate in _candidate_images(image, cfg):
        evidence, confidence, votes, _ = _raw_payload_evidence(
            candidate, cfg, reference
        )
        signs = np.sign(votes)
        agreement = float(np.mean(np.abs(np.sum(signs, axis=0)) == 3))
        evidence_strength = float(np.mean(np.abs(evidence) / 3.0))
        mean_confidence = float(np.mean(confidence))
        score = (
            cfg.candidate_agreement_weight * agreement
            + cfg.candidate_evidence_weight * evidence_strength
            + cfg.candidate_confidence_weight * mean_confidence
        )
        row = {
            "candidate": name,
            "score": float(score),
            "agreement": agreement,
            "evidence_strength": evidence_strength,
            "mean_confidence": mean_confidence,
        }
        details.append(row)
        if best is None or score > best[0]:
            best = (score, name, evidence, confidence, agreement)

    assert best is not None
    score, name, evidence, confidence, agreement = best
    return evidence, {
        "selected_candidate": name,
        "candidate_score": float(score),
        "copy_agreement": float(agreement),
        "mean_confidence": float(np.mean(confidence)),
        "candidate_scores": details,
    }


def _icm_map(evidence: np.ndarray, lam: float, iterations: int) -> np.ndarray:
    data = np.asarray(evidence, dtype=np.float64)
    state = np.where(data >= 0.0, 1.0, -1.0)
    if lam <= 0 or iterations <= 0:
        return (state > 0).astype(np.uint8) * 255
    for _ in range(int(iterations)):
        neighbour = np.zeros_like(state)
        neighbour[1:, :] += state[:-1, :]
        neighbour[:-1, :] += state[1:, :]
        neighbour[:, 1:] += state[:, :-1]
        neighbour[:, :-1] += state[:, 1:]
        updated = np.where(data + float(lam) * neighbour >= 0.0, 1.0, -1.0)
        if np.array_equal(updated, state):
            break
        state = updated
    return (state > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DirectSchurRescueConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    cfg = DirectSchurRescueConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    watermark = (np.asarray(watermark_binary, dtype=np.uint8) > 0).astype(np.uint8)
    if host.shape != (512, 512, 3):
        raise ValueError(f"SP-SCQIM requires a 512x512 RGB host; got {host.shape}")
    if watermark.shape != (64, 64):
        raise ValueError(f"SP-SCQIM requires a 64x64 watermark; got {watermark.shape}")

    payload = arnold_transform(watermark, cfg.arnold_iterations).reshape(-1)
    permutations = _payload_permutations(cfg.seed, PAYLOAD_BITS)
    bits_by_carrier = np.vstack([payload[p] for p in permutations])

    current = host.copy()
    closure_history: list[dict[str, Any]] = []
    invariant_stats: dict[str, Any] = {}
    for iteration in range(cfg.closure_rounds):
        field, coeff = _coefficients(current, cfg.eta)
        before = coeff.copy()
        projection_stats = _embed_coupling_constraints(
            coeff, bits_by_carrier, step=cfg.step
        )
        invariant_stats = _schur_invariant_summary(
            before,
            coeff,
            lift=cfg.schur_lift,
            determinant_epsilon=cfg.determinant_epsilon,
        )
        reconstructed = _unblocks(
            idctn(coeff, type=2, norm="ortho", axes=(-2, -1)), 512, 512
        )
        current = _apply_field_delta(current, reconstructed - field, cfg.eta)

        # Verify physical copies after uint8 projection.  Closure reprojects all
        # three orthogonal constraints from the current lattice point.
        _, rounded_coeff = _coefficients(current, cfg.eta)
        rounded_n = _couplings(rounded_coeff)
        copy_accuracy: list[float] = []
        all_exact = True
        for k in range(3):
            q = np.rint((rounded_n @ SCHUR_COUPLING_BASIS[k]) / cfg.step).astype(np.int64)
            decoded = (q & 1).astype(np.uint8)
            ok = decoded == bits_by_carrier[k]
            copy_accuracy.append(float(np.mean(ok)))
            all_exact = all_exact and bool(np.all(ok))
        closure_history.append(
            {
                "iteration": int(iteration + 1),
                "copy_accuracy": copy_accuracy,
                "all_copies_exact": all_exact,
                **projection_stats,
            }
        )
        if all_exact:
            break

    _, final_coeff = _coefficients(current, cfg.eta)
    spectral_reference = _spectral_scale(final_coeff)
    key = DirectSchurRescueKey(
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in watermark.shape),
        config=cfg.to_dict(),
        spectral_reference=[float(x) for x in spectral_reference],
        arnold_period=int(arnold_period(PAYLOAD_SIDE)),
    )
    metadata = {
        "method_id": METHOD_ID,
        "display_name": DISPLAY_NAME,
        "certificate_mode": "schur",
        "carrier": "three_orthogonal_strict_upper_schur_couplings",
        "embedding_law": "n*=n+H^T(t-Hn), parity(t_k/Delta)=w_k",
        "minimum_frobenius_projection": True,
        "spectrum_preserved_float": invariant_stats.get("max_spectrum_error_float", 1.0) < 1e-8,
        "trace_preserved_float": invariant_stats.get("max_trace_error_float", 1.0) < 1e-8,
        "determinant_preserved_float": invariant_stats.get("max_relative_det_error_float", 1.0) < 1e-8,
        "direct_schur_all_det_nonzero": invariant_stats.get("all_det_nonzero_float", False),
        "direct_schur_min_abs_det": invariant_stats.get("min_abs_det_float", 0.0),
        "legacy_secondary_embedded": False,
        "dct_qr_engine_used": False,
        "blindness_class": "key_assisted_blind",
        "closure_history": closure_history,
        **invariant_stats,
    }
    return (current, key, metadata) if return_metadata else (current, key)


def extract_components(
    possibly_attacked_rgb: np.ndarray,
    key: DirectSchurRescueKey,
) -> dict[str, Any]:
    """Return raw Schur-coupling evidence before candidate search and MAP.

    This keeps the scientific component verifier attached to the active
    SP-SCQIM implementation instead of the inactive legacy rescue module.
    """
    cfg = DirectSchurRescueConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(image.shape) != tuple(key.host_shape):
        raise ValueError(
            f"Image shape {image.shape} does not match key host shape {key.host_shape}"
        )
    reference = np.asarray(key.spectral_reference, dtype=np.float64)
    evidence, confidence, vote_matrix, current_scale = _raw_payload_evidence(
        image, cfg, reference
    )
    evidence_map = _inverse_arnold(
        evidence.reshape(key.watermark_shape),
        cfg.arnold_iterations,
        key.arnold_period,
    )
    confidence_map = _inverse_arnold(
        confidence.reshape(key.watermark_shape),
        cfg.arnold_iterations,
        key.arnold_period,
    )
    return {
        "schur_map": (evidence_map > 0.0).astype(np.uint8),
        "schur_evidence": evidence_map,
        "schur_confidence": confidence_map,
        "vote_matrix": vote_matrix,
        "current_scale": current_scale,
    }


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: DirectSchurRescueKey,
    *,
    return_metadata: bool = False,
):
    cfg = DirectSchurRescueConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(image.shape) != tuple(key.host_shape):
        raise ValueError(
            f"Image shape {image.shape} does not match key host shape {key.host_shape}"
        )
    reference = np.asarray(key.spectral_reference, dtype=np.float64)
    evidence, selection = _select_candidate(image, cfg, reference)
    evidence_map = _inverse_arnold(
        evidence.reshape(key.watermark_shape),
        cfg.arnold_iterations,
        key.arnold_period,
    )
    recovered = _icm_map(evidence_map, cfg.map_lambda, cfg.map_iters)

    _, coeff = _coefficients(image, cfg.eta)
    matrices = _constructed_schur_matrices(coeff, cfg.schur_lift)
    determinants = np.abs(np.linalg.det(matrices))
    metadata = {
        "method_id": METHOD_ID,
        "certificate_mode": "schur",
        "inference_path": "independent_schur_coupling_evidence",
        "dct_qr_engine_used": False,
        "gain_normalization_enabled": cfg.gain_normalization_enabled,
        "det_nonzero": bool(np.all(determinants > cfg.determinant_epsilon)),
        "min_abs_det": float(np.min(determinants)),
        "median_abs_det": float(np.median(determinants)),
        "blindness_class": "key_assisted_blind",
        **selection,
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "METHOD_ID",
    "DISPLAY_NAME",
    "SCHUR_COUPLING_POSITIONS",
    "SCHUR_DIAGONAL_POSITIONS",
    "SCHUR_COUPLING_BASIS",
    "DirectSchurRescueConfig",
    "DirectSchurRescueKey",
    "_constructed_schur_matrices",
    "_embed_coupling_constraints",
    "_embed_schur_coefficients",
    "extract_components",
    "embed",
    "extract",
]
