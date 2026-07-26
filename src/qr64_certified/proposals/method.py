from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import dctn

from qr64_certified.common.types import WatermarkKey
from qr64_certified.common.core import arnold_transform
from qr64_certified._jilp_core import engine as _eng
from qr64_certified._jilp_core.method import (
    METHOD_ID as JILP_METHOD_ID,
    PAY10_BA,
    PAY10_BB,
    _best_candidate,
    _build_key_params,
    _call_with_realtime_thread_budget,
    _extract_one_carrier_bits_conf_kernel,
    _extract_payload_bits_confidence,
    _make_content_adaptive_schedules,
    _normalize_jilp_key,
    _pilot_score_fast,
    _schedules_from_key,
    embed as _base_embed,
)

from .certificate import (
    MatrixCertificate,
    compute_certificate,
    opponent_field,
    qr_gain_scale,
    schur_gain_scale,
)
from .config import QR64Config


@dataclass
class QR64Key:
    base_key: WatermarkKey
    config: dict[str, Any]
    qr_summary: dict[str, Any]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def certificate_mode(self) -> str:
        return str(self.qr_summary.get("certificate_mode", self.config.get("certificate_mode", "qr")))


def _arnold_permute_preserve_dtype(array: np.ndarray, iterations: int) -> np.ndarray:
    src = np.asarray(array).copy()
    if src.ndim != 2 or src.shape[0] != src.shape[1]:
        raise ValueError("Arnold transform requires a square array.")
    n = src.shape[0]
    x, y = np.indices((n, n))
    rr = (x + y) % n
    cc = (x + 2 * y) % n
    for _ in range(int(iterations)):
        dst = np.empty_like(src)
        dst[rr, cc] = src
        src = dst
    return src


def _inverse_arnold_array(array: np.ndarray, key_params: dict[str, Any]) -> np.ndarray:
    if not bool(key_params.get("scramble_payload", True)):
        return np.asarray(array)
    period = int(key_params["arnold_period"])
    iterations = int(key_params["arnold_iter"])
    inverse_iterations = (period - (iterations % period)) % period
    return _arnold_permute_preserve_dtype(np.asarray(array), inverse_iterations)


def _step_classes_from_reliability(
    reliability: np.ndarray,
    levels: tuple[float, float, float],
    fractions: tuple[float, float],
) -> np.ndarray:
    """Allocate stronger quantization only to the least stable blocks.

    The certificate reliability orders blocks from weak to strong.  The first
    fraction receives the largest QIM step, the middle fraction the reference
    step, and the strongest blocks the smallest step.  This is a discrete
    approximation of the variational rule ``Delta_b = f(reliability_b)``.
    """
    values = np.asarray(reliability, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    output = np.empty(values.size, dtype=np.float64)
    cuts = [0, int(round(fractions[0] * values.size)), int(round(fractions[1] * values.size)), values.size]
    for level, start, stop in zip(levels, cuts[:-1], cuts[1:], strict=True):
        output[order[start:stop]] = float(level)
    return output




def _pack_binary_mask(bits: np.ndarray) -> str:
    """Pack a binary payload mask into a JSON-safe base64 string."""
    values = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    packed = np.packbits(values, bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _unpack_binary_mask(encoded: str, count: int) -> np.ndarray:
    """Recover exactly ``count`` bits from a packed JSON-safe mask."""
    raw = base64.b64decode(str(encoded).encode("ascii"), validate=True)
    values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if values.size < int(count):
        raise ValueError("Packed coset mask is shorter than the payload length.")
    return values[: int(count)].astype(np.uint8, copy=False)


def _coset_flip_mask_from_key(key_params: dict[str, Any], count: int) -> np.ndarray:
    """Read the new packed mask while accepting prototype/legacy list keys."""
    if "coset_flip_mask_b64" in key_params:
        stored_count = int(key_params.get("coset_flip_count", count))
        if stored_count != int(count):
            raise ValueError("Coset flip-mask length does not match the payload length.")
        return _unpack_binary_mask(key_params["coset_flip_mask_b64"], count)
    if "coset_flip_by_payload" in key_params:
        values = np.asarray(key_params["coset_flip_by_payload"], dtype=np.uint8).reshape(-1)
        if values.size != int(count):
            raise ValueError("Legacy coset flip-mask length does not match the payload length.")
        return values & 1
    return np.zeros(int(count), dtype=np.uint8)


def _pairwise_coset_optimize(
    carrier: np.ndarray,
    payload_bits: np.ndarray,
    steps: np.ndarray,
    reliability: np.ndarray,
    *,
    rho_frac: float,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Choose the exact least-distorting binary label per QR-homogeneous group.

    For group ``G_j`` the selected label is

        s_j = argmin_{s in {0,1}} sum_{b in G_j}
              |P_{u_b xor s}(v_b) - v_b|^2.

    The original embedding is the feasible case ``s=0``; consequently this
    optimization cannot increase the floating-point QIM projection energy.
    """
    values = np.asarray(carrier, dtype=np.float64).reshape(-1)
    bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1) & 1
    local_steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    local_reliability = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if not (values.size == bits.size == local_steps.size == local_reliability.size):
        raise ValueError("Carrier, payload, step, and reliability arrays must have equal length.")

    cost0 = np.empty(values.size, dtype=np.float64)
    cost1 = np.empty(values.size, dtype=np.float64)
    for i, (value, step) in enumerate(zip(values, local_steps, strict=True)):
        margin = float(step) * float(rho_frac)
        target0 = _eng._project_qim_margin(float(value), float(step), 0, margin)
        target1 = _eng._project_qim_margin(float(value), float(step), 1, margin)
        cost0[i] = float(target0 - value) ** 2
        cost1[i] = float(target1 - value) ** 2

    order = np.argsort(local_reliability, kind="stable")
    flips = np.zeros(values.size, dtype=np.uint8)
    optimized_cost = 0.0
    original_cost = float(np.where(bits == 0, cost0, cost1).sum())
    flipped_groups = 0
    group_count = 0
    for start in range(0, values.size, int(group_size)):
        group = order[start : start + int(group_size)]
        cost_s0 = float(np.where(bits[group] == 0, cost0[group], cost1[group]).sum())
        cost_s1 = float(np.where(bits[group] == 0, cost1[group], cost0[group]).sum())
        label = np.uint8(cost_s1 < cost_s0)
        flips[group] = label
        optimized_cost += min(cost_s0, cost_s1)
        flipped_groups += int(label)
        group_count += 1

    encoded = np.bitwise_xor(bits, flips)
    stats: dict[str, float | int] = {
        "group_size": int(group_size),
        "group_count": int(group_count),
        "flipped_group_count": int(flipped_groups),
        "original_projection_energy": float(original_cost),
        "optimized_projection_energy": float(optimized_cost),
        "projection_energy_ratio": float(optimized_cost / max(original_cost, 1e-18)),
    }
    return encoded, flips, stats

def _embed_reliability_conditioned(
    host: np.ndarray,
    watermark: np.ndarray,
    certificate: MatrixCertificate,
    cfg: QR64Config,
) -> tuple[np.ndarray, WatermarkKey]:
    """Embed by QR-conditioned QIM strength allocation.

    The method preserves the original DCT carrier and integer-lattice closure.
    Only the local QIM radius changes: weak QR blocks receive more separation
    from the decision boundary, while stable blocks receive less distortion.
    """
    params = _build_key_params(
        tuple(int(x) for x in host.shape),
        tuple(int(x) for x in watermark.shape),
        seed=cfg.seed,
        step=cfg.step,
        rho_frac=cfg.rho_frac,
        pilot_count=cfg.pilot_count,
        pilot_step=cfg.pilot_step,
        pilot_rho_frac=cfg.pilot_rho_frac,
        eta=cfg.eta,
        pilot_schedule="stratified",
        use_integer_lattice_closure=True,
        projection_mode="minimum_distortion",
        enable_pilots=True,
        enable_sync_search=True,
        payload_schedule_mode="random",
        scramble_payload=True,
    )
    payload_indices, pilot_indices, pilot_bits, capacity, _hc, wc, _pool = (
        _make_content_adaptive_schedules(
            host,
            tuple(int(x) for x in watermark.shape),
            seed=cfg.seed,
            pilot_count=cfg.pilot_count,
            block_size=8,
            eta=cfg.eta,
            pilot_schedule="stratified",
            payload_schedule_mode="random",
        )
    )
    if capacity != watermark.size:
        raise ValueError(
            "Reliability-conditioned allocation currently assumes one payload bit per 8x8 block."
        )

    params["payload_indices"] = payload_indices.astype(int).tolist()
    params["pilot_indices"] = pilot_indices.astype(int).tolist()
    params["pilot_bits"] = pilot_bits.astype(int).tolist()
    params["schedule_storage"] = "explicit_qr_reliability_conditioned"

    scrambled = arnold_transform(
        (watermark > 127).astype(np.uint8), int(params["arnold_iter"])
    ).ravel()
    levels = cfg.adaptive_step_levels()
    steps = _step_classes_from_reliability(
        certificate.reliability[payload_indices],
        levels,
        tuple(float(x) for x in cfg.adaptive_step_fractions),
    )
    params["adaptive_step_by_payload"] = steps.astype(float).tolist()
    params["adaptive_step_levels"] = [float(x) for x in levels]
    params["adaptive_step_fractions"] = [float(x) for x in cfg.adaptive_step_fractions]
    output = host.copy()
    _eng._embed_one_carrier_md(
        host,
        output,
        pilot_indices,
        pilot_bits,
        _eng.PILOT_BA,
        _eng.PILOT_BB,
        float(params["step_pilot"]),
        float(params["step_pilot"]) * float(params["pilot_rho_frac"]),
        int(wc),
        8,
        float(params["eta"]),
    )

    encoded_bits = scrambled
    if cfg.coset_optimization_enabled:
        pilot_adjusted_carrier = _dct_qim_carrier(output, cfg.eta)[payload_indices]
        encoded_bits, flips, coset_stats = _pairwise_coset_optimize(
            pilot_adjusted_carrier,
            scrambled,
            steps,
            certificate.reliability[payload_indices],
            rho_frac=cfg.rho_frac,
            group_size=cfg.coset_group_size,
        )
        params["coset_optimization_enabled"] = True
        params["coset_group_size"] = int(cfg.coset_group_size)
        params["coset_group_count"] = int(coset_stats["group_count"])
        params["coset_flip_count"] = int(flips.size)
        params["coset_flip_mask_b64"] = _pack_binary_mask(flips)
        params["coset_group_order"] = "ascending_qr_reliability_at_embedding"
        params["coset_projection_stats"] = coset_stats
        params["scientific_rule"] = (
            "QR-homogeneous group label minimizes total squared QIM projection "
            "distance at unchanged local step and guard margin"
        )
    else:
        params["coset_optimization_enabled"] = False
        params["scientific_rule"] = (
            "larger QIM separation for low QR reliability; smaller separation for stable blocks"
        )

    total_updates = 0
    for step in np.unique(steps):
        logical = np.flatnonzero(np.isclose(steps, step))
        block_indices = payload_indices[logical].astype(np.int32)
        bits = encoded_bits[logical].astype(np.uint8)
        execution = np.argsort(block_indices, kind="stable")
        updates, unresolved = _call_with_realtime_thread_budget(
            _eng._embed_one_carrier_jilp,
            output,
            block_indices[execution],
            bits[execution],
            PAY10_BA,
            PAY10_BB,
            float(step),
            float(step) * cfg.rho_frac,
            int(wc),
            8,
            float(params["eta"]),
        )
        if int(unresolved):
            raise RuntimeError(
                f"QR-conditioned integer-lattice embedding left {int(unresolved)} unresolved blocks."
            )
        total_updates += int(updates)
    params["integer_lattice_unit_updates"] = total_updates
    params["integer_lattice_unresolved"] = 0

    key = WatermarkKey(
        method_id=JILP_METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in watermark.shape),
        seed=cfg.seed,
        repeat=1,
        step=cfg.step,
        arnold_iter=int(params["arnold_iter"]),
        arnold_period=int(params["arnold_period"]),
        threshold=127,
        params=params,
        schedule=[],
        fully_blind=True,
        side_information=(
            "QR-conditioned step classes, packed pairwise coset mask, and keyed "
            "schedules; original host is not required"
            if cfg.coset_optimization_enabled
            else "QR-conditioned step classes and keyed schedules; original host is not required"
        ),
    )
    return output, key


def _extract_reliability_conditioned_payload(
    image: np.ndarray, key_params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    payload_indices, *_ = _schedules_from_key(key_params)
    steps = np.asarray(key_params["adaptive_step_by_payload"], dtype=np.float64)
    bits = np.zeros(payload_indices.size, dtype=np.uint8)
    confidence = np.zeros(payload_indices.size, dtype=np.float64)
    for step in np.unique(steps):
        logical = np.flatnonzero(np.isclose(steps, step))
        group_bits, group_confidence = _call_with_realtime_thread_budget(
            _extract_one_carrier_bits_conf_kernel,
            np.asarray(image, dtype=np.uint8),
            payload_indices[logical].astype(np.int32),
            PAY10_BA,
            PAY10_BB,
            float(step),
            int(image.shape[1]),
            8,
            float(key_params.get("eta", 0.07)),
            0,
            0,
        )
        bits[logical] = group_bits
        confidence[logical] = group_confidence
    return bits, confidence


def _dct_qim_carrier(image: np.ndarray, eta: float) -> np.ndarray:
    """Return the blockwise carrier used by the underlying DCT-QIM rule."""
    field = opponent_field(np.asarray(image, dtype=np.uint8), eta=float(eta))
    h, w = field.shape
    blocks = (
        field.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )
    coefficients = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    return 0.5 * (coefficients[:, 0, 1] - coefficients[:, 1, 0])


def _decomposition_gain_scale(image: np.ndarray, cfg: QR64Config) -> np.ndarray:
    if str(cfg.certificate_mode).lower() == "schur":
        return schur_gain_scale(
            image,
            eta=cfg.eta,
            lift=cfg.qr_lift,
            departure_weight=cfg.schur_departure_weight,
        )
    return qr_gain_scale(image, eta=cfg.eta, lift=cfg.qr_lift, mode=cfg.qr_gain_mode)


def _extract_gain_normalized_payload(
    image: np.ndarray,
    key_params: dict[str, Any],
    cfg: QR64Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Decode QIM after decomposition-based local gain compensation.

    The final watermarked image supplies a blind blockwise reference scale
    ``s_b^0``.  For a questioned image, ``alpha_b=s_b/s_b^0`` estimates local
    multiplicative attenuation and the carrier is corrected as

        v_tilde_b = v_b / alpha_b^gamma.

    The reference contains no host coefficients and the original image is not
    needed during extraction.
    """
    payload_indices, *_ = _schedules_from_key(key_params)
    if "adaptive_step_by_payload" in key_params:
        steps = np.asarray(key_params["adaptive_step_by_payload"], dtype=np.float64)
    else:
        steps = np.full(payload_indices.size, float(key_params["step"]), dtype=np.float64)

    reference = np.asarray(
        key_params["decomposition_gain_reference"], dtype=np.float64
    )
    current = _decomposition_gain_scale(image, cfg)
    if reference.shape != current.shape:
        raise ValueError("Decomposition gain reference does not match image capacity.")
    lower, upper = (float(x) for x in cfg.gain_clip)
    ratio = np.clip(current / np.maximum(reference, 1e-12), lower, upper)
    carrier = _dct_qim_carrier(image, cfg.eta)
    corrected = carrier[payload_indices] / np.power(
        ratio[payload_indices], float(cfg.gain_gamma)
    )
    units = corrected / steps
    lattice = np.rint(units)
    bits = (lattice.astype(np.int64) & 1).astype(np.uint8)
    confidence = np.clip(1.0 - 2.0 * np.abs(units - lattice), 0.0, 1.0)
    log_ratio = np.abs(np.log(np.maximum(ratio, 1e-12)))
    stats = {
        "gain_ratio_mean": float(np.mean(ratio[payload_indices])),
        "gain_ratio_std": float(np.std(ratio[payload_indices])),
        "gain_identity_error": float(np.max(log_ratio[payload_indices])),
    }
    return bits, confidence.astype(np.float64), stats


def _icm_map(data: np.ndarray, lam: float, iterations: int) -> np.ndarray:
    """Deterministic blind MAP approximation with an Ising/Potts prior."""
    data = np.asarray(data, dtype=np.float64)
    state = np.where(data >= 0.0, 1, -1)
    for _ in range(int(iterations)):
        neighbours = np.zeros_like(data)
        neighbours[1:] += state[:-1]
        neighbours[:-1] += state[1:]
        neighbours[:, 1:] += state[:, :-1]
        neighbours[:, :-1] += state[:, 1:]
        # Weak diagonal coupling avoids over-erasing thin watermark strokes.
        neighbours[1:, 1:] += 0.35 * state[:-1, :-1]
        neighbours[:-1, :-1] += 0.35 * state[1:, 1:]
        neighbours[1:, :-1] += 0.35 * state[:-1, 1:]
        neighbours[:-1, 1:] += 0.35 * state[1:, :-1]
        updated = np.where(data + float(lam) * neighbours >= 0.0, 1, -1)
        if np.array_equal(updated, state):
            break
        state = updated
    return (state > 0).astype(np.uint8) * 255


def _certificate_consistency(cert: MatrixCertificate, reference: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compare an attacked candidate with compact certificate statistics in the key.

    No original host coefficients are stored. Only six robust scalar summaries
    (median and MAD for log-beta, balance, and coupling) are used.
    """
    log_beta = np.log(cert.beta + 1e-18)
    observed = {
        "median_log_beta": float(np.median(log_beta)),
        "median_balance": float(np.median(cert.balance)),
        "median_coupling": float(np.median(cert.coupling)),
    }

    # Floors prevent an almost constant host statistic from creating an
    # unrealistically sharp rejection boundary.
    beta_scale = max(0.35, 3.0 * float(reference.get("mad_log_beta", 0.0)))
    balance_scale = max(0.04, 3.0 * float(reference.get("mad_balance", 0.0)))
    coupling_scale = max(0.04, 3.0 * float(reference.get("mad_coupling", 0.0)))

    d_beta = abs(observed["median_log_beta"] - float(reference.get("median_log_beta", observed["median_log_beta"]))) / beta_scale
    d_balance = abs(observed["median_balance"] - float(reference.get("median_balance", observed["median_balance"]))) / balance_scale
    d_coupling = abs(observed["median_coupling"] - float(reference.get("median_coupling", observed["median_coupling"]))) / coupling_scale
    distance = (d_beta + d_balance + 0.5 * d_coupling) / 2.5
    score = float(np.exp(-distance))
    details = {
        **observed,
        "distance": float(distance),
        "score": score,
    }
    return score, details


def _geometry_cost(candidate_name: str, shift_y: int, shift_x: int, radius: int) -> float:
    name = str(candidate_name)
    cost = 0.0
    if "rot_inv_" in name:
        try:
            token = name.split("rot_inv_", 1)[1].split("_", 1)[0]
            cost += min(abs(float(token)) / 5.0, 1.0)
        except (ValueError, IndexError):
            cost += 1.0
    if "crop_inv_" in name:
        try:
            token = name.split("crop_inv_", 1)[1].split("_", 1)[0]
            cost += min(abs(float(token)) / 0.25, 1.0)
        except (ValueError, IndexError):
            cost += 1.0
    if shift_y or shift_x:
        cost += min(np.hypot(float(shift_y), float(shift_x)) / max(float(radius), 1.0), 1.0)
    return float(cost)


def _compute_configured_certificate(img: np.ndarray, cfg: QR64Config) -> MatrixCertificate:
    return compute_certificate(
        img,
        eta=cfg.eta,
        lift=cfg.qr_lift,
        det_epsilon=cfg.qr_det_epsilon,
        mode=str(cfg.certificate_mode).lower(),
    )


def _certified_alignment(
    attacked: np.ndarray,
    key_params: dict[str, Any],
    key_summary: dict[str, Any],
    cfg: QR64Config,
) -> tuple[np.ndarray, MatrixCertificate, dict[str, Any]]:
    """Pilot search followed by identity-preserving certificate gating.

    A geometric correction is accepted only when its combined pilot/certificate
    objective improves sufficiently over the null (identity) hypothesis.
    """
    attacked = np.asarray(attacked, dtype=np.uint8)
    raw_pilot, raw_valid = _pilot_score_fast(attacked, key_params, 0, 0)

    candidate_img, shift_y, shift_x, base_meta = _best_candidate(
        attacked,
        key_params,
        quick_accept=cfg.sync_quick_accept,
        translation_radius=cfg.translation_radius,
        translation_subset=cfg.translation_subset,
    )
    candidate_name = str(base_meta.get("candidate", "raw"))

    raw_cert = _compute_configured_certificate(attacked, cfg)
    raw_consistency, raw_details = _certificate_consistency(raw_cert, key_summary)
    raw_objective = float(raw_pilot + cfg.sync_certificate_weight * raw_consistency)

    if candidate_name == "raw" and not shift_y and not shift_x:
        meta = {
            **base_meta,
            "path": "identity_fast_certified",
            "accepted_geometric_correction": False,
            "raw_pilot_score": float(raw_pilot),
            "candidate_pilot_score": float(raw_pilot),
            "raw_certificate_consistency": float(raw_consistency),
            "candidate_certificate_consistency": float(raw_consistency),
            "raw_objective": raw_objective,
            "candidate_objective": raw_objective,
            "objective_gain": 0.0,
            "acceptance_threshold": float(cfg.sync_improvement_threshold),
            "certificate_mode": cfg.certificate_mode,
            "raw_certificate_details": raw_details,
        }
        return attacked, raw_cert, meta

    aligned_candidate = np.asarray(candidate_img, dtype=np.uint8)
    if shift_y or shift_x:
        aligned_candidate = np.roll(
            aligned_candidate,
            shift=(-int(shift_y), -int(shift_x)),
            axis=(0, 1),
        )

    # Re-score all pilots after applying the proposed translation, rather than
    # comparing the subset score returned by the fast search with a full score.
    candidate_pilot, candidate_valid = _pilot_score_fast(aligned_candidate, key_params, 0, 0)
    candidate_cert = _compute_configured_certificate(aligned_candidate, cfg)
    candidate_consistency, candidate_details = _certificate_consistency(candidate_cert, key_summary)
    geometry_cost = _geometry_cost(candidate_name, shift_y, shift_x, cfg.translation_radius)
    candidate_objective = float(
        candidate_pilot
        + cfg.sync_certificate_weight * candidate_consistency
        - cfg.sync_geometry_penalty * geometry_cost
    )
    objective_gain = float(candidate_objective - raw_objective)
    accepted = bool(objective_gain > cfg.sync_improvement_threshold)

    if accepted:
        selected_img = aligned_candidate
        selected_cert = candidate_cert
        selected_score = candidate_pilot
        selected_valid = candidate_valid
        selected_name = candidate_name
        path = "geometric_correction_certified"
    else:
        selected_img = attacked
        selected_cert = raw_cert
        selected_score = raw_pilot
        selected_valid = raw_valid
        selected_name = "raw_identity_gate"
        path = "identity_retained_by_certificate_gate"

    meta = {
        **base_meta,
        "path": path,
        "candidate": selected_name,
        "score": float(selected_score),
        "pilot_valid": int(selected_valid),
        "accepted_geometric_correction": accepted,
        "proposed_candidate": candidate_name,
        "proposed_shift_y": int(shift_y),
        "proposed_shift_x": int(shift_x),
        "raw_pilot_score": float(raw_pilot),
        "candidate_pilot_score": float(candidate_pilot),
        "raw_certificate_consistency": float(raw_consistency),
        "candidate_certificate_consistency": float(candidate_consistency),
        "geometry_cost": float(geometry_cost),
        "raw_objective": raw_objective,
        "candidate_objective": candidate_objective,
        "objective_gain": objective_gain,
        "acceptance_threshold": float(cfg.sync_improvement_threshold),
        "certificate_mode": cfg.certificate_mode,
        "raw_certificate_details": raw_details,
        "candidate_certificate_details": candidate_details,
    }
    return selected_img, selected_cert, meta


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: QR64Config | dict[str, Any] | None = None,
) -> tuple[np.ndarray, QR64Key]:
    """Embed a 64x64 watermark with QR-certified DCT-QIM.

    In the public DCT-QR configuration, QR reliability has two active roles:
    it allocates local QIM spacing and orders homogeneous block pairs.  One
    binary coset label per pair is selected by exact minimum projection energy,
    while the original QIM step and guard margin remain unchanged.
    """
    if config is None:
        cfg = QR64Config.public_dct_qr()
    else:
        cfg = config if isinstance(config, QR64Config) else QR64Config.from_mapping(config)
        cfg = cfg.validated()
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    if wm.shape != (64, 64):
        raise ValueError(f"This protocol keeps the original 64x64 watermark; got {wm.shape}.")

    certificate = _compute_configured_certificate(host, cfg)
    if cfg.adaptive_step_enabled:
        watermarked, base_key = _embed_reliability_conditioned(
            host, wm, certificate, cfg
        )
        embedding_role = (
            "QR-conditioned pairwise coset-optimized QIM"
            if cfg.coset_optimization_enabled
            else "active reliability-conditioned QIM allocation"
        )
    else:
        watermarked, base_key = _base_embed(
            host,
            wm,
            seed=cfg.seed,
            step=cfg.step,
            method_params={
                "step": cfg.step,
                "rho_frac": cfg.rho_frac,
                "pilot_count": cfg.pilot_count,
                "pilot_step": cfg.pilot_step,
                "pilot_rho_frac": cfg.pilot_rho_frac,
                "eta": cfg.eta,
                "payload_schedule_mode": "random",
                "pilot_schedule": "stratified",
                "use_integer_lattice_closure": True,
                "enable_pilots": True,
                "enable_sync_search": True,
                "scramble_payload": True,
            },
        )
        embedding_role = "uniform QIM reference model"

    if cfg.gain_normalization_enabled:
        reference_scale = _decomposition_gain_scale(watermarked, cfg)
        base_key.params["decomposition_gain_reference"] = (
            reference_scale.astype(float).tolist()
        )
        base_key.params["decomposition_gain_mode"] = str(cfg.certificate_mode).lower()
        base_key.params["decomposition_gain_gamma"] = float(cfg.gain_gamma)
        base_key.params["qr_gain_mode"] = str(cfg.qr_gain_mode)
        base_key.params["decomposition_gain_clip"] = [float(x) for x in cfg.gain_clip]
        base_key.params["decomposition_gain_rule"] = (
            "DCT carrier divided by a QR/Schur block-gain ratio raised to gamma"
        )

    coset_active = bool(base_key.params.get("coset_optimization_enabled", False))
    base_key.params["qr64"] = {
        "role": embedding_role,
        "certificate_mode": cfg.certificate_mode,
        "analysis_matrix": "lifted 4x4 AC-DCT matrix per 8x8 block",
        "mathematical_condition": "det(A) != 0",
        "reliability_principle": (
            "weak matrices receive larger class separation; QR-similar pairs share an "
            "exact least-distorting binary coset label"
            if coset_active
            else "weak matrices receive larger class separation; stable matrices receive lower distortion"
        ),
        "coset_optimization_enabled": coset_active,
        "coset_group_size": int(cfg.coset_group_size),
        "embedding_theorem": (
            "pairwise optimized projection energy is no greater than the original "
            "s=0 labeling, while XOR inversion preserves each physical QIM error event"
            if coset_active
            else "not active"
        ),
        "analysis_lift": cfg.qr_lift,
        "det_epsilon": cfg.qr_det_epsilon,
        "map_lambda": cfg.qr_map_lambda,
        "map_iters": cfg.qr_map_iters,
        "sync_rule": "accept T only if J(T)-J(I) exceeds threshold",
        "sync_improvement_threshold": cfg.sync_improvement_threshold,
    }
    return watermarked, QR64Key(
        base_key=base_key,
        config=cfg.to_dict(),
        qr_summary=certificate.robust_summary(),
    )


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: QR64Key,
    *,
    return_metadata: bool = False,
):
    """Fully blind extraction: no original host or original watermark is input."""
    cfg = QR64Config.from_mapping(key.config)
    key_params = _normalize_jilp_key(key.base_key)
    attacked = np.asarray(possibly_attacked_rgb, dtype=np.uint8)

    aligned, certificate, sync_meta = _certified_alignment(
        attacked,
        key_params,
        key.qr_summary,
        cfg,
    )

    wm_shape = tuple(int(x) for x in key_params["watermark_shape"])
    payload_len = int(np.prod(wm_shape))
    bit_array = np.zeros(payload_len, dtype=np.uint8)
    conf_array = np.zeros(payload_len, dtype=np.float64)
    gain_stats = {
        "gain_ratio_mean": 1.0,
        "gain_ratio_std": 0.0,
        "gain_identity_error": float("inf"),
    }
    if cfg.gain_normalization_enabled and "decomposition_gain_reference" in key_params:
        scrambled_bits, qim_conf, gain_stats = _extract_gain_normalized_payload(
            aligned, key_params, cfg
        )
        positions = np.arange(payload_len, dtype=np.int32)
    elif "adaptive_step_by_payload" in key_params:
        scrambled_bits, qim_conf = _extract_reliability_conditioned_payload(
            aligned, key_params
        )
        positions = np.arange(payload_len, dtype=np.int32)
    else:
        scrambled_bits, positions, qim_conf = _extract_payload_bits_confidence(
            aligned, key_params, 0, 0
        )
    if bool(key_params.get("coset_optimization_enabled", False)) or (
        "coset_flip_mask_b64" in key_params or "coset_flip_by_payload" in key_params
    ):
        flips = _coset_flip_mask_from_key(key_params, payload_len)
        scrambled_bits = np.bitwise_xor(scrambled_bits, flips)

    bit_array[positions] = scrambled_bits
    conf_array[positions] = qim_conf
    bit_map = _inverse_arnold_array(bit_array.reshape(wm_shape), key_params)
    conf_map = _inverse_arnold_array(conf_array.reshape(wm_shape), key_params)

    payload_indices, *_ = _schedules_from_key(key_params)
    certificate_scrambled = certificate.reliability[payload_indices].reshape(wm_shape)
    certificate_map = _inverse_arnold_array(certificate_scrambled, key_params)

    sign = np.where(bit_map > 0, 1.0, -1.0)
    data = (
        sign
        * (cfg.evidence_conf_floor + cfg.evidence_conf_scale * np.power(conf_map, cfg.evidence_conf_power))
        * (cfg.evidence_certificate_floor + cfg.evidence_certificate_scale * certificate_map)
    )

    # Integer-lattice embedding is exact on a clean image. Avoid modifying
    # already-certain bits only because of the spatial prior.
    if float(gain_stats["gain_identity_error"]) <= cfg.clean_identity_tolerance:
        recovered = (bit_map > 0).astype(np.uint8) * 255
        inference_path = "exact_decomposition_identity"
    elif float(sync_meta.get("score", 0.0)) >= 0.999 and float(np.mean(conf_map)) >= cfg.exact_confidence_gate:
        recovered = (bit_map > 0).astype(np.uint8) * 255
        inference_path = "exact_high_confidence"
    else:
        recovered = _icm_map(data, cfg.qr_map_lambda, cfg.qr_map_iters)
        inference_path = f"{cfg.certificate_mode}_weighted_map"

    metadata = {
        "fully_blind": True,
        "original_host_used": False,
        "original_watermark_used": False,
        "sync": sync_meta,
        "certificate_mode": cfg.certificate_mode,
        "det_nonzero": bool(certificate.all_nonsingular),
        "min_abs_det": float(certificate.determinant.min()),
        "median_abs_det": float(np.median(certificate.determinant)),
        "mean_qim_confidence": float(np.mean(conf_map)),
        "mean_certificate_reliability": float(np.mean(certificate_map)),
        # Backward-compatible field name for old benchmark readers.
        "mean_qr_reliability": float(np.mean(certificate_map)),
        "inference_path": inference_path,
        "gain_normalization_enabled": bool(
            cfg.gain_normalization_enabled
            and "decomposition_gain_reference" in key_params
        ),
        "coset_optimization_enabled": bool(
            key_params.get("coset_optimization_enabled", False)
            or "coset_flip_mask_b64" in key_params
            or "coset_flip_by_payload" in key_params
        ),
        "coset_group_size": int(key_params.get("coset_group_size", 1)),
        "coset_group_count": int(key_params.get("coset_group_count", 0)),
        **gain_stats,
    }
    return (recovered, metadata) if return_metadata else recovered
