from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np
from PIL import Image

from qr64_certified.common.types import WatermarkKey
from qr64_certified.common.core import arnold_transform
from qr64_certified._jilp_core import engine as _eng
from qr64_certified._jilp_core.method import (
    METHOD_ID as JILP_METHOD_ID,
    PAY10_BA,
    PAY10_BB,
    _build_key_params,
    _call_with_realtime_thread_budget,
    _extract_one_carrier_bits_conf_kernel,
    _make_content_adaptive_schedules,
    _normalize_jilp_key,
    _schedules_from_key,
)
from qr64_certified.certificate import compute_certificate, opponent_field
from qr64_certified.config import QR64Config
from qr64_certified.method import (
    QR64Key,
    _certified_alignment,
    _compute_configured_certificate,
    _icm_map,
    _inverse_arnold_array,
)
from qr64_certified.direct_schur_rescue import (
    DirectSchurRescueConfig,
    DirectSchurRescueKey,
    _blocks_from_field,
    _embed_schur_coefficients,
    _extract_schur_bits_confidence,
    _field_from_blocks,
    _apply_field_delta,
    _payload_bits_by_block,
    _validate_direct_matrices,
)
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------------
# 1. QR-certified adaptive-step DCT-QIM experiment
# ---------------------------------------------------------------------------


def _step_classes_from_reliability(
    reliability: np.ndarray,
    step_levels: tuple[float, ...],
    fractions: tuple[float, ...],
) -> np.ndarray:
    """Assign larger steps to the least reliable blocks.

    fractions are cumulative proportions from weakest to strongest, one per
    step level except the last. Example levels=(16,13,10), fractions=(.25,.65)
    means weakest 25% get 16, next 40% get 13, strongest 35% get 10.
    """
    r = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if len(step_levels) != len(fractions) + 1:
        raise ValueError("len(step_levels) must equal len(fractions)+1")
    order = np.argsort(r, kind="stable")
    out = np.empty(r.size, dtype=np.float64)
    starts = [0] + [int(round(f * r.size)) for f in fractions]
    ends = starts[1:] + [r.size]
    for level, start, end in zip(step_levels, starts, ends, strict=True):
        out[order[start:end]] = float(level)
    return out


def embed_dct_qr_adaptive(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: QR64Config | Mapping[str, Any],
    step_levels: tuple[float, ...] = (16.0, 13.0, 10.5),
    fractions: tuple[float, ...] = (0.25, 0.65),
) -> tuple[np.ndarray, QR64Key]:
    cfg = config if isinstance(config, QR64Config) else QR64Config.from_mapping(config)
    cfg = cfg.validated()
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = np.asarray(watermark_binary, dtype=np.uint8)
    if wm.shape != (64, 64):
        raise ValueError("watermark must be 64x64")

    certificate = _compute_configured_certificate(host, cfg)
    params = _build_key_params(
        tuple(int(x) for x in host.shape),
        tuple(int(x) for x in wm.shape),
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
            tuple(int(x) for x in wm.shape),
            seed=cfg.seed,
            pilot_count=cfg.pilot_count,
            block_size=8,
            eta=cfg.eta,
            pilot_schedule="stratified",
            payload_schedule_mode="random",
        )
    )
    if capacity != wm.size:
        raise ValueError("experiment currently assumes one payload bit per block")

    # Store schedules explicitly because step classes are tied to logical payload positions.
    params["payload_indices"] = payload_indices.astype(int).tolist()
    params["pilot_indices"] = pilot_indices.astype(int).tolist()
    params["pilot_bits"] = pilot_bits.astype(int).tolist()
    params["schedule_storage"] = "explicit_certificate_adaptive"
    params["side_information"] = "certificate-conditioned step classes; no original host"

    bits = arnold_transform((wm > 127).astype(np.uint8), int(params["arnold_iter"])).ravel()
    reliability = certificate.reliability[payload_indices]
    steps = _step_classes_from_reliability(reliability, step_levels, fractions)
    params["adaptive_step_by_payload"] = steps.astype(float).tolist()
    params["adaptive_step_levels"] = [float(x) for x in step_levels]
    params["adaptive_step_fractions"] = [float(x) for x in fractions]
    params["qr_active_embedding_control"] = True

    out = host.copy()
    _eng._embed_one_carrier_md(
        host,
        out,
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

    total_updates = 0
    total_unresolved = 0
    for step in np.unique(steps):
        logical = np.flatnonzero(np.isclose(steps, step))
        block_indices = payload_indices[logical].astype(np.int32)
        group_bits = bits[logical].astype(np.uint8)
        execution = np.argsort(block_indices, kind="stable")
        updates, unresolved = _call_with_realtime_thread_budget(
            _eng._embed_one_carrier_jilp,
            out,
            block_indices[execution],
            group_bits[execution],
            PAY10_BA,
            PAY10_BB,
            float(step),
            float(step) * cfg.rho_frac,
            int(wc),
            8,
            float(params["eta"]),
        )
        total_updates += int(updates)
        total_unresolved += int(unresolved)
    if total_unresolved:
        raise RuntimeError(f"adaptive DCT embedding unresolved={total_unresolved}")
    params["integer_lattice_unit_updates"] = total_updates
    params["integer_lattice_unresolved"] = total_unresolved
    params["qr64"] = {
        "role": "active reliability-conditioned step allocation, extraction weighting, and sync gating",
        "certificate_mode": cfg.certificate_mode,
    }

    key_base = WatermarkKey(
        method_id=JILP_METHOD_ID,
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in wm.shape),
        seed=cfg.seed,
        repeat=1,
        step=cfg.step,
        arnold_iter=int(params["arnold_iter"]),
        arnold_period=int(params["arnold_period"]),
        threshold=127,
        params=params,
        schedule=[],
        fully_blind=True,
        side_information=str(params["side_information"]),
    )
    return out, QR64Key(key_base, cfg.to_dict(), certificate.robust_summary())


def _extract_adaptive_payload(
    image: np.ndarray, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.uint8)
    payload_indices, *_ = _schedules_from_key(params)
    steps = np.asarray(params["adaptive_step_by_payload"], dtype=np.float64)
    bits = np.zeros(payload_indices.size, dtype=np.uint8)
    conf = np.zeros(payload_indices.size, dtype=np.float64)
    for step in np.unique(steps):
        logical = np.flatnonzero(np.isclose(steps, step))
        group_bits, group_conf = _call_with_realtime_thread_budget(
            _extract_one_carrier_bits_conf_kernel,
            image,
            payload_indices[logical].astype(np.int32),
            PAY10_BA,
            PAY10_BB,
            float(step),
            int(image.shape[1]),
            8,
            float(params.get("eta", 0.07)),
            0,
            0,
        )
        bits[logical] = group_bits
        conf[logical] = group_conf
    return bits, conf


def extract_dct_qr_adaptive(
    attacked_rgb: np.ndarray, key: QR64Key, *, return_metadata: bool = False
):
    cfg = QR64Config.from_mapping(key.config)
    params = _normalize_jilp_key(key.base_key)
    aligned, certificate, sync_meta = _certified_alignment(
        np.asarray(attacked_rgb, dtype=np.uint8), params, key.qr_summary, cfg
    )
    scrambled, conf = _extract_adaptive_payload(aligned, params)
    shape = tuple(int(x) for x in params["watermark_shape"])
    bit_map = _inverse_arnold_array(scrambled.reshape(shape), params)
    conf_map = _inverse_arnold_array(conf.reshape(shape), params)
    payload_indices, *_ = _schedules_from_key(params)
    cert_map = _inverse_arnold_array(
        certificate.reliability[payload_indices].reshape(shape), params
    )
    sign = np.where(bit_map > 0, 1.0, -1.0)
    data = (
        sign
        * (cfg.evidence_conf_floor + cfg.evidence_conf_scale * conf_map ** cfg.evidence_conf_power)
        * (cfg.evidence_certificate_floor + cfg.evidence_certificate_scale * cert_map)
    )
    if float(sync_meta.get("score", 0.0)) >= 0.999 and float(np.mean(conf_map)) >= 0.79:
        recovered = (bit_map > 0).astype(np.uint8) * 255
        path = "adaptive_exact_high_confidence"
    else:
        recovered = _icm_map(data, cfg.qr_map_lambda, cfg.qr_map_iters)
        path = "adaptive_qr_weighted_map"
    meta = {
        "det_nonzero": bool(certificate.all_nonsingular),
        "min_abs_det": float(certificate.determinant.min()),
        "inference_path": path,
        "sync": sync_meta,
        "mean_qim_confidence": float(np.mean(conf_map)),
        "mean_certificate_reliability": float(np.mean(cert_map)),
    }
    return (recovered, meta) if return_metadata else recovered


# ---------------------------------------------------------------------------
# 2. Sparse joint Schur rescue experiment
# ---------------------------------------------------------------------------


@dataclass
class SparseSchurKey:
    base: QR64Key
    config: dict[str, Any]
    schur_params: dict[str, Any]


def embed_sparse_joint_schur(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DirectSchurRescueConfig | Mapping[str, Any],
    rescue_count: int = 512,
    joint_iters: int = 4,
    schur_step: float = 0.20,
    selection_mode: str = "primary_confidence",
) -> tuple[np.ndarray, SparseSchurKey, dict[str, Any]]:
    from qr64_certified.method import embed as certified_embed
    from qr64_certified._jilp_core.method import _extract_payload_bits_confidence

    cfg = DirectSchurRescueConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    wm = (np.asarray(watermark_binary, dtype=np.uint8) > 0).astype(np.uint8)
    primary, base_key = certified_embed(host, wm * 255, config=cfg.base_config)
    params = _normalize_jilp_key(base_key.base_key)
    payload_indices, bits_scrambled, bits_by_block = _payload_bits_by_block(wm, params)

    _, _, primary_conf = _extract_payload_bits_confidence(primary, params, 0, 0)
    rescue_count = int(min(max(1, rescue_count), bits_scrambled.size))
    if selection_mode == "primary_confidence":
        selection_score = primary_conf
    elif selection_mode == "certificate":
        host_certificate = _compute_configured_certificate(host, cfg.base_config)
        selection_score = host_certificate.reliability[payload_indices]
    elif selection_mode == "product":
        host_certificate = _compute_configured_certificate(host, cfg.base_config)
        selection_score = primary_conf * host_certificate.reliability[payload_indices]
    else:
        raise ValueError(f"unknown selection_mode={selection_mode}")
    rescue_positions = np.argsort(selection_score, kind="stable")[:rescue_count].astype(np.int32)
    rescue_blocks = payload_indices[rescue_positions].astype(np.int32)

    current = primary.copy()
    eta = float(cfg.base_config.eta)
    history: list[dict[str, Any]] = []
    execution_order = np.argsort(payload_indices, kind="stable")
    indices_execution = payload_indices[execution_order].astype(np.int32)
    bits_execution = bits_scrambled[execution_order].astype(np.uint8)
    primary_step = float(params.get("step_carrier1", params.get("step", 8.25)))
    primary_rho = primary_step * float(params.get("rho_frac", 0.45))

    for iteration in range(int(joint_iters)):
        observed, _, _ = _extract_schur_bits_confidence(
            current, eta=eta, lift=cfg.schur_lift, step=schur_step
        )
        bad_mask = observed[rescue_blocks] != bits_by_block[rescue_blocks]
        bad_blocks = rescue_blocks[bad_mask]
        if bad_blocks.size:
            field = opponent_field(current, eta)
            blocks = _blocks_from_field(field)
            coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
            stats = _embed_schur_coefficients(
                coeff,
                bits_by_block,
                step=schur_step,
                lift=cfg.schur_lift,
                max_log_scale=cfg.schur_max_log_scale,
                only_indices=bad_blocks,
            )
            target_blocks = idctn(coeff, type=2, norm="ortho", axes=(-2, -1))
            target_field = _field_from_blocks(target_blocks, field.shape[0], field.shape[1])
            current = _apply_field_delta(current, target_field - field, eta)
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
        observed_after, conf_after, _ = _extract_schur_bits_confidence(
            current, eta=eta, lift=cfg.schur_lift, step=schur_step
        )
        selected_acc = float(np.mean(observed_after[rescue_blocks] == bits_by_block[rescue_blocks]))
        history.append(
            {
                "iteration": iteration,
                "bad_before": int(bad_blocks.size),
                "selected_accuracy_after_primary_closure": selected_acc,
                "primary_updates": int(updates),
                "primary_unresolved": int(unresolved),
                "selected_mean_confidence": float(np.mean(conf_after[rescue_blocks])),
            }
        )
        if selected_acc >= 1.0 and int(unresolved) == 0:
            break

    validation = _validate_direct_matrices(
        current, eta=eta, lift=cfg.schur_lift, epsilon=cfg.direct_det_epsilon
    )
    schur_params = {
        "eta": eta,
        "lift": float(cfg.schur_lift),
        "step": float(schur_step),
        "payload_indices": payload_indices.astype(int).tolist(),
        "rescue_positions": rescue_positions.astype(int).tolist(),
        "rescue_blocks": rescue_blocks.astype(int).tolist(),
        "fusion_weight": float(cfg.fusion_weight),
        "selection_mode": str(selection_mode),
        "gate_power": float(cfg.gate_power),
        "schur_conf_floor": float(cfg.schur_conf_floor),
        "schur_conf_scale": float(cfg.schur_conf_scale),
    }
    return current, SparseSchurKey(base_key, cfg.to_dict(), schur_params), {
        "history": history,
        **validation,
    }


def extract_sparse_joint_schur(
    attacked_rgb: np.ndarray,
    key: SparseSchurKey,
    *,
    return_metadata: bool = False,
    schur_accept_threshold: float = 0.60,
    primary_erasure_threshold: float = 0.55,
):
    from qr64_certified.direct_schur_rescue import extract_components

    # Reuse the standard component extractor by presenting a compatible key.
    compatible = DirectSchurRescueKey(
        base=key.base,
        config=key.config,
        schur_params={
            **key.schur_params,
            "primary_conf_reference": 0.80,
        },
    )
    components = extract_components(np.asarray(attacked_rgb, dtype=np.uint8), compatible)
    cfg = DirectSchurRescueConfig.from_mapping(key.config)
    primary_map = np.asarray(components["primary_map"])
    primary_conf = np.asarray(components["primary_confidence"], dtype=np.float64)
    schur_map = np.asarray(components["schur_map"])
    schur_conf = np.asarray(components["schur_confidence"], dtype=np.float64)
    cert = np.asarray(components["certificate_map"], dtype=np.float64)
    primary_sign = np.where(primary_map > 0, 1.0, -1.0)
    schur_sign = np.where(schur_map > 0, 1.0, -1.0)
    base = cfg.base_config
    primary_evidence = (
        primary_sign
        * (base.evidence_conf_floor + base.evidence_conf_scale * primary_conf ** base.evidence_conf_power)
        * (base.evidence_certificate_floor + base.evidence_certificate_scale * cert)
    )
    mask_flat = np.zeros(primary_map.size, dtype=bool)
    mask_flat[np.asarray(key.schur_params["rescue_positions"], dtype=np.int32)] = True
    rescue_mask = _inverse_arnold_array(mask_flat.reshape(primary_map.shape), _normalize_jilp_key(key.base.base_key)).astype(bool)
    use = rescue_mask & (primary_conf < primary_erasure_threshold) & (schur_conf >= schur_accept_threshold)
    rescue_evidence = np.where(
        use,
        float(key.schur_params["fusion_weight"])
        * schur_sign
        * (float(key.schur_params["schur_conf_floor"]) + float(key.schur_params["schur_conf_scale"]) * schur_conf**2),
        0.0,
    )
    sync = components["sync"]
    if float(sync.get("score", 0.0)) >= 0.999 and float(np.mean(primary_conf)) >= 0.78:
        recovered = (primary_map > 0).astype(np.uint8) * 255
        path = "exact_primary"
    else:
        recovered = _icm_map(primary_evidence + rescue_evidence, base.qr_map_lambda, base.qr_map_iters)
        path = "sparse_joint_schur_erasure_rescue"
    meta = {
        "det_nonzero": bool(components["certificate"].all_nonsingular),
        "min_abs_det": float(components["certificate"].determinant.min()),
        "inference_path": path,
        "mean_rescue_usage": float(np.mean(use)),
        "mean_schur_confidence": float(np.mean(schur_conf[rescue_mask])) if np.any(rescue_mask) else 0.0,
    }
    return (recovered, meta) if return_metadata else recovered


# ---------------------------------------------------------------------------
# 3. Spatial CD-DetQR pilot-guided affine synchronization experiment
# ---------------------------------------------------------------------------


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(
        pil.rotate(float(degrees), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0)),
        dtype=np.uint8,
    )


def _shear(image: np.ndarray, shear_x: float) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    coeffs = (1.0, -float(shear_x), 0.0, 0.0, 1.0, 0.0)
    return np.asarray(
        pil.transform((w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)),
        dtype=np.uint8,
    )


def spatial_pilot_score(image: np.ndarray, raw_key: dict[str, Any]) -> float:
    from qr64_certified.cd_detqr import BlindCDDetQR, qr_determinant_features

    features = qr_determinant_features(image, int(raw_key["max_patterns"]))
    blocks = np.asarray(raw_key["pilot_blocks"], dtype=np.int64)
    pairs = np.asarray(raw_key["pilot_pair"], dtype=np.int64)
    patterns = np.asarray(raw_key["pilot_pattern"], dtype=np.int64)
    _, _, diff = BlindCDDetQR._selected_component_determinants(features, blocks, pairs, patterns)
    # Soft positive-margin score is more discriminative than majority votes.
    scale = np.median(np.abs(diff)) + 1e-8
    return float(np.mean(np.tanh(diff / scale)))


def extract_spatial_affine_sync(
    questioned_image: np.ndarray,
    key: Any,
    *,
    return_metadata: bool = False,
    angle_grid: tuple[float, ...] = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    shear_grid: tuple[float, ...] = (-0.10, -0.08, -0.06, -0.04, 0.0, 0.04, 0.06, 0.08, 0.10),
):
    from qr64_certified.cd_detqr import BlindCDDetQR

    raw_key = key.payload if hasattr(key, "payload") else key
    image = np.asarray(questioned_image, dtype=np.uint8)
    candidates: list[tuple[str, np.ndarray]] = [("identity", image)]
    candidates.extend((f"rotation_{a:+.1f}", _rotate(image, a)) for a in angle_grid if a != 0.0)
    candidates.extend((f"shear_{s:+.2f}", _shear(image, s)) for s in shear_grid if s != 0.0)
    scored = [(name, img, spatial_pilot_score(img, raw_key)) for name, img in candidates]
    best_name, best_img, best_score = max(scored, key=lambda x: x[2])
    identity_score = scored[0][2]
    # Require a meaningful gain to prevent unnecessary interpolation on nongeometric attacks.
    accepted = best_name != "identity" and best_score - identity_score > 0.30
    selected = best_img if accepted else image
    method = BlindCDDetQR()
    recovered, meta = method.extract_with_metadata(selected, raw_key)
    meta = {
        **meta,
        "sync_candidate": best_name if accepted else "identity",
        "sync_accepted": bool(accepted),
        "identity_pilot_score": float(identity_score),
        "best_pilot_score": float(best_score),
        "pilot_score_gain": float(best_score - identity_score),
    }
    return (recovered, meta) if return_metadata else recovered
