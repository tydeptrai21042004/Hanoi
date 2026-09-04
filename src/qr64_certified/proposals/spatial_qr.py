from __future__ import annotations

"""Spatial QR-assisted pairwise-coset QIM with gain normalization (no DCT).

The payload carrier is a directional luminance difference from two pixels that
lie outside the central QR certificate patch:

    v_b = (Y_b[1,2] - Y_b[2,1]) / 2.

The central mean-centred 4x4 spatial patch supplies QR reliability and an R11
scale reference.  Reliability assigns one of three QIM spacings and forms
homogeneous pairs.  For each pair, one binary coset label is selected by exact
minimum projection energy.  Extraction divides the attacked carrier by the
R11 gain ratio before parity decoding.  No DCT/IDCT is used.
"""

import base64
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .spatial_qr_common import (
    block_order,
    determinant_summary,
    from_blocks,
    project_parity_qim,
    qr_reliability,
    rgb_from_luminance_delta,
    spatial_qr_analysis,
)

METHOD_ID = "spatial_qr"


@dataclass(frozen=True)
class SpatialQRConfig:
    seed: int = 2026
    step: float = 10.0
    weak_step_multiplier: float = 1.25
    strong_step_multiplier: float = 0.75
    weak_fraction: float = 0.30
    strong_fraction: float = 0.30
    gain_gamma: float = 1.0
    gain_clip_min: float = 0.55
    gain_clip_max: float = 1.80
    regularization: float = 1.0
    det_epsilon: float = 1e-10
    closure_rounds: int = 3
    block_size: int = 8
    matrix_size: int = 4
    watermark_size: int = 64
    coset_group_size: int = 2

    def validated(self) -> "SpatialQRConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not (0 < self.strong_step_multiplier <= 1 <= self.weak_step_multiplier):
            raise ValueError("Require 0 < strong_step_multiplier <= 1 <= weak_step_multiplier")
        if not (0 <= self.weak_fraction < 1 and 0 <= self.strong_fraction < 1):
            raise ValueError("step fractions must lie in [0,1)")
        if self.weak_fraction + self.strong_fraction >= 1:
            raise ValueError("weak_fraction + strong_fraction must be < 1")
        if self.gain_gamma < 0:
            raise ValueError("gain_gamma must be nonnegative")
        if not 0 < self.gain_clip_min < self.gain_clip_max:
            raise ValueError("invalid gain clipping interval")
        if self.regularization <= 0 or self.det_epsilon <= 0:
            raise ValueError("regularization and det_epsilon must be positive")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if self.block_size != 8 or self.matrix_size != 4:
            raise ValueError("Spatial QR is defined for 8x8 blocks and a central 4x4 certificate matrix.")
        if self.watermark_size != 64:
            raise ValueError("The repository protocol keeps the watermark at 64x64.")
        if self.coset_group_size < 2:
            raise ValueError("coset_group_size must be at least 2")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: "SpatialQRConfig | Mapping[str, Any] | None") -> "SpatialQRConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(value).items() if k in allowed}).validated()


@dataclass(frozen=True)
class SpatialQRKey:
    config: dict[str, Any]
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]
    step_by_payload: tuple[float, ...]
    gain_reference_r11: tuple[float, ...]
    coset_flip_mask_b64: str

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def certificate_mode(self) -> str:
        return "spatial_qr_pairwise_coset_gain"


def _pack_mask(bits: np.ndarray) -> str:
    packed = np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1) & 1, bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _unpack_mask(encoded: str, count: int) -> np.ndarray:
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    if values.size < count:
        raise ValueError("Stored coset mask is shorter than the payload.")
    return values[:count].astype(np.uint8)


def _analysis(rgb: np.ndarray, cfg: SpatialQRConfig):
    return spatial_qr_analysis(
        rgb,
        block_size=cfg.block_size,
        matrix_size=cfg.matrix_size,
        regularization=cfg.regularization,
        mean_center=True,
    )


def _carrier(blocks: np.ndarray) -> np.ndarray:
    # Outside the central 4x4 certificate patch, so embedding does not directly
    # overwrite the QR reference used for gain estimation.
    return 0.5 * (blocks[:, 1, 2] - blocks[:, 2, 1])


def _step_classes(reliability: np.ndarray, cfg: SpatialQRConfig) -> np.ndarray:
    rel = np.asarray(reliability, dtype=np.float64).reshape(-1)
    order = np.argsort(rel, kind="stable")
    n = rel.size
    weak_n = int(round(cfg.weak_fraction * n))
    strong_n = int(round(cfg.strong_fraction * n))
    out = np.full(n, float(cfg.step), dtype=np.float64)
    out[order[:weak_n]] = float(cfg.step * cfg.weak_step_multiplier)
    if strong_n:
        out[order[n - strong_n :]] = float(cfg.step * cfg.strong_step_multiplier)
    return out


def _pairwise_coset(carrier: np.ndarray, bits: np.ndarray, steps: np.ndarray, reliability: np.ndarray, group_size: int):
    v = np.asarray(carrier, dtype=np.float64).reshape(-1)
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    s = np.asarray(steps, dtype=np.float64).reshape(-1)
    r = np.asarray(reliability, dtype=np.float64).reshape(-1)
    target_b = np.empty_like(v)
    target_flip = np.empty_like(v)
    for i in range(v.size):
        target_b[i] = project_parity_qim([v[i]], [b[i]], s[i])[0]
        target_flip[i] = project_parity_qim([v[i]], [b[i] ^ 1], s[i])[0]
    cost0 = (target_b - v) ** 2
    cost1 = (target_flip - v) ** 2
    order = np.argsort(r, kind="stable")
    flips = np.zeros(v.size, dtype=np.uint8)
    original = float(cost0.sum())
    optimized = 0.0
    flipped_groups = 0
    groups = 0
    for start in range(0, v.size, int(group_size)):
        group = order[start : start + int(group_size)]
        c0 = float(cost0[group].sum())
        c1 = float(cost1[group].sum())
        label = np.uint8(c1 < c0)
        flips[group] = label
        optimized += min(c0, c1)
        flipped_groups += int(label)
        groups += 1
    return b ^ flips, flips, {
        "group_count": groups,
        "flipped_group_count": flipped_groups,
        "original_projection_energy": original,
        "optimized_projection_energy": optimized,
        "projection_energy_ratio": optimized / max(original, 1e-18),
    }


def _embed_once(image: np.ndarray, encoded_bits: np.ndarray, order: np.ndarray, steps: np.ndarray, cfg: SpatialQRConfig):
    y, blocks, matrices, _q, _r, bh, bw, _offsets = _analysis(image, cfg)
    carrier = _carrier(blocks)[order]
    targets = np.empty_like(carrier)
    for i in range(carrier.size):
        targets[i] = project_parity_qim([carrier[i]], [encoded_bits[i]], steps[i])[0]
    delta = targets - carrier
    blocks2 = blocks.copy()
    # v=(a-b)/2; +delta on a and -delta on b changes v by exactly delta.
    blocks2[order, 1, 2] += delta
    blocks2[order, 2, 1] -= delta
    y2 = from_blocks(blocks2, bh, bw)
    output = rgb_from_luminance_delta(image, y2 - y)
    return output, {
        **determinant_summary(matrices, cfg.det_epsilon),
        "mean_abs_qim_update": float(np.mean(np.abs(delta))),
        "max_abs_qim_update": float(np.max(np.abs(delta))),
    }


def embed(host_rgb: np.ndarray, watermark_binary: np.ndarray, *, config=None, return_metadata: bool = False):
    cfg = SpatialQRConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    watermark = np.asarray(watermark_binary)
    if watermark.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"Watermark must be {cfg.watermark_size}x{cfg.watermark_size}; got {watermark.shape}.")
    values = watermark.astype(np.float64).reshape(-1)
    threshold = 0.5 if values.size and float(np.max(values)) <= 1.0 else 127.0
    bits = (values > threshold).astype(np.uint8)
    total_blocks = (host.shape[0] // cfg.block_size) * (host.shape[1] // cfg.block_size)
    order = block_order(total_blocks, bits.size, cfg.seed)

    _y, blocks0, matrices0, _q0, r0, _bh, _bw, _offsets = _analysis(host, cfg)
    reliability = qr_reliability(r0)[order]
    steps = _step_classes(reliability, cfg)
    encoded, flips, coset = _pairwise_coset(_carrier(blocks0)[order], bits, steps, reliability, cfg.coset_group_size)

    output = host.copy()
    rounds: list[dict[str, Any]] = []
    for _ in range(cfg.closure_rounds):
        output, meta = _embed_once(output, encoded, order, steps, cfg)
        rounds.append(meta)

    _yf, blocks_f, matrices_f, _qf, rf, _bhf, _bwf, _offsetsf = _analysis(output, cfg)
    gain_ref = np.maximum(np.abs(rf[order, 0, 0]), 1e-8)
    final_carrier = _carrier(blocks_f)[order]
    decoded_encoded = np.empty(bits.size, dtype=np.uint8)
    for i in range(bits.size):
        decoded_encoded[i] = int(np.rint(final_carrier[i] / steps[i])) & 1
    decoded = decoded_encoded ^ flips

    key = SpatialQRKey(
        config=cfg.to_dict(),
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in watermark.shape),
        step_by_payload=tuple(float(x) for x in steps),
        gain_reference_r11=tuple(float(x) for x in gain_ref),
        coset_flip_mask_b64=_pack_mask(flips),
    )
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_directional_difference_with_qr_certificate_no_dct",
        "embedding_carrier": "(Y[1,2]-Y[2,1])/2",
        "qr_role": "mean_centered_spatial_4x4_reliability_and_R11_gain_reference",
        "pairwise_coset_optimization": True,
        "gain_normalization": True,
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        "cover_dependent_key": True,
        "post_closure_bit_errors": int(np.count_nonzero(decoded != bits)),
        **determinant_summary(matrices_f, cfg.det_epsilon),
        **coset,
        "rounds": rounds,
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(possibly_attacked_rgb: np.ndarray, key: SpatialQRKey, *, return_metadata: bool = False):
    if not isinstance(key, SpatialQRKey):
        raise TypeError("Expected SpatialQRKey.")
    cfg = SpatialQRConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(int(x) for x in image.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {image.shape} does not match key shape {key.host_shape}.")
    _y, blocks, matrices, _q, r, _bh, _bw, _offsets = _analysis(image, cfg)
    payload_len = int(np.prod(key.watermark_shape))
    order = block_order(r.shape[0], payload_len, cfg.seed)
    steps = np.asarray(key.step_by_payload, dtype=np.float64)
    if steps.size != payload_len:
        raise ValueError("Stored step schedule length does not match payload.")
    ref = np.asarray(key.gain_reference_r11, dtype=np.float64)
    if ref.size != payload_len:
        raise ValueError("Stored QR gain-reference length does not match payload.")
    current = np.maximum(np.abs(r[order, 0, 0]), 1e-8)
    gain = np.clip(current / np.maximum(ref, 1e-8), cfg.gain_clip_min, cfg.gain_clip_max)
    normalized = _carrier(blocks)[order] / np.power(gain, cfg.gain_gamma)
    encoded = (np.rint(normalized / steps).astype(np.int64) & 1).astype(np.uint8)
    flips = _unpack_mask(key.coset_flip_mask_b64, payload_len)
    bits = encoded ^ flips
    recovered = (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_directional_difference_with_qr_certificate_no_dct",
        "inference_path": "spatial_pairwise_coset_qim_with_R11_gain_normalization",
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        "cover_dependent_key": True,
        "median_gain": float(np.median(gain)),
        "min_gain": float(np.min(gain)),
        "max_gain": float(np.max(gain)),
        **determinant_summary(matrices, cfg.det_epsilon),
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = ["METHOD_ID", "SpatialQRConfig", "SpatialQRKey", "embed", "extract"]
