from __future__ import annotations

"""Spatial-domain QR direct-R11 parity QIM (no DCT/IDCT)."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .spatial_qr_common import (
    block_order,
    determinant_summary,
    from_blocks,
    project_parity_qim,
    replace_central_patch,
    rgb_from_luminance_delta,
    spatial_qr_analysis,
)

METHOD_ID = "spatial_qr_r11_qim"


@dataclass(frozen=True)
class SpatialQRR11QIMConfig:
    seed: int = 2026
    step: float = 12.0
    regularization: float = 1.0
    det_epsilon: float = 1e-10
    closure_rounds: int = 4
    block_size: int = 8
    matrix_size: int = 4
    watermark_size: int = 64

    def validated(self) -> "SpatialQRR11QIMConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.det_epsilon <= 0:
            raise ValueError("det_epsilon must be positive")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if self.block_size != 8 or self.matrix_size != 4:
            raise ValueError("Spatial QR R11-QIM is defined for 8x8 blocks and a central 4x4 matrix.")
        if self.watermark_size != 64:
            raise ValueError("The repository protocol keeps the watermark at 64x64.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: "SpatialQRR11QIMConfig | Mapping[str, Any] | None") -> "SpatialQRR11QIMConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(value).items() if k in allowed}).validated()


@dataclass(frozen=True)
class SpatialQRR11QIMKey:
    config: dict[str, Any]
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def certificate_mode(self) -> str:
        return "spatial_qr_direct_r11"


def _analysis(rgb: np.ndarray, cfg: SpatialQRR11QIMConfig):
    return spatial_qr_analysis(
        rgb,
        block_size=cfg.block_size,
        matrix_size=cfg.matrix_size,
        regularization=cfg.regularization,
        mean_center=False,
    )


def _positive_targets(
    values: np.ndarray,
    bits: np.ndarray,
    step: float,
    force_lower: np.ndarray | None = None,
) -> np.ndarray:
    """Positive parity targets with an optional same-parity downward rescue.

    The normal target is always the nearest requested-parity lattice point. If
    a previous uint8 closure round proves that an upward target is unreachable
    for a particular payload block (typically due to saturation), that block
    may use the immediately lower same-parity point, two indices away.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    targets = project_parity_qim(v, b, step)
    fallback = np.where(b == 1, float(step), 2.0 * float(step))
    targets = np.where(targets > 0.0, targets, fallback)
    if force_lower is not None:
        mask = np.asarray(force_lower, dtype=bool).reshape(-1)
        if mask.size != v.size:
            raise ValueError("force_lower must match the R11 carrier length")
        lower_same_parity = targets - 2.0 * float(step)
        use_lower = mask & (targets > v) & (lower_same_parity > 0.0)
        targets = np.where(use_lower, lower_same_parity, targets)
    return targets


def _embed_once(
    image: np.ndarray,
    bits: np.ndarray,
    order: np.ndarray,
    cfg: SpatialQRR11QIMConfig,
    force_lower: np.ndarray | None = None,
):
    y, blocks, matrices, q, r, bh, bw, offsets = _analysis(image, cfg)
    selected = r[order, 0, 0]
    targets = _positive_targets(selected, bits, cfg.step, force_lower)
    delta = targets - selected
    r2 = r.copy()
    r2[order, 0, 0] = targets
    reconstructed = q @ r2
    blocks2 = replace_central_patch(
        blocks,
        reconstructed,
        matrix_size=cfg.matrix_size,
        regularization=cfg.regularization,
        offsets=offsets,
    )
    y2 = from_blocks(blocks2, bh, bw)
    output = rgb_from_luminance_delta(image, y2 - y)
    return output, {
        **determinant_summary(matrices, cfg.det_epsilon),
        "mean_abs_r11_update": float(np.mean(np.abs(delta))),
        "max_abs_r11_update": float(np.max(np.abs(delta))),
    }


def embed(host_rgb: np.ndarray, watermark_binary: np.ndarray, *, config=None, return_metadata: bool = False):
    cfg = SpatialQRR11QIMConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    watermark = np.asarray(watermark_binary)
    if watermark.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"Watermark must be {cfg.watermark_size}x{cfg.watermark_size}; got {watermark.shape}.")
    values = watermark.astype(np.float64).reshape(-1)
    threshold = 0.5 if values.size and float(np.max(values)) <= 1.0 else 127.0
    bits = (values > threshold).astype(np.uint8)
    total_blocks = (host.shape[0] // cfg.block_size) * (host.shape[1] // cfg.block_size)
    order = block_order(total_blocks, bits.size, cfg.seed)

    output = host.copy()
    rounds: list[dict[str, Any]] = []
    force_lower = np.zeros(bits.size, dtype=bool)
    for _ in range(cfg.closure_rounds):
        output, meta = _embed_once(output, bits, order, cfg, force_lower)
        _yc, _bc, _mc, _qc, rc, _bhc, _bwc, _oc = _analysis(output, cfg)
        closure_bits = (np.rint(rc[order, 0, 0] / cfg.step).astype(np.int64) & 1).astype(np.uint8)
        force_lower = closure_bits != bits
        meta = dict(meta)
        meta["remaining_bit_errors"] = int(np.count_nonzero(force_lower))
        rounds.append(meta)
        if not np.any(force_lower):
            break

    _y, _blocks, matrices, _q, r, _bh, _bw, _offsets = _analysis(output, cfg)
    selected = r[order, 0, 0]
    decoded = (np.rint(selected / cfg.step).astype(np.int64) & 1).astype(np.uint8)
    key = SpatialQRR11QIMKey(cfg.to_dict(), tuple(int(x) for x in host.shape), tuple(int(x) for x in watermark.shape))
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_qr_direct_r11_no_dct",
        "qr_input": "central_4x4_spatial_luminance_patch_plus_virtual_diagonal",
        "embedding_carrier": "R11_direct_parity_qim",
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        "post_closure_bit_errors": int(np.count_nonzero(decoded != bits)),
        **determinant_summary(matrices, cfg.det_epsilon),
        "rounds": rounds,
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(possibly_attacked_rgb: np.ndarray, key: SpatialQRR11QIMKey, *, return_metadata: bool = False):
    if not isinstance(key, SpatialQRR11QIMKey):
        raise TypeError("Expected SpatialQRR11QIMKey.")
    cfg = SpatialQRR11QIMConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(int(x) for x in image.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {image.shape} does not match key shape {key.host_shape}.")
    _y, _blocks, matrices, _q, r, _bh, _bw, _offsets = _analysis(image, cfg)
    payload_len = int(np.prod(key.watermark_shape))
    order = block_order(r.shape[0], payload_len, cfg.seed)
    carrier = r[order, 0, 0]
    bits = (np.rint(carrier / cfg.step).astype(np.int64) & 1).astype(np.uint8)
    recovered = (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_qr_direct_r11_no_dct",
        "inference_path": "direct_spatial_qr_r11_parity_qim",
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        **determinant_summary(matrices, cfg.det_epsilon),
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = ["METHOD_ID", "SpatialQRR11QIMConfig", "SpatialQRR11QIMKey", "embed", "extract"]
