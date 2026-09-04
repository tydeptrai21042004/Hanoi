from __future__ import annotations

"""Spatial-domain QR direct-R differential QIM (no DCT/IDCT).

Pipeline:
    RGB -> luminance -> 8x8 spatial blocks -> central 4x4 patch
        -> QR(A + lambda I) -> QIM on (R12-R13)/2 -> Q R' -> spatial patch.

For a desired carrier displacement ``delta``, the update
``R12 += delta, R13 -= delta`` is the minimum-Frobenius two-entry update that
realizes the constraint and preserves ``R12+R13`` and every diagonal entry.
"""

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

METHOD_ID = "spatial_qr_direct_r"


@dataclass(frozen=True)
class SpatialQRDirectRConfig:
    seed: int = 2026
    step: float = 8.0
    regularization: float = 1.0
    det_epsilon: float = 1e-10
    closure_rounds: int = 3
    block_size: int = 8
    matrix_size: int = 4
    watermark_size: int = 64

    def validated(self) -> "SpatialQRDirectRConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.det_epsilon <= 0:
            raise ValueError("det_epsilon must be positive")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if self.block_size != 8 or self.matrix_size != 4:
            raise ValueError("Spatial QR direct-R is defined for 8x8 blocks and a central 4x4 matrix.")
        if self.watermark_size != 64:
            raise ValueError("The repository protocol keeps the watermark at 64x64.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: "SpatialQRDirectRConfig | Mapping[str, Any] | None") -> "SpatialQRDirectRConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(value).items() if k in allowed}).validated()


@dataclass(frozen=True)
class SpatialQRDirectRKey:
    config: dict[str, Any]
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def certificate_mode(self) -> str:
        return "spatial_qr_direct_r"


def _analysis(rgb: np.ndarray, cfg: SpatialQRDirectRConfig):
    return spatial_qr_analysis(
        rgb,
        block_size=cfg.block_size,
        matrix_size=cfg.matrix_size,
        regularization=cfg.regularization,
        mean_center=False,
    )


def _embed_once(image: np.ndarray, payload_bits: np.ndarray, order: np.ndarray, cfg: SpatialQRDirectRConfig):
    y, blocks, matrices, q, r, bh, bw, offsets = _analysis(image, cfg)
    det_meta = determinant_summary(matrices, cfg.det_epsilon)
    carrier = 0.5 * (r[:, 0, 1] - r[:, 0, 2])
    selected = carrier[order]
    targets = project_parity_qim(selected, payload_bits, cfg.step)
    delta = targets - selected

    r2 = r.copy()
    r2[order, 0, 1] += delta
    r2[order, 0, 2] -= delta
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
        **det_meta,
        "mean_abs_qim_update": float(np.mean(np.abs(delta))),
        "max_abs_qim_update": float(np.max(np.abs(delta))),
    }


def embed(host_rgb: np.ndarray, watermark_binary: np.ndarray, *, config=None, return_metadata: bool = False):
    cfg = SpatialQRDirectRConfig.from_mapping(config)
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
    for _ in range(cfg.closure_rounds):
        output, meta = _embed_once(output, bits, order, cfg)
        rounds.append(meta)

    _y, _blocks, matrices, _q, final_r, _bh, _bw, _offsets = _analysis(output, cfg)
    carrier = 0.5 * (final_r[order, 0, 1] - final_r[order, 0, 2])
    decoded = (np.rint(carrier / cfg.step).astype(np.int64) & 1).astype(np.uint8)
    key = SpatialQRDirectRKey(cfg.to_dict(), tuple(int(x) for x in host.shape), tuple(int(x) for x in watermark.shape))
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_qr_direct_r_no_dct",
        "qr_input": "central_4x4_spatial_luminance_patch_plus_virtual_diagonal",
        "embedding_carrier": "(R12-R13)/2_first_row_differential",
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        "post_closure_bit_errors": int(np.count_nonzero(decoded != bits)),
        **determinant_summary(matrices, cfg.det_epsilon),
        "rounds": rounds,
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(possibly_attacked_rgb: np.ndarray, key: SpatialQRDirectRKey, *, return_metadata: bool = False):
    if not isinstance(key, SpatialQRDirectRKey):
        raise TypeError("Expected SpatialQRDirectRKey.")
    cfg = SpatialQRDirectRConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(int(x) for x in image.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {image.shape} does not match key shape {key.host_shape}.")
    _y, _blocks, matrices, _q, r, _bh, _bw, _offsets = _analysis(image, cfg)
    payload_len = int(np.prod(key.watermark_shape))
    order = block_order(r.shape[0], payload_len, cfg.seed)
    carrier = 0.5 * (r[order, 0, 1] - r[order, 0, 2])
    bits = (np.rint(carrier / cfg.step).astype(np.int64) & 1).astype(np.uint8)
    recovered = (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)
    metadata = {
        "method_id": METHOD_ID,
        "domain": "spatial_luminance_qr_direct_r_no_dct",
        "inference_path": "direct_spatial_qr_r_parity_qim",
        "transform_used": False,
        "dct_used": False,
        "fully_blind": True,
        **determinant_summary(matrices, cfg.det_epsilon),
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = ["METHOD_ID", "SpatialQRDirectRConfig", "SpatialQRDirectRKey", "embed", "extract"]
