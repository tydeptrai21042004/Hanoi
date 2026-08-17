from __future__ import annotations

"""Transform-domain DCT->QR watermarking by direct modification of R11.

The proposal uses QR only after entering the transform domain:

    RGB -> luminance -> 8x8 DCT -> 4x4 low-frequency matrix A
        -> QR(A + lambda I) -> parity-QIM on R11 -> Q R' -> inverse DCT.

With canonical QR (positive diagonal), R11 is the Euclidean norm of the first
column of the regularized low-frequency DCT matrix.  It therefore acts as a
strong low-frequency energy carrier.  One watermark bit is embedded by forcing
the quantization-index parity of R11 while leaving every other entry of R
unchanged.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.fft import idctn

from .dct_qr_direct_r import (
    _LUMA,
    _LUMA_NORM2,
    _analysis,
    _block_order,
    _det_summary,
    _from_blocks,
    _project_parity_qim,
)

METHOD_ID = "dct_qr_r11_qim"


@dataclass(frozen=True)
class DCTQRR11QIMConfig:
    seed: int = 2026
    step: float = 12.0
    regularization: float = 1.0
    det_epsilon: float = 1e-10
    closure_rounds: int = 2
    block_size: int = 8
    matrix_size: int = 4
    watermark_size: int = 64

    def validated(self) -> "DCTQRR11QIMConfig":
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.det_epsilon <= 0:
            raise ValueError("det_epsilon must be positive")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least 1")
        if self.block_size != 8:
            raise ValueError("The current proposal is defined for 8x8 DCT blocks.")
        if self.matrix_size != 4:
            raise ValueError("The current proposal is defined for a 4x4 DCT-QR matrix.")
        if self.watermark_size != 64:
            raise ValueError("The repository protocol keeps the watermark at 64x64.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, value: "DCTQRR11QIMConfig | Mapping[str, Any] | None"
    ) -> "DCTQRR11QIMConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(value).items() if k in allowed}).validated()


@dataclass(frozen=True)
class DCTQRR11QIMKey:
    config: dict[str, Any]
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def certificate_mode(self) -> str:
        return "transform_dct_qr_direct_r11"


def _analysis_r11(rgb: np.ndarray, cfg: DCTQRR11QIMConfig):
    # Reuse the numerically identical DCT->regularized-QR analysis path while
    # preserving a separate config/key type for this independent proposal.
    from .dct_qr_direct_r import DCTQRDirectRConfig

    common = DCTQRDirectRConfig(
        seed=cfg.seed,
        step=cfg.step,
        regularization=cfg.regularization,
        det_epsilon=cfg.det_epsilon,
        closure_rounds=cfg.closure_rounds,
        block_size=cfg.block_size,
        matrix_size=cfg.matrix_size,
        watermark_size=cfg.watermark_size,
    )
    return _analysis(rgb, common)


def _positive_r11_targets(values: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    """Parity-QIM projection with a strictly positive R11 safety floor."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    targets = _project_parity_qim(values, b, step)
    # Canonical QR requires positive diagonal entries for stable blind parity.
    # For the extremely unlikely near-zero R11 case, choose the smallest
    # strictly positive quantizer index with the requested parity.
    fallback = np.where(b == 1, float(step), 2.0 * float(step))
    return np.where(targets > 0.0, targets, fallback)


def _embed_once(
    image: np.ndarray,
    payload_bits: np.ndarray,
    order: np.ndarray,
    cfg: DCTQRR11QIMConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    y, coeff, matrices, q, r, bh, bw = _analysis_r11(image, cfg)
    det_meta = _det_summary(matrices, cfg.det_epsilon)
    if not det_meta["det_nonzero"]:
        raise ValueError(
            "The regularized transform-domain QR matrix is singular in "
            f"{det_meta['near_singular_blocks']} block(s). Increase regularization."
        )

    selected = r[order, 0, 0]
    targets = _positive_r11_targets(selected, payload_bits, cfg.step)
    delta = targets - selected

    r2 = r.copy()
    r2[order, 0, 0] = targets

    reconstructed = q @ r2
    m = cfg.matrix_size
    reconstructed -= float(cfg.regularization) * np.eye(m, dtype=np.float64)[None, :, :]
    coeff2 = coeff.copy()
    coeff2[:, :m, :m] = reconstructed

    spatial_blocks = idctn(coeff2, type=2, norm="ortho", axes=(-2, -1))
    y2 = _from_blocks(spatial_blocks, bh, bw)

    dy = y2 - y
    output = np.asarray(image, dtype=np.float64) + dy[..., None] * (_LUMA / _LUMA_NORM2)
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)

    rel = np.abs(delta) / np.maximum(np.abs(selected), 1e-12)
    return output, {
        **det_meta,
        "mean_abs_r11_update": float(np.mean(np.abs(delta))),
        "max_abs_r11_update": float(np.max(np.abs(delta))),
        "mean_relative_r11_update": float(np.mean(rel)),
        "max_relative_r11_update": float(np.max(rel)),
    }


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DCTQRR11QIMConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    cfg = DCTQRR11QIMConfig.from_mapping(config)
    host = np.asarray(host_rgb, dtype=np.uint8)
    watermark = np.asarray(watermark_binary)
    if watermark.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(
            f"Watermark must be {cfg.watermark_size}x{cfg.watermark_size}; got {watermark.shape}."
        )

    wm_values = watermark.astype(np.float64).reshape(-1)
    threshold = 0.5 if wm_values.size and float(np.max(wm_values)) <= 1.0 else 127.0
    bits = (wm_values > threshold).astype(np.uint8)

    total_blocks = (host.shape[0] // cfg.block_size) * (host.shape[1] // cfg.block_size)
    order = _block_order(total_blocks, bits.size, cfg.seed)

    output = host.copy()
    round_meta: list[dict[str, Any]] = []
    for _ in range(cfg.closure_rounds):
        output, metadata = _embed_once(output, bits, order, cfg)
        round_meta.append(metadata)

    _y, _coeff, final_matrices, _q, final_r, _bh, _bw = _analysis_r11(output, cfg)
    final_det = _det_summary(final_matrices, cfg.det_epsilon)
    selected_final = final_r[order, 0, 0]
    final_indices = np.rint(selected_final / float(cfg.step)).astype(np.int64)
    final_bits = (final_indices & 1).astype(np.uint8)

    key = DCTQRR11QIMKey(
        config=cfg.to_dict(),
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in watermark.shape),
    )
    metadata = {
        "method_id": METHOD_ID,
        "domain": "transform_domain_dct_then_qr_direct_r11",
        "qr_input": "regularized_4x4_low_frequency_DCT_matrix",
        "embedding_carrier": "R11_direct_parity_qim",
        "r11_modified": True,
        "other_r_entries_modified_explicitly": False,
        "step": float(cfg.step),
        "regularization": float(cfg.regularization),
        "closure_rounds": int(cfg.closure_rounds),
        "fully_blind": True,
        "original_host_used_for_extraction": False,
        "post_closure_bit_errors": int(np.count_nonzero(final_bits != bits)),
        **final_det,
        "rounds": round_meta,
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: DCTQRR11QIMKey,
    *,
    return_metadata: bool = False,
):
    if not isinstance(key, DCTQRR11QIMKey):
        raise TypeError("Expected DCTQRR11QIMKey.")
    cfg = DCTQRR11QIMConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(int(x) for x in image.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {image.shape} does not match key shape {key.host_shape}.")

    _y, _coeff, matrices, _q, r, _bh, _bw = _analysis_r11(image, cfg)
    det_meta = _det_summary(matrices, cfg.det_epsilon)

    payload_len = int(np.prod(key.watermark_shape))
    order = _block_order(r.shape[0], payload_len, cfg.seed)
    carrier = r[order, 0, 0]
    indices = np.rint(carrier / float(cfg.step)).astype(np.int64)
    bits = (indices & 1).astype(np.uint8)
    recovered = (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)

    metadata = {
        "method_id": METHOD_ID,
        "domain": "transform_domain_dct_then_qr_direct_r11",
        "embedding_carrier": "R11_direct_parity_qim",
        "r11_modified_at_embedding": True,
        "inference_path": "direct_transform_qr_r11_parity_qim",
        "fully_blind": True,
        "original_host_used": False,
        **det_meta,
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "METHOD_ID",
    "DCTQRR11QIMConfig",
    "DCTQRR11QIMKey",
    "embed",
    "extract",
]
