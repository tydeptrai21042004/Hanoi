from __future__ import annotations

"""Direct transform-domain DCT->QR watermarking on the first row of R.

This proposal is intentionally different from the existing ``dct_qr`` method.
The existing method embeds its payload on a DCT-QIM carrier and uses QR as a
certificate/reliability mechanism.  Here the payload is embedded *after* QR:

    RGB -> luminance -> 8x8 DCT -> 4x4 low-frequency A -> QR -> modify R
        -> inverse QR -> inverse DCT -> RGB.

For each block, a fixed virtual diagonal regularization is used only while
forming the QR matrix, A_lambda = A + lambda I.  It avoids zero determinants
without consuming blocks, which is important because a 512x512 host contains
exactly 4096 non-overlapping 8x8 blocks for a 64x64 payload.

The direct carrier is the first-row differential

    v = (R[0,1] - R[0,2]) / 2.

A parity-QIM projection embeds one bit.  The update applies +delta to R[0,1]
and -delta to R[0,2], preserving their sum and leaving R[0,0] (R11) untouched
as the dominant low-frequency energy anchor.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn, idctn

METHOD_ID = "dct_qr_direct_r"

_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
_LUMA_NORM2 = float(np.dot(_LUMA, _LUMA))


@dataclass(frozen=True)
class DCTQRDirectRConfig:
    seed: int = 2026
    step: float = 8.0
    regularization: float = 1.0
    det_epsilon: float = 1e-10
    closure_rounds: int = 2
    block_size: int = 8
    matrix_size: int = 4
    watermark_size: int = 64

    def validated(self) -> "DCTQRDirectRConfig":
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
        cls, value: "DCTQRDirectRConfig | Mapping[str, Any] | None"
    ) -> "DCTQRDirectRConfig":
        if value is None:
            return cls().validated()
        if isinstance(value, cls):
            return value.validated()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(value).items() if k in allowed}).validated()


@dataclass(frozen=True)
class DCTQRDirectRKey:
    config: dict[str, Any]
    host_shape: tuple[int, int, int]
    watermark_shape: tuple[int, int]

    @property
    def fully_blind(self) -> bool:
        # The key contains no original-host coefficients or host-dependent map.
        return True

    @property
    def certificate_mode(self) -> str:
        return "transform_dct_qr_direct_r"


def _luminance(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image; got {x.shape}")
    return np.tensordot(x, _LUMA, axes=([-1], [0]))


def _to_blocks(field: np.ndarray, block_size: int) -> tuple[np.ndarray, int, int]:
    y = np.asarray(field, dtype=np.float64)
    h, w = y.shape
    if h % block_size or w % block_size:
        raise ValueError("Image dimensions must be divisible by the DCT block size.")
    bh, bw = h // block_size, w // block_size
    blocks = (
        y.reshape(bh, block_size, bw, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
    )
    return blocks, bh, bw


def _from_blocks(blocks: np.ndarray, bh: int, bw: int) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    block_size = b.shape[-1]
    return (
        b.reshape(bh, bw, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(bh * block_size, bw * block_size)
    )


def _canonical_qr(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(matrices, dtype=np.float64))
    diagonal = np.diagonal(r, axis1=-2, axis2=-1)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q = q * signs[..., None, :]
    r = signs[..., :, None] * r
    return q, r


def _block_order(total_blocks: int, payload_len: int, seed: int) -> np.ndarray:
    if payload_len > total_blocks:
        raise ValueError(
            f"Payload needs {payload_len} blocks, but only {total_blocks} are available."
        )
    return np.random.default_rng(int(seed)).permutation(total_blocks)[:payload_len]


def _project_parity_qim(values: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    if v.size != b.size:
        raise ValueError("Carrier and bit arrays must have equal length.")

    k = np.rint(v / float(step)).astype(np.int64)
    wrong = (k & 1) != b
    lower = k - 1
    upper = k + 1
    choose_upper = np.abs(upper * step - v) < np.abs(lower * step - v)
    corrected = np.where(choose_upper, upper, lower)
    k = np.where(wrong, corrected, k)
    return k.astype(np.float64) * float(step)


def _analysis(
    rgb: np.ndarray, cfg: DCTQRDirectRConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    y = _luminance(rgb)
    blocks, bh, bw = _to_blocks(y, cfg.block_size)
    coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    m = cfg.matrix_size
    eye = np.eye(m, dtype=np.float64)[None, :, :]
    matrices = coeff[:, :m, :m].copy() + float(cfg.regularization) * eye
    q, r = _canonical_qr(matrices)
    return y, coeff, matrices, q, r, bh, bw


def _det_summary(matrices: np.ndarray, epsilon: float) -> dict[str, Any]:
    determinant = np.abs(np.linalg.det(np.asarray(matrices, dtype=np.float64)))
    return {
        "det_nonzero": bool(np.all(determinant > float(epsilon))),
        "min_abs_det": float(np.min(determinant)),
        "median_abs_det": float(np.median(determinant)),
        "near_singular_blocks": int(np.count_nonzero(determinant <= float(epsilon))),
    }


def _embed_once(
    image: np.ndarray,
    payload_bits: np.ndarray,
    order: np.ndarray,
    cfg: DCTQRDirectRConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    y, coeff, matrices, q, r, bh, bw = _analysis(image, cfg)
    det_meta = _det_summary(matrices, cfg.det_epsilon)
    if not det_meta["det_nonzero"]:
        raise ValueError(
            "The regularized transform-domain QR matrix is singular in "
            f"{det_meta['near_singular_blocks']} block(s). Increase regularization."
        )

    carrier = 0.5 * (r[:, 0, 1] - r[:, 0, 2])
    selected = carrier[order]
    targets = _project_parity_qim(selected, payload_bits, cfg.step)
    delta = targets - selected

    r2 = r.copy()
    r2[order, 0, 1] += delta
    r2[order, 0, 2] -= delta

    reconstructed = q @ r2
    m = cfg.matrix_size
    reconstructed -= float(cfg.regularization) * np.eye(m, dtype=np.float64)[None, :, :]
    coeff2 = coeff.copy()
    coeff2[:, :m, :m] = reconstructed
    spatial_blocks = idctn(coeff2, type=2, norm="ortho", axes=(-2, -1))
    y2 = _from_blocks(spatial_blocks, bh, bw)

    # Minimum-L2 RGB correction that realizes the desired luminance change
    # before uint8 clipping/rounding. Closure rounds repair residual parity
    # errors introduced by the integer image lattice and saturation.
    dy = y2 - y
    output = np.asarray(image, dtype=np.float64) + dy[..., None] * (_LUMA / _LUMA_NORM2)
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)

    return output, {
        **det_meta,
        "mean_abs_qim_update": float(np.mean(np.abs(delta))),
        "max_abs_qim_update": float(np.max(np.abs(delta))),
    }


def embed(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: DCTQRDirectRConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    cfg = DCTQRDirectRConfig.from_mapping(config)
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

    # Report determinant information from the final uint8 watermarked image.
    _y, _coeff, final_matrices, _q, _r, _bh, _bw = _analysis(output, cfg)
    final_det = _det_summary(final_matrices, cfg.det_epsilon)
    key = DCTQRDirectRKey(
        config=cfg.to_dict(),
        host_shape=tuple(int(x) for x in host.shape),
        watermark_shape=tuple(int(x) for x in watermark.shape),
    )
    metadata = {
        "method_id": METHOD_ID,
        "domain": "transform_domain_dct_then_qr_direct_r",
        "qr_input": "regularized_4x4_low_frequency_DCT_matrix",
        "embedding_carrier": "(R12-R13)/2_first_row_differential",
        "r11_modified": False,
        "step": float(cfg.step),
        "regularization": float(cfg.regularization),
        "closure_rounds": int(cfg.closure_rounds),
        "fully_blind": True,
        "original_host_used_for_extraction": False,
        **final_det,
        "rounds": round_meta,
    }
    return (output, key, metadata) if return_metadata else (output, key)


def extract(
    possibly_attacked_rgb: np.ndarray,
    key: DCTQRDirectRKey,
    *,
    return_metadata: bool = False,
):
    if not isinstance(key, DCTQRDirectRKey):
        raise TypeError("Expected DCTQRDirectRKey.")
    cfg = DCTQRDirectRConfig.from_mapping(key.config)
    image = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
    if tuple(int(x) for x in image.shape) != tuple(key.host_shape):
        raise ValueError(f"Image shape {image.shape} does not match key shape {key.host_shape}.")

    _y, _coeff, matrices, _q, r, _bh, _bw = _analysis(image, cfg)
    det_meta = _det_summary(matrices, cfg.det_epsilon)
    carrier = 0.5 * (r[:, 0, 1] - r[:, 0, 2])

    payload_len = int(np.prod(key.watermark_shape))
    order = _block_order(carrier.size, payload_len, cfg.seed)
    indices = np.rint(carrier[order] / float(cfg.step)).astype(np.int64)
    bits = (indices & 1).astype(np.uint8)
    recovered = (bits.reshape(key.watermark_shape) * 255).astype(np.uint8)

    metadata = {
        "method_id": METHOD_ID,
        "domain": "transform_domain_dct_then_qr_direct_r",
        "embedding_carrier": "(R12-R13)/2_first_row_differential",
        "inference_path": "direct_transform_qr_r_parity_qim",
        "fully_blind": True,
        "original_host_used": False,
        **det_meta,
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "METHOD_ID",
    "DCTQRDirectRConfig",
    "DCTQRDirectRKey",
    "embed",
    "extract",
]
