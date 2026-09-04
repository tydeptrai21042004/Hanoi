from __future__ import annotations

"""Shared spatial-domain QR utilities for the non-DCT proposal family.

The functions in this module deliberately avoid DCT/IDCT.  Each 8x8 RGB block
is converted only to luminance.  QR certificates are formed from the central
4x4 spatial luminance patch (optionally mean-centred), and reconstructed spatial
patches are written back directly when a proposal modifies R.
"""

from typing import Any

import numpy as np

_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
_LUMA_NORM2 = float(np.dot(_LUMA, _LUMA))


def luminance(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image; got {x.shape}")
    return np.tensordot(x, _LUMA, axes=([-1], [0]))


def to_blocks(field: np.ndarray, block_size: int = 8) -> tuple[np.ndarray, int, int]:
    y = np.asarray(field, dtype=np.float64)
    h, w = y.shape
    if h % block_size or w % block_size:
        raise ValueError("Image dimensions must be divisible by the spatial block size.")
    bh, bw = h // block_size, w // block_size
    blocks = (
        y.reshape(bh, block_size, bw, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
    )
    return blocks, bh, bw


def from_blocks(blocks: np.ndarray, bh: int, bw: int) -> np.ndarray:
    b = np.asarray(blocks, dtype=np.float64)
    block_size = b.shape[-1]
    return (
        b.reshape(bh, bw, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(bh * block_size, bw * block_size)
    )


def canonical_qr(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(matrices, dtype=np.float64))
    diagonal = np.diagonal(r, axis1=-2, axis2=-1)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q = q * signs[..., None, :]
    r = signs[..., :, None] * r
    return q, r


def block_order(total_blocks: int, payload_len: int, seed: int) -> np.ndarray:
    if payload_len > total_blocks:
        raise ValueError(
            f"Payload needs {payload_len} blocks, but only {total_blocks} are available."
        )
    return np.random.default_rng(int(seed)).permutation(total_blocks)[:payload_len]


def project_parity_qim(values: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
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


def spatial_qr_analysis(
    rgb: np.ndarray,
    *,
    block_size: int = 8,
    matrix_size: int = 4,
    regularization: float = 1.0,
    mean_center: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Return luminance blocks and QR of the central spatial matrix.

    The central 4x4 patch occupies rows/columns 2..5 in an 8x8 block.  For
    reliability/gain certificates, ``mean_center=True`` makes the QR statistic
    invariant to a constant luminance offset before the small diagonal virtual
    regularization is added.
    """
    if block_size != 8 or matrix_size != 4:
        raise ValueError("Current spatial QR proposals require 8x8 blocks and a 4x4 QR matrix.")
    y = luminance(rgb)
    blocks, bh, bw = to_blocks(y, block_size)
    start = (block_size - matrix_size) // 2
    patch = blocks[:, start : start + matrix_size, start : start + matrix_size].copy()
    offsets = np.zeros((patch.shape[0], 1, 1), dtype=np.float64)
    if mean_center:
        offsets[:, 0, 0] = patch.mean(axis=(1, 2))
        patch = patch - offsets
    matrices = patch + float(regularization) * np.eye(matrix_size, dtype=np.float64)[None, :, :]
    q, r = canonical_qr(matrices)
    return y, blocks, matrices, q, r, bh, bw, offsets


def determinant_summary(matrices: np.ndarray, epsilon: float) -> dict[str, Any]:
    determinant = np.abs(np.linalg.det(np.asarray(matrices, dtype=np.float64)))
    return {
        "det_nonzero": bool(np.all(determinant > float(epsilon))),
        "min_abs_det": float(np.min(determinant)),
        "median_abs_det": float(np.median(determinant)),
        "near_singular_blocks": int(np.count_nonzero(determinant <= float(epsilon))),
    }


def rgb_from_luminance_delta(image: np.ndarray, dy: np.ndarray) -> np.ndarray:
    output = np.asarray(image, dtype=np.float64) + np.asarray(dy, dtype=np.float64)[..., None] * (
        _LUMA / _LUMA_NORM2
    )
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def replace_central_patch(
    blocks: np.ndarray,
    reconstructed: np.ndarray,
    *,
    matrix_size: int = 4,
    regularization: float = 1.0,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(blocks, dtype=np.float64).copy()
    start = (out.shape[-1] - matrix_size) // 2
    patch = np.asarray(reconstructed, dtype=np.float64) - float(regularization) * np.eye(
        matrix_size, dtype=np.float64
    )[None, :, :]
    if offsets is not None:
        patch = patch + np.asarray(offsets, dtype=np.float64)
    out[:, start : start + matrix_size, start : start + matrix_size] = patch
    return out


def qr_reliability(r: np.ndarray) -> np.ndarray:
    """Dimensionless QR stability proxy used only for scheduling/step classes."""
    rr = np.asarray(r, dtype=np.float64)
    diag = np.abs(np.diagonal(rr, axis1=-2, axis2=-1))
    numerator = np.min(diag, axis=1)
    denominator = np.linalg.norm(rr, axis=(1, 2)) + 1e-12
    return numerator / denominator


__all__ = [
    "_LUMA",
    "_LUMA_NORM2",
    "luminance",
    "to_blocks",
    "from_blocks",
    "canonical_qr",
    "block_order",
    "project_parity_qim",
    "spatial_qr_analysis",
    "determinant_summary",
    "rgb_from_luminance_delta",
    "replace_central_patch",
    "qr_reliability",
]
