from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.fft import dctn
from scipy.linalg import schur

CertificateMode = Literal["qr", "schur"]


@dataclass(frozen=True)
class MatrixCertificate:
    """Blockwise nonsingularity and stability certificate.

    ``balance`` is decomposition-specific:
      * QR: canonical-R diagonal balance.
      * Schur: complex-Schur eigenvalue-modulus balance.

    ``coupling`` is also decomposition-specific:
      * QR: normalized strict-upper-triangular energy of R.
      * Schur: normalized strict-upper-triangular energy of T.
    """

    determinant: np.ndarray
    beta: np.ndarray
    balance: np.ndarray
    coupling: np.ndarray
    reliability: np.ndarray
    lift: float
    mode: CertificateMode

    @property
    def diagonal_balance(self) -> np.ndarray:
        """Backward-compatible alias used by the original repository."""
        return self.balance

    @property
    def all_nonsingular(self) -> bool:
        return bool(np.all(self.determinant > 0.0))

    def robust_summary(self) -> dict[str, float | int | bool | str]:
        log_beta = np.log(self.beta + 1e-18)
        return {
            "certificate_mode": self.mode,
            "block_count": int(self.determinant.size),
            "all_det_nonzero": bool(self.all_nonsingular),
            "min_abs_det": float(self.determinant.min()),
            "median_abs_det": float(np.median(self.determinant)),
            "min_beta": float(self.beta.min()),
            "median_beta": float(np.median(self.beta)),
            "median_log_beta": float(np.median(log_beta)),
            "mad_log_beta": float(_mad(log_beta)),
            "median_balance": float(np.median(self.balance)),
            "mad_balance": float(_mad(self.balance)),
            "median_coupling": float(np.median(self.coupling)),
            "mad_coupling": float(_mad(self.coupling)),
            # Kept for compatibility with earlier result parsers.
            "median_diagonal_balance": float(np.median(self.balance)),
        }


# Backward-compatible public name.
QRCertificate = MatrixCertificate


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def opponent_field(rgb: np.ndarray, eta: float = 0.07) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {x.shape}.")
    return (
        0.299 * x[..., 0]
        + 0.587 * x[..., 1]
        + 0.114 * x[..., 2]
        + float(eta) * (x[..., 0] - 0.5 * x[..., 1] - 0.5 * x[..., 2])
    )


def analysis_matrices(rgb: np.ndarray, eta: float = 0.07, lift: float = 1.0) -> np.ndarray:
    """Build one lifted 4x4 low-frequency analysis matrix per 8x8 block.

    The DC term is replaced by an antisymmetric low-frequency difference so the
    certificate measures local directional structure rather than mean intensity.
    The diagonal lift is virtual: it is used only for certificate analysis and
    is never embedded into the image.
    """
    f = opponent_field(rgb, eta=eta)
    h, w = f.shape
    if h % 8 or w % 8:
        raise ValueError("Image dimensions must be divisible by 8.")
    blocks = f.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    coeff = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    a = coeff[:, :4, :4].copy()
    a[:, 0, 0] = coeff[:, 0, 1] - coeff[:, 1, 0]
    a += float(lift) * np.eye(4, dtype=np.float64)[None, :, :]
    return a


def canonical_qr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(a, dtype=np.float64))
    diag = np.diagonal(r, axis1=-2, axis2=-1)
    sign = np.where(diag < 0.0, -1.0, 1.0)
    q = q * sign[..., None, :]
    r = sign[..., :, None] * r
    return q, r


def _common_terms(a: np.ndarray, det_epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    det = np.abs(np.linalg.det(a))
    if np.any(det <= float(det_epsilon)):
        bad = np.flatnonzero(det <= float(det_epsilon))
        raise ValueError(
            f"Certificate condition det(A) != 0 failed for {bad.size} block(s); "
            f"first indices: {bad[:10].tolist()}"
        )
    frob = np.linalg.norm(a, axis=(1, 2))
    # For 4x4 A, |det(A)| / ||A||_F^3 is a computable lower bound on sigma_min(A).
    beta = det / (frob**3 + 1e-12)
    return det, beta


def _normalize_reliability(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    lo, hi = np.quantile(raw, [0.05, 0.95])
    if hi - lo <= 1e-12:
        return np.ones_like(raw) * 0.5
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def compute_qr_certificate(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
    det_epsilon: float = 1e-12,
) -> MatrixCertificate:
    a = analysis_matrices(rgb, eta=eta, lift=lift)
    det, beta = _common_terms(a, det_epsilon)
    _q, r = canonical_qr(a)
    diag = np.abs(np.diagonal(r, axis1=-2, axis2=-1))
    balance = diag.min(axis=1) / (diag.max(axis=1) + 1e-12)
    upper = np.triu(r, k=1)
    coupling = np.linalg.norm(upper, axis=(1, 2)) / (np.linalg.norm(r, axis=(1, 2)) + 1e-12)

    raw = np.log1p(beta) + 2.0 * balance + 0.20 * (1.0 - coupling)
    reliability = _normalize_reliability(raw)
    return MatrixCertificate(det, beta, balance, coupling, reliability, float(lift), "qr")


def canonical_schur(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the complex Schur factors for one analysis matrix."""
    t, z = schur(np.asarray(a, dtype=np.float64), output="complex", check_finite=False)
    return t, z


def compute_schur_certificate(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
    det_epsilon: float = 1e-12,
) -> MatrixCertificate:
    """Compute a Schur spectral certificate on the same analysis matrices.

    For the complex Schur form A = Z T Z^H, the diagonal of T contains the
    eigenvalues and det(A)=det(T). Instead of repeatedly calling LAPACK on all
    4096 tiny blocks, the implementation obtains those diagonal invariants with
    batched eigenvalues. The Schur strict-upper energy is recovered exactly from
    unitary Frobenius invariance:

        ||strict_upper(T)||_F^2 = ||A||_F^2 - sum_i |lambda_i|^2.

    ``canonical_schur`` remains available for direct factor validation.
    """
    a = analysis_matrices(rgb, eta=eta, lift=lift)
    det, beta = _common_terms(a, det_epsilon)
    eig = np.linalg.eigvals(a)
    eig_abs = np.abs(eig)
    balance = eig_abs.min(axis=1) / (eig_abs.max(axis=1) + 1e-12)

    frob_sq = np.sum(np.square(np.abs(a)), axis=(1, 2))
    eig_sq = np.sum(np.square(eig_abs), axis=1)
    strict_upper_sq = np.maximum(frob_sq - eig_sq, 0.0)
    coupling = np.sqrt(strict_upper_sq) / (np.sqrt(frob_sq) + 1e-12)

    raw = np.log1p(beta) + 2.0 * balance + 0.20 * (1.0 - coupling)
    reliability = _normalize_reliability(raw)
    return MatrixCertificate(det, beta, balance, coupling, reliability, float(lift), "schur")


def compute_certificate(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
    det_epsilon: float = 1e-12,
    mode: CertificateMode = "qr",
) -> MatrixCertificate:
    normalized = str(mode).strip().lower()
    if normalized == "qr":
        return compute_qr_certificate(rgb, eta=eta, lift=lift, det_epsilon=det_epsilon)
    if normalized == "schur":
        return compute_schur_certificate(rgb, eta=eta, lift=lift, det_epsilon=det_epsilon)
    raise ValueError("mode must be 'qr' or 'schur'.")


def qr_gain_scale(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
) -> np.ndarray:
    """Return a positive QR scale for local multiplicative-gain estimation.

    For the canonical factorization ``A_b = Q_b R_b``, the scale is

        s_b^QR = ||diag(R_b)||_2.

    It is invariant to the sign ambiguity of QR and responds smoothly to local
    contrast attenuation.  The value is used only as a blind reference ratio;
    it is not an additional watermark carrier.
    """
    matrices = analysis_matrices(rgb, eta=eta, lift=lift)
    _q, r = canonical_qr(matrices)
    diagonal = np.abs(np.diagonal(r, axis1=-2, axis2=-1))
    return np.sqrt(np.sum(diagonal * diagonal, axis=1) + 1e-12)


def schur_gain_scale(
    rgb: np.ndarray,
    *,
    eta: float = 0.07,
    lift: float = 1.0,
    departure_weight: float = 0.50,
) -> np.ndarray:
    """Return the Schur spectral-departure scale used for DCT compensation.

    Let ``lambda_i(A_b)`` be the eigenvalues of the lifted DCT analysis matrix
    and let the Henrici departure satisfy

        dep(A_b)^2 = ||A_b||_F^2 - sum_i |lambda_i(A_b)|^2.

    The scale

        s_b^S = sqrt(sum_i |lambda_i|^2 + omega dep(A_b)^2)

    combines diagonal Schur energy with strict-upper Schur energy.  It therefore
    remains genuinely Schur-based while avoiding the unreliable independent
    secondary-bit channel used by the exploratory predecessor.
    """
    if departure_weight < 0:
        raise ValueError("departure_weight must be nonnegative")
    matrices = analysis_matrices(rgb, eta=eta, lift=lift)
    eigenvalues = np.linalg.eigvals(matrices)
    eigenvalue_energy = np.sum(np.abs(eigenvalues) ** 2, axis=1)
    frobenius_energy = np.sum(np.abs(matrices) ** 2, axis=(1, 2))
    departure_sq = np.maximum(frobenius_energy - eigenvalue_energy, 0.0)
    return np.sqrt(
        eigenvalue_energy + float(departure_weight) * departure_sq + 1e-12
    )
