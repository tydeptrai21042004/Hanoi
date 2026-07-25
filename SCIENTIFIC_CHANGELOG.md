# Scientific revision changelog

## 2026 redesigned proposal release

### DCT–QR

- Preserved the DCT-QIM carrier and integer-lattice closure.
- Changed the validated QIM levels to `(20, 15, 12)` over QR reliability groups.
- Added canonical-`R` diagonal-energy reference and local gain normalization:
  `v_tilde = v / (s_QR / s_QR_ref)^0.75`.
- The gain reference is measured from the final watermarked image, not from the
  original host.

### DCT–Schur

- Removed the low-accuracy independent Schur secondary channel from the public
  embedding/extraction path; retained it only in legacy/ablation code.
- Added Schur spectral reliability-conditioned DCT-QIM levels `(16.5, 12.5, 9)`.
- Added a homogeneous eigenvalue/departure scale for local DCT gain correction.

### Spatial DetQR

- Replaced the algebraically redundant raw determinant carrier by the normalized
  QR residual `z(A)=det(Q)r22=det(A)/r11`.
- Added a hard embedded determinant margin.
- Derived and implemented the exact minimum nonnegative integer amplitude that
  satisfies both the signed normalized-QR margin and `det(A) != 0`.
- Reduced the pilot set to 31 and moved payload and affine pilots into the same
  normalized-QR domain.

### Evaluation and reproducibility

- Added theorem-level unit tests; current result: `29 passed`.
- Added 13-host × 15-attack validation with clean NC exactly 1 for all 39 clean
  round-trips.
- Added `scripts/run_redesigned_validation.py` and machine-readable result files.
- Archived the pre-redesign method descriptions under `docs/legacy_original/`.
