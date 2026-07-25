# Scientific revision changelog

## 2026 redesigned proposal release

### DCT–QR

- Preserved the DCT-QIM carrier, QR reliability allocation, and exact
  integer-lattice clean closure.
- Replaced the global scale `||diag(R)||_2` by the carrier-specific canonical
  factor `r11=||A[:,0]||_2` of the unlifted DCT analysis matrix.
- Added regularized carrier gain normalization
  `v_tilde = v / (r11/r11_ref)^0.90` and theorem-level tests for its exact
  homogeneity and reverse-triangle perturbation bound.
- Reduced validated QIM levels from `(20, 15, 12)` to
  `(18.25, 13.25, 10.25)`, increasing 13-host mean PSNR from 46.213584 dB to
  47.250421 dB while preserving clean NC=1 and increasing aggregate attacked NC.
- Preserved the previous global-QR configuration for controlled ablation in
  `configs/dct_qr_previous_global_qr.json`.

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

- Added theorem-level unit tests; current result: `33 passed`.
- Added 13-host × 15-attack validation with clean NC exactly 1 for all 39 clean
  round-trips.
- Added `scripts/run_redesigned_validation.py` and machine-readable result files.
- Archived the pre-redesign method descriptions under `docs/legacy_original/`.

## 2026 ablation and hyperparameter flag release (v2.1.0)

- Added a unified, validated flag interface in
  `qr64_certified.proposals.flags` for all three proposals.
- Added repeatable `--ablation NAME` support and explicit CLI hyperparameter
  overrides to `scripts/run_proposal_benchmark.py`.
- Added DCT-QR ablations for uniform QIM, gain normalization, carrier-specific
  versus global QR gain, spatial MAP, certificate evidence, and synchronization
  certificate gating.
- Added DCT-Schur ablations for uniform QIM, spectral gain, Henrici departure,
  spatial MAP, certificate evidence, and synchronization certificate gating.
- Added Spatial CD-DetQR core flags for payload-pattern search, determinant
  safety margin, payload masking, boundary regularization, affine sync, and
  closure rounds.
- Added `scripts/run_ablation_study.py`, including `--clean-only`, and
  `scripts/run_hyperparameter_sweep.py` for deterministic Cartesian sweeps.
- Added `scripts/list_proposal_flags.py` and complete usage documentation under
  `docs/ABLATION_AND_HYPERPARAMETER_FLAGS.md`.
- Added unit tests for every ablation configuration, nested Schur overrides,
  DCT-QR override precedence, and a Spatial fixed-pattern/no-mask clean
  round-trip. Current test result: `37 passed`.
