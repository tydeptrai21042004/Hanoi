# Scientific revision changelog

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

## 2026-09-04 — Three non-DCT Spatial-QR proposals

Added three independent proposal IDs that remove DCT/IDCT from the corresponding QR research questions:

- `spatial_qr`: directional spatial luminance carrier, QR reliability step classes, exact pairwise coset selection, and `R11` gain normalization.
- `spatial_qr_direct_r`: direct QR of a central 4×4 spatial luminance patch with minimum-Frobenius antisymmetric QIM on `(R12-R13)/2`.
- `spatial_qr_r11_qim`: direct parity-QIM on spatial-domain `R11`, with a selective same-parity downward closure rescue for saturated blocks.

Engineering/scientific integration:

- extended the public proposal registry from five to eight methods;
- extended unified benchmark selectors and adapters;
- extended ABC configuration dispatch and fixed the direct-R fall-through to the Spatial CD-DetQR schema;
- added `before` and `after_abc` config files for all three spatial methods;
- added clean/no-DCT tests;
- updated all Markdown documentation, with historical documents explicitly labeled rather than rewriting old result claims;
- added machine-readable `results/spatial_qr_family_clean_13host.json` and `results/spatial_qr_family_lenna_15attack.json`.

Validation: all three methods achieve clean NC = 1.0 on all 13 supplied hosts. Mean clean PSNR is approximately 51.81, 50.73, and 50.58 dB respectively. On Lenna/15 moderate attacks, current mean NC is approximately 0.9141, 0.8507, and 0.7059. These are new proposal results, not publication-level claims.

## 2026 Artificial Bee Colony optimizer migration

- Replaced the proposal-level Particle Swarm Optimization driver with a
  deterministic Artificial Bee Colony (ABC) implementation.
- Added employed-bee, onlooker-bee, and scout phases with bounded neighbour
  generation, deterministic seeding, complete evaluation traces, and global-best
  retention.
- Inserted the existing validated configuration as food source zero so ABC cannot
  return a worse score than the starting point on the evaluated objective.
- Switched active proposal configuration paths from `*_after_pso.json` to
  `*_after_abc.json` across benchmark registries, validation scripts, ablations,
  documentation, and reproducibility tools.
- Corrected the Spatial CD-DetQR search bounds to the active normalized carrier
  scale around `0.005`; the obsolete large-margin range was incompatible with the
  current method.
- Added `tests/test_abc.py` for baseline retention, deterministic replay, scout
  behavior, and bound safety.
- Added an active SP-SCQIM `extract_components()` implementation so the scientific
  component verifier no longer imports inactive legacy code.
- Full repository result: `51 passed`.
- DCT-QR full 15-attack ABC check on Lenna: PSNR `51.719975 dB`, clean NC `1.0`,
  mean attacked NC `0.998248`, minimum NC `0.988790`.
- Bounded end-to-end ABC smoke checks completed for DCT-Schur and Spatial
  CD-DetQR, with clean NC `1.0` in both active configurations.

## 2026 independent DCT–Schur SP-SCQIM replacement (v2.3.0)

- Removed duplicate public `embed()`/`extract()` definitions from the DCT–Schur path.
- Replaced the QR-engine-backed Schur-gain hybrid with an independent strict-upper Schur coupling carrier.
- Added three orthonormal coupling projections and three interleaved payload copies.
- Added the exact minimum-Frobenius update `n*=n+H^T(t-Hn)`.
- Added spectrum, trace, determinant, engine-independence, parameter-activity, and clean-round-trip tests.
- Full test result: `49 passed`.
- 13-host step-9 result: mean PSNR 48.173152 dB, clean NC 1.0, mean attacked NC 0.993814.
- The former Schur-gain hybrid remains stronger in attacked NC (0.996058), so the new method is explicitly marked as not yet passing the robustness promotion gate.


## 2026 DCT–QR pairwise-coset replacement

- Replaced only the public DCT–QR payload parity assignment; DCT–Schur and Spatial DetQR formulas remain unchanged.
- Added QR-homogeneous groups of two payload blocks and exact binary coset minimization at unchanged local QIM steps and guard margins.
- Added packed coset flip-mask extraction with backward compatibility for old DCT–QR keys.
- Added the `no_coset_optimization` ablation and direct coset hyperparameter flags.
- Validated 13 hosts × 15 attacks: mean PSNR 50.399139 dB, clean NC 1.0, mean attacked NC 0.998028, mean worst NC 0.988097.
- Added theorem-level and integration tests; current result: `45 passed`.

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
