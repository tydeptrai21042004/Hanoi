# Scientific revision changelog

## DCT-QR

- Added QR reliability-conditioned QIM levels using the validated ratios `(1.28125, 1.0, 0.8125)` and reliability partitions `(0.20, 0.60)`.
- Retained integer-lattice closure and blind extraction.
- Set the validated spatial-prior coefficient to `0.33` and exact clean-confidence gate to `0.79` in the after-PSO configuration.
- Preserved a uniform-strength mode for controlled before/after experiments.

## DCT-Schur Rescue

- Reclassified the method as an exploratory secondary-channel hypothesis.
- Disabled adaptive primary strength by default in this method so the Schur contribution can be isolated scientifically.
- Added the explicit acceptance criterion: independent clean Schur accuracy must be at least `0.99` before fusion is credited.
- Kept the spectrum/trace/determinant-preserving floating-point construction unchanged.

## Spatial CD-DetQR

- Reduced the pilot count from `79` to the validated `71`.
- Added determinant-pilot affine hypothesis testing over small rotations and shears.
- Added two acceptance conditions: pilot-score gain above `0.30` and absolute score at least `0.50`.
- Preserved the exact antisymmetric update law `x' = x + 4sa`.

## Evaluation

- Replaced a mean-only optimization score by a combined mean, lower-decile, minimum-NC, and PSNR criterion.
- Added the complete 13-host before/after tables under `results/scientific_validation/`.
- Added English and Vietnamese mathematical method descriptions.
