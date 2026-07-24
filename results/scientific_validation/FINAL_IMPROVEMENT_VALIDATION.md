# Final before/after validation of the three proposal methods

## Test protocol

- 13 supplied 512x512 RGB host images.
- One 64x64 binary watermark.
- 15 repository moderate attacks for every method.
- Rotation 2 degrees and shear 0.08 additionally tested for the spatial synchronization change.
- The existing repository test suite also passed: 9/9 tests.

## DCT-QR — accepted change

Change: QR reliability actively assigns the QIM step: weakest 20% = 20.5, middle 40% = 16, strongest 40% = 13. MAP lambda is retuned from 0.2634 to 0.33. The clean exact gate is 0.79.

| Metric | Before | After | Difference |
|---|---:|---:|---:|
| Mean PSNR | 45.635875 | 45.684599 | +0.048724 dB |
| Clean NC | 0.999981 | 1.000000 | +0.000019 |
| Mean NC (15 attacks) | 0.990520 | 0.990990 | +0.000471 |
| Mean worst-attack NC per host | 0.959743 | 0.962965 | +0.003222 |

Decision: accept. It improves PSNR and both average and worst-attack NC, while making the QR certificate an active embedding controller.

## DCT-Schur Rescue — rejected change

Tested change: sparse rescue of the 64 weakest primary positions, Schur step 0.15, 5 joint closure iterations, fusion weight 0.4.

| Metric | Before | After | Difference |
|---|---:|---:|---:|
| Mean PSNR | 46.937031 | 47.031285 | +0.094254 dB |
| Clean NC | 0.999981 | 1.000000 | +0.000019 |
| Mean NC (15 attacks) | 0.988850 | 0.988802 | -0.000048 |
| Mean worst-attack NC per host | 0.953918 | 0.957539 | +0.003621 |
| Selected Schur clean accuracy | — | 0.858173 | target is at least 0.99 |

Decision: reject. PSNR and worst-case NC improve, but mean NC decreases slightly and the secondary channel fails the 0.99 clean criterion. A primary-null-space Schur projection must be implemented and re-tested before this can be a valid rescue proposal.

## Spatial CD-DetQR — accepted geometry change

Change: reduce pilot count from 79 to 71 and add pilot-score-gated affine synchronization. A correction is accepted only when pilot score improves by more than 0.30.

| Metric | Before | After | Difference |
|---|---:|---:|---:|
| Mean PSNR | 53.454653 | 53.636001 | +0.181348 dB |
| Moderate mean NC | 0.974359 | 0.974351 | -0.000008 |
| Rotation/shear mean NC | 0.546462 | 0.986670 | +0.440207 |
| Rotation 2-degree NC | 0.568466 | 0.992574 | — |
| Shear 0.08 NC | 0.524459 | 0.980765 | — |
| Combined 17-attack mean NC | 0.924018 | 0.975800 | +0.051782 |

Decision: accept for geometric robustness. Ordinary-attack NC is statistically unchanged, PSNR increases, and rotation/shear NC improves dramatically on every host.
