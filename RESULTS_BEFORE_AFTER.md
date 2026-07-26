# Validated before/after results

Protocol: 13 supplied hosts, one 64×64 watermark, 15 moderate attacks for all methods, and rotation/shear tests for the spatial synchronization component.

## DCT-QR

Accepted replacement: QR-conditioned pairwise coset optimization at unchanged QIM steps and margins.

| Metric | Previous carrier-r11 | Active pairwise-coset | Difference |
|---|---:|---:|---:|
| Mean PSNR | 47.250421 | **50.399139** | **+3.148718 dB** |
| Clean NC | 1.000000 | **1.000000** | 0 |
| Mean attacked NC | 0.997867 | **0.998028** | +0.000161 |
| Mean worst-attack NC | 0.986162 | **0.988097** | +0.001935 |
| Global worst NC | 0.964654 | **0.964723** | +0.000069 |

Scientific interpretation: each QR-homogeneous pair selects the least-distorting shared binary coset. The previous mapping is the feasible case `s=0`, so continuous projection energy cannot increase, while XOR inversion preserves the physical QIM error event.

## DCT-Schur Rescue

Tested sparse rescue was rejected.

| Metric | Before | Candidate | Difference |
|---|---:|---:|---:|
| Mean PSNR | 46.937031 | 47.031285 | +0.094254 dB |
| Mean NC | 0.988850 | 0.988802 | -0.000048 |
| Mean worst-attack NC | 0.953918 | 0.957539 | +0.003621 |
| Selected Schur clean accuracy | — | 0.858173 | target 0.99 |

Scientific interpretation: preservation of spectrum, trace, and determinant is mathematically valid in floating point, but it does not by itself prove that the secondary bit channel is reliable.

## Spatial CD-DetQR

Accepted change: pilot count 79→71 and determinant-pilot affine hypothesis testing.

| Metric | Before | After | Difference |
|---|---:|---:|---:|
| Mean PSNR | 53.454653 | 53.636001 | +0.181348 dB |
| Moderate mean NC | 0.974359 | 0.974351 | -0.000008 |
| Rotation/shear mean NC | 0.546462 | 0.986670 | +0.440207 |
| Combined 17-attack mean NC | 0.924018 | 0.975800 | +0.051782 |

Scientific interpretation: the same signed determinant statistic is used for payload and synchronization, so geometric alignment is justified by the carrier model rather than added as a separate black-box stage.

Full per-host data are in `results/scientific_validation/`.
