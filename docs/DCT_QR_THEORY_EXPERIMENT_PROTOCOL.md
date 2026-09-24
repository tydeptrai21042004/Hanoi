# DCT-QR Theory: ablation and robustness protocol

## Purpose

This protocol evaluates the contribution of each theorem-backed component of `dct_qr_theory` without modifying the default full method. Ablations are **one factor at a time** and are exposed through `DCTQRTheoryConfig` / `apply_proposal_flags`, so every variant is reproducible from the command line.

## One-factor ablations

| Ablation | Full component | Ablated behavior | Scientific question |
|---|---|---|---|
| `no_opponent_term` | norm-preserving nonzero opponent coefficient `eta*` | `eta=0` (BT.601 luminance only) | Does the opponent component add robustness beyond luminance while preserving gradient norm? |
| `uniform_step` | `Delta_b = Delta_0 GM(beta)/beta_b` | `Delta_b = Delta_0` | Does the beta certificate provide useful continuous strength allocation? |
| `no_global_coset` | global exact binary coset minimization | fixed coset `s=0` | Does exact global coset choice reduce embedding projection energy / distortion? |
| `no_gain_normalization` | exact `r11` ratio correction (`gamma=1`) | decode raw carrier | How much robustness is attributable to decomposition-gain normalization? |
| `no_spatial_icm` | degree-normalized four-neighbour ICM | hard QIM decisions | Does spatial regularization recover uncertain attacked bits? |
| `no_sync_search` | pilot-supported alignment search | decode attacked image directly | How much robustness comes from synchronization rather than the carrier itself? |

Run all ablations on the curated 20-attack suite:

```bash
PYTHONPATH=src python scripts/run_ablation_study.py \
  --method dct_qr_theory \
  --config configs/dct_qr_theory_after_abc.json \
  --attack-suite common20 \
  --output-dir results/ablation_common20
```

A single-host numerical illustration is stored in `results/dct_qr_theory_ablation_common20/`. Treat it as an illustration only; publication claims should aggregate the full host set and report dispersion / worst-host behavior.

## Curated common-20 attack suite

The main comparison suite contains exactly 20 standard image-processing attacks, grouped into five categories with four attacks each. This keeps the headline benchmark interpretable and avoids mixing severe real-world simulations or compound attacks into the same mean.

### Compression / quantization

1. JPEG compression, quality 70
2. JPEG2000 compression, quality layer 7
3. 6-bit depth reduction
4. 64-color quantization

### Noise

5. Additive Gaussian noise, sigma 1
6. Salt-and-pepper noise, amount 0.005
7. Speckle noise, variance 0.005
8. Poisson noise, peak 255

### Filtering / enhancement

9. Median filter, 3x3
10. Average filter, 3x3
11. Gaussian blur, radius 1
12. Sharpening, factor 2

### Geometric / resampling

13. Rotation and inverse rotation, 1 degree
14. Down-scaling / restoration, factor 0.75
15. Crop-and-resize, 90% retained
16. Translation, 4 pixels in x and y

### Photometric / point processing

17. Brightness scaling, factor 0.9
18. Contrast scaling, factor 1.1
19. Gamma correction, gamma 1.2
20. Histogram equalization

These attack families are standard in robust-image-watermarking studies. Geometric RST/cropping and common signal-processing attacks such as JPEG, additive noise, and median filtering are repeatedly used in published evaluations; broader reviews likewise discuss robustness against filtering, noising, geometric transforms, JPEG, histogram equalization, brightness/contrast and related enhancement operations.

The repository retains `stress`, `publication`, `real_world`, deformation, structured-loss, and combined suites for secondary experiments. Do not combine those severe suites with `common20` into one undifferentiated headline mean.

## Numerical method illustrations

Run:

```bash
PYTHONPATH=src python scripts/run_dct_qr_theory_numerics.py
```

The script writes:

- `numerical_summary.json`: derived eta norm identity, Neumann margin, beta-vs-sigma-min check, adaptive step distribution, embedding quality, and grouped common-20 robustness;
- `common20_per_attack.csv`: per-attack NC/BER and gain-normalization diagnostics;
- `NUMERICAL_ILLUSTRATION.md`: compact manuscript-facing summary.

The included Lenna run gives:

- `eta* = 0.068666667` with numerical norm-preservation error at machine precision;
- Neumann lift `lambda = 1012.399928` and `max ||A||_2/lambda = 0.986064 < 1`;
- `beta <= sigma_min` on every block in the numerical check;
- adaptive steps from about `5.1356` to `8.4480`, median `5.5923`;
- PSNR `53.6035 dB`, clean NC `1.0`;
- common-20 mean NC `0.975482`, minimum NC `0.731832`.

For global coset optimization, Lenna selects `s=0`, so its one-host coset ablation is inactive. The 13-host scan in `results/dct_qr_theory_numerics/coset_host_scan.csv` shows nonzero coset selection on multiple hosts; e.g. Girl obtains a projection-energy ratio about `0.95991`, which directly illustrates the distortion non-increase proposition.

## Reporting rules

1. Report the full model first, followed by one-factor ablations.
2. Do not describe a single-host difference as statistically significant.
3. Aggregate by attack category as well as globally; geometric failures should not be hidden by strong photometric performance.
4. Keep `common20` separate from severe/compound/real-world stress tests.
5. The Neumann certificate is an embedding-time nonsingularity guarantee for the constructed analysis matrix; do not present it as a universal attacked-image robustness theorem.
6. `dct_qr_theory` is host-image-free at extraction but carries host-derived side information in the key; describe it as key-assisted / host-image-free extraction when comparing blindness assumptions.
