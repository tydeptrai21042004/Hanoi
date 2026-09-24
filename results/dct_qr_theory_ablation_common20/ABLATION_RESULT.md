# DCT-QR theory one-factor ablation — Lenna / common20

| Variant | PSNR (dB) | Clean NC | Mean NC | Q10 NC | Min NC | Mean BER |
|---|---:|---:|---:|---:|---:|---:|
| full | 53.6035 | 1.000000 | 0.975482 | 0.953995 | 0.731832 | 0.024902 |
| no_opponent_term | 53.6762 | 1.000000 | 0.975154 | 0.961246 | 0.716404 | 0.024976 |
| uniform_step | 53.6658 | 1.000000 | 0.974825 | 0.954123 | 0.717990 | 0.025269 |
| no_global_coset | 53.6035 | 1.000000 | 0.975482 | 0.953995 | 0.731832 | 0.024902 |
| no_gain_normalization | 53.6035 | 1.000000 | 0.864249 | 0.614617 | 0.490947 | 0.131360 |
| no_spatial_icm | 53.6035 | 1.000000 | 0.931251 | 0.840085 | 0.639154 | 0.069128 |
| no_sync_search | 53.6035 | 1.000000 | 0.964952 | 0.918501 | 0.737128 | 0.035303 |

Interpretation: `no_gain_normalization` and `no_spatial_icm` produce the largest robustness losses on this host. `no_global_coset` is inactive for Lenna because the full optimizer selects the zero coset. This single-host table is an illustration, not a statistical claim; the manuscript table should aggregate all hosts.
