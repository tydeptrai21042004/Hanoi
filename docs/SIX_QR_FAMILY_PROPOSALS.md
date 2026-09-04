# Six QR-family proposal methods: DCT and non-DCT counterparts

The repository now exposes six directly comparable QR-family proposals in two groups.

| DCT family | Non-DCT spatial counterpart | Main carrier/role |
|---|---|---|
| `dct_qr` | `spatial_qr` | differential carrier + QR reliability + pairwise coset + gain normalization |
| `dct_qr_direct_r` | `spatial_qr_direct_r` | direct QIM on `(R12-R13)/2` |
| `dct_qr_r11_qim` | `spatial_qr_r11_qim` | direct QIM on `R11` |

The spatial counterparts do **not** call DCT or IDCT. This permits a controlled research question: whether the transform stage itself is necessary once QR-based carrier design, QIM, and gain handling are fixed.

The public repository also contains two additional independent proposals, `dct_schur_rescue` and `spatial_cd_detqr`, so the complete public proposal registry contains **eight methods**.

## Current maturity

| Method | Clean fidelity | Current robustness evidence | Role |
|---|---|---|---|
| `dct_qr` | Excellent | strongest validated evidence | main DCT proposal |
| `dct_qr_direct_r` | Excellent | moderate/weak | mathematically clean DCT direct-R proposal |
| `dct_qr_r11_qim` | Excellent | weak under gain | DCT R11 ablation/proposal |
| `spatial_qr` | Excellent | smoke-validated; gain attacks promising | main non-DCT counterpart |
| `spatial_qr_direct_r` | Excellent | smoke-validated | non-DCT direct-R counterpart |
| `spatial_qr_r11_qim` | Excellent | smoke-validated | non-DCT R11 counterpart |

Do not treat the three newly added spatial methods as publication-validated until the same multi-host/attack protocol is completed.

## Current spatial-family evidence

Clean-only validation on all 13 supplied BMP hosts gives exact NC = 1.0 for all three new methods. Mean PSNR is approximately 51.81 dB (`spatial_qr`), 50.73 dB (`spatial_qr_direct_r`), and 50.58 dB (`spatial_qr_r11_qim`).

On Lenna with the repository's 15 moderate attacks, the current default mean attacked NC values are approximately 0.9141, 0.8507, and 0.7059 respectively. These measurements make `spatial_qr` the strongest of the newly added non-DCT family, while all three remain below the validated `dct_qr` robustness level. Machine-readable evidence is stored in `results/spatial_qr_family_clean_13host.json` and `results/spatial_qr_family_lenna_15attack.json`.
