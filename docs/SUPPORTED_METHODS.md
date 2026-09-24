# Supported proposal methods

The public proposal registry exposes **nine scientific proposal methods**. Six methods form the directly paired QR family (three DCT-based methods and three non-DCT spatial counterparts), with `dct_qr_theory` as an additional theory-grounded DCT-QR proposal.

| Method | Domain / carrier | DCT used? | Status |
|---|---|---:|---|
| `dct_qr` | DCT differential carrier + QR reliability + pairwise coset + gain normalization | Yes | Validated main DCT proposal |
| `dct_qr_theory` | DCT differential carrier + derived QR certificate + beta-adaptive QIM + global coset | Yes | Theory-grounded proposal; ablation/common-20 protocol available |
| `dct_qr_direct_r` | Direct QIM on `(R12-R13)/2` after QR of a 4×4 low-frequency DCT matrix | Yes | New proposal; smoke-validated |
| `dct_qr_r11_qim` | Direct parity-QIM on `R11` after transform-domain QR | Yes | New proposal / ablation; smoke-validated |
| `spatial_qr` | Spatial directional-difference carrier + central spatial QR reliability + pairwise coset + `R11` gain normalization | **No** | New non-DCT proposal; smoke-validated |
| `spatial_qr_direct_r` | Direct QIM on `(R12-R13)/2` after QR of a central 4×4 spatial luminance patch | **No** | New non-DCT proposal; smoke-validated |
| `spatial_qr_r11_qim` | Direct parity-QIM on `R11` after QR of a central 4×4 spatial luminance patch | **No** | New non-DCT proposal / ablation; smoke-validated |
| `dct_schur_rescue` | DCT strict-upper Schur coupling QIM | Yes | Independent proposal; promotion gate under validation |
| `spatial_cd_detqr` | Spatial normalized QR-residual + determinant constraint | No | Validated spatial proposal |

## Six-method QR family

| DCT method | Non-DCT counterpart | Controlled research question |
|---|---|---|
| `dct_qr` | `spatial_qr` | Is DCT necessary when QR reliability, pairwise coset optimization and gain normalization are retained? |
| `dct_qr_direct_r` | `spatial_qr_direct_r` | How does the same direct-R differential QIM behave when QR is applied to spatial pixels instead of DCT coefficients? |
| `dct_qr_r11_qim` | `spatial_qr_r11_qim` | How does raw `R11` parity-QIM behave in transform and spatial domains? |

Detailed equations:

- `DCT_QR_DIRECT_R_PROPOSAL_VI.md`
- `DCT_QR_R11_QIM_PROPOSAL_VI.md`
- `SPATIAL_QR_PROPOSAL_VI.md`
- `SPATIAL_QR_DIRECT_R_PROPOSAL_VI.md`
- `SPATIAL_QR_R11_QIM_PROPOSAL_VI.md`
- `SIX_QR_FAMILY_PROPOSALS.md`

The original historical three-method documents remain for reproducibility, but their method counts must not be interpreted as the current public registry size.
