# Supported proposal methods

The public registry exposes four scientific models.

| Method | Mathematical role | Status |
|---|---|---|
| `dct_qr` | QR reliability conditions local DCT-QIM separation | Validated improvement |
| `dct_qr_direct_r` | **Direct QR decomposition of a 4×4 low-frequency DCT matrix; payload modifies R after QR** | New proposal; smoke-validated |
| `dct_schur_rescue` | Spectrum-preserving Schur non-normality as secondary evidence | Exploratory; independent clean criterion not met |
| `spatial_cd_detqr` | Channel-differential signed QR determinant with affine determinant-pilot synchronization | Validated geometric improvement |

The new direct-R equations are documented in `DCT_QR_DIRECT_R_PROPOSAL_VI.md`.
The earlier three methods remain documented in `SCIENTIFIC_METHODS.md` and
`SCIENTIFIC_METHODS_VI.md`.
