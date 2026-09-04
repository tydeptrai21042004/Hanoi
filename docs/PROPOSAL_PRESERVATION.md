# Historical proposal preservation verification

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

> **Current-release notice:** this document records the earlier namespace-only
> migration. The present release intentionally changes `proposals/method.py`,
> `proposals/config.py`, `proposals/flags.py`, and `proposal_registry.py` for the
> new DCT–QR pairwise-coset embedding law. DCT–Schur, Spatial DetQR, and baseline
> mathematical implementations remain unchanged.

The original three proposal methods from the uploaded project are still
preserved:

1. `dct_qr`
2. `dct_schur_rescue`
3. `spatial_cd_detqr`

The current repository additionally exposes five methods outside that historical core:
`dct_qr_direct_r`, `dct_qr_r11_qim`, `spatial_qr`, `spatial_qr_direct_r`, and
`spatial_qr_r11_qim`. These additions are **not** part of the historical byte/numerical-
preservation claim below. The three `spatial_qr*` methods explicitly remove DCT/IDCT.

## Location

The actual implementations are now visible under:

```text
src/qr64_certified/proposals/
```

The old modules under `src/qr64_certified/` are compatibility wrappers, so old
commands and imports continue to work.

## Source preservation

The following original source files are byte-for-byte unchanged after being
moved into the explicit proposal package:

- `config.py`
- `certificate.py`
- `cd_detqr.py`
- `optimization.py`
- `proposal_registry.py`

Only dependency import paths changed in:

- `method.py`
- `direct_schur_rescue.py`

Those imports previously referenced the removed nested `realtime_watermark`
package. They now reference the identical utilities under `qr64_certified`.
No embedding, extraction, matrix, quantization, determinant, QR, Schur, pilot,
or synchronization equation was changed.

## Numerical equivalence

For Lenna and the included 64×64 watermark, the SHA-256 hashes of both the
watermarked image and recovered watermark are identical before and after the
proposal namespace correction for all three methods.

The machine-readable hashes are stored in:

```text
docs/PROPOSAL_PRESERVATION.json
```
