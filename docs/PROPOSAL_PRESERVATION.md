# Historical proposal preservation verification

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

The current repository additionally exposes `dct_qr_direct_r`. This fourth
method is a new transform-domain DCT->QR->R proposal and is not part of the
historical byte/numerical-preservation claim below.

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
