# ABC Optimizer Migration Report

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

## Completed changes

- Replaced proposal-level PSO with deterministic Artificial Bee Colony (ABC).
- Implemented employed-bee, onlooker-bee, and scout phases.
- Preserved the existing validated configuration as food source zero.
- Retained the global best independently of scout abandonment.
- Added deterministic seeds and complete per-evaluation JSON traces.
- Switched active configuration names and benchmark paths to `*_after_abc.json`.
- Updated optimization, benchmark, validation, ablation, registry, README, and documentation paths.
- Added corrected Spatial CD-DetQR search bounds around the active normalized margin scale.
- Added active Schur component extraction so scientific verification no longer uses legacy code.

## Tests

```text
pytest: 51 passed
checksum verification: passed
scientific component verification: passed
```

## Active one-host validation

| Method | PSNR (dB) | Clean NC | Validation scope |
|---|---:|---:|---|
| DCT-QR ABC | 51.719975 | 1.000000 | Full 15 moderate attacks |
| DCT-Schur ABC | 48.585684 | 1.000000 | Bounded ABC search + component verification |
| Spatial CD-DetQR ABC | 55.872211 | 1.000000 | Bounded ABC search + geometric component verification |

DCT-QR full-suite metrics:

```text
Mean attacked NC: 0.998248
Minimum attacked NC: 0.988790
```

## Old PSO versus active ABC smoke comparison

| Method | PSO PSNR | ABC PSNR | PSO attacked NC | ABC attacked NC |
|---|---:|---:|---:|---:|
| DCT-QR | 50.277718 | 51.719975 | 1.000000 | 1.000000 |
| DCT-Schur | 48.156553 | 48.585684 | 1.000000 | 0.999514 |
| Spatial CD-DetQR | 55.872211 | 55.872211 | 0.998542 | 0.999028 |

The comparison above uses the first deterministic moderate attack. Full ABC colony searches for Schur and Spatial are computationally expensive because each objective evaluation invokes their candidate-search extraction paths. The repository therefore includes a bounded smoke protocol through `--attack-limit`, while the default command still evaluates the complete moderate attack suite.

## Reproduction

Full search:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 20 \
  --cycles 20 \
  --seed 2026
```

Bounded integration test:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 2 \
  --cycles 1 \
  --onlookers 1 \
  --attack-limit 1 \
  --config-output-dir results/abc_smoke_configs
```

## Post-migration non-DCT QR additions

After the historical ABC migration above, the optimizer dispatcher was extended to handle `dct_qr_direct_r`, `dct_qr_r11_qim`, `spatial_qr`, `spatial_qr_direct_r`, and `spatial_qr_r11_qim` explicitly. This also fixes the earlier fall-through where direct-R configurations could be mistaken for Spatial CD-DetQR configuration. The three new spatial methods have checked-in smoke defaults; full ABC optimization results have not yet been claimed.
