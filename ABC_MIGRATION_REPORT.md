# ABC Optimizer Migration Report

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
