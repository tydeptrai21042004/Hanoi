# Artificial Bee Colony Optimization

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

## Purpose

The proposal pipeline now uses a deterministic **Artificial Bee Colony (ABC)**
algorithm for offline hyperparameter selection. ABC replaces the previous
Particle Swarm Optimization driver; it does not change the mathematical
embedding or extraction law of any proposal.

## Search phases

For each cycle, the implementation performs:

1. **Employed-bee phase:** each food source generates one single-dimension
   neighbour and greedily keeps it only when the objective improves.
2. **Onlooker-bee phase:** sources are sampled according to positive ABC
   fitness probabilities, followed by the same greedy neighbour test.
3. **Scout phase:** a source that exceeds the abandonment limit is replaced by
   a uniformly sampled candidate inside the declared bounds.

The existing validated configuration is always inserted as food source zero.
The global best is stored independently of scout replacement, so the returned
score is never worse than the starting configuration on the evaluated
objective.

## Reproducibility

The optimizer is deterministic for a fixed seed. Every objective evaluation is
recorded with its phase, cycle, source, partner, changed dimension, random
coefficient, score, position, and metric details.

Default settings:

```text
food_sources = 20
cycles = 20
onlooker_bees = food_sources
limit = food_sources * number_of_dimensions
seed = 2026
```

## Commands

Full search:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 20 \
  --cycles 20 \
  --seed 2026
```

Bounded integration smoke test:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 2 \
  --cycles 1 \
  --onlookers 1 \
  --attack-limit 1 \
  --config-output-dir results/abc_smoke_configs
```

## Output

- Active configurations: `configs/<method>_after_abc.json`
- Per-method traces: `results/optimization_abc/<method>_abc_trace.json`
- Overall summary: `results/optimization_abc/optimization_summary.json`

## Proposal-specific search spaces

| Proposal | Optimized variables |
|---|---|
| `dct_qr` | global QIM step, QR map weight, evidence confidence exponent |
| `dct_qr_direct_r` | QIM step, QR regularization, closure rounds |
| `dct_qr_r11_qim` | QIM step, QR regularization, closure rounds |
| `spatial_qr` | QIM step, gain exponent, spatial-QR regularization, closure rounds |
| `spatial_qr_direct_r` | QIM step, spatial-QR regularization, closure rounds |
| `spatial_qr_r11_qim` | QIM step, spatial-QR regularization, closure rounds |
| `dct_schur_rescue` | coupling step, MAP weight, gain exponent, closure rounds |
| `spatial_cd_detqr` | normalized target margin, boundary penalty, determinant margin, pilot margin |

The three new `spatial_qr*` methods use the same deterministic ABC driver but no DCT/IDCT code path. Their checked-in `*_after_abc.json` files currently equal the smoke-validated defaults until a full multi-host ABC run is completed.

The Spatial CD-DetQR bounds use the active normalized-carrier scale around `0.005`.
This corrects the obsolete large-margin range that was incompatible with the
validated spatial model and could severely reduce imperceptibility.
