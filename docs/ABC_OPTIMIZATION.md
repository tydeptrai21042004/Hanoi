# Artificial Bee Colony Optimization

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
| DCT–QR | global QIM step, MAP weight, evidence confidence exponent |
| DCT–Schur | coupling step, MAP weight, gain exponent, closure rounds |
| Spatial CD–DetQR | normalized target margin, boundary penalty, determinant margin, pilot margin |

The spatial bounds use the active normalized-carrier scale around `0.005`.
This corrects the obsolete large-margin range that was incompatible with the
validated spatial model and could severely reduce imperceptibility.
