# Ablation and Hyperparameter Flags

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

The public proposals expose reproducible component ablations and direct
hyperparameter overrides. No source-code editing is required.

## Discover the available flags

```bash
python scripts/list_proposal_flags.py
python scripts/run_proposal_benchmark.py --help
```

`--ablation` is repeatable. Hyperparameter flags are applied first, then the
requested ablations are applied. The effective validated configuration and a
`flag_report` are written to `summary.json`.

## DCT-QR Pairwise Coset-Optimized QIM

### Ablations

| Flag | Scientific component removed or replaced |
|---|---|
| `no_coset_optimization` | Restores the previous direct payload-to-parity mapping while retaining the same QR steps and gain model. |
| `uniform_step` | Replaces QR reliability-conditioned local QIM steps with one global step. |
| `no_gain_normalization` | Removes carrier-subspace QR gain compensation. |
| `global_qr_gain` | Replaces carrier-specific `r11` gain with the previous global `||diag(R)||2` gain. |
| `no_spatial_map` | Removes ICM spatial regularization by setting MAP strength and iterations to zero. |
| `no_certificate_evidence` | Removes QR reliability from the evidence weight while keeping QIM confidence. |
| `no_sync_certificate` | Removes the QR-certificate term from synchronization candidate selection. |

Example:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --ablation global_qr_gain \
  --ablation no_spatial_map
```

Important hyperparameters include:

```text
--step
--coset-optimization / --no-coset-optimization
--coset-group-size
--adaptive-step-ratios WEAK,MID,STRONG
--adaptive-step-fractions Q1,Q2
--rho-frac
--gain-gamma
--gain-clip LOW,HIGH
--qr-gain-mode carrier_r11|diag_l2
--qr-map-lambda
--qr-map-iters
--evidence-conf-power
--sync-certificate-weight
```

Example:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --step 13.0 \
  --adaptive-step-ratios 1.38,1.0,0.77 \
  --gain-gamma 0.90 \
  --print-effective-config
```

## DCT-QR Direct-R Differential QIM

This proposal performs QR on the regularized 4×4 low-frequency DCT matrix and
embeds directly into the first row of `R`.

### Ablation

| Flag | Scientific component removed or simplified |
|---|---|
| `single_closure` | Uses one uint8 closure round instead of the default two. |

Direct hyperparameters are:

```text
--seed
--step
--regularization
--det-epsilon
--closure-rounds
```

Example:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr_direct_r \
  --step 8.0 \
  --regularization 1.0 \
  --closure-rounds 2
```

## DCT-Schur SP-SCQIM

### Ablations

| Flag | Scientific component removed |
|---|---|
| `single_coupling` | Removes confidence differentiation between the three interleaved observations. |
| `no_gain_normalization` | Removes stored Schur-scale gain normalization. |
| `no_spatial_map` | Removes ICM spatial regularization. |
| `no_candidate_search` | Uses the attacked image directly without blind sharpening selection. |
| `single_closure` | Uses one uint8 lattice-closure round. |

Active hyperparameters include:

```text
--step
--eta
--gain-normalization / --no-gain-normalization
--gain-gamma
--gain-clip
--schur-lift
--schur-closure-iters
--qr-map-lambda        # compatibility CLI name for map_lambda
--qr-map-iters         # compatibility CLI name for map_iters
```

The public method no longer accepts a nested DCT-QR configuration. The historical
Schur-gain hybrid remains in `direct_schur_rescue_legacy.py` for audit only.

## Spatial CD-DetQR

### Ablations

| Flag | Scientific component removed or simplified |
|---|---|
| `fixed_payload_pattern` | Disables per-block pattern search and uses `fixed_payload_pattern`. |
| `no_boundary_penalty` | Removes the spatial-boundary regularizer from carrier selection. |
| `minimal_det_safety` | Removes the extra determinant safety margin; the positive normalized QR margin still enforces `det(A) != 0`. |
| `no_payload_mask` | Removes the secret XOR payload mask. |
| `no_affine_sync` | Disables pilot-based rotation/shear search. |
| `single_closure` | Uses one closure round instead of the configured number. |

Important hyperparameters:

```text
--target-margin
--boundary-penalty
--max-patterns
--fixed-payload-pattern
--determinant-floor
--embedded-determinant-margin
--pilot-count
--pilot-margin
--sync-acceptance-gain
--sync-min-pilot-score
--closure-rounds
--rotation-grid
--shear-grid
--payload-pattern-search / --no-payload-pattern-search
--determinant-safety / --no-determinant-safety
--payload-mask / --no-payload-mask
--affine-sync / --no-affine-sync
```

Example:

```bash
python scripts/run_proposal_benchmark.py \
  --method spatial_cd_detqr \
  --ablation fixed_payload_pattern \
  --fixed-payload-pattern 0 \
  --ablation minimal_det_safety
```

## One-component-at-a-time ablation tables

```bash
python scripts/run_ablation_study.py --method dct_qr
python scripts/run_ablation_study.py --method dct_schur_rescue
python scripts/run_ablation_study.py --method spatial_cd_detqr
```

For a fast configuration and clean-round-trip check, especially for the
affine-search spatial method:

```bash
python scripts/run_ablation_study.py --method spatial_cd_detqr --clean-only
```

A subset can be selected:

```bash
python scripts/run_ablation_study.py \
  --method dct_qr \
  --only global_qr_gain,no_spatial_map
```

Outputs:

```text
results/ablation/<method>/ablation_summary.json
results/ablation/<method>/ablation_summary.csv
```

## Hyperparameter grid sweeps

Grid values use `|` between alternatives. Tuple-valued parameters use commas
inside one alternative.

```bash
python scripts/run_hyperparameter_sweep.py \
  --method dct_qr \
  --grid 'step=12.75|13.0|13.25' \
  --grid 'gain_gamma=0.8|0.9|1.0' \
  --output results/dct_qr_step_gamma.json
```

Tuple example:

```bash
python scripts/run_hyperparameter_sweep.py \
  --method dct_qr \
  --grid 'adaptive_step_ratios=1.30,1.0,0.80|1.38,1.0,0.77'
```

An ablation can be held fixed during a sweep:

```bash
python scripts/run_hyperparameter_sweep.py \
  --method dct_schur_rescue \
  --ablation no_schur_departure \
  --grid 'gain_gamma=0.6|0.7|0.8'
```

## Python API

```python
from qr64_certified import DCT_QR, apply_proposal_flags
from qr64_certified.proposals.proposal_registry import default_config_for_method

base = default_config_for_method(DCT_QR)
config, report = apply_proposal_flags(
    DCT_QR,
    base,
    {
        "ablation": ["global_qr_gain"],
        "step": 13.0,
        "gain_gamma": 0.9,
    },
)
```

Every returned configuration is validated before use. Unknown or
method-incompatible ablations raise an explicit error rather than being
silently ignored.
