# Three proposal blind-watermark research package

This repository contains three redesigned 64×64 blind-watermark proposals,
16 research baselines, and one deterministic attack library under the unified
`qr64_certified` package.

## Validated proposal methods

| Canonical ID | Domain constraint | Redesigned contribution |
|---|---|---|
| `dct_qr` | DCT + QR | QR-conditioned pairwise coset-optimized QIM with carrier-specific `r11` gain normalization |
| `dct_schur_rescue` | DCT + Schur | spectrum-preserving orthogonal strict-upper Schur coupling QIM with three interleaved payload copies |
| `spatial_cd_detqr` | Spatial QR and `det(A) != 0` | normalized QR-residual carrier with a closed-form minimum integer update and hard determinant floor |

The old exploratory implementations are retained as `*_legacy.py` modules for
ablation and auditability, but the public proposal registry exposes the new
validated paths.

## Main validation result

Protocol: 13 RGB hosts at 512×512, one 64×64 watermark, 15 deterministic
moderate attacks, and 39 clean round-trips.

| Method | Mean PSNR | Clean NC | Mean attacked NC | Mean worst NC/host |
|---|---:|---:|---:|---:|
| DCT–QR | **50.399139 dB** | 1.000000 | **0.998028** | **0.988097** |
| DCT–Schur SP-SCQIM | 48.173152 dB | 1.000000 | 0.993814 | 0.942074 |
| Spatial DetQR | 56.670744 dB | 1.000000 | 0.991345 | 0.942829 |

The active DCT–Schur path is now mathematically independent of DCT–QR and improves mean PSNR over the former Schur-gain hybrid, but it does **not** yet improve aggregate attacked NC. Median filtering on Baboon is the present promotion blocker. See the Schur report for the full gate, proofs, and exact limitation.

Detailed mathematics, novelty boundaries, proof sketches, experiment protocol,
and limitations:

- [`docs/DCT_SCHUR_SP_SCQIM_REPORT.md`](docs/DCT_SCHUR_SP_SCQIM_REPORT.md)
- [`docs/DCT_QR_CARRIER_SUBSPACE_PROPOSAL.md`](docs/DCT_QR_CARRIER_SUBSPACE_PROPOSAL.md)
- [`docs/REDESIGNED_PROPOSALS_VI.md`](docs/REDESIGNED_PROPOSALS_VI.md)
- [`results/dct_qr_previous_vs_carrier_r11.json`](results/dct_qr_previous_vs_carrier_r11.json)
- [`results/dct_qr_pairwise_coset_13host.json`](results/dct_qr_pairwise_coset_13host.json)
- [`results/redesigned_comparison.json`](results/redesigned_comparison.json)
- [`results/redesigned_13host_validation.json`](results/redesigned_13host_validation.json)

## Structure

```text
src/qr64_certified/
├── proposals/
│   ├── method.py              DCT–QR proposal
│   ├── schur_coupling_qim.py  independent DCT–Schur SP-SCQIM proposal
│   ├── direct_schur_rescue.py compatibility wrapper
│   ├── cd_detqr.py            spatial normalized-residual DetQR
│   ├── certificate.py         QR/Schur certificates and gain scales
│   ├── config.py              DCT proposal configuration
│   └── proposal_registry.py   exactly three public proposals
├── baselines/                 16 unified baseline registrations
├── attacks/                   deterministic attack operators and suites
├── common/                    shared I/O and metrics
└── _jilp_core/                integer-lattice closure dependency
```

Compatibility wrappers keep the former root imports available:

```python
from qr64_certified.proposals.method import embed
from qr64_certified.method import embed
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JILP_NUM_THREADS=1
```

Windows PowerShell equivalents:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\scripts"
$env:JILP_NUM_THREADS="1"
```

## Use the proposal API

```python
from qr64_certified.proposals import embed_proposal, extract_proposal

watermarked, key = embed_proposal("dct_qr", host, watermark)
recovered = extract_proposal(watermarked, key)
```

List the proposals:

```bash
python scripts/list_proposals.py
```

## Verify

```bash
pytest -q
python scripts/smoke_test_proposals.py
python scripts/list_baselines.py
python scripts/list_attacks.py
```

Expected test result is reported by the current `pytest -q` run; the suite includes dedicated carrier-QR homogeneity, perturbation-bound, ablation-flag, and clean-round-trip checks. Current result: `49 passed`.



### Reproduce the independent DCT–Schur validation

```bash
python scripts/benchmark_schur_sp_scqim.py \
  --step 9.0 \
  --output results/schur_sp_scqim/reproduced.json
```

The active configuration is `configs/dct_schur_sp_scqim.json`. The benchmark is promotion-gated against the former Schur result; the current implementation passes the clean-NC and PSNR gates but not the attacked-NC gate.

### Reproduce the active DCT–QR validation

Quick one-host validation:

```bash
python scripts/validate_dct_qr_pairwise_coset.py --host-limit 1
```

Full 13-host validation:

```bash
python scripts/validate_dct_qr_pairwise_coset.py
```

The active configuration is `configs/dct_qr_after_abc.json`. It is selected by
the deterministic ABC optimizer while retaining the validated pairwise-coset
embedding law (`coset_group_size=2`). Historical configurations remain
available for controlled before/after comparisons.

## Artificial Bee Colony parameter optimization

The proposal hyperparameters are selected with a deterministic Artificial Bee
Colony (ABC) optimizer. The existing validated configuration is inserted as
food source zero, and the global best is retained even if a scout abandons that
source. Therefore, optimization cannot return a score below the supplied
starting configuration on the evaluated objective.

Full search:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 20 \
  --cycles 20 \
  --seed 2026
```

Fast end-to-end smoke search without overwriting the active configurations:

```bash
python scripts/optimize_parameters.py \
  --method all \
  --food-sources 2 \
  --cycles 1 \
  --onlookers 1 \
  --attack-limit 1 \
  --config-output-dir results/abc_smoke_configs
```

ABC writes `configs/*_after_abc.json` and detailed traces under
`results/optimization_abc/`. The legacy `--particles` and `--iterations`
arguments remain accepted as aliases for `--food-sources` and `--cycles`.

## Reproduce the redesigned validation

Quick one-host check:

```bash
python scripts/run_redesigned_validation.py --host-limit 1
```

Full 13-host validation:

```bash
python scripts/run_redesigned_validation.py
```

The script reads `configs/*_after_abc.json`, fails if any clean NC is below
`1 - 1e-12`, and writes a complete JSON report.

## Scientific scope

The package demonstrates proposal-ready laws and reproducible improvements, but
it is not yet a final publication benchmark. A paper-grade study still needs
multiple watermark patterns and seeds, a held-out parameter-selection protocol,
statistical significance testing, runtime/key-size reporting, and comparison
with independent state-of-the-art implementations under the same attacks.

## Ablation and hyperparameter flags

All three proposals now expose repeatable scientific ablations and direct
hyperparameter overrides:

```bash
python scripts/list_proposal_flags.py
python scripts/run_proposal_benchmark.py --help
```

Example DCT-QR ablation and override:

```bash
python scripts/run_proposal_benchmark.py \
  --method dct_qr \
  --ablation no_coset_optimization \
  --gain-gamma 0.90 \
  --step 13.25
```

Generate one-component-at-a-time ablation tables:

```bash
python scripts/run_ablation_study.py --method dct_qr
python scripts/run_ablation_study.py --method dct_schur_rescue
python scripts/run_ablation_study.py --method spatial_cd_detqr
```

Run a deterministic hyperparameter grid:

```bash
python scripts/run_hyperparameter_sweep.py \
  --method dct_qr \
  --grid 'step=12.75|13.0|13.25' \
  --grid 'gain_gamma=0.8|0.9|1.0'
```

See [`docs/ABLATION_AND_HYPERPARAMETER_FLAGS.md`](docs/ABLATION_AND_HYPERPARAMETER_FLAGS.md)
for the complete method-specific flag tables and examples.

## Unified proposal–baseline benchmark

The repository now includes a shared benchmark layer under
`src/qr64_certified/benchmark/`. It evaluates proposals and baselines with the
same attack objects, metrics, output columns, runtime measurement, key-size
measurement, error handling and aggregation. No proposal or baseline
embedding/extraction formula was changed.

Fast complete check:

```bash
./scripts/run_complete_pipeline.sh
```

Windows PowerShell:

```powershell
.\scripts\run_complete_pipeline.ps1
```

Direct quick benchmark:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/quick.json
```

Publication benchmark with atomic per-trial resume:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/publication.json \
  --resume
```

The attack library now exposes a balanced 84-attack `publication` suite and a
113-attack `extended` suite while preserving the old frozen suites. Expanded
metrics include PSNR, SSIM, UIQI, SNR, MSE/RMSE/MAE, image fidelity, edge and
histogram measures, NC/NCC, BER, bit accuracy, Hamming distance, precision,
recall, specificity, F1, balanced accuracy, timings, key size and payload rate.

See [`docs/UNIFIED_BENCHMARK.md`](docs/UNIFIED_BENCHMARK.md) for method selectors,
attack definitions, result schemas and interpretation rules.
