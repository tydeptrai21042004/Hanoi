# Three proposal blind-watermark research package

This repository contains three redesigned 64×64 blind-watermark proposals,
16 research baselines, and one deterministic attack library under the unified
`qr64_certified` package.

## Validated proposal methods

| Canonical ID | Domain constraint | Redesigned contribution |
|---|---|---|
| `dct_qr` | DCT + QR | carrier-subspace QR transfer normalization with reliability-conditioned DCT-QIM |
| `dct_schur_rescue` | DCT + Schur | Schur spectral-reliability DCT-QIM with eigenvalue/departure gain compensation |
| `spatial_cd_detqr` | Spatial QR and `det(A) != 0` | normalized QR-residual carrier with a closed-form minimum integer update and hard determinant floor |

The old exploratory implementations are retained as `*_legacy.py` modules for
ablation and auditability, but the public proposal registry exposes the new
validated paths.

## Main validation result

Protocol: 13 RGB hosts at 512×512, one 64×64 watermark, 15 deterministic
moderate attacks, and 39 clean round-trips.

| Method | Mean PSNR | Clean NC | Mean attacked NC | Mean worst NC/host |
|---|---:|---:|---:|---:|
| DCT–QR | **47.250421 dB** | 1.000000 | **0.997867** | **0.986162** |
| DCT–Schur | 48.014086 dB | 1.000000 | 0.996058 | 0.975353 |
| Spatial DetQR | 56.670744 dB | 1.000000 | 0.991345 | 0.942829 |

Compared with the stored previous proposal references, all three improve mean
PSNR and aggregate attacked NC. Spatial DetQR improves the aggregate result but
its separate 2° rotation and 0.08 shear measurements are slightly below the old
version; see the report for the exact limitation.

Detailed mathematics, novelty boundaries, proof sketches, experiment protocol,
and limitations:

- [`docs/DCT_QR_CARRIER_SUBSPACE_PROPOSAL.md`](docs/DCT_QR_CARRIER_SUBSPACE_PROPOSAL.md)
- [`docs/REDESIGNED_PROPOSALS_VI.md`](docs/REDESIGNED_PROPOSALS_VI.md)
- [`results/dct_qr_previous_vs_carrier_r11.json`](results/dct_qr_previous_vs_carrier_r11.json)
- [`results/redesigned_comparison.json`](results/redesigned_comparison.json)
- [`results/redesigned_13host_validation.json`](results/redesigned_13host_validation.json)

## Structure

```text
src/qr64_certified/
├── proposals/
│   ├── method.py              DCT–QR proposal
│   ├── direct_schur_rescue.py DCT–Schur proposal
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

Expected test result is reported by the current `pytest -q` run; the suite includes dedicated carrier-QR homogeneity, perturbation-bound, ablation-flag, and clean-round-trip checks. Current result: `41 passed`.


### Reproduce the focused DCT–QR comparison

Quick one-host comparison:

```bash
python scripts/run_dct_qr_carrier_comparison.py --host-limit 1
```

Full 13-host comparison:

```bash
python scripts/run_dct_qr_carrier_comparison.py
```

The previous global QR configuration is preserved in
`configs/dct_qr_previous_global_qr.json`; the validated carrier-specific
configuration is `configs/dct_qr_after_pso.json`.

## Reproduce the redesigned validation

Quick one-host check:

```bash
python scripts/run_redesigned_validation.py --host-limit 1
```

Full 13-host validation:

```bash
python scripts/run_redesigned_validation.py
```

The script reads `configs/*_after_pso.json`, fails if any clean NC is below
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
  --ablation global_qr_gain \
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
