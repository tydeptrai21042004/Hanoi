# Three proposal blind-watermark research package

This repository contains three redesigned 64×64 blind-watermark proposals,
16 research baselines, and one deterministic attack library under the unified
`qr64_certified` package.

## Validated proposal methods

| Canonical ID | Domain constraint | Redesigned contribution |
|---|---|---|
| `dct_qr` | DCT + QR | QR reliability-conditioned DCT-QIM with canonical-`R` gain normalization |
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
| DCT–QR | 46.213584 dB | 1.000000 | 0.997246 | 0.982613 |
| DCT–Schur | 48.014086 dB | 1.000000 | 0.996058 | 0.975353 |
| Spatial DetQR | 56.670744 dB | 1.000000 | 0.991345 | 0.942829 |

Compared with the stored previous proposal references, all three improve mean
PSNR and aggregate attacked NC. Spatial DetQR improves the aggregate result but
its separate 2° rotation and 0.08 shear measurements are slightly below the old
version; see the report for the exact limitation.

Detailed mathematics, novelty boundaries, proof sketches, experiment protocol,
and limitations:

- [`docs/REDESIGNED_PROPOSALS_VI.md`](docs/REDESIGNED_PROPOSALS_VI.md)
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

Expected test result for this package:

```text
29 passed
```

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
