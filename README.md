# Unified watermarking research package

This repository contains the **three original 64×64 proposal methods**, 16
research baselines, and one deterministic attack library under the single
`qr64_certified` package.

## Structure

```text
src/qr64_certified/
├── proposals/                 the three original proposal implementations
│   ├── method.py              DCT-QR reliability-conditioned method
│   ├── direct_schur_rescue.py DCT-Schur invariant rescue method
│   ├── cd_detqr.py            spatial CD-DetQR method
│   ├── config.py              original DCT proposal configuration
│   ├── certificate.py         original QR/Schur certificates
│   ├── optimization.py        proposal PSO utility
│   └── proposal_registry.py   registry exposing exactly three proposals
├── baselines/                 16 unified baseline registrations
├── attacks/                   49 attack operators and named suites
├── common/                    shared utilities
└── _jilp_core/                internal dependency used by two proposals
```

The old root import paths remain available as compatibility wrappers. For
example, both imports resolve to the same implementation:

```python
from qr64_certified.proposals.method import embed
from qr64_certified.method import embed
```

## The three proposal methods

| Canonical ID | Implementation | Description |
|---|---|---|
| `dct_qr` | `qr64_certified/proposals/method.py` | QR reliability-conditioned DCT-QIM |
| `dct_schur_rescue` | `qr64_certified/proposals/direct_schur_rescue.py` | invariant-preserving Schur rescue hypothesis |
| `spatial_cd_detqr` | `qr64_certified/proposals/cd_detqr.py` | spatial channel-differential determinant embedding and synchronization |

Use the explicit proposal API:

```python
from qr64_certified.proposals import embed_proposal, extract_proposal

watermarked, key = embed_proposal("dct_qr", host, watermark)
recovered = extract_proposal(watermarked, key)
```

List the proposals:

```bash
python scripts/list_proposals.py
```

The mathematical implementation was not rewritten. The proposal source was
relocated into `proposals/`; only dependency import paths were adjusted where
the removed nested packages had previously been referenced. Numerical
watermarked-image and recovered-watermark hashes are unchanged for all three
methods. See `docs/PROPOSAL_PRESERVATION.md`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JILP_NUM_THREADS=1
```

## Verify

```bash
pytest -q
python scripts/list_proposals.py
python scripts/smoke_test_proposals.py
python scripts/list_baselines.py
python scripts/list_attacks.py
python scripts/smoke_integrated_baselines.py
```

## Baseline naming

Canonical baseline IDs use:

```text
<domain>_<algorithm>_<citation>_<access>
```

Example:

```python
from qr64_certified.baselines import embed_baseline, extract_baseline

watermarked, key = embed_baseline(
    "dct_dm_qim_chen2001_blind",
    host,
    watermark,
)
recovered = extract_baseline(watermarked, key)
```

Legacy baseline IDs remain accepted as aliases, but canonical IDs are emitted
in new results. See `docs/BASELINE_INTEGRATION.md`.

## Attack evaluation

```bash
python scripts/run_baseline_attack_suite.py \
  dct_dm_qim_chen2001_blind \
  --suite sanity
```

Available suites include `sanity`, `compression`, `noise`, `filtering`,
`geometric`, `photometric`, `occlusion`, `combined`, `common`, and `stress`.
See `docs/ATTACKS.md`.

## Existing results

All pre-existing files under `results/` remain unchanged. Their SHA-256
manifest is retained in `docs/RESULTS_PRESERVED.sha256`.

Scientific method descriptions remain in:

- `docs/SCIENTIFIC_METHODS.md`
- `docs/SCIENTIFIC_METHODS_VI.md`
- `results/scientific_validation/`
