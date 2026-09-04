# Unified baseline architecture

> **Repository update — 4 September 2026.** The current public registry contains **eight proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the new non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The independent `dct_schur_rescue` and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

All proposal and baseline code now lives under one Python package:

```text
src/qr64_certified/
├── common/                         shared I/O, metrics, keys, and math helpers
├── attacks/                        one deterministic attack registry
├── baselines/
│   ├── common/
│   │   ├── dct/                    shared DCT/QIM utilities
│   │   └── transform/              shared DWT/IWT/QWT/SVD utilities
│   ├── implementations/
│   │   ├── dct/                    six DCT/QIM/DEW implementations
│   │   └── transform/              ten transform-domain implementations
│   ├── vendor/                     required Fireworks helper only
│   ├── evidence/                   paper-fidelity evidence files
│   ├── metadata.py                 unified scientific disclosures
│   ├── registry.py                 one public baseline API
│   └── evaluation.py               baseline-under-attack evaluation
├── _jilp_core/                     internal core required by two proposals
└── proposals/                      eight current proposal methods (six QR-family + Schur + DetQR)
```

There are no separate `realtime_watermark` or `watermarklab` packages. The
numerical formulas were moved without alteration; only imports, public IDs, and
registry wiring changed.

## Canonical baseline-name rule

Every public baseline ID follows:

```text
<domain>_<algorithm>_<citation>_<access>
```

The final token is always one of:

```text
blind | keyassisted | semiblind | nonblind
```

Examples:

```text
dct_dm_qim_chen2001_blind
dwt_qr_fa_guo2017_keyassisted
dwt_svd_roy2018_semiblind
dwt_entropy_kumar2021_nonblind
```

Old IDs such as `dct_dm_qim_chen2001`, `guo2017_dwt_qr_fa`, and `kumar2021`
remain accepted aliases, but all new reports emit canonical IDs.

## Unified API

```python
from qr64_certified.baselines import embed_baseline, extract_baseline

watermarked, key = embed_baseline(
    "dct_dm_qim_chen2001_blind",
    host_rgb,
    watermark_64x64,
    seed=2026,
)
recovered = extract_baseline(watermarked, key)
```

For a non-blind method:

```python
watermarked, key = embed_baseline(
    "dwt_entropy_kumar2021_nonblind",
    host_rgb,
    watermark_64x64,
)
recovered = extract_baseline(
    watermarked,
    key,
    original_host=host_rgb,
)
```

## Scientific grouping

Every registration records:

- blindness tier;
- fidelity/adaptation tier;
- original-host requirement;
- 4096-bit payload compatibility;
- primary blind-table eligibility;
- original native ID and backward-compatible aliases.

Blind, key-assisted, semi-blind, and non-blind methods must be summarized in
separate tables. `PRIMARY_BLIND_BASELINE_IDS` contains only baselines eligible
for the main blind comparison.

## Verification

```bash
python scripts/list_baselines.py
python scripts/smoke_integrated_baselines.py
pytest -q
```

The reorganization was checked by hashing the watermarked and recovered arrays
from all 16 baselines before and after the move. Every pair was identical.
