# Completion report

## Delivered changes

- Added one adapter-based benchmark path for all three proposals and all 16 baselines.
- Added deterministic `publication` (84 attacks) and `extended` (113 attacks) suites.
- Added real-world recompression, screen-capture, print–scan, copy–move,
  random-erasing, scan-line, elastic-warp and lens-distortion attacks.
- Added expanded image, watermark, timing, payload and key-size metrics.
- Added atomic per-trial output, resume validation and CSV/JSON aggregation.
- Added quick, all-method smoke and publication protocol configurations.
- Added Bash and PowerShell launchers.
- Replaced the former configuration-mutating complete pipeline with a read-only
  test/smoke/unified-benchmark pipeline.

## Mathematical preservation

No file under either of these numerical implementation trees was modified:

```text
src/qr64_certified/proposals/
src/qr64_certified/baselines/implementations/
```

Changes are limited to attacks, metrics, evaluation adapters, benchmark runners,
configuration, tests, reports and documentation. Therefore, embedding and
extraction equations for every proposal and baseline are unchanged.

## Validation performed

- `pytest -q`: **41 passed**.
- Extended attack execution: **113/113 succeeded**, with RGB `uint8` type and
  original image shape preserved.
- Unified all-method clean smoke: **19/19 methods completed**.
- Unified quick comparison: **8 methods × 3 attacks = 24 successful rows**.
- `scripts/run_complete_pipeline.sh`: completed successfully after the quick
  protocol was frozen to three sanity attacks.
- `sha256sum -c CHECKSUMS.sha256`: all packaged files verified.

## Primary commands

```bash
./scripts/run_complete_pipeline.sh
python scripts/run_unified_benchmark.py --protocol configs/benchmark/quick.json
python scripts/run_unified_benchmark.py --protocol configs/benchmark/all_methods_smoke.json
python scripts/run_unified_benchmark.py --protocol configs/benchmark/publication.json --resume
```

Windows PowerShell:

```powershell
.\scripts\run_complete_pipeline.ps1
.\scripts\run_publication_benchmark.ps1
```

See `docs/UNIFIED_BENCHMARK.md` for the full protocol and output schema.
