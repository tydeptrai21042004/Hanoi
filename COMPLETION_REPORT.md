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

## Mathematical change scope

This release intentionally replaces only the public DCT–QR payload parity
assignment with QR-conditioned pairwise coset optimization. The existing DCT
carrier, QR step allocation, QIM margins, gain normalization, pilots,
synchronization, and integer-lattice closure are retained.

The DCT–Schur and Spatial DetQR embedding/extraction formulas and all baseline
implementations are unchanged. Historical preservation records remain under
`docs/PROPOSAL_PRESERVATION.*` and are explicitly marked as records of the
earlier namespace-only migration.

## Validation performed

- Pairwise-coset DCT–QR validation: **13 hosts × 15 attacks**, mean PSNR **50.399139 dB**, clean NC **1.0**, mean attacked NC **0.998028**.
- `pytest -q`: **45 passed**.
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
