$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:JILP_NUM_THREADS = if ($env:JILP_NUM_THREADS) { $env:JILP_NUM_THREADS } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }
$env:PYTHONPATH = "$Root\src;$Root\scripts"

python -m pytest -q
python scripts/smoke_test_proposals.py
python scripts/smoke_integrated_baselines.py
python scripts/run_unified_benchmark.py --protocol configs/benchmark/quick.json --resume
