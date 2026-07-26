$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:JILP_NUM_THREADS = if ($env:JILP_NUM_THREADS) { $env:JILP_NUM_THREADS } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }
$env:PYTHONPATH = "$Root\src;$Root\scripts"

python -m pytest -q
python scripts/run_unified_benchmark.py --protocol configs/benchmark/publication.json --resume
