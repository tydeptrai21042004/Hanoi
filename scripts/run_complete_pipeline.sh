#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export JILP_NUM_THREADS="${JILP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python scripts/smoke_test_proposals.py
python scripts/optimize_parameters.py --particles "${PSO_PARTICLES:-5}" --iterations "${PSO_ITERATIONS:-3}"
python scripts/benchmark_before_after.py --save-images
python scripts/generate_confidential_pdf.py
pytest -q
