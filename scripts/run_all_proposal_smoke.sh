#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export JILP_NUM_THREADS="${JILP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python scripts/list_supported_methods.py
python scripts/smoke_test_proposals.py
pytest -q
