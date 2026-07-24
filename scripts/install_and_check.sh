#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pip install -r requirements.txt
python -m pip install -e .
bash scripts/run_all_proposal_smoke.sh
