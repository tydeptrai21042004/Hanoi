#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.attacks import available_attack_groups, list_attack_suites

print(json.dumps({
    "attack_groups": list(available_attack_groups()),
    "suites": list_attack_suites(),
}, indent=2, ensure_ascii=False))
