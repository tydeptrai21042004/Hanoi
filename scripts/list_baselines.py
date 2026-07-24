#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.baselines import list_baselines

print(json.dumps({"baselines": list_baselines()}, indent=2, ensure_ascii=False))
