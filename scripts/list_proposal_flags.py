#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.proposals.flags import ABLATION_FLAGS, HYPERPARAMETER_FLAGS


def main() -> None:
    report = {
        method: {
            "ablation_flags": list(ABLATION_FLAGS[method]),
            "hyperparameter_flags": list(HYPERPARAMETER_FLAGS[method]),
        }
        for method in ABLATION_FLAGS
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
