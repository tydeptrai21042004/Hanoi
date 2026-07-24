#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.proposals import list_supported_methods


def main() -> None:
    print(json.dumps({"proposal_methods": list_supported_methods()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
