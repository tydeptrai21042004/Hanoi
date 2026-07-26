#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.benchmark import write_aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild CSV/JSON summaries from unified trial files")
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.trial_dir.parent / "aggregate"
    report = write_aggregate(args.trial_dir, output)
    print(json.dumps({"trial_count": report["trial_count"], "row_count": report["row_count"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
