#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qr64_certified.attacks import get_attack_suite, list_attack_suites
from qr64_certified.baselines import evaluate_baseline_attacks, get_baseline_spec


def load_binary(path: Path, size: int) -> np.ndarray:
    arr = np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return np.where(arr > 127, 255, 0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one canonical baseline under a unified attack suite")
    parser.add_argument("baseline_id")
    parser.add_argument("--suite", default="sanity", choices=tuple(list_attack_suites()))
    parser.add_argument("--host", type=Path, default=ROOT / "data/host/lenna.bmp")
    parser.add_argument("--watermark", type=Path, default=ROOT / "data/watermark/wm.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/baseline_attacks")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    spec = get_baseline_spec(args.baseline_id)
    wm_size = 64 if spec.common_4096_payload else 16
    host = np.asarray(Image.open(args.host).convert("RGB"), dtype=np.uint8)
    watermark = load_binary(args.watermark, wm_size)
    rows = evaluate_baseline_attacks(
        spec.canonical_id,
        host,
        watermark,
        get_attack_suite(args.suite),
        seed=args.seed,
        repeat=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.canonical_id}__{args.suite}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
