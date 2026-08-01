#!/usr/bin/env python3
"""Reproduce the redesigned three-proposal validation.

The default protocol evaluates every BMP host under ``data/host`` with the
64x64 watermark under ``data/watermark/wm.png`` and the 15 deterministic
moderate attacks exposed by the package.  It writes per-host records and one
aggregate record for each selected proposal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import (  # noqa: E402
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary  # noqa: E402
from three_method_utils import evaluate_method, read_config  # noqa: E402

METHODS = (DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR)


def _aggregate(host_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not host_rows:
        raise ValueError("No host results were produced.")
    return {
        "host_count": len(host_rows),
        "psnr": float(np.mean([row["psnr"] for row in host_rows])),
        "ssim": float(np.mean([row["ssim"] for row in host_rows])),
        "clean_nc": float(np.mean([row["clean_nc"] for row in host_rows])),
        "mean_nc": float(np.mean([row["mean_nc"] for row in host_rows])),
        "mean_worst_nc": float(np.mean([row["min_nc"] for row in host_rows])),
        "global_worst_nc": float(np.min([row["min_nc"] for row in host_rows])),
        "seconds": float(np.sum([row["seconds"] for row in host_rows])),
    }


def _parse_methods(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [value for value in values if value not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown method(s): {', '.join(unknown)}. Valid: {', '.join(METHODS)}"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the redesigned three watermark proposals on all hosts."
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=METHODS,
        help="Comma-separated proposal IDs; default evaluates all three.",
    )
    parser.add_argument(
        "--host-dir",
        type=Path,
        default=ROOT / "data" / "host",
        help="Directory containing 512x512 RGB BMP hosts.",
    )
    parser.add_argument(
        "--watermark",
        type=Path,
        default=ROOT / "data" / "watermark" / "wm.png",
        help="Binary watermark; it is resized/loaded as 64x64.",
    )
    parser.add_argument(
        "--host-limit",
        type=int,
        default=None,
        help="Optional number of alphabetically sorted hosts for a quick check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "redesigned_validation_reproduced.json",
    )
    parser.add_argument(
        "--strict-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any clean extraction has NC below 1-1e-12 (default: enabled).",
    )
    args = parser.parse_args()

    os.environ.setdefault("JILP_NUM_THREADS", "1")

    hosts = sorted(args.host_dir.glob("*.bmp"), key=lambda p: p.name.lower())
    if args.host_limit is not None:
        if args.host_limit <= 0:
            parser.error("--host-limit must be positive.")
        hosts = hosts[: args.host_limit]
    if not hosts:
        parser.error(f"No BMP host images found under {args.host_dir}")

    watermark = load_watermark_binary(args.watermark, size=64)
    report: dict[str, Any] = {
        "protocol": {
            "host_count": len(hosts),
            "hosts": [path.name for path in hosts],
            "watermark": str(args.watermark),
            "watermark_size": [64, 64],
            "attack_suite": "moderate_attacks (15 deterministic attacks)",
            "jilp_num_threads": os.environ.get("JILP_NUM_THREADS", "1"),
        },
        "methods": {},
    }

    for method in args.methods:
        config_path = ROOT / "configs" / f"{method}_after_abc.json"
        config = read_config(config_path, method)
        host_rows: list[dict[str, Any]] = []
        print(f"\n[{method}] config={config_path.relative_to(ROOT)}", flush=True)

        for index, host_path in enumerate(hosts, start=1):
            host = load_host_rgb(host_path)
            started = time.perf_counter()
            summary, attack_rows, _watermarked, _key, _clean = evaluate_method(
                method, config, host, watermark
            )
            elapsed = time.perf_counter() - started
            row = {
                "host": host_path.name,
                "psnr": float(summary["host_psnr"]),
                "ssim": float(summary["host_ssim"]),
                "clean_nc": float(summary["clean_nc"]),
                "mean_nc": float(summary["mean_nc"]),
                "q10_nc": float(summary["q10_nc"]),
                "min_nc": float(summary["min_nc"]),
                "worst_attack": min(attack_rows, key=lambda item: item["nc"])["attack"],
                "attacks": {item["attack"]: float(item["nc"]) for item in attack_rows},
                "seconds": float(elapsed),
            }
            if args.strict_clean and row["clean_nc"] < 1.0 - 1e-12:
                raise RuntimeError(
                    f"Clean NC requirement failed for {method} on {host_path.name}: "
                    f"{row['clean_nc']:.16f}"
                )
            host_rows.append(row)
            print(
                f"  {index:02d}/{len(hosts):02d} {host_path.name:<16} "
                f"PSNR={row['psnr']:.4f} clean={row['clean_nc']:.6f} "
                f"meanNC={row['mean_nc']:.6f} worst={row['min_nc']:.6f}",
                flush=True,
            )

        aggregate = _aggregate(host_rows)
        report["methods"][method] = {
            "configuration_file": str(config_path.relative_to(ROOT)),
            "hosts": host_rows,
            "aggregate": aggregate,
        }
        print(
            f"[{method}] aggregate PSNR={aggregate['psnr']:.6f}, "
            f"cleanNC={aggregate['clean_nc']:.6f}, "
            f"meanNC={aggregate['mean_nc']:.6f}, "
            f"meanWorst={aggregate['mean_worst_nc']:.6f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"\nSaved reproducible report to: {args.output}")


if __name__ == "__main__":
    main()
