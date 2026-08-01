#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import (
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    embed_proposal,
    extract_proposal,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr, ssim
from qr64_certified.proposals.flags import ABLATION_FLAGS, apply_proposal_flags
from three_method_utils import config_to_dict, evaluate_method, read_config

METHODS = (DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full model and one-component-at-a-time ablations."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("before", "after_abc"), default="after_abc")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", type=Path, default=ROOT / "data" / "host" / "lenna.bmp")
    parser.add_argument(
        "--watermark", type=Path, default=ROOT / "data" / "watermark" / "wm.png"
    )
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated ablations or 'all'. The full model is always included.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "ablation"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Run embedding and clean extraction only; useful for fast flag smoke tests.",
    )
    args = parser.parse_args()
    os.environ.setdefault("JILP_NUM_THREADS", "1")

    available = ABLATION_FLAGS[args.method]
    if args.only == "all":
        selected = available
    else:
        selected = tuple(x.strip() for x in args.only.split(",") if x.strip())
        invalid = [x for x in selected if x not in available]
        if invalid:
            parser.error(
                f"Invalid ablation(s): {', '.join(invalid)}. Valid: {', '.join(available)}"
            )

    config_path = args.config or ROOT / "configs" / f"{args.method}_{args.stage}.json"
    base_config = read_config(config_path, args.method)
    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)

    rows: list[dict[str, Any]] = []
    variants = (("full", ()),) + tuple((name, (name,)) for name in selected)
    for label, ablations in variants:
        config, flag_report = apply_proposal_flags(
            args.method, base_config, {"ablation": list(ablations)}
        )
        if args.clean_only:
            watermarked, key = embed_proposal(
                args.method, host, watermark, config=config
            )
            clean = extract_proposal(watermarked, key)
            clean_value = float(nc(watermark, clean))
            row = {
                "variant": label,
                "ablations": list(ablations),
                "psnr": float(psnr(host, watermarked)),
                "ssim": float(ssim(host, watermarked)),
                "clean_nc": clean_value,
                "mean_nc": clean_value,
                "q10_nc": clean_value,
                "min_nc": clean_value,
                "mean_ber": float(1.0 - np.mean((watermark > 0) == (clean > 0))),
                "flag_report": flag_report,
                "configuration": config_to_dict(config),
            }
        else:
            summary, _attack_rows, _watermarked, _key, _clean = evaluate_method(
                args.method, config, host, watermark
            )
            row = {
                "variant": label,
                "ablations": list(ablations),
                "psnr": float(summary["host_psnr"]),
                "ssim": float(summary["host_ssim"]),
                "clean_nc": float(summary["clean_nc"]),
                "mean_nc": float(summary["mean_nc"]),
                "q10_nc": float(summary["q10_nc"]),
                "min_nc": float(summary["min_nc"]),
                "mean_ber": float(summary["mean_ber"]),
                "flag_report": flag_report,
                "configuration": summary["configuration"],
            }
        rows.append(row)
        print(
            f"{label:<28} PSNR={row['psnr']:.5f} clean={row['clean_nc']:.6f} "
            f"meanNC={row['mean_nc']:.6f} minNC={row['min_nc']:.6f}",
            flush=True,
        )

    out = args.output_dir / args.method
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "method_id": args.method,
        "host": args.host.name,
        "watermark": args.watermark.name,
        "configuration_source": str(config_path),
        "protocol": "clean_only" if args.clean_only else "moderate_attacks_15",
        "variants": rows,
    }
    (out / "ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("variant", "psnr", "ssim", "clean_nc", "mean_nc", "q10_nc", "min_nc", "mean_ber")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


if __name__ == "__main__":
    main()
