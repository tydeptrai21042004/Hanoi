#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.proposals.flags import (
    ABLATION_FLAGS,
    HYPERPARAMETER_FLAGS,
    apply_proposal_flags,
)
from three_method_utils import evaluate_method, read_config

METHODS = (DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR)
BOOL_NAMES = {
    "adaptive_step_enabled",
    "gain_normalization_enabled",
    "payload_pattern_search_enabled",
    "determinant_safety_enabled",
    "payload_mask_enabled",
    "affine_sync_enabled",
}
INT_NAMES = {
    "pilot_count",
    "qr_map_iters",
    "schur_closure_iters",
    "max_patterns",
    "fixed_payload_pattern",
    "mask_seed",
    "pilot_seed",
    "closure_rounds",
}
TUPLE_LENGTHS = {
    "adaptive_step_ratios": 3,
    "adaptive_step_fractions": 2,
    "gain_clip": 2,
}
LIST_NAMES = {"rotation_grid", "shear_grid"}
STRING_NAMES = {"qr_gain_mode"}


def _parse_scalar(name: str, text: str) -> Any:
    value = text.strip()
    if name in BOOL_NAMES:
        normalized = value.lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"{name} expects true/false, received {value!r}")
        return normalized in {"true", "1", "yes"}
    if name in INT_NAMES:
        return int(value)
    if name in TUPLE_LENGTHS:
        result = tuple(float(x.strip()) for x in value.split(","))
        if len(result) != TUPLE_LENGTHS[name]:
            raise ValueError(f"{name} expects {TUPLE_LENGTHS[name]} comma-separated values")
        return result
    if name in LIST_NAMES:
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if name in STRING_NAMES:
        return value
    return float(value)


def _parse_grid(spec: str) -> tuple[str, tuple[Any, ...]]:
    if "=" not in spec:
        raise ValueError("Grid must use NAME=VALUE1|VALUE2 syntax.")
    name, values = spec.split("=", 1)
    name = name.strip().replace("-", "_")
    parsed = tuple(_parse_scalar(name, item) for item in values.split("|"))
    if not parsed:
        raise ValueError(f"No values supplied for {name}")
    return name, parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic grid sweep using the proposal flag interface."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage", choices=("before", "after_abc"), default="after_abc")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", type=Path, default=ROOT / "data" / "host" / "lenna.bmp")
    parser.add_argument(
        "--watermark", type=Path, default=ROOT / "data" / "watermark" / "wm.png"
    )
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="NAME=V1|V2|...",
        help="Repeat for a Cartesian sweep; tuple values use commas inside one V.",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help="Optional repeatable ablation applied to every trial.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "hyperparameter_sweep.json"
    )
    args = parser.parse_args()
    os.environ.setdefault("JILP_NUM_THREADS", "1")

    invalid_ablation = [x for x in args.ablation if x not in ABLATION_FLAGS[args.method]]
    if invalid_ablation:
        parser.error(
            f"Invalid ablation(s): {', '.join(invalid_ablation)}. "
            f"Valid: {', '.join(ABLATION_FLAGS[args.method])}"
        )
    try:
        grids = tuple(_parse_grid(spec) for spec in args.grid)
    except ValueError as exc:
        parser.error(str(exc))
    unknown = [name for name, _ in grids if name not in HYPERPARAMETER_FLAGS[args.method]]
    if unknown:
        parser.error(
            f"Unsupported parameter(s) for {args.method}: {', '.join(unknown)}. "
            f"Valid: {', '.join(HYPERPARAMETER_FLAGS[args.method])}"
        )

    config_path = args.config or ROOT / "configs" / f"{args.method}_{args.stage}.json"
    base_config = read_config(config_path, args.method)
    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)

    names = tuple(name for name, _ in grids)
    rows: list[dict[str, Any]] = []
    for index, combination in enumerate(product(*(values for _, values in grids)), start=1):
        flag_values = {name: value for name, value in zip(names, combination, strict=True)}
        flag_values["ablation"] = args.ablation
        try:
            config, flag_report = apply_proposal_flags(args.method, base_config, flag_values)
        except ValueError as exc:
            rows.append({"index": index, "parameters": flag_values, "status": "invalid", "error": str(exc)})
            continue
        summary, _attack_rows, _watermarked, _key, _clean = evaluate_method(
            args.method, config, host, watermark
        )
        row = {
            "index": index,
            "status": "ok",
            "parameters": {name: flag_values[name] for name in names},
            "ablations": list(args.ablation),
            "psnr": float(summary["host_psnr"]),
            "ssim": float(summary["host_ssim"]),
            "clean_nc": float(summary["clean_nc"]),
            "mean_nc": float(summary["mean_nc"]),
            "q10_nc": float(summary["q10_nc"]),
            "min_nc": float(summary["min_nc"]),
            "mean_ber": float(summary["mean_ber"]),
            "flag_report": flag_report,
        }
        rows.append(row)
        values_text = ", ".join(f"{name}={flag_values[name]}" for name in names)
        print(
            f"{index:03d} {values_text}: PSNR={row['psnr']:.5f} "
            f"clean={row['clean_nc']:.6f} meanNC={row['mean_nc']:.6f}",
            flush=True,
        )

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if valid_rows:
        best = max(
            valid_rows,
            key=lambda row: (
                row["clean_nc"] >= 1.0 - 1e-12,
                row["mean_nc"],
                row["psnr"],
            ),
        )
    else:
        best = None
    payload = {
        "method_id": args.method,
        "host": args.host.name,
        "watermark": args.watermark.name,
        "configuration_source": str(config_path),
        "grid": {name: list(values) for name, values in grids},
        "ablations": list(args.ablation),
        "best": best,
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    if valid_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["index", *names, "psnr", "ssim", "clean_nc", "mean_nc", "q10_nc", "min_nc", "mean_ber"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in valid_rows:
                flat = {"index": row["index"], **row["parameters"]}
                for metric in ("psnr", "ssim", "clean_nc", "mean_nc", "q10_nc", "min_nc", "mean_ber"):
                    flat[metric] = row[metric]
                writer.writerow(flat)


if __name__ == "__main__":
    main()
