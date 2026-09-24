#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import DCT_QR, DCT_QR_DIRECT_R, DCT_QR_R11_QIM, DCT_QR_THEORY, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR
from three_method_utils import evaluate_method, read_config, write_rows
from qr64_certified.common.io import load_host_rgb, load_watermark_binary, save_image
from qr64_certified.proposals.flags import add_proposal_flag_arguments, apply_proposal_flags
from qr64_certified.attacks.presets import get_attack_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one proposal method on one host and 15 attacks.")
    parser.add_argument("--method", choices=[DCT_QR, DCT_QR_DIRECT_R, DCT_QR_R11_QIM, DCT_QR_THEORY, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR], required=True)
    parser.add_argument("--stage", choices=["before", "after_abc"], default="after_abc")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config; overrides --stage when supplied.",
    )
    parser.add_argument("--host", default=str(ROOT / "data" / "host" / "lenna.bmp"))
    parser.add_argument("--watermark", default=str(ROOT / "data" / "watermark" / "wm.png"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--attack-suite", default="common20", help="Named attack suite (default: common20).")
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print the validated configuration after all flags are applied.",
    )
    add_proposal_flag_arguments(parser)
    args = parser.parse_args()
    os.environ.setdefault("JILP_NUM_THREADS", "1")
    config_path = args.config or ROOT / "configs" / f"{args.method}_{args.stage}.json"
    cfg = read_config(config_path, args.method)
    try:
        cfg, flag_report = apply_proposal_flags(args.method, cfg, args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_effective_config:
        from three_method_utils import config_to_dict
        print(json.dumps(config_to_dict(cfg), indent=2))
    host = load_host_rgb(args.host)
    wm = load_watermark_binary(args.watermark, size=64)
    summary, rows, watermarked, _key, clean = evaluate_method(
        args.method, cfg, host, wm, attacks=get_attack_suite(args.attack_suite)
    )
    summary["attack_suite"] = args.attack_suite
    summary["stage"] = args.stage
    summary["configuration_source"] = str(config_path)
    summary["flag_report"] = flag_report
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / "single" / args.method / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_rows(out / "per_attack.csv", rows)
    if args.save_images:
        save_image(out / "watermarked.png", watermarked)
        save_image(out / "clean_extracted.png", clean)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
