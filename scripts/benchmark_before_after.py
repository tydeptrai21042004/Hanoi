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

from qr64_certified import DCT_QR, DCT_SCHUR_RESCUE, SPATIAL_CD_DETQR
from three_method_utils import METHODS, evaluate_method, read_config, write_rows
from qr64_certified.common.io import load_host_rgb, load_watermark_binary, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export before/after ABC results for all three methods.")
    parser.add_argument("--host", default=str(ROOT / "data" / "host" / "lenna.bmp"))
    parser.add_argument("--watermark", default=str(ROOT / "data" / "watermark" / "wm.png"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "before_after_abc"))
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    comparison = []
    summaries: dict[str, object] = {}
    for method in METHODS:
        summaries[method] = {}
        stage_summaries = {}
        for stage, suffix in (("before", "before"), ("after_abc", "after_abc")):
            config_path = ROOT / "configs" / f"{method}_{suffix}.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Missing {config_path}. Run scripts/optimize_parameters.py first."
                )
            config = read_config(config_path, method)
            summary, rows, watermarked, _key, recovered_clean = evaluate_method(
                method, config, host, watermark
            )
            summary["stage"] = stage
            summary["host"] = Path(args.host).name
            summary["watermark"] = Path(args.watermark).name
            stage_summaries[stage] = summary
            for row in rows:
                row = dict(row)
                row["stage"] = stage
                combined_rows.append(row)
            stage_dir = output / method / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            write_rows(stage_dir / "per_attack.csv", rows)
            if args.save_images:
                save_image(stage_dir / "watermarked.png", watermarked)
                save_image(stage_dir / "clean_extracted.png", recovered_clean)

        before = stage_summaries["before"]
        after = stage_summaries["after_abc"]
        comparison.append(
            {
                "method_id": method,
                "before_psnr": before["host_psnr"],
                "after_psnr": after["host_psnr"],
                "delta_psnr": after["host_psnr"] - before["host_psnr"],
                "before_clean_nc": before["clean_nc"],
                "after_clean_nc": after["clean_nc"],
                "before_mean_nc": before["mean_nc"],
                "after_mean_nc": after["mean_nc"],
                "delta_mean_nc": after["mean_nc"] - before["mean_nc"],
                "before_mean_ber": before["mean_ber"],
                "after_mean_ber": after["mean_ber"],
                "delta_mean_ber": after["mean_ber"] - before["mean_ber"],
            }
        )
        summaries[method] = stage_summaries

    write_rows(output / "comparison_summary.csv", comparison)
    write_rows(output / "all_per_attack.csv", combined_rows)
    (output / "all_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
