#!/usr/bin/env python3
"""Reproduce the previous-vs-proposed DCT-QR comparison.

The script uses the same 64x64 watermark, the same host set, and the same
moderate attack preset for both configurations.  It writes complete per-host
and aggregate JSON results, including per-attack means.
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

from qr64_certified.attacks import apply_attack  # noqa: E402
from qr64_certified.attacks.presets import moderate_attacks  # noqa: E402
from qr64_certified.common.io import load_host_rgb, load_watermark_binary  # noqa: E402
from qr64_certified.common.metrics import nc, psnr  # noqa: E402
from qr64_certified.proposals.config import QR64Config  # noqa: E402
from qr64_certified.proposals.method import embed, extract  # noqa: E402


def load_config(path: Path) -> QR64Config:
    return QR64Config.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def evaluate_config(
    config: QR64Config,
    host_paths: list[Path],
    watermark: np.ndarray,
) -> dict[str, Any]:
    attacks = moderate_attacks()
    rows: list[dict[str, Any]] = []
    for host_path in host_paths:
        started = time.perf_counter()
        host = load_host_rgb(host_path)
        watermarked, key = embed(host, watermark, config=config)
        clean = extract(watermarked, key)
        attack_values: dict[str, float] = {}
        for attack in attacks:
            attacked = apply_attack(watermarked, attack)
            attack_values[attack.name] = float(nc(watermark, extract(attacked, key)))
        values = np.asarray(list(attack_values.values()), dtype=np.float64)
        rows.append(
            {
                "host": host_path.name,
                "psnr": float(psnr(host, watermarked)),
                "clean_nc": float(nc(watermark, clean)),
                "mean_nc": float(np.mean(values)),
                "q10_nc": float(np.quantile(values, 0.10)),
                "min_nc": float(np.min(values)),
                "attacks": attack_values,
                "seconds": float(time.perf_counter() - started),
            }
        )
        print(
            f"{host_path.name}: PSNR={rows[-1]['psnr']:.6f}, "
            f"clean NC={rows[-1]['clean_nc']:.6f}, "
            f"mean NC={rows[-1]['mean_nc']:.6f}",
            flush=True,
        )

    attack_names = [attack.name for attack in attacks]
    per_attack_mean = {
        name: float(np.mean([row["attacks"][name] for row in rows]))
        for name in attack_names
    }
    return {
        "configuration": config.to_dict(),
        "hosts": rows,
        "aggregate": {
            "host_count": len(rows),
            "attack_count": len(attacks),
            "psnr": float(np.mean([row["psnr"] for row in rows])),
            "clean_nc": float(np.mean([row["clean_nc"] for row in rows])),
            "mean_nc": float(np.mean([row["mean_nc"] for row in rows])),
            "mean_q10_nc": float(np.mean([row["q10_nc"] for row in rows])),
            "mean_worst_nc": float(np.mean([row["min_nc"] for row in rows])),
            "global_worst_nc": float(np.min([row["min_nc"] for row in rows])),
            "per_attack_mean_nc": per_attack_mean,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-limit", type=int, default=0, help="0 means all hosts")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/dct_qr_carrier_subspace_comparison.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host_paths = sorted((ROOT / "data/host").glob("*.bmp"), key=lambda p: p.name.lower())
    if args.host_limit > 0:
        host_paths = host_paths[: args.host_limit]
    if not host_paths:
        raise RuntimeError("No BMP host images were found in data/host.")

    watermark = load_watermark_binary(ROOT / "data/watermark/wm.png", size=64)
    previous = load_config(ROOT / "configs/dct_qr_previous_global_qr.json")
    proposed = load_config(ROOT / "configs/dct_qr_after_pso.json")

    result = {
        "protocol": {
            "watermark": "data/watermark/wm.png",
            "watermark_size": [64, 64],
            "host_files": [path.name for path in host_paths],
            "attack_preset": "moderate_attacks",
            "seed": proposed.seed,
        },
        "previous_global_qr": evaluate_config(previous, host_paths, watermark),
        "proposed_carrier_r11": evaluate_config(proposed, host_paths, watermark),
    }

    old = result["previous_global_qr"]["aggregate"]
    new = result["proposed_carrier_r11"]["aggregate"]
    result["delta_proposed_minus_previous"] = {
        "psnr": new["psnr"] - old["psnr"],
        "clean_nc": new["clean_nc"] - old["clean_nc"],
        "mean_nc": new["mean_nc"] - old["mean_nc"],
        "mean_q10_nc": new["mean_q10_nc"] - old["mean_q10_nc"],
        "mean_worst_nc": new["mean_worst_nc"] - old["mean_worst_nc"],
        "global_worst_nc": new["global_worst_nc"] - old["global_worst_nc"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["delta_proposed_minus_previous"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
