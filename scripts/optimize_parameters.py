#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qr64_certified import (
    CDDetQRConfig,
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    DirectSchurRescueConfig,
    QR64Config,
)
from qr64_certified.optimization import particle_swarm_maximize
from three_method_utils import (
    METHODS,
    evaluate_method,
    optimization_score,
    read_config,
    write_config,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSO parameter selection for the three proposal methods.")
    parser.add_argument("--host", default=str(ROOT / "data" / "host" / "lenna.bmp"))
    parser.add_argument("--watermark", default=str(ROOT / "data" / "watermark" / "wm.png"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "optimization"))
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    return parser.parse_args()


def _problem(method: str, before):
    if method == DCT_QR:
        bounds = [(8.25, 16.0), (0.20, 0.60), (1.5, 2.5)]
        initial = [before.step, before.qr_map_lambda, before.evidence_conf_power]

        def make(position: np.ndarray):
            return replace(
                before,
                step=float(position[0]),
                pilot_step=10.0,
                qr_map_lambda=float(position[1]),
                evidence_conf_power=float(position[2]),
                certificate_mode="qr",
            ).validated()

    elif method == DCT_SCHUR_RESCUE:
        bounds = [(8.25, 16.0), (0.02, 0.08), (0.0, 0.40), (0.70, 1.50)]
        initial = [
            before.base_config.step,
            before.schur_step,
            before.fusion_weight,
            before.gate_power,
        ]

        def make(position: np.ndarray):
            base = replace(
                before.base_config,
                step=float(position[0]),
                pilot_step=10.0,
                certificate_mode="schur",
            ).validated()
            return replace(
                before,
                base_config=base,
                schur_step=float(position[1]),
                fusion_weight=float(position[2]),
                gate_power=float(position[3]),
            ).validated()

    else:
        # Scientific search space for the spatial determinant model.  The
        # validated 71-pilot synchronization design is held fixed so the search
        # isolates payload margin, spatial regularity, determinant safety, and
        # pilot separation.
        bounds = [(1.5, 2.75), (0.0, 0.05), (0.35, 0.75), (6.0, 10.0)]
        initial = [2.0, 0.01, 0.5, 8.0]

        def make(position: np.ndarray):
            cfg = CDDetQRConfig(
                target_margin=float(position[0]),
                boundary_penalty=float(position[1]),
                max_patterns=before.max_patterns,
                determinant_floor=before.determinant_floor,
                embedded_determinant_margin=float(position[2]),
                mask_seed=before.mask_seed,
                pilot_seed=before.pilot_seed,
                pilot_count=71,
                pilot_margin=float(position[3]),
                affine_sync_enabled=before.affine_sync_enabled,
                sync_acceptance_gain=before.sync_acceptance_gain,
                sync_min_pilot_score=before.sync_min_pilot_score,
                rotation_grid=tuple(before.rotation_grid),
                shear_grid=tuple(before.shear_grid),
            )
            cfg.validate()
            return cfg

    return bounds, initial, make


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected = METHODS if args.method == "all" else (args.method,)
    overall: dict[str, object] = {
        "host": str(Path(args.host).name),
        "watermark": str(Path(args.watermark).name),
        "optimizer": "deterministic particle swarm optimization",
        "particles": args.particles,
        "iterations": args.iterations,
        "methods": {},
    }

    for offset, method in enumerate(selected):
        before_path = ROOT / "configs" / f"{method}_before.json"
        before = read_config(before_path, method)
        bounds, initial, make_config = _problem(method, before)
        cache: dict[tuple[float, ...], tuple[float, dict[str, object]]] = {}

        def objective(position: np.ndarray):
            cache_key = tuple(float(round(x, 6)) for x in position)
            if cache_key in cache:
                return cache[cache_key]
            config = make_config(position)
            summary, _rows, _watermarked, _key, _clean = evaluate_method(
                method, config, host, watermark
            )
            if method == SPATIAL_CD_DETQR:
                # Preserve the requested quality floor while maximizing attack
                # NC. A configuration below 54 dB is strongly penalized.
                score = (
                    float(summary["mean_nc"])
                    + 0.0001 * float(summary["host_psnr"])
                    - 10.0 * max(0.0, 54.0 - float(summary["host_psnr"]))
                    - 50.0 * max(0.0, 1.0 - float(summary["clean_nc"]))
                )
            else:
                score = optimization_score(summary)
            details: dict[str, object] = {
                "host_psnr": summary["host_psnr"],
                "clean_nc": summary["clean_nc"],
                "mean_nc": summary["mean_nc"],
                "mean_ber": summary["mean_ber"],
            }
            cache[cache_key] = (score, details)
            print(
                f"{method}: position={np.round(position, 5).tolist()} "
                f"score={score:.8f} PSNR={summary['host_psnr']:.4f} "
                f"cleanNC={summary['clean_nc']:.6f} meanNC={summary['mean_nc']:.6f}",
                flush=True,
            )
            return score, details

        result = particle_swarm_maximize(
            objective,
            bounds,
            initial_position=initial,
            particles=args.particles,
            iterations=args.iterations,
            seed=args.seed + offset,
        )
        best_config = make_config(result.best_position)
        after_path = ROOT / "configs" / f"{method}_after_pso.json"
        write_config(after_path, best_config)
        trace = {
            "method_id": method,
            "bounds": bounds,
            "initial_position": initial,
            "best_position": result.best_position.tolist(),
            "best_score": result.best_score,
            "history": result.history,
            "evaluations": result.evaluations,
            "after_config": str(after_path.relative_to(ROOT)),
        }
        (output / f"{method}_pso_trace.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        overall["methods"][method] = {
            "best_position": result.best_position.tolist(),
            "best_score": result.best_score,
            "after_config": str(after_path.relative_to(ROOT)),
        }

    (output / "optimization_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
