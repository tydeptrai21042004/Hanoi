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
    DCT_QR,
    DCT_QR_DIRECT_R,
    DCT_QR_R11_QIM,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    SPATIAL_QR,
    SPATIAL_QR_DIRECT_R,
    SPATIAL_QR_R11_QIM,
)
from qr64_certified.attacks.presets import moderate_attacks
from qr64_certified.optimization import artificial_bee_colony_maximize
from three_method_utils import (
    METHODS,
    evaluate_method,
    optimization_score,
    read_config,
    write_config,
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artificial Bee Colony (ABC) parameter selection for the three "
            "registered proposal methods."
        )
    )
    parser.add_argument("--host", default=str(ROOT / "data" / "host" / "lenna.bmp"))
    parser.add_argument(
        "--watermark", default=str(ROOT / "data" / "watermark" / "wm.png")
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "optimization_abc")
    )
    parser.add_argument(
        "--config-output-dir",
        default=str(ROOT / "configs"),
        help="Directory receiving *_after_abc.json configurations.",
    )
    parser.add_argument(
        "--food-sources",
        "--particles",
        dest="food_sources",
        type=int,
        default=20,
        help="ABC food sources/employed bees (legacy alias: --particles).",
    )
    parser.add_argument(
        "--cycles",
        "--iterations",
        dest="cycles",
        type=int,
        default=20,
        help="ABC optimization cycles (legacy alias: --iterations).",
    )
    parser.add_argument(
        "--onlookers",
        type=int,
        default=0,
        help="Onlooker bees per cycle; 0 uses the food-source count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="ABC abandonment limit; 0 uses food_sources * dimensions.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument(
        "--attack-limit",
        type=int,
        default=0,
        help="Use only the first N moderate attacks; 0 uses the full suite.",
    )
    return parser.parse_args()


def _starting_config(method: str):
    """Load the strongest available validated configuration as ABC source zero."""
    candidates = (
        ROOT / "configs" / f"{method}_after_abc.json",
        ROOT / "configs" / f"{method}_after_pso.json",
        ROOT / "configs" / f"{method}_before.json",
    )
    for path in candidates:
        if path.exists():
            return read_config(path, method), path
    raise FileNotFoundError(f"No starting configuration found for {method}")


def _problem(method: str, starting):
    if method == DCT_QR:
        bounds = [(8.25, 16.0), (0.20, 0.60), (1.5, 2.5)]
        initial = [
            starting.step,
            starting.qr_map_lambda,
            starting.evidence_conf_power,
        ]

        def make(position: np.ndarray):
            return replace(
                starting,
                step=float(position[0]),
                qr_map_lambda=float(position[1]),
                evidence_conf_power=float(position[2]),
                certificate_mode="qr",
            ).validated()

    elif method == DCT_SCHUR_RESCUE:
        bounds = [(7.5, 10.5), (0.40, 0.90), (0.50, 1.00), (1.0, 3.0)]
        initial = [
            starting.step,
            starting.map_lambda,
            starting.gain_gamma,
            float(starting.closure_rounds),
        ]

        def make(position: np.ndarray):
            return replace(
                starting,
                step=float(position[0]),
                map_lambda=float(position[1]),
                gain_gamma=float(position[2]),
                closure_rounds=max(1, int(round(float(position[3])))),
            ).validated()

    elif method in {DCT_QR_DIRECT_R, DCT_QR_R11_QIM, SPATIAL_QR_DIRECT_R, SPATIAL_QR_R11_QIM}:
        bounds = [(4.0, 16.0), (0.25, 4.0), (1.0, 5.0)]
        initial = [starting.step, starting.regularization, float(starting.closure_rounds)]

        def make(position: np.ndarray):
            return replace(
                starting,
                step=float(position[0]),
                regularization=float(position[1]),
                closure_rounds=max(1, int(round(float(position[2])))),
            ).validated()

    elif method == SPATIAL_QR:
        bounds = [(4.0, 14.0), (0.70, 1.20), (0.25, 4.0), (1.0, 5.0)]
        initial = [
            starting.step,
            starting.gain_gamma,
            starting.regularization,
            float(starting.closure_rounds),
        ]

        def make(position: np.ndarray):
            return replace(
                starting,
                step=float(position[0]),
                gain_gamma=float(position[1]),
                regularization=float(position[2]),
                closure_rounds=max(1, int(round(float(position[3])))),
            ).validated()

    else:
        # Bounds match the active normalized spatial carrier scale. The former
        # PSO script used margins around 1.5--2.75, which was inconsistent with
        # the validated 0.005 configuration and could destroy imperceptibility.
        bounds = [(0.0025, 0.0200), (0.0, 0.0020), (0.05, 0.50), (0.0025, 0.0200)]
        initial = [
            starting.target_margin,
            starting.boundary_penalty,
            starting.embedded_determinant_margin,
            starting.pilot_margin,
        ]

        def make(position: np.ndarray):
            config = replace(
                starting,
                target_margin=float(position[0]),
                boundary_penalty=float(position[1]),
                embedded_determinant_margin=float(position[2]),
                pilot_margin=float(position[3]),
            )
            config.validate()
            return config

    return bounds, initial, make


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    if args.attack_limit < 0:
        raise SystemExit("--attack-limit cannot be negative")

    os.environ.setdefault("JILP_NUM_THREADS", "1")
    host = load_host_rgb(args.host)
    watermark = load_watermark_binary(args.watermark, size=64)
    output = Path(args.output_dir)
    config_output = Path(args.config_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_output.mkdir(parents=True, exist_ok=True)

    attacks = moderate_attacks()
    if args.attack_limit:
        attacks = attacks[: args.attack_limit]
    if not attacks:
        raise SystemExit("The selected attack set is empty")

    selected = METHODS if args.method == "all" else (args.method,)
    overall: dict[str, object] = {
        "host": str(Path(args.host).name),
        "watermark": str(Path(args.watermark).name),
        "optimizer": "deterministic artificial bee colony",
        "food_sources": args.food_sources,
        "cycles": args.cycles,
        "onlooker_bees": args.onlookers or args.food_sources,
        "limit": args.limit or "automatic: food_sources * dimensions",
        "seed": args.seed,
        "attack_count": len(attacks),
        "methods": {},
    }

    for offset, method in enumerate(selected):
        starting, starting_path = _starting_config(method)
        bounds, initial, make_config = _problem(method, starting)
        cache: dict[tuple[float, ...], tuple[float, dict[str, object]]] = {}

        def objective(position: np.ndarray):
            cache_key = tuple(float(round(x, 8)) for x in position)
            if cache_key in cache:
                return cache[cache_key]
            config = make_config(position)
            summary, _rows, _watermarked, _key, _clean = evaluate_method(
                method, config, host, watermark, attacks=attacks
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
                "q10_nc": summary["q10_nc"],
                "min_nc": summary["min_nc"],
                "mean_ber": summary["mean_ber"],
            }
            cache[cache_key] = (score, details)
            print(
                f"{method}: position={np.round(position, 7).tolist()} "
                f"score={score:.8f} PSNR={summary['host_psnr']:.4f} "
                f"cleanNC={summary['clean_nc']:.6f} "
                f"meanNC={summary['mean_nc']:.6f}",
                flush=True,
            )
            return score, details

        result = artificial_bee_colony_maximize(
            objective,
            bounds,
            initial_position=initial,
            food_sources=args.food_sources,
            cycles=args.cycles,
            seed=args.seed + offset,
            limit=None if args.limit == 0 else args.limit,
            onlooker_bees=None if args.onlookers == 0 else args.onlookers,
        )
        best_config = make_config(result.best_position)
        after_path = config_output / f"{method}_after_abc.json"
        write_config(after_path, best_config)
        trace = {
            "method_id": method,
            "optimizer": "artificial_bee_colony",
            "bounds": bounds,
            "starting_config": _display_path(starting_path),
            "initial_position": initial,
            "best_position": result.best_position.tolist(),
            "best_score": result.best_score,
            "history": result.history,
            "evaluations": result.evaluations,
            "after_config": _display_path(after_path),
        }
        (output / f"{method}_abc_trace.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        overall["methods"][method] = {
            "starting_config": _display_path(starting_path),
            "best_position": result.best_position.tolist(),
            "best_score": result.best_score,
            "after_config": _display_path(after_path),
        }

    (output / "optimization_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
