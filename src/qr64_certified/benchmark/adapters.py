from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from qr64_certified.baselines.registry import embed_baseline, extract_baseline
from qr64_certified.proposals.cd_detqr import CDDetQRConfig
from qr64_certified.proposals.config import QR64Config
from qr64_certified.proposals.direct_schur_rescue import DirectSchurRescueConfig
from qr64_certified.proposals.proposal_registry import (
    DCT_QR,
    DCT_SCHUR_RESCUE,
    SPATIAL_CD_DETQR,
    default_config_for_method,
    embed_proposal,
    extract_proposal,
)

from .types import BaselineRunParameters, BenchmarkMethodSpec


def load_baseline_parameters(path: Path | None) -> dict[str, BaselineRunParameters]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, BaselineRunParameters] = {}
    for method_id, values in raw.items():
        row = dict(values or {})
        output[str(method_id)] = BaselineRunParameters(
            repeat=row.get("repeat", 1),
            step=row.get("step"),
            method_params=dict(row.get("method_params", {})),
        )
    return output


def _load_proposal_config(spec: BenchmarkMethodSpec, root: Path):
    path = root / str(spec.config_path) if spec.config_path else None
    if path is None or not path.exists():
        return default_config_for_method(spec.method_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if spec.method_id == DCT_QR:
        raw["certificate_mode"] = "qr"
        return QR64Config.from_mapping(raw)
    if spec.method_id == DCT_SCHUR_RESCUE:
        return DirectSchurRescueConfig.from_mapping(raw)
    allowed = set(CDDetQRConfig.__dataclass_fields__)
    config = CDDetQRConfig(**{key: value for key, value in raw.items() if key in allowed})
    config.validate()
    return config


def _proposal_config_for_seed(method_id: str, config: Any, seed: int):
    """Vary scheduling seeds without changing any embedding/extraction equation.

    Seed 2026 exactly preserves the supplied final configuration. Other seeds
    shift only the pseudorandom schedules/masks used by the same method.
    """
    offset = int(seed) - 2026
    if method_id == DCT_QR:
        return replace(config, seed=int(seed)).validated()
    if method_id == DCT_SCHUR_RESCUE:
        base = replace(config.base_config, seed=int(seed))
        return replace(config, base_config=base).validated()
    if method_id == SPATIAL_CD_DETQR:
        varied = replace(
            config,
            mask_seed=int(config.mask_seed) + offset,
            pilot_seed=int(config.pilot_seed) + offset,
        )
        varied.validate()
        return varied
    return config


class BenchmarkAdapter:
    def __init__(
        self,
        spec: BenchmarkMethodSpec,
        root: Path,
        baseline_parameters: BaselineRunParameters | None = None,
    ) -> None:
        self.spec = spec
        self.root = root
        self.baseline_parameters = baseline_parameters or BaselineRunParameters()
        self._proposal_config = _load_proposal_config(spec, root) if spec.method_kind == "proposal" else None

    def configuration_summary(self, seed: int) -> dict[str, Any]:
        if self.spec.method_kind == "baseline":
            return {
                "repeat": self.baseline_parameters.repeat,
                "step": self.baseline_parameters.step,
                "method_params": dict(self.baseline_parameters.method_params),
            }
        config = _proposal_config_for_seed(self.spec.method_id, self._proposal_config, seed)
        if hasattr(config, "to_dict"):
            return dict(config.to_dict())
        from dataclasses import asdict
        return asdict(config)

    def embed(
        self,
        host: np.ndarray,
        watermark: np.ndarray,
        *,
        seed: int,
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        if self.spec.method_kind == "proposal":
            config = _proposal_config_for_seed(self.spec.method_id, self._proposal_config, seed)
            watermarked, key, metadata = embed_proposal(
                self.spec.method_id,
                host,
                watermark,
                config=config,
                return_metadata=True,
            )
            return np.asarray(watermarked, dtype=np.uint8), key, dict(metadata or {})
        watermarked, key = embed_baseline(
            self.spec.method_id,
            host,
            watermark,
            seed=int(seed),
            repeat=self.baseline_parameters.repeat,
            step=self.baseline_parameters.step,
            method_params=dict(self.baseline_parameters.method_params),
        )
        return np.asarray(watermarked, dtype=np.uint8), key, {}

    def extract(
        self,
        image: np.ndarray,
        key: Any,
        *,
        original_host: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.spec.method_kind == "proposal":
            recovered, metadata = extract_proposal(image, key, return_metadata=True)
            return np.asarray(recovered, dtype=np.uint8), dict(metadata or {})
        recovered = extract_baseline(
            image,
            key,
            original_host=original_host if self.spec.requires_original_host else None,
        )
        return np.asarray(recovered, dtype=np.uint8), {}


__all__ = ["BenchmarkAdapter", "load_baseline_parameters"]
