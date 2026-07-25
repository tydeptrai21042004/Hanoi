from __future__ import annotations

"""Spatial QR determinant watermarking with a closed-form minimum update.

The proposal remains entirely in the spatial domain.  For every 8x8 block and
Hadamard partition, luminance means ``u`` and ``v`` form

    A = [[u, 1],
         [v, 1]] = Q R.

Its signed determinant is ``det(A)=u-v`` and the signed normalized QR residual
is

    z(A) = det(Q) r_22 = det(A) / r_11
         = (u-v) / sqrt(u^2+v^2).

An antisymmetric integer update changes ``u`` by ``+s a`` and ``v`` by
``-s a``.  The smallest integer ``a`` that guarantees both a normalized margin
and ``det(A) != 0`` is available in closed form.  Thus QR and nonsingularity are
active mathematical constraints rather than labels added after embedding.
"""

from dataclasses import asdict as _asdict, dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

_H4 = np.array(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=np.float64,
)


def _spatial_partitions() -> tuple[np.ndarray, np.ndarray]:
    candidates: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            pattern = np.outer(_H4[i], _H4[j])
            boundary = float(
                np.abs(np.diff(pattern, axis=0)).sum()
                + np.abs(np.diff(pattern, axis=1)).sum()
            )
            candidates.append((boundary, pattern))
    candidates.sort(key=lambda item: item[0])
    return (
        np.stack([item[1] for item in candidates]),
        np.asarray([item[0] for item in candidates], dtype=np.float64),
    )


_PATTERNS, _BOUNDARIES = _spatial_partitions()


def _expanded_patterns(max_patterns: int) -> np.ndarray:
    return np.repeat(
        np.repeat(_PATTERNS[:max_patterns], 2, axis=1), 2, axis=2
    ).astype(np.int8)


def _luminance(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _blocks8(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    if h % 8 or w % 8:
        raise ValueError("Host dimensions must be divisible by 8")
    return (
        channel.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def qr_residual_features(
    image: np.ndarray, max_patterns: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``u``, ``v``, signed determinant and normalized QR residual.

    The determinant is evaluated from a canonical QR factorization, not from a
    direct subtraction.  Canonicalization makes the diagonal of R nonnegative,
    eliminating QR sign ambiguity.
    """
    if not 1 <= int(max_patterns) <= len(_PATTERNS):
        raise ValueError(f"max_patterns must be in [1, {len(_PATTERNS)}]")
    blocks = _blocks8(_luminance(image))
    patterns = _expanded_patterns(int(max_patterns))
    plus = np.stack([blocks[:, p > 0].mean(axis=1) for p in patterns], axis=1)
    minus = np.stack([blocks[:, p < 0].mean(axis=1) for p in patterns], axis=1)

    matrices = np.empty((*plus.shape, 2, 2), dtype=np.float64)
    matrices[..., 0, 0] = plus
    matrices[..., 0, 1] = 1.0
    matrices[..., 1, 0] = minus
    matrices[..., 1, 1] = 1.0
    q, r = np.linalg.qr(matrices)
    diagonal = np.diagonal(r, axis1=-2, axis2=-1)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q = q * signs[..., None, :]
    r = signs[..., :, None] * r
    det_q = q[..., 0, 0] * q[..., 1, 1] - q[..., 0, 1] * q[..., 1, 0]
    determinant = det_q * r[..., 0, 0] * r[..., 1, 1]
    normalized_residual = det_q * r[..., 1, 1]
    return plus, minus, determinant, normalized_residual


def qr_determinant_features(image: np.ndarray, max_patterns: int = 10) -> np.ndarray:
    """Backward-compatible determinant feature, now for the proposal luminance matrix."""
    _u, _v, determinant, _z = qr_residual_features(image, max_patterns)
    return determinant


def minimum_integer_amplitude(
    u: np.ndarray,
    v: np.ndarray,
    sign: np.ndarray,
    normalized_margin: float,
    determinant_margin: float,
) -> np.ndarray:
    """Closed-form smallest integer update satisfying both constraints.

    For ``d=u-v``, ``k=u+v`` and desired sign ``s``, an antisymmetric update
    gives ``d'=d+2sa`` while preserving ``k``.  The condition

        s z(A') >= mu,  z(A') = sqrt(2)d'/sqrt(k^2+d'^2)

    is equivalent to

        s d' >= mu |k| / sqrt(2-mu^2).

    Together with ``|det(A')| >= delta``, the exact minimum is

        a* = max(0, ceil((max(delta, mu|k|/sqrt(2-mu^2)) - s d)/2)).
    """
    mu = float(normalized_margin)
    if not 0 < mu < np.sqrt(2.0):
        raise ValueError("normalized_margin must lie in (0, sqrt(2))")
    delta = float(determinant_margin)
    if delta <= 0:
        raise ValueError("determinant_margin must be positive")
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sign = np.asarray(sign, dtype=np.float64)
    d = u - v
    k = u + v
    threshold = np.maximum(
        delta,
        mu * np.abs(k) / np.sqrt(2.0 - mu * mu),
    )
    return np.maximum(
        0.0, np.ceil((threshold - sign[..., None] * d) / 2.0 - 1e-12)
    ).astype(np.int64)


@dataclass(frozen=True)
class CDDetQRConfig:
    # Tested proposal configuration.  The normalized QR margin is small because
    # pattern selection and the exact minimum update carry most of the burden.
    target_margin: float = 0.005
    boundary_penalty: float = 0.0002
    max_patterns: int = 10
    determinant_floor: float = 1e-6
    embedded_determinant_margin: float = 0.10

    mask_seed: int = 918273
    pilot_seed: int = 202607
    pilot_count: int = 31
    pilot_margin: float = 0.005

    affine_sync_enabled: bool = True
    sync_acceptance_gain: float = 0.30
    sync_min_pilot_score: float = 0.50
    rotation_grid: tuple[float, ...] = (
        -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
    )
    shear_grid: tuple[float, ...] = (
        -0.10, -0.08, -0.06, -0.04,
        0.04, 0.06, 0.08, 0.10,
    )
    closure_rounds: int = 3

    def validate(self) -> None:
        if not 0 < self.target_margin < np.sqrt(2.0):
            raise ValueError("target_margin must be in (0, sqrt(2))")
        if self.boundary_penalty < 0:
            raise ValueError("boundary_penalty must be nonnegative")
        if not 1 <= self.max_patterns <= len(_PATTERNS):
            raise ValueError(f"max_patterns must be in [1, {len(_PATTERNS)}]")
        if self.determinant_floor <= 0:
            raise ValueError("determinant_floor must be positive")
        if self.embedded_determinant_margin <= self.determinant_floor:
            raise ValueError("embedded_determinant_margin must exceed determinant_floor")
        if self.pilot_count <= 0 or self.pilot_count % 2 == 0:
            raise ValueError("pilot_count must be a positive odd integer")
        if not 0 < self.pilot_margin < np.sqrt(2.0):
            raise ValueError("pilot_margin must be in (0, sqrt(2))")
        if self.sync_acceptance_gain < 0:
            raise ValueError("sync_acceptance_gain must be nonnegative")
        if not -1 <= self.sync_min_pilot_score <= 1:
            raise ValueError("sync_min_pilot_score must be in [-1,1]")
        if self.closure_rounds < 1:
            raise ValueError("closure_rounds must be at least one")


def _rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(
        pil.rotate(
            float(degrees),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        ),
        dtype=np.uint8,
    )


def _shear_image(image: np.ndarray, shear_x: float) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    h, w = array.shape[:2]
    pil = Image.fromarray(array)
    coefficients = (1.0, -float(shear_x), 0.0, 0.0, 1.0, 0.0)
    return np.asarray(
        pil.transform(
            (w, h),
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        ),
        dtype=np.uint8,
    )


class BlindCDDetQR:
    """Blind spatial normalized-QR-residual determinant watermarking."""

    def __init__(self, config: CDDetQRConfig | None = None):
        self.config = config or CDDetQRConfig()
        self.config.validate()
        self.key: dict[str, Any] | None = None
        self._report: dict[str, Any] | None = None

    @staticmethod
    def _block_origin(block: int, width_blocks: int) -> tuple[int, int]:
        return (block // width_blocks) * 8, (block % width_blocks) * 8

    def _apply_update(
        self,
        output: np.ndarray,
        block: int,
        pattern: int,
        sign: int,
        amplitude: int,
    ) -> int:
        if amplitude <= 0:
            return 0
        width_blocks = output.shape[1] // 8
        row, col = self._block_origin(int(block), width_blocks)
        p = _expanded_patterns(self.config.max_patterns)[int(pattern)].astype(np.int16)
        proposed = (
            output[row : row + 8, col : col + 8].astype(np.int16)
            + int(sign) * int(amplitude) * p[..., None]
        )
        clipped = int(np.count_nonzero((proposed < 0) | (proposed > 255)))
        output[row : row + 8, col : col + 8] = np.clip(proposed, 0, 255).astype(np.uint8)
        return clipped

    def _select_payload(
        self, host: np.ndarray, coded_bits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u, v, determinant, normalized = qr_residual_features(
            host, self.config.max_patterns
        )
        sign = np.where(coded_bits > 0, 1, -1).astype(np.int64)
        amplitudes = minimum_integer_amplitude(
            u,
            v,
            sign,
            self.config.target_margin,
            self.config.embedded_determinant_margin,
        )
        cost = (
            amplitudes.astype(np.float64) ** 2
            + self.config.boundary_penalty
            * _BOUNDARIES[: self.config.max_patterns][None, :]
            - 1e-6 * sign[:, None] * normalized
        )
        selected = np.argmin(cost, axis=1).astype(np.int64)
        idx = np.arange(coded_bits.size)
        return selected, amplitudes[idx, selected], determinant[idx, selected], sign

    def _select_pilots(
        self, image: np.ndarray, payload_patterns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, v, determinant, _normalized = qr_residual_features(
            image, self.config.max_patterns
        )
        sign = np.ones(determinant.shape[0], dtype=np.int64)
        amplitudes = minimum_integer_amplitude(
            u,
            v,
            sign,
            self.config.pilot_margin,
            self.config.embedded_determinant_margin,
        )
        permutation = np.random.default_rng(self.config.pilot_seed).permutation(
            determinant.shape[0]
        )
        blocks: list[int] = []
        patterns: list[int] = []
        selected_amplitudes: list[int] = []
        for block in permutation:
            choices = np.arange(self.config.max_patterns, dtype=np.int64)
            choices = choices[choices != int(payload_patterns[block])]
            # The original host pilot must represent the negative hypothesis.
            choices = choices[
                determinant[block, choices]
                < -self.config.embedded_determinant_margin
            ]
            if choices.size == 0:
                continue
            cost = (
                amplitudes[block, choices].astype(np.float64) ** 2
                + self.config.boundary_penalty * _BOUNDARIES[choices]
            )
            selected = int(choices[int(np.argmin(cost))])
            blocks.append(int(block))
            patterns.append(selected)
            selected_amplitudes.append(int(amplitudes[block, selected]))
            if len(blocks) == self.config.pilot_count:
                break
        if len(blocks) != self.config.pilot_count:
            raise RuntimeError(
                "Not enough negative nonsingular QR determinant carriers for pilots."
            )
        return (
            np.asarray(blocks, dtype=np.int64),
            np.asarray(patterns, dtype=np.int64),
            np.asarray(selected_amplitudes, dtype=np.int64),
        )

    def _close_schedule(
        self,
        output: np.ndarray,
        blocks: np.ndarray,
        patterns: np.ndarray,
        signs: np.ndarray,
        margin: float,
    ) -> tuple[int, int]:
        u, v, _det, _z = qr_residual_features(output, self.config.max_patterns)
        selected_u = u[blocks, patterns][:, None]
        selected_v = v[blocks, patterns][:, None]
        amplitude = minimum_integer_amplitude(
            selected_u,
            selected_v,
            signs,
            margin,
            self.config.embedded_determinant_margin,
        )[:, 0]
        clipped = 0
        updates = 0
        for block, pattern, sign, a in zip(
            blocks, patterns, signs, amplitude, strict=True
        ):
            if int(a) > 0:
                clipped += self._apply_update(
                    output, int(block), int(pattern), int(sign), int(a)
                )
                updates += int(a)
        return updates, clipped

    @staticmethod
    def _pilot_score_from_features(
        normalized: np.ndarray, blocks: np.ndarray, patterns: np.ndarray
    ) -> float:
        values = normalized[blocks, patterns]
        scale = float(np.median(np.abs(values))) + 1e-8
        return float(np.mean(np.tanh(values / scale)))

    def _pilot_score(self, image: np.ndarray, key: dict[str, Any]) -> float:
        _u, _v, _d, normalized = qr_residual_features(
            image, int(key["max_patterns"])
        )
        return self._pilot_score_from_features(
            normalized,
            np.asarray(key["pilot_blocks"], dtype=np.int64),
            np.asarray(key["pilot_pattern"], dtype=np.int64),
        )

    def _align_from_pilots(
        self, image: np.ndarray, key: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = np.asarray(image, dtype=np.uint8)
        identity_score = self._pilot_score(image, key)
        if not bool(key.get("affine_sync_enabled", self.config.affine_sync_enabled)):
            return image, {
                "sync_accepted": False,
                "sync_candidate": "identity",
                "identity_pilot_score": identity_score,
                "best_pilot_score": identity_score,
                "pilot_score_gain": 0.0,
            }
        candidates: list[tuple[str, np.ndarray]] = []
        candidates.extend(
            (f"rotation_{angle:+.1f}", _rotate_image(image, angle))
            for angle in key.get("rotation_grid", self.config.rotation_grid)
            if abs(float(angle)) > 1e-12
        )
        candidates.extend(
            (f"shear_{shear:+.2f}", _shear_image(image, shear))
            for shear in key.get("shear_grid", self.config.shear_grid)
            if abs(float(shear)) > 1e-12
        )
        best_name = "identity"
        best_image = image
        best_score = identity_score
        for name, candidate in candidates:
            score = self._pilot_score(candidate, key)
            if score > best_score:
                best_name, best_image, best_score = name, candidate, score
        gain = float(best_score - identity_score)
        accepted = bool(
            best_name != "identity"
            and gain > float(key["sync_acceptance_gain"])
            and best_score >= float(key["sync_min_pilot_score"])
        )
        return (best_image if accepted else image), {
            "sync_accepted": accepted,
            "sync_candidate": best_name if accepted else "identity",
            "proposed_candidate": best_name,
            "identity_pilot_score": float(identity_score),
            "best_pilot_score": float(best_score),
            "pilot_score_gain": gain,
            "acceptance_gain": float(key["sync_acceptance_gain"]),
            "minimum_pilot_score": float(key["sync_min_pilot_score"]),
        }

    def embed(
        self, host_rgb: np.ndarray, watermark_binary: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        host = np.asarray(host_rgb, dtype=np.uint8)
        watermark = (np.asarray(watermark_binary, dtype=np.uint8) > 0).astype(np.uint8)
        if host.shape != (512, 512, 3):
            raise ValueError(f"Spatial CD-DetQR requires 512x512 RGB; got {host.shape}")
        if watermark.shape != (64, 64):
            raise ValueError(f"Watermark must be 64x64; got {watermark.shape}")
        bits = watermark.ravel()
        mask = np.random.default_rng(self.config.mask_seed).integers(
            0, 2, bits.size, dtype=np.uint8
        )
        coded = bits ^ mask
        payload_patterns, payload_amplitudes, original_det, signs = self._select_payload(
            host, coded
        )
        blocks = np.arange(bits.size, dtype=np.int64)
        output = host.copy()
        clipped = 0
        for block, pattern, sign, amplitude in zip(
            blocks, payload_patterns, signs, payload_amplitudes, strict=True
        ):
            clipped += self._apply_update(
                output, int(block), int(pattern), int(sign), int(amplitude)
            )

        pilot_blocks, pilot_patterns, pilot_amplitudes = self._select_pilots(
            output, payload_patterns
        )
        for block, pattern, amplitude in zip(
            pilot_blocks, pilot_patterns, pilot_amplitudes, strict=True
        ):
            clipped += self._apply_update(
                output, int(block), int(pattern), 1, int(amplitude)
            )

        closure_updates = 0
        for _ in range(self.config.closure_rounds):
            updates, added_clipping = self._close_schedule(
                output,
                blocks,
                payload_patterns,
                signs,
                self.config.target_margin,
            )
            closure_updates += updates
            clipped += added_clipping
            updates, added_clipping = self._close_schedule(
                output,
                pilot_blocks,
                pilot_patterns,
                np.ones(pilot_blocks.size, dtype=np.int64),
                self.config.pilot_margin,
            )
            closure_updates += updates
            clipped += added_clipping
        # Payload exactness has priority in the final closure.
        updates, added_clipping = self._close_schedule(
            output,
            blocks,
            payload_patterns,
            signs,
            self.config.target_margin,
        )
        closure_updates += updates
        clipped += added_clipping

        _u, _v, determinant, normalized = qr_residual_features(
            output, self.config.max_patterns
        )
        payload_det = determinant[blocks, payload_patterns]
        pilot_det = determinant[pilot_blocks, pilot_patterns]
        clean_coded = (payload_det > 0).astype(np.uint8)
        pilot_votes = int(np.count_nonzero(pilot_det > 0))
        pilot_detected = pilot_votes > pilot_blocks.size // 2
        clean_bits = clean_coded ^ mask if pilot_detected else clean_coded
        clean_errors = int(np.count_nonzero(clean_bits != bits))
        if clean_errors:
            raise RuntimeError(
                f"Spatial QR closure failed to make clean extraction exact: {clean_errors} bit(s)."
            )

        # Evaluate the untouched host with the same public decision path.
        _hu, _hv, host_det, _hz = qr_residual_features(host, self.config.max_patterns)
        original_coded = (host_det[blocks, payload_patterns] > 0).astype(np.uint8)
        original_pilot_votes = int(
            np.count_nonzero(host_det[pilot_blocks, pilot_patterns] > 0)
        )
        original_pilot = original_pilot_votes > pilot_blocks.size // 2
        original_bits = original_coded ^ mask if original_pilot else original_coded

        selected_abs = np.abs(np.concatenate([payload_det, pilot_det]))
        activated_singular = int(
            np.count_nonzero(np.abs(original_det) <= self.config.determinant_floor)
        )
        self.key = {
            "method": "spatial_normalized_qr_residual_det_nonzero",
            "method_class": "key_assisted_fully_blind",
            "domain": "single_spatial_qr_domain",
            "image_shape": list(host.shape),
            "watermark_shape": list(watermark.shape),
            "max_patterns": int(self.config.max_patterns),
            "target_margin": float(self.config.target_margin),
            "determinant_floor": float(self.config.determinant_floor),
            "embedded_determinant_margin": float(
                self.config.embedded_determinant_margin
            ),
            "mask_seed": int(self.config.mask_seed),
            "pilot_seed": int(self.config.pilot_seed),
            "payload_pattern": payload_patterns.astype(np.uint8).tolist(),
            "pilot_blocks": pilot_blocks.astype(np.uint16).tolist(),
            "pilot_pattern": pilot_patterns.astype(np.uint8).tolist(),
            "affine_sync_enabled": bool(self.config.affine_sync_enabled),
            "sync_acceptance_gain": float(self.config.sync_acceptance_gain),
            "sync_min_pilot_score": float(self.config.sync_min_pilot_score),
            "rotation_grid": [float(x) for x in self.config.rotation_grid],
            "shear_grid": [float(x) for x in self.config.shear_grid],
            "embedding_rule": (
                "a*=max(0,ceil((max(delta,mu|u+v|/sqrt(2-mu^2))-s(u-v))/2))"
            ),
            "qr_carrier": "z=det(Q)r22=det(A)/r11",
        }
        self._report = {
            "clean_bit_errors": clean_errors,
            "clean_pilot_votes": pilot_votes,
            "clean_pilot_detected": bool(pilot_detected),
            "original_host_pilot_votes": original_pilot_votes,
            "original_host_pilot_detected": bool(original_pilot),
            "original_host_bit_agreement": float(np.mean(original_bits == bits)),
            "all_selected_component_determinants_nonzero": bool(
                np.all(selected_abs > self.config.determinant_floor)
            ),
            "minimum_abs_component_determinant": float(np.min(selected_abs)),
            "minimum_abs_payload_difference": float(np.min(np.abs(payload_det))),
            "minimum_abs_pilot_difference": float(np.min(np.abs(pilot_det))),
            "minimum_signed_normalized_payload_margin": float(
                np.min(signs * normalized[blocks, payload_patterns])
            ),
            "mean_payload_amplitude": float(np.mean(payload_amplitudes)),
            "median_payload_amplitude": float(np.median(payload_amplitudes)),
            "maximum_payload_amplitude": int(np.max(payload_amplitudes)),
            "zero_payload_amplitude_fraction": float(
                np.mean(payload_amplitudes == 0)
            ),
            "mean_pilot_amplitude": float(np.mean(pilot_amplitudes)),
            "maximum_pilot_amplitude": int(np.max(pilot_amplitudes)),
            "pilot_count": int(pilot_blocks.size),
            "activated_singular_payload_blocks": activated_singular,
            "closure_integer_updates": int(closure_updates),
            "clipped_channel_values": int(clipped),
        }
        return output, self.key

    def extract_with_metadata(
        self, questioned_image: np.ndarray, key: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        key = key or self.key
        if key is None:
            raise ValueError("embedding key is required")
        image = np.asarray(questioned_image, dtype=np.uint8)
        expected = tuple(int(x) for x in key["image_shape"])
        if image.shape != expected:
            raise ValueError(f"questioned image shape {image.shape} != {expected}")
        aligned, sync = self._align_from_pilots(image, key)
        _u, _v, determinant, normalized = qr_residual_features(
            aligned, int(key["max_patterns"])
        )
        n = int(np.prod(key["watermark_shape"]))
        blocks = np.arange(n, dtype=np.int64)
        payload_patterns = np.asarray(key["payload_pattern"], dtype=np.int64)
        pilot_blocks = np.asarray(key["pilot_blocks"], dtype=np.int64)
        pilot_patterns = np.asarray(key["pilot_pattern"], dtype=np.int64)
        payload_det = determinant[blocks, payload_patterns]
        pilot_det = determinant[pilot_blocks, pilot_patterns]
        coded = (payload_det > 0).astype(np.uint8)
        pilot_votes = int(np.count_nonzero(pilot_det > 0))
        pilot_detected = pilot_votes > pilot_blocks.size // 2
        if pilot_detected:
            mask = np.random.default_rng(int(key["mask_seed"])).integers(
                0, 2, n, dtype=np.uint8
            )
            bits = coded ^ mask
            path = "qr_determinant_pilot_then_unmask"
        else:
            bits = coded
            path = "pilot_absent_keep_masked_payload"
        selected_abs = np.abs(np.concatenate([payload_det, pilot_det]))
        floor = float(key["determinant_floor"])
        metadata = {
            "pilot_detected": bool(pilot_detected),
            "pilot_votes": pilot_votes,
            "pilot_count": int(pilot_blocks.size),
            "det_nonzero": bool(np.all(selected_abs > floor)),
            "min_abs_det": float(np.min(selected_abs)),
            "minimum_abs_payload_difference": float(np.min(np.abs(payload_det))),
            "minimum_abs_pilot_difference": float(np.min(np.abs(pilot_det))),
            "mean_abs_normalized_qr_residual": float(
                np.mean(np.abs(normalized[blocks, payload_patterns]))
            ),
            "inference_path": path,
            "sync": sync,
        }
        recovered = (
            bits.reshape(tuple(int(x) for x in key["watermark_shape"])) * 255
        ).astype(np.uint8)
        return recovered, metadata

    def extract(
        self, questioned_image: np.ndarray, key: dict[str, Any] | None = None
    ) -> np.ndarray:
        recovered, _metadata = self.extract_with_metadata(questioned_image, key)
        return recovered

    def embedding_report(self) -> dict[str, Any]:
        if self._report is None:
            raise ValueError("embed must be called first")
        return dict(self._report)


METHOD_ID = "spatial_cd_detqr"
DISPLAY_NAME = "Spatial normalized-QR residual DetQR (det(A) != 0)"


def _config_from_mapping(
    value: CDDetQRConfig | Mapping[str, Any] | None,
) -> CDDetQRConfig:
    if value is None:
        cfg = CDDetQRConfig()
    elif isinstance(value, CDDetQRConfig):
        cfg = value
    else:
        allowed = set(CDDetQRConfig.__dataclass_fields__)
        raw = {k: v for k, v in dict(value).items() if k in allowed}
        if "rotation_grid" in raw:
            raw["rotation_grid"] = tuple(raw["rotation_grid"])
        if "shear_grid" in raw:
            raw["shear_grid"] = tuple(raw["shear_grid"])
        cfg = CDDetQRConfig(**raw)
    cfg.validate()
    return cfg


@dataclass
class CDDetQRKey:
    payload: dict[str, Any]
    config: dict[str, Any]
    report: dict[str, Any]

    @property
    def fully_blind(self) -> bool:
        return True

    @property
    def method_id(self) -> str:
        return METHOD_ID

    @property
    def certificate_mode(self) -> str:
        return "qr_spatial_normalized_residual_det_nonzero"


def embed_cd_detqr(
    host_rgb: np.ndarray,
    watermark_binary: np.ndarray,
    *,
    config: CDDetQRConfig | Mapping[str, Any] | None = None,
    return_metadata: bool = False,
):
    cfg = _config_from_mapping(config)
    method = BlindCDDetQR(cfg)
    watermarked, raw_key = method.embed(host_rgb, watermark_binary)
    report = method.embedding_report()
    key = CDDetQRKey(payload=raw_key, config=_asdict(cfg), report=report)
    metadata = {
        "method_id": METHOD_ID,
        "certificate_mode": "qr_spatial_normalized_residual_det_nonzero",
        "direct_schur_carrier": False,
        "fully_blind": True,
        "domain": "single_spatial_domain",
        "configuration": key.config,
        "det_nonzero": report["all_selected_component_determinants_nonzero"],
        "min_abs_det": report["minimum_abs_component_determinant"],
        "clean_pilot_detected": report["clean_pilot_detected"],
        "embedding_rule": raw_key["embedding_rule"],
    }
    return (watermarked, key, metadata) if return_metadata else (watermarked, key)


def extract_cd_detqr(
    possibly_attacked_rgb: np.ndarray,
    key: CDDetQRKey,
    *,
    return_metadata: bool = False,
):
    cfg = _config_from_mapping(key.config)
    method = BlindCDDetQR(cfg)
    recovered, method_metadata = method.extract_with_metadata(
        possibly_attacked_rgb, key.payload
    )
    metadata = {
        "method_id": METHOD_ID,
        "certificate_mode": "qr_spatial_normalized_residual_det_nonzero",
        "fully_blind": True,
        "original_host_used": False,
        "original_watermark_used": False,
        "det_nonzero": method_metadata["det_nonzero"],
        "min_abs_det": method_metadata["min_abs_det"],
        "mean_qim_confidence": "",
        "mean_certificate_reliability": "",
        "pilot_detected": method_metadata["pilot_detected"],
        "pilot_votes": method_metadata["pilot_votes"],
        "pilot_count": method_metadata["pilot_count"],
        "inference_path": method_metadata["inference_path"],
        "sync": method_metadata.get("sync", {}),
        "mean_abs_normalized_qr_residual": method_metadata[
            "mean_abs_normalized_qr_residual"
        ],
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "BlindCDDetQR",
    "CDDetQRConfig",
    "CDDetQRKey",
    "minimum_integer_amplitude",
    "qr_residual_features",
    "qr_determinant_features",
    "embed_cd_detqr",
    "extract_cd_detqr",
    "METHOD_ID",
    "DISPLAY_NAME",
]
