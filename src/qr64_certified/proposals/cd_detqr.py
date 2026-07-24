from __future__ import annotations

from dataclasses import asdict as _asdict, dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image

_H4 = np.array(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
    dtype=np.float64,
)
_CHANNEL_PAIRS = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)


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


def _blocks8(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    if h % 8 or w % 8:
        raise ValueError("Host dimensions must be divisible by 8")
    return channel.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)


def _cell_means4(blocks: np.ndarray) -> np.ndarray:
    return blocks.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4))


def qr_determinant_features(image: np.ndarray, max_patterns: int = 10) -> np.ndarray:
    """Compute signed 2x2 determinants through a real QR factorization.

    For each 8x8 spatial block, RGB channel and Hadamard partition, construct

        A = [[mean(S+), 1],
             [mean(S-), 1]].

    The signed determinant is evaluated from A=QR as
    det(Q)*prod(diag(R)). No DCT, DWT, FFT or another transform domain is used.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    if not 1 <= max_patterns <= len(_PATTERNS):
        raise ValueError(f"max_patterns must be in [1, {len(_PATTERNS)}]")

    patterns = _PATTERNS[:max_patterns]
    features: list[np.ndarray] = []
    for channel in range(3):
        cells = _cell_means4(_blocks8(image[..., channel]))
        plus = np.stack([cells[:, p > 0].mean(axis=1) for p in patterns], axis=1)
        minus = np.stack([cells[:, p < 0].mean(axis=1) for p in patterns], axis=1)
        matrices = np.empty((*plus.shape, 2, 2), dtype=np.float64)
        matrices[..., 0, 0] = plus
        matrices[..., 0, 1] = 1.0
        matrices[..., 1, 0] = minus
        matrices[..., 1, 1] = 1.0
        q, r = np.linalg.qr(matrices)
        det_q = q[..., 0, 0] * q[..., 1, 1] - q[..., 0, 1] * q[..., 1, 0]
        determinant = det_q * r[..., 0, 0] * r[..., 1, 1]
        features.append(determinant)
    return np.stack(features, axis=1)


def _difference_tensor(features: np.ndarray, max_patterns: int) -> np.ndarray:
    return np.stack(
        [
            features[:, c1, :max_patterns] - features[:, c2, :max_patterns]
            for c1, c2 in _CHANNEL_PAIRS
        ],
        axis=1,
    )


@dataclass(frozen=True)
class CDDetQRConfig:
    # Payload margin is deliberately small. Robustness is retained mainly by
    # masked, sign-matched carrier selection instead of large pixel updates.
    target_margin: float = 2.0
    boundary_penalty: float = 0.01
    max_patterns: int = 10
    determinant_floor: float = 1e-6
    embedded_determinant_margin: float = 0.5

    # A small, genuinely embedded QR-determinant pilot prevents the selected
    # carrier map from decoding the watermark from the untouched host image.
    mask_seed: int = 918273
    pilot_seed: int = 202607
    pilot_count: int = 71
    pilot_margin: float = 8.0

    # The determinant pilot defines a statistical synchronization criterion.
    # A geometric correction is accepted only when it raises the normalized
    # pilot score by a nontrivial amount relative to the identity hypothesis.
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

    def validate(self) -> None:
        if self.target_margin <= 0:
            raise ValueError("target_margin must be positive")
        if self.boundary_penalty < 0:
            raise ValueError("boundary_penalty must be nonnegative")
        if not 1 <= self.max_patterns <= len(_PATTERNS):
            raise ValueError(f"max_patterns must be in [1, {len(_PATTERNS)}]")
        if self.determinant_floor <= 0:
            raise ValueError("determinant_floor must be positive")
        if self.embedded_determinant_margin <= 0:
            raise ValueError("embedded_determinant_margin must be positive")
        if self.pilot_count <= 0 or self.pilot_count % 2 == 0:
            raise ValueError("pilot_count must be a positive odd integer")
        if self.pilot_margin <= 0:
            raise ValueError("pilot_margin must be positive")
        if self.sync_acceptance_gain < 0:
            raise ValueError("sync_acceptance_gain must be nonnegative")
        if not -1 <= self.sync_min_pilot_score <= 1:
            raise ValueError("sync_min_pilot_score must be in [-1,1]")
        if any(abs(float(x)) > 15 for x in self.rotation_grid):
            raise ValueError("rotation_grid is restricted to small affine perturbations")
        if any(abs(float(x)) >= 0.5 for x in self.shear_grid):
            raise ValueError("shear_grid must contain moderate affine shears")


def _rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    """Apply a same-size inverse-geometry hypothesis for pilot evaluation."""
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
    """Apply a horizontal affine-shear hypothesis without changing image size."""
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
    """Blind single-spatial-domain channel-difference QR watermarking.

    Version 2 reduces distortion by selecting, for each masked payload bit, a
    nonsingular QR-determinant carrier whose existing sign already supports the
    bit whenever possible. A small odd QR-determinant pilot is then physically
    embedded in separate spatial partitions. At extraction, the pilot decides
    whether the recovered masked payload is unmasked. Thus the extractor uses
    only the questioned image and key; neither the host nor original watermark
    is required.
    """

    def __init__(self, config: CDDetQRConfig | None = None):
        self.config = config or CDDetQRConfig()
        self.config.validate()
        self.key: dict[str, Any] | None = None
        self._report: dict[str, Any] | None = None

    def _valid_component_pairs(self, features: np.ndarray) -> np.ndarray:
        valid: list[np.ndarray] = []
        for c1, c2 in _CHANNEL_PAIRS:
            valid.append(
                (
                    np.abs(features[:, c1, : self.config.max_patterns])
                    > self.config.determinant_floor
                )
                & (
                    np.abs(features[:, c2, : self.config.max_patterns])
                    > self.config.determinant_floor
                )
            )
        return np.stack(valid, axis=1)

    def _select_payload_carriers(
        self, features: np.ndarray, coded_bits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select low-distortion nonsingular carriers for masked bits."""
        differences = _difference_tensor(features, self.config.max_patterns)
        valid = self._valid_component_pairs(features)
        desired = np.where(coded_bits.astype(bool), 1.0, -1.0)
        signed_margin = desired[:, None, None] * differences
        boundary_cost = (
            self.config.boundary_penalty
            * _BOUNDARIES[: self.config.max_patterns][None, None, :]
        )

        # Prefer a carrier whose untouched sign already represents the coded
        # bit. Among those, maximize signed margin while mildly penalizing high
        # spatial boundary patterns.
        preferred_score = np.where(
            valid & (signed_margin > 0.0),
            signed_margin - boundary_cost,
            -np.inf,
        )
        score_flat = preferred_score.reshape(len(coded_bits), -1)
        flat = np.argmax(score_flat, axis=1)
        best = score_flat[np.arange(len(coded_bits)), flat]

        # If no sign-matched nonsingular carrier exists, first use the best
        # valid carrier. For a completely singular/flat block, permit a
        # deterministic activation candidate; _safe_amplitude then moves both
        # component determinants away from zero in the same closed-form update.
        # This removes host-specific embedding failures while preserving the
        # post-embedding condition det(A) != 0.
        missing = ~np.isfinite(best)
        if np.any(missing):
            fallback_score = np.where(
                valid[missing],
                signed_margin[missing] - boundary_cost,
                -np.inf,
            )
            fallback_flat = fallback_score.reshape(np.count_nonzero(missing), -1)
            fallback_choice = np.argmax(fallback_flat, axis=1)
            fallback_best = fallback_flat[
                np.arange(np.count_nonzero(missing)), fallback_choice
            ]
            still_missing_local = ~np.isfinite(fallback_best)
            if np.any(still_missing_local):
                unrestricted = (
                    signed_margin[missing][still_missing_local] - boundary_cost
                ).reshape(np.count_nonzero(still_missing_local), -1)
                fallback_choice[still_missing_local] = np.argmax(unrestricted, axis=1)
            flat[missing] = fallback_choice

        return flat // self.config.max_patterns, flat % self.config.max_patterns

    def _safe_amplitude(
        self,
        features: np.ndarray,
        block: int,
        pair_index: int,
        pattern_index: int,
        desired_sign: int,
        margin: float,
    ) -> int:
        c1, c2 = _CHANNEL_PAIRS[int(pair_index)]
        d1 = float(features[block, c1, pattern_index])
        d2 = float(features[block, c2, pattern_index])
        difference = d1 - d2
        base = max(
            0,
            int(np.ceil((margin - desired_sign * difference) / 4.0 - 1e-12)),
        )
        # Each unit changes the two component determinants by +/-2 and their
        # difference by 4. Search a few neighboring integer amplitudes so both
        # post-embedding QR matrices stay safely nonsingular.
        for amplitude in range(base, base + 6):
            candidate_d1 = d1 + 2.0 * desired_sign * amplitude
            candidate_d2 = d2 - 2.0 * desired_sign * amplitude
            if (
                abs(candidate_d1) >= self.config.embedded_determinant_margin
                and abs(candidate_d2) >= self.config.embedded_determinant_margin
            ):
                return amplitude
        raise RuntimeError("No determinant-safe integer embedding amplitude was found")

    @staticmethod
    def _apply_carrier(
        output: np.ndarray,
        image_width: int,
        block: int,
        pair_index: int,
        pattern_index: int,
        desired_sign: int,
        amplitude: int,
    ) -> int:
        if amplitude <= 0:
            return 0
        width_blocks = image_width // 8
        y0 = (int(block) // width_blocks) * 8
        x0 = (int(block) % width_blocks) * 8
        c1, c2 = _CHANNEL_PAIRS[int(pair_index)]
        pattern8 = np.repeat(
            np.repeat(_PATTERNS[int(pattern_index)], 2, axis=0), 2, axis=1
        ).astype(np.int16)
        clipped = 0
        for channel, update_sign in ((int(c1), desired_sign), (int(c2), -desired_sign)):
            before = output[y0 : y0 + 8, x0 : x0 + 8, channel]
            proposed = before + update_sign * int(amplitude) * pattern8
            after = np.clip(proposed, 0, 255)
            clipped += int(np.count_nonzero(proposed != after))
            output[y0 : y0 + 8, x0 : x0 + 8, channel] = after
        return clipped

    def _select_pilot_carriers(
        self,
        host_features: np.ndarray,
        payload_pattern: np.ndarray,
        n_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.pilot_count > n_blocks:
            raise ValueError("pilot_count cannot exceed the number of 8x8 blocks")

        differences = _difference_tensor(host_features, self.config.max_patterns)
        valid = self._valid_component_pairs(host_features)
        permutation = np.random.default_rng(self.config.pilot_seed).permutation(n_blocks)
        selected_blocks: list[int] = []
        selected_pairs: list[int] = []
        selected_patterns: list[int] = []

        for block in permutation:
            candidate_valid = valid[block].copy()
            # An orthogonal pattern in the same block prevents the pilot from
            # overwriting the payload carrier in the no-clipping case.
            candidate_valid[:, int(payload_pattern[block])] = False
            negative = candidate_valid & (differences[block] < 0.0)
            if not np.any(negative):
                continue
            score = np.where(
                negative,
                differences[block]
                - self.config.boundary_penalty
                * _BOUNDARIES[: self.config.max_patterns][None, :],
                -np.inf,
            )
            flat = int(np.argmax(score.reshape(-1)))
            selected_blocks.append(int(block))
            selected_pairs.append(flat // self.config.max_patterns)
            selected_patterns.append(flat % self.config.max_patterns)
            if len(selected_blocks) == self.config.pilot_count:
                break

        if len(selected_blocks) != self.config.pilot_count:
            raise RuntimeError(
                "Not enough nonsingular negative QR-determinant carriers were "
                "available for the blind pilot"
            )

        return (
            np.asarray(selected_blocks, dtype=np.int64),
            np.asarray(selected_pairs, dtype=np.int64),
            np.asarray(selected_patterns, dtype=np.int64),
        )

    @staticmethod
    def _selected_component_determinants(
        features: np.ndarray,
        blocks: np.ndarray,
        pairs: np.ndarray,
        patterns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c1 = _CHANNEL_PAIRS[pairs, 0]
        c2 = _CHANNEL_PAIRS[pairs, 1]
        d1 = features[blocks, c1, patterns]
        d2 = features[blocks, c2, patterns]
        return d1, d2, d1 - d2

    def _pilot_score(self, image: np.ndarray, key: dict[str, Any]) -> float:
        """Normalized signed determinant evidence of the synchronization pilot.

        For pilot differences x_i, the score is

            S = mean(tanh(x_i / median(|x|))).

        Normalization makes the statistic less sensitive to global contrast, and
        the odd saturating function preserves the sign information required by
        the positive pilot hypothesis.
        """
        features = qr_determinant_features(image, int(key["max_patterns"]))
        blocks = np.asarray(key["pilot_blocks"], dtype=np.int64)
        pairs = np.asarray(key["pilot_pair"], dtype=np.int64)
        patterns = np.asarray(key["pilot_pattern"], dtype=np.int64)
        _, _, difference = self._selected_component_determinants(
            features, blocks, pairs, patterns
        )
        scale = float(np.median(np.abs(difference))) + 1e-8
        return float(np.mean(np.tanh(difference / scale)))

    def _align_from_pilots(
        self, image: np.ndarray, key: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select an affine hypothesis by a pilot-only likelihood surrogate.

        The identity image is the null hypothesis.  A rotated or sheared image is
        accepted only when its determinant-pilot score exceeds the identity score
        by ``sync_acceptance_gain``.  Thus synchronization does not use the host or
        the original watermark.
        """
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
        rotation_grid = tuple(
            float(x) for x in key.get("rotation_grid", self.config.rotation_grid)
        )
        shear_grid = tuple(
            float(x) for x in key.get("shear_grid", self.config.shear_grid)
        )
        candidates.extend(
            (f"rotation_{angle:+.1f}", _rotate_image(image, angle))
            for angle in rotation_grid
            if abs(angle) > 1e-12
        )
        candidates.extend(
            (f"shear_{shear:+.2f}", _shear_image(image, shear))
            for shear in shear_grid
            if abs(shear) > 1e-12
        )

        best_name = "identity"
        best_image = image
        best_score = identity_score
        for name, candidate in candidates:
            score = self._pilot_score(candidate, key)
            if score > best_score:
                best_name = name
                best_image = candidate
                best_score = score

        threshold = float(
            key.get("sync_acceptance_gain", self.config.sync_acceptance_gain)
        )
        gain = float(best_score - identity_score)
        minimum_score = float(
            key.get("sync_min_pilot_score", self.config.sync_min_pilot_score)
        )
        accepted = bool(
            best_name != "identity"
            and gain > threshold
            and best_score >= minimum_score
        )
        return (best_image if accepted else image), {
            "sync_accepted": accepted,
            "sync_candidate": best_name if accepted else "identity",
            "proposed_candidate": best_name,
            "identity_pilot_score": float(identity_score),
            "best_pilot_score": float(best_score),
            "pilot_score_gain": gain,
            "acceptance_threshold": threshold,
            "minimum_accepted_pilot_score": minimum_score,
        }

    def embed(self, host: np.ndarray, watermark: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        host = np.asarray(host, dtype=np.uint8)
        if host.ndim != 3 or host.shape[2] != 3:
            raise ValueError("host must be HxWx3 RGB")
        h, w, _ = host.shape
        if h % 8 or w % 8:
            raise ValueError("host dimensions must be divisible by 8")

        watermark_array = np.asarray(watermark, dtype=np.uint8)
        bits = (watermark_array >= 127).astype(np.uint8).ravel()
        n_blocks = (h // 8) * (w // 8)
        if bits.size != n_blocks:
            raise ValueError(f"watermark must contain exactly {n_blocks} bits")

        mask = np.random.default_rng(self.config.mask_seed).integers(
            0, 2, size=bits.size, dtype=np.uint8
        )
        coded_bits = bits ^ mask
        host_features = qr_determinant_features(host, self.config.max_patterns)
        payload_pair, payload_pattern = self._select_payload_carriers(
            host_features, coded_bits
        )
        blocks = np.arange(n_blocks, dtype=np.int64)
        original_payload_d1, original_payload_d2, original_payload_difference = (
            self._selected_component_determinants(
                host_features, blocks, payload_pair, payload_pattern
            )
        )
        activated_singular_payload_blocks = int(
            np.count_nonzero(
                (np.abs(original_payload_d1) <= self.config.determinant_floor)
                | (np.abs(original_payload_d2) <= self.config.determinant_floor)
            )
        )

        output = host.astype(np.int16).copy()
        payload_amplitudes = np.zeros(n_blocks, dtype=np.int64)
        clipped_values = 0
        desired = np.where(coded_bits.astype(bool), 1, -1)
        for block in blocks:
            amplitude = self._safe_amplitude(
                host_features,
                int(block),
                int(payload_pair[block]),
                int(payload_pattern[block]),
                int(desired[block]),
                self.config.target_margin,
            )
            payload_amplitudes[block] = amplitude
            clipped_values += self._apply_carrier(
                output,
                w,
                int(block),
                int(payload_pair[block]),
                int(payload_pattern[block]),
                int(desired[block]),
                amplitude,
            )

        payload_image = output.astype(np.uint8)
        post_payload_features = qr_determinant_features(
            payload_image, self.config.max_patterns
        )
        pilot_blocks, pilot_pairs, pilot_patterns = self._select_pilot_carriers(
            host_features, payload_pattern, n_blocks
        )
        pilot_amplitudes = np.zeros(self.config.pilot_count, dtype=np.int64)
        for i, (block, pair_index, pattern_index) in enumerate(
            zip(pilot_blocks, pilot_pairs, pilot_patterns, strict=True)
        ):
            amplitude = self._safe_amplitude(
                post_payload_features,
                int(block),
                int(pair_index),
                int(pattern_index),
                1,
                self.config.pilot_margin,
            )
            pilot_amplitudes[i] = amplitude
            clipped_values += self._apply_carrier(
                output,
                w,
                int(block),
                int(pair_index),
                int(pattern_index),
                1,
                amplitude,
            )

        watermarked = output.astype(np.uint8)
        embedded_features = qr_determinant_features(
            watermarked, self.config.max_patterns
        )
        payload_d1, payload_d2, payload_difference = self._selected_component_determinants(
            embedded_features, blocks, payload_pair, payload_pattern
        )
        pilot_d1, pilot_d2, pilot_difference = self._selected_component_determinants(
            embedded_features, pilot_blocks, pilot_pairs, pilot_patterns
        )
        clean_coded_bits = (payload_difference > 0.0).astype(np.uint8)
        clean_pilot_votes = int(np.count_nonzero(pilot_difference > 0.0))
        clean_pilot = clean_pilot_votes > self.config.pilot_count // 2
        clean_bits = clean_coded_bits ^ mask if clean_pilot else clean_coded_bits

        # The untouched host must not directly decode the watermark through the
        # public extraction path. Its pilot carriers were deliberately selected
        # with negative signs, so the mask is not removed.
        original_raw = (original_payload_difference > 0.0).astype(np.uint8)
        _, _, original_pilot_difference = self._selected_component_determinants(
            host_features, pilot_blocks, pilot_pairs, pilot_patterns
        )
        original_pilot_votes = int(np.count_nonzero(original_pilot_difference > 0.0))
        original_pilot = original_pilot_votes > self.config.pilot_count // 2
        original_output_bits = original_raw ^ mask if original_pilot else original_raw

        all_component_determinants = np.concatenate(
            [np.abs(payload_d1), np.abs(payload_d2), np.abs(pilot_d1), np.abs(pilot_d2)]
        )
        self.key = {
            "method": "blind_cd_detqr_v2_masked_qr_pilot",
            "method_class": "key_assisted_actual_blind_embedding",
            "domain": "single_spatial_domain",
            "image_shape": list(host.shape),
            "watermark_shape": list(watermark_array.shape),
            "target_margin": self.config.target_margin,
            "boundary_penalty": self.config.boundary_penalty,
            "max_patterns": self.config.max_patterns,
            "determinant_floor": self.config.determinant_floor,
            "embedded_determinant_margin": self.config.embedded_determinant_margin,
            "mask_seed": int(self.config.mask_seed),
            "pilot_seed": int(self.config.pilot_seed),
            "pilot_margin": self.config.pilot_margin,
            "affine_sync_enabled": bool(self.config.affine_sync_enabled),
            "sync_acceptance_gain": float(self.config.sync_acceptance_gain),
            "sync_min_pilot_score": float(self.config.sync_min_pilot_score),
            "rotation_grid": [float(x) for x in self.config.rotation_grid],
            "shear_grid": [float(x) for x in self.config.shear_grid],
            "synchronization_principle": (
                "accept an affine hypothesis only when determinant-pilot evidence improves over identity"
            ),
            "pair_for_bit": payload_pair.astype(np.uint8).tolist(),
            "pattern_for_bit": payload_pattern.astype(np.uint8).tolist(),
            "pilot_blocks": pilot_blocks.astype(np.uint16).tolist(),
            "pilot_pair": pilot_pairs.astype(np.uint8).tolist(),
            "pilot_pattern": pilot_patterns.astype(np.uint8).tolist(),
        }
        self._report = {
            "clean_bit_errors": int(np.count_nonzero(clean_bits != bits)),
            "clean_pilot_votes": clean_pilot_votes,
            "clean_pilot_detected": bool(clean_pilot),
            "original_host_pilot_votes": original_pilot_votes,
            "original_host_pilot_detected": bool(original_pilot),
            "original_host_bit_agreement": float(np.mean(original_output_bits == bits)),
            "original_host_hamming_errors": int(
                np.count_nonzero(original_output_bits != bits)
            ),
            "all_selected_component_determinants_nonzero": bool(
                np.all(all_component_determinants > self.config.determinant_floor)
            ),
            "minimum_abs_component_determinant": float(
                np.min(all_component_determinants)
            ),
            "minimum_abs_payload_difference": float(
                np.min(np.abs(payload_difference))
            ),
            "minimum_abs_pilot_difference": float(np.min(np.abs(pilot_difference))),
            "mean_payload_amplitude": float(np.mean(payload_amplitudes)),
            "median_payload_amplitude": float(np.median(payload_amplitudes)),
            "maximum_payload_amplitude": int(np.max(payload_amplitudes)),
            "zero_payload_amplitude_fraction": float(
                np.mean(payload_amplitudes == 0)
            ),
            "mean_pilot_amplitude": float(np.mean(pilot_amplitudes)),
            "maximum_pilot_amplitude": int(np.max(pilot_amplitudes)),
            "pilot_count": int(self.config.pilot_count),
            "activated_singular_payload_blocks": activated_singular_payload_blocks,
            "clipped_channel_values": int(clipped_values),
        }
        return watermarked, self.key

    def extract_with_metadata(
        self, questioned_image: np.ndarray, key: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        key = key or self.key
        if key is None:
            raise ValueError("embedding key is required")
        image = np.asarray(questioned_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("questioned_image must be HxWx3 RGB")
        expected_shape = tuple(int(v) for v in key["image_shape"])
        if image.shape != expected_shape:
            raise ValueError(
                f"questioned_image shape {image.shape} does not match key shape {expected_shape}"
            )

        h, w, _ = image.shape
        n_blocks = (h // 8) * (w // 8)
        blocks = np.arange(n_blocks, dtype=np.int64)
        payload_pair = np.asarray(key["pair_for_bit"], dtype=np.int64)
        payload_pattern = np.asarray(key["pattern_for_bit"], dtype=np.int64)
        pilot_blocks = np.asarray(key["pilot_blocks"], dtype=np.int64)
        pilot_pairs = np.asarray(key["pilot_pair"], dtype=np.int64)
        pilot_patterns = np.asarray(key["pilot_pattern"], dtype=np.int64)
        if payload_pair.size != n_blocks or payload_pattern.size != n_blocks:
            raise ValueError("payload carrier schedule does not match the image block count")

        aligned_image, sync_metadata = self._align_from_pilots(image, key)
        features = qr_determinant_features(aligned_image, int(key["max_patterns"]))
        payload_d1, payload_d2, payload_difference = self._selected_component_determinants(
            features, blocks, payload_pair, payload_pattern
        )
        pilot_d1, pilot_d2, pilot_difference = self._selected_component_determinants(
            features, pilot_blocks, pilot_pairs, pilot_patterns
        )
        raw_coded_bits = (payload_difference > 0.0).astype(np.uint8)
        pilot_votes = int(np.count_nonzero(pilot_difference > 0.0))
        pilot_detected = pilot_votes > pilot_blocks.size // 2
        if pilot_detected:
            mask = np.random.default_rng(int(key["mask_seed"])).integers(
                0, 2, size=n_blocks, dtype=np.uint8
            )
            bits = raw_coded_bits ^ mask
            inference_path = "qr_pilot_majority_then_unmask"
        else:
            bits = raw_coded_bits
            inference_path = "qr_pilot_absent_keep_masked_payload"

        all_component_determinants = np.concatenate(
            [np.abs(payload_d1), np.abs(payload_d2), np.abs(pilot_d1), np.abs(pilot_d2)]
        )
        determinant_floor = float(key.get("determinant_floor", self.config.determinant_floor))
        metadata = {
            "pilot_detected": bool(pilot_detected),
            "pilot_votes": pilot_votes,
            "pilot_count": int(pilot_blocks.size),
            "det_nonzero": bool(
                np.all(all_component_determinants > determinant_floor)
            ),
            "min_abs_det": float(np.min(all_component_determinants)),
            "minimum_abs_payload_difference": float(
                np.min(np.abs(payload_difference))
            ),
            "minimum_abs_pilot_difference": float(np.min(np.abs(pilot_difference))),
            "inference_path": inference_path,
            "sync": sync_metadata,
        }
        recovered = (
            bits.reshape(tuple(int(v) for v in key["watermark_shape"])) * 255
        ).astype(np.uint8)
        return recovered, metadata

    def extract(
        self, questioned_image: np.ndarray, key: dict[str, Any] | None = None
    ) -> np.ndarray:
        recovered, _ = self.extract_with_metadata(questioned_image, key)
        return recovered

    def embedding_report(self) -> dict[str, Any]:
        if self._report is None:
            raise ValueError("embed must be called first")
        return dict(self._report)


# ---------------------------------------------------------------------------
# Unified proposal API adapters
# ---------------------------------------------------------------------------

METHOD_ID = "spatial_cd_detqr"
DISPLAY_NAME = "Blind CD-DetQR v2 (single spatial QR determinant)"


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
        return "qr_spatial_determinant_masked_pilot"


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
    if return_metadata:
        metadata = {
            "method_id": METHOD_ID,
            "certificate_mode": "qr_spatial_determinant_masked_pilot",
            "direct_schur_carrier": False,
            "fully_blind": True,
            "domain": "single_spatial_domain",
            "configuration": key.config,
            "det_nonzero": report["all_selected_component_determinants_nonzero"],
            "min_abs_det": report["minimum_abs_component_determinant"],
            "clean_pilot_detected": report["clean_pilot_detected"],
        }
        return watermarked, key, metadata
    return watermarked, key


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
        "certificate_mode": "qr_spatial_determinant_masked_pilot",
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
    }
    return (recovered, metadata) if return_metadata else recovered


__all__ = [
    "BlindCDDetQR",
    "CDDetQRConfig",
    "CDDetQRKey",
    "qr_determinant_features",
    "embed_cd_detqr",
    "extract_cd_detqr",
    "METHOD_ID",
    "DISPLAY_NAME",
]
