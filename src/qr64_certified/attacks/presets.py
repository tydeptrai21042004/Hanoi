from __future__ import annotations

from .types import AttackConfig


def _a(attack_id: str, group: str, params: dict, category: str, severity: str) -> AttackConfig:
    return AttackConfig(attack_id, group, params, category, severity)


SANITY = (
    _a("none__clean", "none", {}, "clean", "none"),
    _a("jpeg__q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("noise_gaussian__sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("filter_gaussian__radius0p5", "gaussian_blur", {"radius": 0.5}, "filtering", "mild"),
    _a("photometric_brightness__factor0p95", "brightness", {"factor": 0.95}, "photometric", "mild"),
    _a("geometric_resize__factor0p9", "resize", {"factor": 0.9}, "geometric", "mild"),
)

COMPRESSION = (
    _a("jpeg__q95", "jpeg", {"quality": 95}, "compression", "mild"),
    _a("jpeg__q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("jpeg__q80", "jpeg", {"quality": 80}, "compression", "moderate"),
    _a("jpeg__q70", "jpeg", {"quality": 70}, "compression", "moderate"),
    _a("jpeg__q50", "jpeg", {"quality": 50}, "compression", "strong"),
    _a("jpeg__q30", "jpeg", {"quality": 30}, "compression", "strong"),
    _a("webp__q90", "webp", {"quality": 90}, "compression", "mild"),
    _a("webp__q70", "webp", {"quality": 70}, "compression", "moderate"),
    _a("webp__q50", "webp", {"quality": 50}, "compression", "strong"),
    _a("jpeg2000__layer7", "jpeg2000", {"quality_layer": 7.0}, "compression", "moderate"),
    _a("chroma_subsampling__factor2", "chroma_subsampling", {"factor": 2}, "compression", "moderate"),
    _a("chroma_subsampling__factor4", "chroma_subsampling", {"factor": 4}, "compression", "strong"),
    _a("color_quantization__colors64", "color_quantization", {"colors": 64}, "compression", "moderate"),
    _a("bit_depth__bits5", "bit_depth", {"bits": 5}, "compression", "moderate"),
    _a("bit_depth__bits3", "bit_depth", {"bits": 3}, "compression", "strong"),
)

NOISE = (
    _a("noise_gaussian__sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("noise_gaussian__sigma3", "gaussian_noise", {"sigma": 3.0, "seed": 123}, "noise", "moderate"),
    _a("noise_gaussian__sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}, "noise", "strong"),
    _a("noise_gaussian_var__0p001", "gaussian_noise_var", {"variance": 0.001, "seed": 123}, "noise", "mild"),
    _a("noise_gaussian_var__0p003", "gaussian_noise_var", {"variance": 0.003, "seed": 123}, "noise", "strong"),
    _a("noise_salt_pepper__0p001", "salt_pepper", {"amount": 0.001, "seed": 123}, "noise", "mild"),
    _a("noise_salt_pepper__0p005", "salt_pepper", {"amount": 0.005, "seed": 123}, "noise", "moderate"),
    _a("noise_salt_pepper__0p01", "salt_pepper", {"amount": 0.01, "seed": 123}, "noise", "strong"),
    _a("noise_speckle__var0p005", "speckle_noise", {"variance": 0.005, "seed": 123}, "noise", "moderate"),
    _a("noise_poisson__peak255", "poisson_noise", {"peak": 255.0, "seed": 123}, "noise", "mild"),
    _a("noise_poisson__peak64", "poisson_noise", {"peak": 64.0, "seed": 123}, "noise", "strong"),
    _a("pixel_dropout__amount0p01", "pixel_dropout", {"amount": 0.01, "seed": 123}, "noise", "moderate"),
)

FILTERING = (
    _a("filter_median__size3", "median_filter", {"size": 3}, "filtering", "moderate"),
    _a("filter_median__size5", "median_filter", {"size": 5}, "filtering", "strong"),
    _a("filter_average__size3", "average_filter", {"size": 3}, "filtering", "moderate"),
    _a("filter_average__size5", "average_filter", {"size": 5}, "filtering", "strong"),
    _a("filter_gaussian__radius0p5", "gaussian_blur", {"radius": 0.5}, "filtering", "mild"),
    _a("filter_gaussian__radius1", "gaussian_blur", {"radius": 1.0}, "filtering", "moderate"),
    _a("filter_motion__size7_horizontal", "motion_blur", {"size": 7, "angle": "horizontal"}, "filtering", "moderate"),
    _a("filter_motion__size9_vertical", "motion_blur", {"size": 9, "angle": "vertical"}, "filtering", "strong"),
    _a("filter_bilateral__d7", "bilateral_filter", {"diameter": 7, "sigma_color": 50, "sigma_space": 50}, "filtering", "moderate"),
    _a("filter_unsharp__r2_p150", "unsharp_mask", {"radius": 2.0, "percent": 150, "threshold": 3}, "filtering", "moderate"),
    _a("filter_sharpen__factor2", "sharpen", {"factor": 2.0}, "filtering", "moderate"),
)

GEOMETRIC = (
    _a("geometric_rotation__deg1", "rotation", {"degrees": 1.0}, "geometric", "mild"),
    _a("geometric_rotation__deg2", "rotation", {"degrees": 2.0}, "geometric", "moderate"),
    _a("geometric_rotation__deg5", "rotation", {"degrees": 5.0}, "geometric", "strong"),
    _a("geometric_rotation_back__deg1", "rotation_back", {"degrees": 1.0}, "geometric", "mild"),
    _a("geometric_translation__x4_y4", "translation", {"shift_x": 4, "shift_y": 4}, "geometric", "mild"),
    _a("geometric_translation__x8_y8", "translation", {"shift_x": 8, "shift_y": 8}, "geometric", "moderate"),
    _a("geometric_resize__factor0p75", "resize", {"factor": 0.75}, "geometric", "moderate"),
    _a("geometric_resize__factor1p5", "resize", {"factor": 1.5}, "geometric", "moderate"),
    _a("geometric_resample_cycle__0p67_x2", "resample_cycle", {"down_factor": 0.67, "cycles": 2}, "geometric", "strong"),
    _a("geometric_crop_resize__keep0p9", "crop_resize", {"keep": 0.9}, "geometric", "moderate"),
    _a("geometric_crop_resize__keep0p75", "crop_resize", {"keep": 0.75}, "geometric", "strong"),
    _a("geometric_shear__x0p05", "shear", {"shear_x": 0.05}, "geometric", "moderate"),
    _a("geometric_shear__x0p08", "shear", {"shear_x": 0.08}, "geometric", "strong"),
    _a("geometric_affine__r2_s0p98_sh0p02", "affine", {"rotation": 2.0, "scale": 0.98, "shear_x": 0.02}, "geometric", "strong"),
    _a("geometric_perspective__strength0p02", "perspective", {"strength": 0.02, "seed": 123}, "geometric", "moderate"),
    _a("geometric_perspective__strength0p04", "perspective", {"strength": 0.04, "seed": 123}, "geometric", "strong"),
    _a("geometric_row_col_delete__4_4", "row_col_delete", {"rows": 4, "cols": 4, "seed": 123}, "geometric", "moderate"),
    _a("geometric_border_crop_pad__px16", "border_crop_pad", {"pixels": 16}, "geometric", "moderate"),
)

PHOTOMETRIC = (
    _a("photometric_gamma__0p8", "gamma", {"gamma": 0.8}, "photometric", "moderate"),
    _a("photometric_gamma__1p2", "gamma", {"gamma": 1.2}, "photometric", "moderate"),
    _a("photometric_brightness__factor0p8", "brightness", {"factor": 0.8}, "photometric", "strong"),
    _a("photometric_brightness__factor1p2", "brightness", {"factor": 1.2}, "photometric", "strong"),
    _a("photometric_contrast__factor0p8", "contrast", {"factor": 0.8}, "photometric", "strong"),
    _a("photometric_contrast__factor1p2", "contrast", {"factor": 1.2}, "photometric", "strong"),
    _a("photometric_saturation__factor0p5", "saturation", {"factor": 0.5}, "photometric", "moderate"),
    _a("photometric_hue__deg15", "hue_shift", {"degrees": 15.0}, "photometric", "moderate"),
    _a("photometric_white_balance__warm", "white_balance", {"red": 1.08, "green": 1.0, "blue": 0.92}, "photometric", "moderate"),
    _a("photometric_hist_equalization", "hist_equalization", {}, "photometric", "strong"),
    _a("photometric_clahe_like", "clahe_like", {}, "photometric", "strong"),
    _a("photometric_channel_dropout__blue", "channel_dropout", {"channel": 2, "value": 0}, "photometric", "strong"),
    _a("photometric_adaptive_threshold", "adaptive_threshold", {"block_size": 31, "c": 3.0}, "photometric", "extreme"),
)

OCCLUSION = (
    _a("occlusion_random__block64", "occlusion", {"block": 64, "seed": 123}, "occlusion", "mild"),
    _a("occlusion_center__fraction0p1", "occlusion_fraction", {"fraction": 0.10}, "occlusion", "mild"),
    _a("occlusion_center__fraction0p25", "occlusion_fraction", {"fraction": 0.25}, "occlusion", "strong"),
    _a("occlusion_center__fraction0p5", "occlusion_fraction", {"fraction": 0.50}, "occlusion", "extreme"),
    _a("occlusion_checkerboard__tile32", "checkerboard_cutout", {"tile": 32}, "occlusion", "extreme"),
    _a("occlusion_mosaic__factor8", "mosaic", {"factor": 8}, "occlusion", "strong"),
)

COMBINED = (
    _a("combined__jpeg70_gaussian1", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_noise", "params": {"sigma": 1.0, "seed": 123}},
    ]}, "combined", "strong"),
    _a("combined__jpeg70_blur1", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_blur", "params": {"radius": 1.0}},
    ]}, "combined", "strong"),
    _a("combined__rotate2_jpeg70", "combined", {"steps": [
        {"group": "rotation", "params": {"degrees": 2.0}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__resize0p75_jpeg70", "combined", {"steps": [
        {"group": "resize", "params": {"factor": 0.75}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__crop0p9_jpeg70", "combined", {"steps": [
        {"group": "crop_resize", "params": {"keep": 0.9}},
        {"group": "jpeg", "params": {"quality": 70}},
    ]}, "combined", "strong"),
    _a("combined__perspective0p02_jpeg80", "combined", {"steps": [
        {"group": "perspective", "params": {"strength": 0.02, "seed": 123}},
        {"group": "jpeg", "params": {"quality": 80}},
    ]}, "combined", "strong"),
)

def _deduplicate(configs: tuple[AttackConfig, ...]) -> tuple[AttackConfig, ...]:
    seen: set[str] = set()
    output: list[AttackConfig] = []
    for cfg in configs:
        if cfg.attack_id not in seen:
            seen.add(cfg.attack_id)
            output.append(cfg)
    return tuple(output)

# The common suite is intentionally broad but excludes extreme destructive tests.
COMMON = _deduplicate(
    SANITY + COMPRESSION[:10] + NOISE[:9] + FILTERING[:9] + GEOMETRIC[:14] + PHOTOMETRIC[:11] + OCCLUSION[:4]
)
STRESS = _deduplicate(COMPRESSION + NOISE + FILTERING + GEOMETRIC + PHOTOMETRIC + OCCLUSION + COMBINED)

# Frozen suites used by the original proposal result scripts. These are kept
# byte-for-byte equivalent in parameterization so existing reported results can
# be reproduced; new experiments should use get_attack_suite("common") or
# get_attack_suite("stress").
LEGACY_MODERATE = (
    _a("jpeg_q90", "jpeg", {"quality": 90}, "compression", "mild"),
    _a("jpeg_q70", "jpeg", {"quality": 70}, "compression", "moderate"),
    _a("gaussian_noise_sigma1", "gaussian_noise", {"sigma": 1.0, "seed": 123}, "noise", "mild"),
    _a("salt_pepper_0p005", "salt_pepper", {"amount": 0.005, "seed": 123}, "noise", "moderate"),
    _a("median_3x3", "median_filter", {"size": 3}, "filtering", "moderate"),
    _a("gaussian_blur_r1", "gaussian_blur", {"radius": 1.0}, "filtering", "moderate"),
    _a("rotation_back_1deg", "rotation_back", {"degrees": 1.0}, "geometric", "mild"),
    _a("gamma_0p8", "gamma", {"gamma": 0.8}, "photometric", "moderate"),
    _a("gamma_1p2", "gamma", {"gamma": 1.2}, "photometric", "moderate"),
    _a("brightness_0p9", "brightness", {"factor": 0.9}, "photometric", "mild"),
    _a("brightness_1p1", "brightness", {"factor": 1.1}, "photometric", "mild"),
    _a("contrast_0p9", "contrast", {"factor": 0.9}, "photometric", "mild"),
    _a("contrast_1p1", "contrast", {"factor": 1.1}, "photometric", "mild"),
    _a("resize_0p75", "resize", {"factor": 0.75}, "geometric", "moderate"),
    _a("resize_1p5", "resize", {"factor": 1.5}, "geometric", "moderate"),
)

LEGACY_STRESS = (
    _a("jpeg_q50", "jpeg", {"quality": 50}, "compression", "strong"),
    _a("gaussian_noise_sigma5", "gaussian_noise", {"sigma": 5.0, "seed": 123}, "noise", "strong"),
    _a("gaussian_var_0p003", "gaussian_noise_var", {"variance": 0.003, "seed": 123}, "noise", "strong"),
    _a("speckle_0p01", "speckle_noise", {"variance": 0.01, "seed": 123}, "noise", "strong"),
    _a("median_5x5", "median_filter", {"size": 5}, "filtering", "strong"),
    _a("motion_blur_9", "motion_blur", {"size": 9}, "filtering", "strong"),
    _a("hist_equalization", "hist_equalization", {}, "photometric", "strong"),
    _a("rotation_2deg", "rotation", {"degrees": 2.0}, "geometric", "moderate"),
    _a("crop_resize_90", "crop_resize", {"keep": 0.90}, "geometric", "moderate"),
    _a("occlusion_25pct", "occlusion_fraction", {"fraction": 0.25, "position": "center"}, "occlusion", "strong"),
    _a("shear_0p08", "shear", {"shear_x": 0.08}, "geometric", "strong"),
    _a("jpeg70_plus_blur", "combined", {"steps": [
        {"group": "jpeg", "params": {"quality": 70}},
        {"group": "gaussian_blur", "params": {"radius": 1.0}},
    ]}, "combined", "strong"),
)

ATTACK_SUITES = {
    "sanity": SANITY,
    "compression": COMPRESSION,
    "noise": NOISE,
    "filtering": FILTERING,
    "geometric": GEOMETRIC,
    "photometric": PHOTOMETRIC,
    "occlusion": OCCLUSION,
    "combined": COMBINED,
    "common": COMMON,
    "stress": STRESS,
}


def get_attack_suite(name: str) -> tuple[AttackConfig, ...]:
    key = str(name).strip().lower()
    if key not in ATTACK_SUITES:
        raise KeyError(f"Unknown attack suite '{name}'. Valid: {', '.join(ATTACK_SUITES)}")
    return ATTACK_SUITES[key]


def list_attack_suites() -> dict[str, int]:
    return {name: len(configs) for name, configs in ATTACK_SUITES.items()}

def moderate_attacks() -> list[AttackConfig]:
    """Frozen original proposal suite retained for result reproducibility."""
    return list(LEGACY_MODERATE)

def stress_attacks() -> list[AttackConfig]:
    """Frozen original stress suite retained for result reproducibility."""
    return list(LEGACY_STRESS)

__all__ = [
    "ATTACK_SUITES", "get_attack_suite", "list_attack_suites",
    "moderate_attacks", "stress_attacks",
]
