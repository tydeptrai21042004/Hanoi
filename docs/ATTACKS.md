# Unified attack library

> **Repository update — 4 September 2026.** The current public registry contains **nine proposals**. The six-method QR family consists of `dct_qr`, `dct_qr_direct_r`, `dct_qr_r11_qim` and the non-DCT counterparts `spatial_qr`, `spatial_qr_direct_r`, `spatial_qr_r11_qim`. The additional theory-grounded `dct_qr_theory`, independent `dct_schur_rescue`, and `spatial_cd_detqr` proposals remain available. Historical three-method/five-method results below are preserved as historical evidence and do not describe the current registry size.

All attacks are exposed from:

```python
from qr64_certified.attacks import (
    AttackConfig,
    apply_attack,
    get_attack_suite,
    list_attack_suites,
)
```

The library contains 49 attack operators and 81 deterministic stress presets.
Every attack preserves RGB shape and returns `uint8` data.

## Categories

- compression: JPEG, JPEG2000, WebP, chroma subsampling, quantization;
- noise: Gaussian, normalized-variance Gaussian, salt-and-pepper, speckle,
  Poisson, and pixel dropout;
- filtering: median, average, Gaussian, motion, bilateral, sharpen, unsharp;
- geometric: rotation, translation, resize, resampling cycles, crop-resize,
  shear, affine, perspective, line deletion, and border loss;
- photometric: gamma, brightness, contrast, saturation, hue, white balance,
  histogram changes, channel loss, and adaptive thresholding;
- occlusion: random blocks, area masks, checkerboard masks, and mosaic;
- combined: compression plus noise/filtering and geometry plus compression.

## Suites

```text
sanity       6 attacks
compression 15 attacks
noise       12 attacks
filtering   11 attacks
geometric   18 attacks
photometric 13 attacks
occlusion    6 attacks
combined     6 attacks
common      60 unique attacks
common20    20 curated paper-facing attacks (5 groups x 4)
stress      81 unique attacks
real_world  11 attacks
deformation 6 attacks
structured_loss 7 attacks
publication 84 attacks
extended    113 attacks
```

The original proposal scripts still use frozen legacy 15-attack and 12-attack
suites so the existing published result files remain reproducible. New baseline
experiments may use `common` or `stress`; the main `dct_qr_theory` manuscript-facing comparison uses `common20` so the headline result stays balanced and interpretable.

## Run one baseline

```bash
python scripts/run_baseline_attack_suite.py \
  dct_dm_qim_chen2001_blind \
  --suite common
```

The output rows always include the canonical baseline ID, blindness tier,
fidelity tier, attack category, attack severity, NC, BER, PSNR, and timings.
