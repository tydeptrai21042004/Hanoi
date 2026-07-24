# Unified attack library

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
stress      81 unique attacks
```

The original proposal scripts still use frozen legacy 15-attack and 12-attack
suites so the existing published result files remain reproducible. New baseline
experiments should use `common` or `stress`.

## Run one baseline

```bash
python scripts/run_baseline_attack_suite.py \
  dct_dm_qim_chen2001_blind \
  --suite common
```

The output rows always include the canonical baseline ID, blindness tier,
fidelity tier, attack category, attack severity, NC, BER, PSNR, and timings.
