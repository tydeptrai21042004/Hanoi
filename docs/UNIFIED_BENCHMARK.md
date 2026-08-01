# Unified proposal–baseline benchmark

## Purpose

The unified benchmark evaluates every proposal and baseline through the same:

- host and watermark loader;
- deterministic attack objects;
- image-quality metrics;
- watermark-recovery metrics;
- runtime and key-size measurements;
- error handling, result schema, aggregation, and resume mechanism.

It does **not** modify any proposal or baseline embedding/extraction equation. The
method modules under `src/qr64_certified/proposals/` and
`src/qr64_certified/baselines/implementations/` remain the numerical source of
truth.

## Ready-to-run commands

Install dependencies and expose the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts"
export JILP_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src;$PWD\scripts"
$env:JILP_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

Run tests and a quick shared benchmark:

```bash
./scripts/run_complete_pipeline.sh
```

Windows:

```powershell
.\scripts\run_complete_pipeline.ps1
```

Run only the unified benchmark:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/quick.json
```

Run all 19 methods on one host and the clean sanity case:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/all_methods_smoke.json
```

Run the publication protocol with resume support:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/publication.json \
  --resume
```

The publication protocol is intentionally large. Every completed
method/host/watermark/seed trial is saved atomically, so an interrupted run can
continue without repeating valid trials.

## Method selectors

The `--methods` option accepts a selector or comma-separated canonical IDs:

| Selector | Included methods |
|---|---|
| `proposals` | Three proposal methods |
| `baselines` | All 16 baselines |
| `paper_comparison` | Three proposals plus five primary blind baselines |
| `primary_blind_baselines` | Five primary blind baselines |
| `strict_blind` | All methods registered as blind |
| `key_assisted` | Key-assisted blind methods |
| `semi_blind` | Semi-blind methods |
| `non_blind` | Non-blind methods |
| `common_4096` | Methods supporting the common 64×64 payload |
| `all` | All 19 methods |

Examples:

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/quick.json \
  --methods proposals \
  --suite publication \
  --host-limit 2
```

```bash
python scripts/run_unified_benchmark.py \
  --protocol configs/benchmark/quick.json \
  --methods dct_qr,dct_dm_qim_chen2001_blind \
  --suite extended \
  --attack-limit 20
```

## Attack suites

The original frozen suites remain unchanged for backward reproducibility.
Additional suites are additive.

| Suite | Attack count | Intended use |
|---|---:|---|
| `sanity` | 6 | Fast installation and API check |
| `common` | 60 | Broad moderate comparison |
| `stress` | 81 | Existing severe suite |
| `real_world` | 11 | Recompression, screen capture, print–scan, tampering |
| `deformation` | 6 | Elastic and lens distortion |
| `structured_loss` | 7 | Random erasing and scan-line loss |
| `publication` | 84 | Balanced paper-oriented suite |
| `extended` | 113 | Largest deterministic robustness suite |

New attack families include:

- repeated JPEG recompression;
- WebP, JPEG2000, chroma subsampling, quantization and dithering;
- Gaussian, normalized-variance Gaussian, speckle, Poisson and impulse noise;
- median, average, Gaussian, motion, bilateral and sharpening filters;
- rotation, translation, resize, crop, shear, affine and perspective changes;
- elastic deformation and radial lens distortion;
- grayscale conversion, channel permutation, hue, white balance and thresholding;
- center/random occlusion, checkerboard loss, random erasing and stripe dropout;
- copy–move tampering;
- screen-capture and print–scan approximations;
- multi-stage combined attacks.

All stochastic attacks use explicit seeds and preserve the original image shape
and RGB `uint8` type.

## Metrics

### Embedding and attacked-image quality

- MSE, RMSE, MAE and maximum absolute error;
- PSNR and SNR;
- SSIM and UIQI;
- correlation coefficient;
- normalized absolute error;
- structural content and image fidelity;
- histogram intersection;
- edge-preservation index;
- reference/test entropy and entropy difference.

For each attack, quality is measured both:

1. between the original host and attacked watermarked image; and
2. between the clean watermarked image and attacked image.

### Watermark recovery

- NC and zero-mean NCC;
- BER and bit accuracy;
- Hamming distance;
- precision, recall and specificity;
- F1 and balanced accuracy;
- false-positive and false-negative rates;
- TP, TN, FP and FN counts.

### System and protocol

- embedding, attack and extraction time;
- clean extraction time;
- serialized key size;
- payload bits and payload bits per host pixel;
- method kind, blindness tier, fidelity tier and comparison group;
- attack category and severity;
- per-row success/error status and exception information.

List metrics from the command line:

```bash
python scripts/list_metrics.py
```

## Output structure

A run creates:

```text
results/unified_benchmark/<protocol_id>/
├── manifest.json
├── run_summary.json
├── trials/
│   └── <method>__<host>__<watermark>__seed<seed>.json
└── aggregate/
    ├── rows.csv
    ├── method_summary.csv
    ├── category_summary.csv
    ├── severity_summary.csv
    └── summary.json
```

Each trial embeds only once, performs clean extraction, then applies every
selected attack to the same watermarked image. Trial files are written through a
temporary file and atomically renamed.

Rebuild aggregation without rerunning methods:

```bash
python scripts/aggregate_unified_benchmark.py \
  results/unified_benchmark/<protocol_id>/trials
```

## Fair interpretation

The aggregator does not silently rank incompatible methods. Every row retains:

- blindness tier;
- original-host requirement;
- cover-dependent-key flag;
- payload size;
- fidelity tier;
- comparison group.

Use these fields to make separate strict-blind, key-assisted, semi-blind and
non-blind tables. The 16×16 DEW payload must not be ranked directly against the
64×64 common-payload group without an explicit capacity qualification.

## Configuration files

- `configs/benchmark/quick.json`: fast shared proposal/baseline run.
- `configs/benchmark/all_methods_smoke.json`: all-method integration check.
- `configs/benchmark/publication.json`: two watermarks, three seeds, all hosts,
  84 attacks and the paper-comparison method group.
- `configs/benchmark/baseline_parameters.json`: explicit baseline parameters.

Proposal configurations continue to come from the existing frozen
`configs/*_after_abc.json` files. Seed 2026 reproduces their stored scheduling.
Other seeds vary only pseudorandom scheduling/masking, not the mathematical
embedding or detector laws.
