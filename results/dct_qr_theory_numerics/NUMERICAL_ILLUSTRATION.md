# DCT-QR theory numerical illustration

Host: `lenna.bmp`; watermark: `wm.png`.

## Theory diagnostics

- Derived opponent coefficient: `0.068666667`.
- Norm preservation error: `0.000e+00`.
- Neumann lift: `1012.399928`; max spectral/lift ratio: `0.986064` (< 1 is the sufficient condition).
- beta <= sigma_min for every block: `True`; max beta/sigma_min: `0.171098`.
- Adaptive QIM step range: `5.1356` to `8.4480`; median `5.5923`.

## Embedding and common-20 robustness

- PSNR: `53.6035 dB`; SSIM: `0.998495`; clean NC: `1.000000`.
- Global-coset projection-energy ratio: `1.000000` (<= 1 by construction).
- Common-20 mean NC: `0.975482`; q10 NC: `0.953995`; minimum NC: `0.731832`.

## Grouped robustness

| Group | n | Mean NC | Min NC | Mean BER |
|---|---:|---:|---:|---:|
| compression_quantization | 4 | 0.992628 | 0.983402 | 0.007385 |
| noise | 4 | 0.966648 | 0.933515 | 0.033142 |
| filtering_enhancement | 4 | 0.988886 | 0.985669 | 0.011169 |
| geometric_resampling | 4 | 0.931864 | 0.731832 | 0.070190 |
| photometric_point_processing | 4 | 0.997386 | 0.990273 | 0.002625 |
