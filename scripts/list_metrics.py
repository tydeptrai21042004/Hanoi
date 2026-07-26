#!/usr/bin/env python3
from __future__ import annotations

IMAGE_METRICS = [
    "MSE", "RMSE", "MAE", "maximum absolute error", "PSNR", "SNR", "SSIM",
    "UIQI", "correlation coefficient", "normalized absolute error",
    "structural content", "image fidelity", "histogram intersection",
    "edge preservation index", "entropy and entropy difference",
]
WATERMARK_METRICS = [
    "NC", "zero-mean NCC", "BER", "bit accuracy", "Hamming distance",
    "precision", "recall", "specificity", "F1", "balanced accuracy",
    "false-positive rate", "false-negative rate", "TP/TN/FP/FN",
]
SYSTEM_METRICS = [
    "embedding time", "attack time", "extraction time", "serialized key size",
    "payload bits", "payload bits per host pixel", "success/error rate",
]

print("Image quality metrics:")
for value in IMAGE_METRICS:
    print(f"  - {value}")
print("\nWatermark recovery metrics:")
for value in WATERMARK_METRICS:
    print(f"  - {value}")
print("\nSystem and protocol metrics:")
for value in SYSTEM_METRICS:
    print(f"  - {value}")
