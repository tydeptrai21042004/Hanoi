from __future__ import annotations

import numpy as np

from qr64_certified.common.metrics import image_quality_metrics, watermark_metrics


def test_expanded_image_metrics_are_exact_for_identical_images():
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    values = image_quality_metrics(image, image)
    assert values["mse"] == 0.0
    assert values["rmse"] == 0.0
    assert values["mae"] == 0.0
    assert values["ssim"] == 1.0
    assert values["uiqi"] == 1.0
    assert values["correlation"] == 1.0
    assert values["histogram_intersection"] == 1.0
    assert values["edge_preservation"] == 1.0


def test_watermark_metrics_include_confusion_and_accuracy():
    truth = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    recovered = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    values = watermark_metrics(truth, recovered)
    assert values["tp"] == 1
    assert values["tn"] == 2
    assert values["fp"] == 0
    assert values["fn"] == 1
    assert values["hamming_distance"] == 1
    assert values["ber"] == 0.25
    assert values["bit_accuracy"] == 0.75
