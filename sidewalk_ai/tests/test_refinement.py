import cv2
import numpy as np

from sidewalk_ai.processing.refinement import refine_sidewalk_mask


def test_refine_returns_connected_non_empty_mask():
    raw = np.zeros((80, 120), np.uint8)
    raw[30:70, 25:45] = 1
    raw[30:70, 70:90] = 1

    clean, lines = refine_sidewalk_mask(raw, min_keep_area_px=100)

    num_labels, _ = cv2.connectedComponents(clean.astype("uint8"), connectivity=8)

    assert clean.any()
    assert num_labels == 2
    assert len(lines) == 2
