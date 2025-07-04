import numpy as np
from sidewalk_ai.processing.refinement import refine_sidewalk_mask

def test_refine_improves_connectivity():
    raw = np.zeros((40, 60), np.uint8)
    raw[10:30, 15:20] = 1
    raw[10:30, 40:45] = 1          #  gap → should be bridged

    clean = refine_sidewalk_mask(raw, min_keep_area_px=200)
    # after refinement, both blobs must be connected
    assert np.count_nonzero(clean) > np.count_nonzero(raw)