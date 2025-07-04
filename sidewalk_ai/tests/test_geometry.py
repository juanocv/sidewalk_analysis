import numpy as np
from sidewalk_ai.processing.geometry import compute_width

def test_compute_width_plane():
    H, W = 120, 160
    depth = np.full((H, W), 5.0, dtype="float32")

    mask = np.zeros((H, W), bool)
    mask[:, 10:-10] = True                    # 140-px wide band

    res = compute_width(mask, depth, FOV_deg=60, band_frac=(0.0, 1.0))

    # analytical ground-truth for pin-hole camera
    fx = W / (2.0 * np.tan(np.radians(60 / 2)))
    expected = 140 * 5.0 / fx                 # ≈ 5.05 m

    rel_err = abs(res.width_m - expected) / expected
    assert rel_err < 0.08                     # within 8 %