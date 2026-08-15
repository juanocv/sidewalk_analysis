"""Characterization test for `compute_width`.

`compute_width` is a long heuristic with many interacting gates, so it is pinned
against recorded outputs over a grid of synthetic scenes. The values were
captured before the scan loop was deduplicated and must not drift silently.

When a change to the estimator is *intended*, regenerate the file with:

    python -m sidewalk_ai.tests.test_compute_width_golden --update

and review the diff — every changed number is a changed measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sidewalk_ai.processing.geometry import compute_width

GOLDEN_PATH = Path(__file__).parent / "data" / "compute_width_golden.json"

H, W = 400, 600


# --------------------------------------------------------------------------- #
# Scene grid                                                                  #
# --------------------------------------------------------------------------- #
def _scenes():
    scenes = []

    for top, bot, x0, x1 in [
        (300, 390, 100, 500),
        (250, 340, 60, 540),
        (320, 395, 200, 460),
        (280, 360, 20, 580),
        (352, 362, 100, 500),
        (300, 365, 165, 435),
    ]:
        mask = np.zeros((H, W), bool)
        mask[top:bot, x0:x1] = True
        scenes.append((f"band_{top}_{bot}_{x0}_{x1}", mask))

    for base_half, growth in [(30, 1.6), (60, 0.8), (20, 2.4)]:
        mask = np.zeros((H, W), bool)
        for y in range(240, 390):
            half = int(base_half + (y - 240) * growth)
            mask[y, max(0, 300 - half) : min(W, 300 + half)] = True
        scenes.append((f"trapezoid_{base_half}_{growth}", mask))

    for gap_lo, gap_hi in [(250, 330), (200, 300), (280, 400)]:
        mask = np.zeros((H, W), bool)
        mask[300:390, 80:520] = True
        mask[300:390, gap_lo:gap_hi] = False
        scenes.append((f"gap_{gap_lo}_{gap_hi}", mask))

    # Thin slivers: every row fails the Δu gate, which is what drives the
    # band-retry pass.
    for x0, x1 in [(290, 300), (280, 305), (100, 115)]:
        mask = np.zeros((H, W), bool)
        mask[300:360, x0:x1] = True
        scenes.append((f"sliver_{x0}_{x1}", mask))

    rng = np.random.default_rng(7)
    for i in range(3):
        mask = np.zeros((H, W), bool)
        mask[300:390, 100:500] = True
        mask[300:390, 100:500] &= ~(rng.random((90, 400)) < 0.15)
        scenes.append((f"speckle_{i}", mask))

    scenes.append(("empty", np.zeros((H, W), bool)))
    scenes.append(("full", np.ones((H, W), bool)))
    return scenes


def _depths():
    flat = np.full((H, W), 5.0, dtype="float32")
    holed = flat.copy()
    holed[-20:, :] = np.nan  # the logo strip the pipeline blanks out
    far = np.full((H, W), 12.0, dtype="float32")
    far[:, 300:] = 12.6
    return [
        ("none", None),
        ("flat", flat),
        ("grad", np.tile(np.linspace(12.0, 2.0, H, dtype="float32")[:, None], (1, W))),
        ("holed", holed),
        ("far", far),
    ]


_PARAM_SETS = [
    ("default", {}),
    ("strict_du", {"du_range_px": (50, 220)}),
    ("wide_du", {"du_range_px": (20, 250)}),
    ("fixed_band", {"band_mode": "fixed", "band_frac": (0.5, 1.0)}),
    ("fov60", {"fov_deg": 60}),
    ("pitch_-25", {"pitch_deg": -25}),
    ("no_divergence", {"divergence_pct": None}),
    ("tight_divergence", {"divergence_pct": 0.05}),
    ("min_rows_12", {"min_valid_rows": 12}),
    ("seed7", {"seed": 7}),
]


def _measure():
    """Run the grid and return {key: [width, margin, n_pixels]}."""
    results = {}
    for scene_name, mask in _scenes():
        for depth_name, depth in _depths():
            for param_name, params in _PARAM_SETS:
                kwargs = dict(pitch_deg=-10, fov_deg=90, bottom_ignore_px=0, min_valid_rows=3)
                kwargs.update(params)
                res = compute_width(mask, depth, **kwargs)
                results[f"{scene_name}|{depth_name}|{param_name}"] = [
                    round(float(res.width_m), 10),
                    round(float(res.margin_m), 10),
                    int(res.n_pixels),
                ]
    return results


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_compute_width_matches_recorded_behaviour():
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = _measure()

    assert set(actual) == set(expected), "scene grid drifted from the recorded one"

    drifted = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    assert not drifted, "compute_width output changed for:\n" + "\n".join(
        f"  {k}: recorded={old} now={new}" for k, (old, new) in list(drifted.items())[:20]
    )


def test_the_grid_still_exercises_the_band_retry_path():
    # Guards the guard: if no scene reaches the retry pass, the golden file
    # would stop covering the loop that once raised UnboundLocalError.
    from sidewalk_ai.processing import geometry

    seen = []
    original = geometry._swai_log

    def spy(tag, payload):
        seen.append(tag)
        return original(tag, payload)

    geometry._swai_log = spy
    try:
        # Only the slivers reach the retry; re-running the whole grid here would
        # double the suite's runtime for no extra coverage.
        for name, mask in _scenes():
            if not name.startswith("sliver_"):
                continue
            compute_width(
                mask, None, pitch_deg=-10, fov_deg=90, bottom_ignore_px=0, min_valid_rows=3
            )
    finally:
        geometry._swai_log = original

    assert "band_retry" in seen


@pytest.mark.parametrize("key", ["empty|none|default", "empty|flat|default"])
def test_empty_masks_report_no_measurement(key):
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert expected[key] == [0.0, 0.0, 0]


if __name__ == "__main__":
    if "--update" in sys.argv:
        GOLDEN_PATH.write_text(json.dumps(_measure(), indent=0, sort_keys=True), encoding="utf-8")
        print(f"rewrote {GOLDEN_PATH}")
    else:
        print(__doc__)
