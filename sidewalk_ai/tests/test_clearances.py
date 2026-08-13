"""Clearance geometry: span is read from the mask, not extrapolated from lines."""

import numpy as np

from sidewalk_ai.processing.geometry import compute_clearances

SIDEWALK_WIDTH_M = 2.0
LEFT_PX, RIGHT_PX = 60, 540  # 480 px of sidewalk == SIDEWALK_WIDTH_M


def _sidewalk() -> np.ndarray:
    """Horizontal sidewalk band, i.e. the near-perpendicular view."""
    mask = np.zeros((400, 600), bool)
    mask[250:330, LEFT_PX:RIGHT_PX] = True
    return mask


def _obstacle(x0: int, x1: int) -> np.ndarray:
    obstacle = np.zeros((400, 600), bool)
    obstacle[300:330, x0:x1] = True
    return obstacle


def _clearance(obstacle: np.ndarray):
    results = compute_clearances(
        _sidewalk(),
        obstacles=[("tree#1", obstacle)],
        sidewalk_width_m=SIDEWALK_WIDTH_M,
    )
    assert len(results) == 1
    return results[0]


def test_centred_obstacle_splits_the_corridor_evenly():
    # Regression: a horizontal band makes both fitted curb lines flat, which the
    # previous line-inversion path could not invert, collapsing every clearance
    # to zero.
    res = _clearance(_obstacle(290, 310))

    assert res.L_m > 0.9
    assert res.R_m > 0.9
    assert res.L_m == res.R_m
    # 20 px of a 480 px span, scaled to metres.
    assert abs(res.obs_width - 20 / 480 * SIDEWALK_WIDTH_M) < 0.05


def test_obstacle_against_a_curb_leaves_one_sided_clearance():
    left = _clearance(_obstacle(LEFT_PX, LEFT_PX + 60))
    right = _clearance(_obstacle(RIGHT_PX - 60, RIGHT_PX))

    assert left.L_m < 0.05
    assert left.R_m > 1.5
    # Mirrored geometry must give mirrored clearances.
    assert abs(left.L_m - right.R_m) < 1e-9
    assert abs(left.R_m - right.L_m) < 1e-9


def test_full_width_obstacle_leaves_no_corridor():
    res = _clearance(_obstacle(LEFT_PX, RIGHT_PX))

    assert res.total_m < 0.05
    assert res.obs_width > 0.95 * SIDEWALK_WIDTH_M


def test_clearances_never_exceed_the_sidewalk_width():
    for x0, x1 in ((0, 40), (LEFT_PX, 200), (280, 320), (500, 600)):
        res = _clearance(_obstacle(x0, x1))
        assert 0.0 <= res.L_m <= SIDEWALK_WIDTH_M
        assert 0.0 <= res.R_m <= SIDEWALK_WIDTH_M
        assert res.total_m <= SIDEWALK_WIDTH_M + 1e-9
        assert res.obs_width >= 0.0
