"""Which segment labels count as obstacles.

The vocabulary was a single hardcoded set written against ADE20K's 150 classes,
applied unchanged to Cityscapes' 19 -- which is part of why the two back-ends
disagree about what sits on a sidewalk.
"""

from __future__ import annotations

import numpy as np
import pytest

from sidewalk_ai.models._obstacles import _label_matches, extract_obstacles


def _scene():
    """A sidewalk band with a tree and a car touching it."""
    seg_map = np.zeros((60, 80), np.int32)
    seg_map[30:50, 5:75] = 1  # sidewalk
    seg_map[40:50, 20:30] = 2  # tree standing on it
    seg_map[40:50, 55:70] = 3  # car overlapping it
    sidewalk = seg_map == 1
    return seg_map, sidewalk


_INFO = [(1, "sidewalk"), (2, "tree"), (3, "car")]


def _labels(obstacles):
    return sorted({label.split("#")[0] for label, _ in obstacles})


def test_the_shared_default_ignores_cars():
    seg_map, sidewalk = _scene()

    assert _labels(extract_obstacles(seg_map, _INFO, sidewalk)) == ["tree"]


def test_a_caller_can_widen_the_vocabulary():
    # Nothing in the module has to change for a project that counts a car
    # parked on the sidewalk as the barrier it is.
    seg_map, sidewalk = _scene()

    obstacles = extract_obstacles(seg_map, _INFO, sidewalk, ignore_labels=frozenset())

    assert _labels(obstacles) == ["car", "tree"]


def test_the_target_class_is_never_an_obstacle_of_itself():
    seg_map, sidewalk = _scene()

    obstacles = extract_obstacles(
        seg_map, _INFO, sidewalk, ignore_labels=frozenset(), sidewalk_labels=frozenset({"sidewalk"})
    )

    assert "sidewalk" not in _labels(obstacles)


# --------------------------------------------------------------------------- #
# label matching                                                              #
# --------------------------------------------------------------------------- #
def test_ade20k_synonym_lists_match_on_the_first_name():
    # ADE20K labels look like "building, edifice"; the vocabulary holds "building".
    assert _label_matches("building, edifice", {"building"})


def test_a_longer_name_sharing_a_prefix_is_not_swallowed():
    # Regression: `startswith` made "skyscraper" match "sky", so a skyscraper
    # was discarded as background.
    assert not _label_matches("skyscraper", {"sky"})
    assert _label_matches("sky", {"sky"})


@pytest.mark.parametrize("label", ["cardboard", "carpet", "cart"])
def test_names_beginning_with_an_ignored_class_survive(label):
    assert not _label_matches(label, {"car"})
