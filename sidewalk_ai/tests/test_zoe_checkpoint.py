"""ZoeDepth checkpoint unwrapping.

The published ZoeDepth `.pt` files are *training* checkpoints: the parameters
sit under a "model" key next to "optimizer" and "epoch". Loading the wrapper
directly matches nothing, and because the load uses strict=False that failed
silently -- 511 missing keys, no exception, a randomly initialised network.

No weights are downloaded here: only the unwrapping is under test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sidewalk_ai.models.zoe import _unwrap_checkpoint  # noqa: E402


def _tensor(value: float):
    return torch.tensor([value])


def test_training_checkpoint_is_unwrapped():
    checkpoint = {
        "model": {"core.conv.weight": _tensor(1.0), "head.bias": _tensor(2.0)},
        "optimizer": {"state": {}},
        "epoch": 12,
    }

    weights = _unwrap_checkpoint(checkpoint)

    assert set(weights) == {"core.conv.weight", "head.bias"}
    assert "optimizer" not in weights
    assert "epoch" not in weights


def test_a_bare_state_dict_is_passed_through():
    checkpoint = {"core.conv.weight": _tensor(1.0)}

    assert set(_unwrap_checkpoint(checkpoint)) == {"core.conv.weight"}


def test_dataparallel_prefix_is_stripped():
    checkpoint = {"model": {"module.core.conv.weight": _tensor(1.0)}}

    assert set(_unwrap_checkpoint(checkpoint)) == {"core.conv.weight"}


def test_unwrapping_is_what_makes_the_keys_match():
    # Spells out the failure mode: without unwrapping, a strict=False load
    # reports every parameter as missing and raises nothing at all.
    module = torch.nn.Linear(2, 1)
    checkpoint = {"model": module.state_dict(), "optimizer": {}, "epoch": 3}

    wrapped = module.load_state_dict(checkpoint, strict=False)
    assert len(wrapped.missing_keys) == len(module.state_dict())

    unwrapped = module.load_state_dict(_unwrap_checkpoint(checkpoint), strict=False)
    assert unwrapped.missing_keys == []
