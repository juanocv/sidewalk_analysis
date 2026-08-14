"""DeepLab checkpoint/backbone agreement.

A checkpoint for the wrong backbone still matches a few tensors by coincidence
(a mobilenet checkpoint fills 44 of resnet101's 674), so the previous
"raise only when nothing matched" test let a 93% randomly initialised network
through. That segmented almost nothing, and the only symptom was a 0.00 m width
with a "No sidewalk support" warning far downstream.

No weights are downloaded here: the loader is driven with tiny stand-in modules.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from sidewalk_ai.models.deeplab import load_deeplab_checkpoint  # noqa: E402


def _make_modeling(monkeypatch):
    """Install a stand-in `network.modeling` exposing two unrelated shapes."""

    def small(num_classes=19, output_stride=16):
        return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))

    def big(num_classes=19, output_stride=16):
        return torch.nn.Sequential(
            torch.nn.Linear(4, 4),  # shared shape: matches by coincidence
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 8),
        )

    modeling = types.ModuleType("network.modeling")
    modeling.small_net = small
    modeling.big_net = big
    network = types.ModuleType("network")
    network.modeling = modeling
    monkeypatch.setitem(sys.modules, "network", network)
    monkeypatch.setitem(sys.modules, "network.modeling", modeling)
    return small, big


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    small, _ = _make_modeling(monkeypatch)
    path = tmp_path / "small_net.pth"
    torch.save({"model_state": small().state_dict()}, path)
    return path


def test_matching_backbone_loads(checkpoint):
    model = load_deeplab_checkpoint(str(checkpoint), model_name="small_net", device="cpu")

    assert model.training is False


def test_mismatched_backbone_is_rejected(checkpoint):
    with pytest.raises(RuntimeError) as excinfo:
        load_deeplab_checkpoint(str(checkpoint), model_name="big_net", device="cpu")

    message = str(excinfo.value)
    assert "does not fit model_name='big_net'" in message
    # The counts are the actionable part: they say how much would be random.
    assert "tensors matched" in message
    assert "--deeplab-model" in message


def test_partial_match_does_not_slip_through(checkpoint, monkeypatch):
    # Guards the exact regression: some tensors match, so the old
    # `if not filtered` check passed and a mostly-random model was returned.
    _, big = _make_modeling(monkeypatch)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model_keys = big().state_dict()
    overlap = {k for k in state if k in model_keys and state[k].shape == model_keys[k].shape}

    assert overlap, "test setup must produce a partial match"
    assert len(overlap) < len(model_keys)

    with pytest.raises(RuntimeError):
        load_deeplab_checkpoint(str(checkpoint), model_name="big_net", device="cpu")


def test_missing_file_is_reported_as_such(tmp_path, monkeypatch):
    _make_modeling(monkeypatch)

    with pytest.raises(FileNotFoundError):
        load_deeplab_checkpoint(str(tmp_path / "absent.pth"), model_name="small_net", device="cpu")
