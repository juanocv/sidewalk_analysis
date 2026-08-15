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


# --------------------------------------------------------------------------- #
# device selection                                                            #
# --------------------------------------------------------------------------- #
# Every test above names `device="cpu"` explicitly, which is exactly why none of
# them caught the CLI path being broken: `--device cpu` was dropped on the way to
# the loader, whose default was a hardcoded "cuda". On a machine without a GPU
# the DeepLab back-end was unreachable, failing with a bare
# "RuntimeError: No CUDA GPUs are available".


def test_default_device_does_not_assume_cuda(checkpoint):
    model = load_deeplab_checkpoint(str(checkpoint), model_name="small_net")

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert next(model.parameters()).device.type == expected


def test_factory_forwards_the_requested_device_to_the_loader(checkpoint, monkeypatch):
    from sidewalk_ai.models import deeplab as deeplab_module
    from sidewalk_ai.models.factory import build_segmenter

    seen = {}
    real_loader = deeplab_module.load_deeplab_checkpoint

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real_loader(path, **kwargs)

    monkeypatch.setattr(deeplab_module, "load_deeplab_checkpoint", spy)

    build_segmenter(
        "deeplab",
        ckpt_path=str(checkpoint),
        model_name="small_net",
        device="cpu",
    )

    assert seen.get("device") == "cpu"


# --------------------------------------------------------------------------- #
# architecture inference from the checkpoint name                             #
# --------------------------------------------------------------------------- #
from sidewalk_ai.models.deeplab import infer_model_name  # noqa: E402


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("best_deeplabv3plus_mobilenet_cityscapes_os16.pth", "deeplabv3plus_mobilenet"),
        ("best_deeplabv3plus_resnet101_cityscapes_os16.pth", "deeplabv3plus_resnet101"),
        ("best_deeplabv3_resnet50_voc_os16.pth", "deeplabv3_resnet50"),
        ("deeplabv3plus_hrnetv2_48_cityscapes.pth", "deeplabv3plus_hrnetv2_48"),
        ("DeepLabV3Plus_MobileNet.PTH", "deeplabv3plus_mobilenet"),
    ],
)
def test_architecture_is_read_from_the_filename(filename, expected):
    assert infer_model_name(f"/weights/{filename}") == expected


def test_plus_is_not_mistaken_for_the_plain_architecture():
    # "deeplabv3plus_mobilenet" must never resolve to "deeplabv3_mobilenet".
    assert infer_model_name("best_deeplabv3plus_mobilenet_cityscapes.pth").startswith(
        "deeplabv3plus"
    )


def test_unrecognisable_name_returns_none():
    # The caller turns this into "pass --deeplab-model explicitly".
    assert infer_model_name("/weights/my_finetuned_weights.pth") is None
