"""CLI entry point.

These tests exist because `play.py` became importable: it used to run the whole
pipeline at module scope, so importing it parsed sys.argv and loaded models.
"""

import json

import numpy as np
import pytest

from sidewalk_ai.cli import play
from sidewalk_ai.processing.geometry import ClearanceResult, WidthResult


class FakeResult:
    def __init__(self, width_m=2.4, heading=90):
        self.width = WidthResult(width_m, 0.3, 42)
        self.clearances = [
            ClearanceResult("tree#1:base1", L_m=1.1, R_m=0.9, total_m=2.0, obs_width=0.4)
        ]
        self.heading = heading
        self.img_path = None
        self.rgb_image = None
        self.sidewalk_mask = np.zeros((8, 8), bool)
        self.obstacles = []


class FakePipe:
    def __init__(self):
        self.calls = []

    def analyse_image(self, img_path, *, pitch, fov, heading=None, depth_scale=None):
        self.calls.append({"img_path": img_path, "pitch": pitch, "fov": fov, "heading": heading})
        return FakeResult()


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub out model construction so no weights are loaded."""
    pipe = FakePipe()
    monkeypatch.setattr(play, "_build", lambda args: (object(), pipe))
    return pipe


# --------------------------------------------------------------------------- #
# argument handling                                                           #
# --------------------------------------------------------------------------- #
def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        play.main(["--help"])

    assert excinfo.value.code == 0
    assert "--no-refine" in capsys.readouterr().out


def test_missing_target_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        play.main([])

    assert excinfo.value.code == 2
    assert "provide an address" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --image runs                                                                #
# --------------------------------------------------------------------------- #
def test_image_run_prints_width_and_returns_zero(tmp_path, fake_pipeline, capsys):
    image = tmp_path / "frame.jpg"
    image.touch()

    code = play.main(["--image", str(image), "--outdir", str(tmp_path / "out")])

    assert code == 0
    out = capsys.readouterr().out
    assert "WIDTH  2.40" in out
    assert "CLEAR  tree#1:base1" in out


def test_image_run_forwards_pitch_and_fov(tmp_path, fake_pipeline):
    image = tmp_path / "frame.jpg"
    image.touch()

    play.main(
        ["--image", str(image), "--pitch", "-20", "--fov", "75", "--outdir", str(tmp_path / "out")]
    )

    call = fake_pipeline.calls[0]
    # Regression: these were dropped, so every --image run silently analysed the
    # frame with pitch=0/fov=90 regardless of what was asked for.
    assert call["pitch"] == -20
    assert call["fov"] == 75


def test_image_run_uses_the_argparse_pitch_default(tmp_path, fake_pipeline):
    image = tmp_path / "frame.jpg"
    image.touch()

    play.main(["--image", str(image), "--outdir", str(tmp_path / "out")])

    assert fake_pipeline.calls[0]["pitch"] == -10


def test_metrics_json_is_written(tmp_path, fake_pipeline):
    image = tmp_path / "frame.jpg"
    image.touch()
    metrics = tmp_path / "metrics.json"

    play.main(
        [
            "--image",
            str(image),
            "--metrics-json",
            str(metrics),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["min_clear_required_m"] == 1.20
    assert payload["global"]["total_obstacles"] == 1


def test_outdir_is_created(tmp_path, fake_pipeline):
    image = tmp_path / "frame.jpg"
    image.touch()
    outdir = tmp_path / "nested" / "debug"

    play.main(["--image", str(image), "--outdir", str(outdir)])

    assert outdir.is_dir()
