"""ZoeDepth's torch.hub call must not need a terminal.

`torch.hub.load` asks for confirmation the first time it caches a GitHub repo
unless `trust_repo` is set. With no TTY -- a script, a CI job, or any run whose
output is redirected -- that prompt fails as a bare

    EOFError: EOF when reading a line

raised from deep inside torch.hub, naming neither ZoeDepth nor trust. Since
`--depth zoe` is the pipeline default, this made the README's own first example
unrunnable non-interactively.

No repository is cloned and no weights are downloaded here: `torch.hub.load` is
replaced by a spy that records its arguments and stops construction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sidewalk_ai.models.zoe import ZoeDepthEstimator  # noqa: E402


class _StopConstruction(Exception):
    """Raised by the spy so nothing after the hub call runs."""


@pytest.fixture
def hub_call(monkeypatch):
    """Capture the arguments of the single torch.hub.load call."""
    recorded = {}

    def spy(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        raise _StopConstruction

    monkeypatch.setattr(torch.hub, "load", spy)
    return recorded


def test_github_source_is_trusted_up_front(hub_call):
    with pytest.raises(_StopConstruction):
        ZoeDepthEstimator(device="cpu")

    assert hub_call["kwargs"].get("trust_repo") is True


def test_the_trusted_repo_is_the_hardcoded_upstream_one(hub_call):
    # trust_repo=True is only defensible because the repo is not caller-supplied.
    with pytest.raises(_StopConstruction):
        ZoeDepthEstimator(device="cpu", repo_or_path="attacker/evil")

    assert hub_call["args"][0] == "isl-org/ZoeDepth"


def test_a_local_source_still_points_at_the_given_path(hub_call, tmp_path):
    with pytest.raises(_StopConstruction):
        ZoeDepthEstimator(device="cpu", source="local", repo_or_path=tmp_path)

    assert hub_call["args"][0] == str(tmp_path.resolve())
    assert hub_call["kwargs"].get("source") == "local"
