"""Web API: request routing, per-request isolation and concurrency limits.

The heavy models are stubbed out; what is under test is the HTTP layer and the
pipeline registry, not the estimator.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sidewalk_ai import webapi  # noqa: E402
from sidewalk_ai.processing.geometry import ClearanceResult, WidthResult  # noqa: E402


class FakeResult:
    def __init__(self):
        self.width = WidthResult(2.4, 0.3, 42)
        self.clearances = [
            ClearanceResult("tree#1:base1", L_m=1.1, R_m=0.9, total_m=2.0, obs_width=0.4)
        ]
        self.heading = 90
        self.img_path = None
        self.rgb_image = None
        self.sidewalk_mask = np.zeros((8, 8), bool)
        self.obstacles = []


class FakeDepth:
    """Records how many times a depth back-end was constructed."""

    builds: list[tuple[str, str | None]] = []

    def __init__(self, backend, variant):
        FakeDepth.builds.append((backend, variant))
        self.backend = backend
        self.variant = variant


class FakePipe:
    def __init__(self, depth, refine):
        self.depth = depth
        self.refine = refine


@pytest.fixture
def client(monkeypatch):
    FakeDepth.builds = []

    monkeypatch.setattr(webapi.sw, "build_segmenter", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(webapi.sw, "StreetViewClient", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(
        webapi, "build_depth", lambda backend, variant=None: FakeDepth(backend, variant)
    )
    monkeypatch.setattr(
        webapi.sw,
        "SidewalkPipeline",
        lambda *, segmenter, depth, streetview, refine: FakePipe(depth, refine),
        raising=False,
    )
    monkeypatch.setattr(webapi, "run_pipeline", lambda pipe, cfg: FakeResult())

    with TestClient(webapi.app) as test_client:
        yield test_client


def test_ping(client):
    assert client.get("/ping").json() == {"ok": True}


def test_single_view_returns_width_and_clearances(client):
    response = client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5})

    assert response.status_code == 200
    body = response.json()
    assert body["width_m"] == pytest.approx(2.4)
    assert body["clearances"][0]["label"] == "tree#1:base1"
    assert body["accessibility"]["global_stats"]["total_obstacles"] == 1


# --------------------------------------------------------------------------- #
# per-request isolation                                                       #
# --------------------------------------------------------------------------- #
def test_zoe_variant_does_not_leak_into_later_requests(client):
    """
    Regression: a request carrying `zoe_variant` used to overwrite the shared
    pipeline for (depth, refine), so every later request silently inherited that
    variant — and rebuilt the model each time.
    """
    client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5, "zoe_variant": "ZoeD_K"})
    client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5})

    registry = webapi.app.state.registry
    default_pipe = registry.get("zoe", webapi.DEFAULT_ZOE_VARIANT, True)
    override_pipe = registry.get("zoe", "zoed_k", True)

    assert default_pipe is not override_pipe
    assert default_pipe.depth.variant == webapi.DEFAULT_ZOE_VARIANT
    assert override_pipe.depth.variant == "zoed_k"


def test_a_variant_is_only_built_once(client):
    for _ in range(3):
        client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5, "zoe_variant": "ZoeD_K"})

    # Previously each request rebuilt the model; now the registry caches it.
    assert FakeDepth.builds.count(("zoe", "zoed_k")) == 1


def test_refine_and_variant_get_separate_pipelines(client):
    registry = webapi.app.state.registry

    pipes = {
        (variant, refine): registry.get("zoe", variant, refine)
        for variant in ("zoed_n", "zoed_k")
        for refine in (True, False)
    }

    assert len({id(p) for p in pipes.values()}) == 4
    assert pipes[("zoed_n", True)].refine is True
    assert pipes[("zoed_n", False)].refine is False
    # One model per variant, shared by both refine settings.
    assert FakeDepth.builds.count(("zoe", "zoed_n")) == 1


def test_unknown_zoe_variant_is_rejected(client):
    response = client.post(
        "/analyse/single", json={"lat": -23.6, "lon": -46.5, "zoe_variant": "ZoeD_XL"}
    )

    assert response.status_code == 400
    assert "zoe_variant must be one of" in response.json()["detail"]


def test_missing_location_is_a_422(client, monkeypatch):
    def boom(pipe, cfg):
        raise ValueError("Either address or lat+lon required")

    monkeypatch.setattr(webapi, "run_pipeline", boom)

    response = client.post("/analyse/single", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# concurrency                                                                 #
# --------------------------------------------------------------------------- #
def test_model_work_is_serialised(client, monkeypatch):
    """Only MAX_CONCURRENCY requests may be inside the pipeline at once."""
    inside = 0
    peak = 0
    guard = threading.Lock()

    def slow_pipeline(pipe, cfg):
        nonlocal inside, peak
        with guard:
            inside += 1
            peak = max(peak, inside)
        try:
            threading.Event().wait(0.05)
            return FakeResult()
        finally:
            with guard:
                inside -= 1

    monkeypatch.setattr(webapi, "run_pipeline", slow_pipeline)

    threads = [
        threading.Thread(
            target=lambda: client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5})
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= webapi.MAX_CONCURRENCY


def test_queue_timeout_reports_busy(client, monkeypatch):
    monkeypatch.setenv("SWAI_API_QUEUE_TIMEOUT_S", "0.05")
    exhausted = threading.BoundedSemaphore(1)
    exhausted.acquire()
    monkeypatch.setattr(webapi, "_inference_slots", exhausted)

    response = client.post("/analyse/single", json={"lat": -23.6, "lon": -46.5})

    assert response.status_code == 503
    assert "busy" in response.json()["detail"].lower()
