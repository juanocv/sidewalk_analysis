# sidewalk_ai/io/streetview.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from pydantic import Field
from pydantic_settings import BaseSettings
from joblib import Memory

# --------------------------------------------------------------------------- #
# 1)  Settings – central place for API key, cache size, paths, default params
# --------------------------------------------------------------------------- #

class Settings(BaseSettings):
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    cache_dir: Path = Path.home() / ".sidewalk_ai" / "cache" / "streetview"
    default_fov: int = 120
    default_size: str = "600x400"
    timeout_s: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


cfg = Settings()

# --------------------------------------------------------------------------- #
# 2)  Request “DTO” – keeps the public API explicit and typed
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class ImageRequest:
    lat: float
    lon: float
    heading: int = 0
    pitch: int = 0
    fov: int = cfg.default_fov
    size: str = cfg.default_size


# --------------------------------------------------------------------------- #
# 3)  Street View client – thin I/O layer, **no business logic**
# --------------------------------------------------------------------------- #

class StreetViewClient:
    """
    Fetch Google Street View images + metadata with local on-disk caching.

    Usage
    -----
    >>> client = StreetViewClient()
    >>> img_path = client.fetch(ImageRequest(lat=-23.68, lon=-46.54))
    >>> meta     = client.metadata(-23.68, -46.54)
    """

    _BASE = "https://maps.googleapis.com/maps/api/streetview"
    _META = "https://maps.googleapis.com/maps/api/streetview/metadata"
    _GEO  = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        settings: Settings = cfg,
    ):
        self.settings = settings
        self.cache = Memory(location=cache_dir or settings.cache_dir, compress=True)
        self.session = session or requests.Session()
        self.session.params = {"key": settings.google_api_key}
        self.session.headers.update({"User-Agent": "sidewalk-ai/0.1"})
        os.makedirs(self.cache.location, exist_ok=True)

    # ------------------------- public helper -------------------------------- #

    def geocode(self, address: str) -> tuple[float, float]:
        """Return (lat, lon) for a free-form address string."""
        resp = self._get(self._GEO, params={"address": address})
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"No coordinates found for address: “{address}”.")
        loc = results[0]["geometry"]["location"]
        return loc["lat"], loc["lng"]

    # ------------------------- image download ------------------------------- #

    def fetch(self, req: ImageRequest | str) -> Path:
        """
        Download a Street View image (or return the cached copy).

        If *req* is a str we treat it as an address and geocode it first.
        """
        if isinstance(req, str):
            lat, lon = self.geocode(req)
            req = ImageRequest(lat, lon)  # type: ignore[arg-type]

        filename = (
            f"sv_{req.lat:.6f}_{req.lon:.6f}_{req.heading:03d}_"
            f"{req.pitch:02d}_{req.fov}_{req.size}.jpg"
        )
        local_path = self.cache.location / filename

        if local_path.exists():
            return local_path

        params = {
            "location": f"{req.lat},{req.lon}",
            "heading": req.heading,
            "pitch": req.pitch,
            "fov": req.fov,
            "size": req.size,
        }
        # Google occasionally returns HTTP 200 + empty body → guard with len()
        content = self._get(self._BASE, params=params).content
        if len(content) < 1024:  # heuristics for an error placeholder image
            raise RuntimeError("Street View returned an empty image.")

        local_path.write_bytes(content)
        return local_path

    # ------------------------- metadata lookup ------------------------------ #

    #@Memory.cache(ignore=["self"], verbose=0)
    def metadata(self, lat: float, lon: float) -> dict:
        """Cached call – Google’s quota counts metadata requests as well."""
        resp = self._get(self._META, params={"location": f"{lat},{lon}"})
        return resp.json()

    # ------------------------- private helpers ------------------------------ #

    def _get(self, url: str, params: dict) -> requests.Response:
        start = time.perf_counter()
        resp = self.session.get(url, params=params, timeout=self.settings.timeout_s)
        try:
            resp.raise_for_status()
        except Exception:  # pragma: no cover
            msg = resp.json().get("error_message", "")
            raise RuntimeError(f"Street View API error: {msg or resp.text}") from None
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            # You may plug a proper logger here
            print(f"[streetview] GET {url.split('/')[-1]} – {elapsed:5.1f} ms")
        return resp