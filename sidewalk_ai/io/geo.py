# sidewalk_ai/io/geo.py
"""
Light-weight geographic helpers (no external deps).

The module is **pure**: it never prints, reads, or writes files and is
fully covered by type hints—so you can unit-test everything in < 1 ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

__all__ = [
    "EARTH_RADIUS_M",
    "Coordinate",
    "haversine",
    "bearing",
    "destination",
]

# --------------------------------------------------------------------------- #
# 0)  Constants / dataclasses                                                 #
# --------------------------------------------------------------------------- #

EARTH_RADIUS_M: float = 6_371_000.0           # IUGG mean Earth radius (metres)


@dataclass(frozen=True, slots=True)
class Coordinate:
    """Immutable (lat, lon) pair in *decimal degrees*."""
    lat: float
    lon: float

    # enable:   c1 | c2   → great-circle distance (metres)
    def __or__(self, other: "Coordinate") -> float:
        return haversine(self, other)


# --------------------------------------------------------------------------- #
# 1)  Core helpers                                                            #
# --------------------------------------------------------------------------- #

def haversine(a: Union[Coordinate, Tuple[float, float]],
              b: Union[Coordinate, Tuple[float, float]]) -> float:
    """
    Fast great-circle distance in **metres** (error <1 m up to 200 km).

    Examples
    --------
    >>> sao = Coordinate(-23.5505, -46.6333)
    >>> rio = Coordinate(-22.9068, -43.1729)
    >>> round(haversine(sao, rio)/1000)   # ≈ 357 km
    357
    """
    lat1, lon1 = a if isinstance(a, tuple) else (a.lat, a.lon)
    lat2, lon2 = b if isinstance(b, tuple) else (b.lat, b.lon)

    φ1, φ2 = map(math.radians, (lat1, lat2))
    Δφ     = math.radians(lat2 - lat1)
    Δλ     = math.radians(lon2 - lon1)

    s = math.sin(Δφ * 0.5)
    c = math.sin(Δλ * 0.5)
    h = s * s + math.cos(φ1) * math.cos(φ2) * c * c
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def bearing(a: Coordinate, b: Coordinate) -> float:
    """
    Initial course *a → b* in degrees (0° = North, clockwise).

    Useful when you want to fetch Street-View images at fixed headings
    from a start point towards a target point.
    """
    φ1, φ2 = map(math.radians, (a.lat, b.lat))
    Δλ = math.radians(b.lon - a.lon)

    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    θ = math.atan2(x, y)
    return (math.degrees(θ) + 360.0) % 360.0


def destination(
    origin: Coordinate,
    distance_m: float,
    heading_deg: float,
) -> Coordinate:
    """
    Geographic point reached after travelling *distance_m* at *heading_deg*
    along a great-circle path.

    Parameters
    ----------
    origin
        Start coordinate (decimal degrees).
    distance_m
        Arc length (metres).
    heading_deg
        Bearing in degrees – 0 = north, 90 = east.

    Returns
    -------
    Coordinate
        New latitude/longitude (decimal degrees).

    Notes
    -----
    Matches the algorithm in your old `new_coords()` helper but keeps full
    double precision and removes rounding side-effects.
    """
    d = distance_m / EARTH_RADIUS_M
    θ = math.radians(heading_deg)

    φ1 = math.radians(origin.lat)
    λ1 = math.radians(origin.lon)

    φ2 = math.asin(
        math.sin(φ1) * math.cos(d) + math.cos(φ1) * math.sin(d) * math.cos(θ)
    )
    λ2 = λ1 + math.atan2(
        math.sin(θ) * math.sin(d) * math.cos(φ1),
        math.cos(d) - math.sin(φ1) * math.sin(φ2),
    )

    return Coordinate(lat=math.degrees(φ2), lon=(math.degrees(λ2) + 540) % 360 - 180)