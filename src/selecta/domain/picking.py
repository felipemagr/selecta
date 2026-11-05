"""Rules for picking a record."""

from typing import Literal

from selecta.domain.collection import ReleaseSummary

Fit = Literal["no_limit", "fits", "unknown", "too_long"]
_FIT_ORDER: dict[Fit, int] = {"no_limit": 0, "fits": 0, "unknown": 1, "too_long": 2}


def fit(duration_seconds: int | None, minutes: int | None) -> Fit:
    """Return how a record's length fits the listening time."""
    if minutes is None:
        return "no_limit"
    if not duration_seconds:
        return "unknown"
    return "fits" if duration_seconds <= minutes * 60 else "too_long"


def rank(candidates: list[ReleaseSummary], minutes: int | None) -> list[ReleaseSummary]:
    """Return candidates with those that fit the time first. Time never excludes a record."""
    return sorted(candidates, key=lambda c: _FIT_ORDER[fit(c.duration_seconds, minutes)])
