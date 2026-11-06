"""Rules for picking a record."""

from selecta.domain.collection import ReleaseSummary

# One side can usually be played, so only a record this many times longer than the listening
# time is worth asking about.
MUCH_LONGER_FACTOR = 1.5


def runs_past(duration_seconds: int | None, minutes: int | None, factor: float = 1.0) -> bool:
    """Return whether a record of known length runs past `factor` times the listening time."""
    if not (minutes and duration_seconds):
        return False
    return duration_seconds > minutes * 60 * factor


def rank(candidates: list[ReleaseSummary], remembered: set[int]) -> list[ReleaseSummary]:
    """Return candidates with fresh records before remembered ones, keeping the order otherwise."""
    return sorted(candidates, key=lambda c: c.id in remembered)
