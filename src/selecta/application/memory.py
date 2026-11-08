"""What the listener did in past sessions, read from the stored sessions."""

from datetime import datetime, timedelta

from selecta.application.conversation import Recalled, Session
from selecta.application.ports import SessionStore

MEMORY_DAYS = 30


def recall(store: SessionStore, now: datetime) -> list[Recalled]:
    """Return the latest outcome of each record picked or rejected in the last MEMORY_DAYS."""
    latest: dict[int, Recalled] = {}
    for session in store.sessions_since(now - timedelta(days=MEMORY_DAYS)):
        for r in session.rejections:
            latest[r.release_id] = Recalled(
                release_id=r.release_id,
                title=r.title,
                artists=r.artists,
                outcome="rejected",
                on=session.started_at,
                reason=r.reason,
            )
        if pick := session.pick:
            latest[pick.release_id] = Recalled(
                release_id=pick.release_id,
                title=pick.title,
                artists=pick.artists,
                outcome="picked",
                on=session.started_at,
            )
    return sorted(latest.values(), key=lambda r: r.on)


def last_session(session: Session) -> str | None:
    """Return a line about the most recent session with a pick or rejection, if any."""
    if not session.memory:
        return None
    on = session.memory[-1].on
    events = sorted((r for r in session.memory if r.on == on), key=lambda r: r.outcome != "picked")
    parts = [
        f"{'picked' if r.outcome == 'picked' else 'they turned down'} {_record(r)}" for r in events
    ]
    return f"Most recent session with a pick or rejection ({_day(on)}): {'; '.join(parts)}."


def note(recalled: Recalled) -> str:
    """Return the candidate note for a remembered record."""
    if recalled.outcome == "picked":
        return (
            f"It was picked for them on {_day(recalled.on)}: if you offer it again, say so and ask "
            "if they'd like it again."
        )
    return (
        f'They turned it down on {_day(recalled.on)} ("{recalled.reason}"): if you offer it, say '
        "so and ask if today is different."
    )


def _record(r: Recalled) -> str:
    return f"{', '.join(r.artists)} - {r.title}" if r.title else f"record {r.release_id}"


def _day(on: datetime) -> str:
    local = on.astimezone()
    return f"{local:%b} {local.day}"
