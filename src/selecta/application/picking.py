"""Choosing the candidates the agent may pick from."""

import random

from pydantic import BaseModel

from selecta.application.conversation import Session
from selecta.application.memory import note
from selecta.application.ports import CollectionStore
from selecta.domain.collection import ReleaseSummary
from selecta.domain.picking import MUCH_LONGER_FACTOR, rank, runs_past

MAX_CANDIDATES = 8
MIN_CANDIDATES = 3
MUCH_LONGER_NOTE = (
    "It runs well past their time: ask whether one side is fine or they'd rather have something "
    "that fits."
)


class Candidates(BaseModel):
    """Records offered to the model, and the filters that produced them."""

    releases: list[ReleaseSummary]
    filters: dict[str, str]
    relaxed: list[str]


def find_candidates(store: CollectionStore, session: Session) -> Candidates:
    """Return owned records matching the brief, minus the ones rejected in this session.

    If the listener wants whole albums, records known to run past their time are left out.
    With too few matches, energy then mood are relaxed; stricter matches stay first. Each step is
    shuffled with the session id as seed, then fresh records go before ones from past sessions.
    """
    brief = session.brief
    wanted = {
        slot: answer.value
        for slot in ("mood", "energy")
        if (answer := getattr(brief, slot)) and answer.value != "UNKNOWN"
    }
    attempts = [wanted, {k: v for k, v in wanted.items() if k != "energy"}, {}]
    releases: list[ReleaseSummary] = []
    used = wanted
    for i, filters in enumerate(attempts):
        if filters in attempts[:i]:
            continue
        used = filters
        taken = set(session.rejected_ids()) | {r.id for r in releases}
        minutes = brief.listening_minutes() if brief.wants_whole_album() else None
        found = [
            r
            for r in store.owned_summaries(**filters)
            if r.id not in taken and not runs_past(r.duration_seconds, minutes)
        ]
        random.Random(session.id).shuffle(found)
        remembered = {r.release_id for r in session.memory}
        releases += rank(found, remembered)[: MAX_CANDIDATES - len(releases)]
        if len(releases) >= MIN_CANDIDATES:
            break
    return Candidates(releases=releases, filters=used, relaxed=[s for s in wanted if s not in used])


def describe(candidates: Candidates, session: Session) -> str:
    """Return the candidates as lines for the system prompt."""
    if not candidates.releases:
        return "No record is left to offer: tell the listener, and ask what else they'd like."
    filters = ", ".join(f"{k}={v}" for k, v in candidates.filters.items()) or "none"
    header = f"Pick one with pick_record. Candidates chosen by code (filters: {filters}"
    if candidates.relaxed:
        header += f"; relaxed {', '.join(candidates.relaxed)}: too few records matched"
    lines = [header + "):"]
    minutes = session.brief.listening_minutes()
    memory = {r.release_id: r for r in session.memory}
    for r in candidates.releases:
        length = f"{r.duration_seconds // 60} min" if r.duration_seconds else "length unknown"
        year = f" ({r.original_year})" if r.original_year else ""
        feel = f"{r.mood}/{r.energy or 'energy not tagged'}" if r.mood else "mood not tagged"
        long = runs_past(r.duration_seconds, minutes, MUCH_LONGER_FACTOR)
        notes = [MUCH_LONGER_NOTE] if long else []
        if recalled := memory.get(r.id):
            notes.append(note(recalled))
        artists = ", ".join(r.artists)
        suffix = f" {' '.join(notes)}" if notes else ""
        lines.append(f"- {r.id}: {artists} - {r.title}{year}, {feel}, {length}.{suffix}")
    return "\n".join(lines)
