"""Choosing the candidates the agent may pick from."""

import random

from pydantic import BaseModel

from selecta.application.conversation import Session
from selecta.application.ports import CollectionStore
from selecta.domain.collection import ReleaseSummary
from selecta.domain.picking import Fit, fit, rank

MAX_CANDIDATES = 8
MIN_CANDIDATES = 3

FIT_NOTES: dict[Fit, str] = {
    "no_limit": "Their listening time isn't known.",
    "fits": "It fits their time.",
    "unknown": "Its length is unknown: say you're not sure how long it runs.",
    "too_long": "It runs longer than their time: say so, and suggest playing one side.",
}


class Candidates(BaseModel):
    """Records offered to the model, and the filters that produced them."""

    releases: list[ReleaseSummary]
    filters: dict[str, str]
    relaxed: list[str]


def find_candidates(store: CollectionStore, session: Session) -> Candidates:
    """Return owned, not rejected records matching the brief.

    With too few matches, energy then mood are relaxed; stricter matches stay first. Each step is
    shuffled with the session id as seed, then ranked by time, so fitting records are never lost.
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
        found = [r for r in store.owned_summaries(**filters) if r.id not in taken]
        random.Random(session.id).shuffle(found)
        releases += rank(found, brief.listening_minutes())[: MAX_CANDIDATES - len(releases)]
        if len(releases) >= MIN_CANDIDATES:
            break
    return Candidates(releases=releases, filters=used, relaxed=[s for s in wanted if s not in used])


def describe(candidates: Candidates, minutes: int | None) -> str:
    """Return the candidates as lines for the system prompt."""
    if not candidates.releases:
        return "No record is left to offer: tell the listener, and ask what else they'd like."
    filters = ", ".join(f"{k}={v}" for k, v in candidates.filters.items()) or "none"
    header = f"Pick one with pick_record. Candidates chosen by code (filters: {filters}"
    if candidates.relaxed:
        header += f"; relaxed {', '.join(candidates.relaxed)}: too few records matched"
    lines = [header + "):"]
    for r in candidates.releases:
        length = f"{r.duration_seconds // 60} min" if r.duration_seconds else "length unknown"
        year = f" ({r.original_year})" if r.original_year else ""
        feel = f"{r.mood}/{r.energy or 'energy not tagged'}" if r.mood else "mood not tagged"
        note = FIT_NOTES[fit(r.duration_seconds, minutes)]
        artists = ", ".join(r.artists)
        lines.append(f"- {r.id}: {artists} - {r.title}{year}, {feel}, {length}. {note}")
    return "\n".join(lines)
