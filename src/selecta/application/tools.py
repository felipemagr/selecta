"""The agent's tools."""

import json
import logging
from typing import Any, get_args

from pydantic import ValidationError

from selecta.application.conversation import (
    Language,
    Pick,
    Rejection,
    Session,
    ToolCall,
    ToolSpec,
)
from selecta.application.picking import FIT_NOTES, Candidates, find_candidates
from selecta.application.ports import CollectionQuery, CollectionStore, Embedder
from selecta.application.retrieval import semantic_search
from selecta.domain.brief import SLOT_ORDER, Brief, Occasion
from selecta.domain.enrichment import Energy, Mood
from selecta.domain.picking import fit

logger = logging.getLogger(__name__)


def _answer(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": value,
            "evidence": {"type": "string", "description": "The listener's own words."},
        },
        "required": ["value", "evidence"],
    }


SPECS = [
    ToolSpec(
        name="update_brief",
        description=(
            "Save what the listener just told you about what they want to hear. Pass only the "
            "slots they answered, each with the words they used. Use UNKNOWN when they can't or "
            "won't say. Always pass the language the listener speaks. Returns what was saved, any "
            "value that was rejected and why, and the next slot to ask about."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mood": _answer({"enum": list(get_args(Mood))}),
                "energy": _answer({"enum": list(get_args(Energy))}),
                "minutes": _answer(
                    {
                        "anyOf": [{"type": "integer"}, {"enum": ["UNKNOWN"]}],
                        "description": "How long they want to listen, in minutes.",
                    }
                ),
                "occasion": _answer({"enum": list(get_args(Occasion))}),
                "language": {"enum": list(get_args(Language))},
            },
            "required": ["language"],
        },
    ),
    ToolSpec(
        name="search_collection",
        description=(
            "Search the listener's records by exact facts: names (misspellings are fine), genre, "
            "style, label, years, mood, energy, or sorted by age or when they were added. Use it "
            "for questions like 'my oldest record' or 'anything on Warp'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Artist, title, track or person."},
                "genre": {"type": "string"},
                "style": {"type": "string"},
                "label": {"type": "string"},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "mood": {"enum": [m for m in get_args(Mood) if m != "UNKNOWN"]},
                "energy": {"enum": [e for e in get_args(Energy) if e != "UNKNOWN"]},
                "sort": {"enum": ["relevance", "oldest", "newest", "recently_added"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
    ),
    ToolSpec(
        name="get_release",
        description="Get one record's full details: tracklist, credits, label, year and notes.",
        parameters={
            "type": "object",
            "properties": {"release_id": {"type": "integer"}},
            "required": ["release_id"],
        },
    ),
    ToolSpec(
        name="semantic_search",
        description=(
            "Search record facts and artist biographies by meaning, for open questions like "
            "'a band from Düsseldorf' or 'something with saxophone'. Works in any language."
        ),
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
    ToolSpec(
        name="pick_record",
        description=(
            "Choose the record for the listener. Call it before you announce a pick: only a pick "
            "this tool accepted may be announced. Usually one of the candidates in the interview "
            "state; any owned record if the listener asks for it by name. Only one record is "
            "picked at a time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "release_id": {"type": "integer"},
                "reason": {"type": "string", "description": "Why it suits them, in a sentence."},
                "asked_by_name": {
                    "type": "boolean",
                    "description": "True if the listener asked for this record themselves.",
                },
            },
            "required": ["release_id", "reason"],
        },
    ),
    ToolSpec(
        name="reject_pick",
        description="Record that the listener turned down the current pick.",
        parameters={"type": "object", "properties": {}},
    ),
]


class Tools:
    """Tool executor."""

    def __init__(self, store: CollectionStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def candidates(self, session: Session) -> Candidates:
        """Return the records the agent may pick from now."""
        return find_candidates(self._store, session)

    def run(self, call: ToolCall, session: Session) -> str:
        """Return the result of a tool call. Failures are returned too, never raised."""
        args = call.arguments
        try:
            match call.name:
                case "update_brief":
                    return _update_brief(session, args)
                case "search_collection":
                    hits = self._store.search(CollectionQuery.model_validate(args))
                    return _dump([h.model_dump(mode="json") for h in hits]) if hits else "No match."
                case "get_release":
                    release = self._store.get_release(args["release_id"])
                    return release.model_dump_json() if release else "No such record."
                case "pick_record":
                    asked = args.get("asked_by_name") is True
                    return self._pick(session, int(args["release_id"]), args["reason"], asked)
                case "reject_pick":
                    return _reject(session)
                case "semantic_search":
                    passages = semantic_search(self._store, self._embedder, args["text"])
                    return _dump([p.model_dump() for p in passages])
                case _:
                    return f"Unknown tool: {call.name}"
        except (ValidationError, KeyError, TypeError, ValueError) as e:
            return f"Invalid arguments for {call.name}: {e}"
        except Exception:
            logger.exception("tool %s failed", call.name)
            return f"{call.name} failed. Tell the listener you can't check that right now."

    def _pick(self, session: Session, release_id: int, reason: str, asked_by_name: bool) -> str:
        if session.pick:
            return (
                "A record is already picked. Keep it, unless the listener turned it down: then "
                "call reject_pick."
            )
        if not asked_by_name:
            if session.brief.next_slot():
                return "Finish the interview first, unless the listener asked for a record by name."
            if release_id in session.rejected_ids():
                return "The listener already rejected that record. Pick another."
            if release_id not in {r.id for r in self.candidates(session).releases}:
                return "Pick from the candidates, unless the listener asked for a record by name."
        release = self._store.owned_summary(release_id)
        if release is None:
            return "That record isn't in the listener's collection. Pick another."
        session.pick = Pick(
            release_id=release.id, title=release.title, artists=release.artists, reason=reason
        )
        minutes = session.brief.listening_minutes()
        return f"Picked. {FIT_NOTES[fit(release.duration_seconds, minutes)]}"


def _reject(session: Session) -> str:
    """Reject the pick, with the listener's last message as evidence rather than the model's."""
    if session.pick is None:
        return "Nothing is picked yet."
    said = next(m.content for m in reversed(session.messages) if m.role == "user" and m.content)
    session.rejections.append(Rejection(release_id=session.pick.release_id, reason=said))
    session.pick = None
    return "Rejected. The candidates in the interview state are updated."


def _update_brief(session: Session, args: dict[str, Any]) -> str:
    """Save each valid slot; one bad value doesn't drop the others."""
    if args.get("language") in get_args(Language):
        session.language = args["language"]
    before = session.brief.values()
    saved, rejected = [], {}
    for slot in SLOT_ORDER:
        if args.get(slot) is None:
            continue
        try:
            candidate = Brief.model_validate({slot: args[slot]})
        except ValidationError as e:
            rejected[slot] = e.errors()[0]["msg"]
            continue
        setattr(session.brief, slot, getattr(candidate, slot))
        saved.append(slot)
    after = session.brief.values()
    changed = any(before[s] is not None and before[s] != after[s] for s in before)
    if session.pick and changed:
        session.pick = None
    return _dump({"saved": saved, "rejected": rejected, "next_slot": session.brief.next_slot()})


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
