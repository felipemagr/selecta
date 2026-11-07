"""The agent's tools."""

import json
import logging
from collections.abc import Callable
from typing import Any, get_args
from urllib.parse import urlparse

from pydantic import ValidationError

from selecta.application.conversation import (
    Language,
    Pick,
    Rejection,
    Session,
    ToolCall,
    ToolSpec,
)
from selecta.application.lookup import Lookup
from selecta.application.picking import MUCH_LONGER_NOTE, Candidates, find_candidates
from selecta.application.ports import (
    CollectionQuery,
    CollectionStore,
    Embedder,
    PlaybackResult,
    RerankError,
    WebSearchError,
)
from selecta.application.retrieval import semantic_search
from selecta.domain.brief import OPTIONAL_SLOTS, SLOT_ORDER, Answer, Brief, Occasion
from selecta.domain.collection import Release
from selecta.domain.enrichment import Energy, Mood
from selecta.domain.library import PlayableTrack, norm
from selecta.domain.picking import MUCH_LONGER_FACTOR, runs_past

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
                "whole_album": _answer(
                    {
                        "type": "boolean",
                        "description": "True if they want the whole album, not just one side.",
                    }
                ),
                "language": {"enum": list(get_args(Language))},
            },
            "required": ["language"],
        },
    ),
    ToolSpec(
        name="skip_questions",
        description=(
            "Stop the interview when the listener wants you to just pick, or is getting "
            "impatient with the questions. Pass their words. The slots not answered yet are "
            "saved as UNKNOWN and the candidates appear."
        ),
        parameters={
            "type": "object",
            "properties": {"evidence": {"type": "string", "description": "The listener's words."}},
            "required": ["evidence"],
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
            "properties": {
                "release_id": {"type": "integer"},
            },
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
        name="look_up",
        description=(
            "Look up something about one of the listener's records that its details don't say: "
            "its story, how it was recorded, how it was received. For who plays on it, the "
            "tracks, the label or the year, use get_release instead. Returns passages with the "
            "URL each comes from."
        ),
        parameters={
            "type": "object",
            "properties": {
                "release_id": {"type": "integer"},
                "question": {
                    "type": "string",
                    "description": "What they want to know, in a few words, in English.",
                },
            },
            "required": ["release_id", "question"],
        },
    ),
    ToolSpec(
        name="pick_record",
        description=(
            "Choose the record for the listener. Call it before you announce a pick: only a pick "
            "this tool accepted may be announced. Usually one of the candidates in the interview "
            "state; any owned record the listener asks for by name, at any point. Only one "
            "record is picked at a time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "release_id": {"type": "integer"},
                "title": {"type": "string", "description": "The record's title, as listed."},
                "reason": {"type": "string", "description": "Why it suits them, in a sentence."},
                "asked_by_name": {
                    "type": "boolean",
                    "description": "True if the listener asked for this record themselves.",
                },
            },
            "required": ["release_id", "title", "reason"],
        },
    ),
    ToolSpec(
        name="reject_pick",
        description="Record that the listener turned down the current pick.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="play_record",
        description=(
            "Play a record on the listener's speakers: the current pick once they want to hear "
            "it, or a record they name, right away and with no questions first. A record not "
            "picked yet is picked first. Use an id from the candidates or from "
            "search_collection. Returns whether it started: say it's playing only after that."
        ),
        parameters={
            "type": "object",
            "properties": {
                "release_id": {"type": "integer"},
                "title": {"type": "string", "description": "The record's title, as listed."},
                "track": {
                    "type": "string",
                    "description": "Where to start: a vinyl position (B1) or a track number (3).",
                },
            },
            "required": ["release_id", "title"],
        },
    ),
]

Play = Callable[[Release, list[PlayableTrack]], PlaybackResult]
NOT_PLAYING = ", not playing yet."


class Tools:
    """Tool executor."""

    def __init__(
        self,
        store: CollectionStore,
        embedder: Embedder,
        lookup: Lookup | None = None,
        music_server: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lookup = lookup
        self._music_server = music_server

    def specs(self, can_play: bool) -> list[ToolSpec]:
        """Return the tools the model may call: look_up and play_record only when set up."""
        unavailable = {"look_up"} if self._lookup is None else set()
        if not (can_play and self._music_server):
            unavailable.add("play_record")
        return [s for s in SPECS if s.name not in unavailable]

    def candidates(self, session: Session) -> Candidates:
        """Return the records the agent may pick from now: playable ones, with a music server."""
        return find_candidates(self._store, session, playable=self._music_server)

    def run(
        self,
        call: ToolCall,
        session: Session,
        before_search: Callable[[], None] = lambda: None,
        play: Play | None = None,
    ) -> str:
        """Return the result of a tool call. Failures are returned too, never raised.

        `before_search` runs right before a slow web search, so the listener can be told to wait.
        `play` sends tracks to the listener's player and returns what the player reported.
        """
        args = call.arguments
        try:
            match call.name:
                case "update_brief":
                    return _update_brief(session, args)
                case "skip_questions":
                    return _skip_questions(session, str(args["evidence"]))
                case "search_collection":
                    hits = self._store.search(CollectionQuery.model_validate(args))
                    return _dump([h.model_dump(mode="json") for h in hits]) if hits else "No match."
                case "get_release":
                    release = self._store.get_release(args["release_id"])
                    return release.model_dump_json() if release else "No such record."
                case "pick_record":
                    asked = args.get("asked_by_name") is True
                    release_id, reason = int(args["release_id"]), args["reason"]
                    return self._pick(session, release_id, reason, asked, args.get("title"))
                case "reject_pick":
                    return _reject(session)
                case "play_record" if play and self._music_server:
                    track = str(args["track"]) if args.get("track") else None
                    release_id, title = int(args["release_id"]), args.get("title")
                    return self._play(session, release_id, play, track, title)
                case "look_up":
                    release_id, question = int(args["release_id"]), str(args["question"])
                    return self._look_up(release_id, question, before_search)
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

    def _play(
        self,
        session: Session,
        release_id: int,
        play: Play,
        track: str | None = None,
        title: str | None = None,
    ) -> str:
        """Play the record from the given track, picking it first if it isn't the current pick."""
        release = self._store.get_release(release_id)
        if release is None or self._store.owned_summary(release_id) is None:
            return "That isn't one of the listener's records. Find it with search_collection."
        if wrong := _wrong_title(release_id, release.artists, release.title, title):
            return wrong
        tracks = self._store.playable(release_id)
        if not tracks:
            return "That record isn't on the music server, only on vinyl. Tell the listener."
        if track:
            start = _track_index(tracks, track)
            if start is None:
                return f"There's no track {track} on the server for that record. Tell the listener."
            tracks = tracks[start:]
        notes = ""
        if session.pick is None or session.pick.release_id != release_id:
            picked = self._pick(session, release_id, "They asked to hear it.", asked_by_name=True)
            if session.pick is None or session.pick.release_id != release_id:
                return picked
            notes = picked.split(NOT_PLAYING, 1)[-1].strip()
        match play(release, tracks):
            case "playing":
                following = f"; next is {_track(tracks[1])}" if len(tracks) > 1 else ""
                name = _name(release.artists, release.title)
                first = _track(tracks[0])
                playing = f"Playing {name}: {len(tracks)} tracks, from {first}{following}."
                return f"{playing} {notes}".strip()
            case "cancelled":
                return "The listener spoke over you, so it wasn't started. Listen to them first."
            case _:
                return "The player didn't start. Tell the listener it can't play right now."

    def _look_up(self, release_id: int, question: str, before_search: Callable[[], None]) -> str:
        if self._lookup is None:
            return "Unknown tool: look_up"
        release = self._store.owned_summary(release_id)
        if release is None:
            return "That record isn't in the listener's collection."
        try:
            passages = self._lookup.passages(release, question, before_search)
        except (WebSearchError, RerankError):
            logger.exception("web lookup failed")
            return "The web search failed. Tell the listener you can't check that right now."
        if not passages:
            return (
                "Nothing found on the web. Who plays on it is in get_release; otherwise say you "
                "don't know."
            )
        found = _dump(
            [
                {"site": urlparse(p.url).hostname or p.url, **p.model_dump(exclude={"position"})}
                for p in passages
            ]
        )
        site = urlparse(passages[0].url).hostname or "the site"
        return (
            f"Answer only from these passages, starting with where it comes from, like "
            f"\"According to {site}, ...\". If they don't answer the question, say you don't "
            f"know.\n{found}"
        )

    def _pick(
        self,
        session: Session,
        release_id: int,
        reason: str,
        asked_by_name: bool,
        title: str | None = None,
    ) -> str:
        release = self._store.owned_summary(release_id)
        if release is None:
            return "That record isn't in the listener's collection. Pick another."
        if wrong := _wrong_title(release_id, release.artists, release.title, title):
            return wrong
        replaced = session.pick
        if not asked_by_name:
            if session.pick:
                return (
                    "A record is already picked. Keep it, unless the listener turned it down: "
                    "then call reject_pick."
                )
            if session.brief.next_slot():
                return "Finish the interview first, unless the listener asked for a record by name."
            if release_id in session.rejected_ids():
                return "The listener already rejected that record. Pick another."
            if release_id not in {r.id for r in self.candidates(session).releases}:
                return "Pick from the candidates, unless the listener asked for a record by name."
        session.pick = Pick(
            release_id=release.id, title=release.title, artists=release.artists, reason=reason
        )
        minutes = session.brief.listening_minutes()
        note = f" It replaces {replaced.title}." if replaced else ""
        picked = f"Picked {_name(release.artists, release.title)}{NOT_PLAYING}{note}"
        if runs_past(release.duration_seconds, minutes, MUCH_LONGER_FACTOR):
            return f"{picked} {MUCH_LONGER_NOTE}"
        return picked


def _wrong_title(release_id: int, artists: list[str], title: str, said: str | None) -> str | None:
    """Return why the id isn't the record the model named, or None if they match."""
    named, actual = set(norm(said or "").split()), set(norm(title).split())
    if not named or not actual or named <= actual or actual <= named:
        return None
    return f"{release_id} is {_name(artists, title)}, not {said}. Use the right id."


def _name(artists: list[str], title: str) -> str:
    return f"{', '.join(artists)} - {title}"


def _track(track: PlayableTrack) -> str:
    return f"{track.position} {track.title}"


def _track_index(tracks: list[PlayableTrack], track: str) -> int | None:
    """Return where to start: a vinyl position (B1) or a track number (3) among those that play."""
    wanted = track.strip().upper()
    for i, t in enumerate(tracks):
        if t.position.upper() == wanted:
            return i
    if wanted.isdigit() and 1 <= int(wanted) <= len(tracks):
        return int(wanted) - 1
    return None


def _reject(session: Session) -> str:
    """Reject the pick, with the listener's last message as evidence rather than the model's."""
    if session.pick is None:
        return "Nothing is picked yet."
    said = next(m.content for m in reversed(session.messages) if m.role == "user" and m.content)
    pick = session.pick
    session.rejections.append(
        Rejection(release_id=pick.release_id, reason=said, title=pick.title, artists=pick.artists)
    )
    session.pick = None
    return "Rejected. The candidates in the interview state are updated."


def _skip_questions(session: Session, evidence: str) -> str:
    """Save every unanswered slot as UNKNOWN and mark the listener as in a hurry."""
    skipped = [slot for slot in SLOT_ORDER if getattr(session.brief, slot) is None]
    for slot in skipped:
        setattr(session.brief, slot, Answer(value="UNKNOWN", evidence=evidence))
    session.hurried = True
    return _dump({"skipped": skipped, "next_slot": None})


def _update_brief(session: Session, args: dict[str, Any]) -> str:
    """Save each valid slot; one bad value doesn't drop the others."""
    if args.get("language") in get_args(Language):
        session.language = args["language"]
    before = session.brief.values()
    saved, rejected = [], {}
    for slot in (*SLOT_ORDER, *OPTIONAL_SLOTS):
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
