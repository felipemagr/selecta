"""The eval world: the synthetic sample collection in memory, a player and the web, all fake."""

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from selecta.application.agent import Agent
from selecta.application.conversation import Message, Reply, Session, ToolSpec
from selecta.application.lookup import Lookup
from selecta.application.ports import (
    CollectionQuery,
    Passage,
    Ranked,
    WebPage,
    WebPassage,
)
from selecta.application.tools import Tools
from selecta.domain.collection import Collection, Release, ReleaseSummary
from selecta.domain.library import PlayableTrack

ROOT = Path(__file__).parents[2]
COLLECTION = Collection.model_validate_json((ROOT / "examples" / "collection.json").read_text())
RELEASES = {r.id: r for r in COLLECTION.releases}
TAGS = json.loads((Path(__file__).parent / "tags.json").read_text())
NOW = datetime(2026, 11, 10, 21, 30, tzinfo=UTC)

# What the web knows: stored passages for a few records, with their source.
WEB = {
    1001: [
        WebPassage(
            url="https://folk-archive.example/low-lanterns",
            text="Harbour Lights was recorded in a converted chapel and was praised by critics for "
            "its harmonies; it reached the top 20 in 1971.",
            position=0,
        )
    ],
    1007: [
        WebPassage(
            url="https://jazz-notes.example/theo-marsh",
            text="Late Set at the Corner captures the trio on a Sunday night; critics called Lou "
            "Barretto's brushwork the quiet star of the record.",
            position=0,
        )
    ],
}


def fold(text: str) -> str:
    """Return text lowercased and without accents, for loose matching."""
    text = unicodedata.normalize("NFKD", text.replace("\u2019", "'"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


class World:
    """The listener's collection and sessions, in memory. Honours the store contracts."""

    def __init__(self) -> None:
        self.releases = RELEASES

    # Collection store.

    def _summary(self, release: Release) -> ReleaseSummary:
        tags = TAGS[str(release.id)]
        return ReleaseSummary(
            id=release.id,
            title=release.title,
            artists=release.artists,
            original_year=release.original_year,
            label=release.label,
            format=release.format,
            genres=release.genres,
            styles=release.styles,
            mood=tags["mood"],
            energy=tags["energy"],
            duration_seconds=release.total_seconds(),
            track_count=len(release.tracks),
            playable_count=len(tags["playable"]),
        )

    def owned_summaries(
        self, mood: str | None = None, energy: str | None = None, playable: bool = False
    ) -> list[ReleaseSummary]:
        summaries = [self._summary(r) for r in self.releases.values()]
        return [
            s
            for s in summaries
            if mood in (None, s.mood)
            and energy in (None, s.energy)
            and (s.playable_count or not playable)
        ]

    def owned_summary(self, release_id: int) -> ReleaseSummary | None:
        release = self.releases.get(release_id)
        return self._summary(release) if release else None

    def get_release(self, release_id: int) -> Release | None:
        return self.releases.get(release_id)

    def search(self, query: CollectionQuery) -> list[ReleaseSummary]:
        found = []
        for release in self.releases.values():
            summary = self._summary(release)
            people = [c.name for c in release.credits]
            titles = [t.title for t in release.tracks]
            text = fold(" ".join([*release.artists, release.title, *people, *titles]))
            if query.text and not all(w in text for w in fold(query.text).split()):
                continue
            if query.genre and fold(query.genre) not in fold(" ".join(release.genres)):
                continue
            if query.style and fold(query.style) not in fold(" ".join(release.styles)):
                continue
            if query.label and fold(query.label) not in fold(release.label or ""):
                continue
            year = release.original_year or 0
            if (query.year_from and year < query.year_from) or (
                query.year_to and year > query.year_to
            ):
                continue
            if query.mood not in (None, summary.mood) or query.energy not in (None, summary.energy):
                continue
            found.append(summary)
        if query.sort in ("oldest", "newest"):
            found.sort(key=lambda s: s.original_year or 0, reverse=query.sort == "newest")
        return found[: query.limit]

    def semantic_search(self, text: str, embedding: list[float], limit: int) -> list[Passage]:
        """Score each record and artist profile by the words it shares with the question."""
        words = _words(text)
        passages = [
            Passage(kind="release", ref_id=r.id, text=_describe(r), score=0.0)
            for r in self.releases.values()
        ] + [
            Passage(kind="artist", ref_id=a.id, text=f"{a.name}. {a.profile}", score=0.0)
            for a in COLLECTION.artists
        ]
        for p in passages:
            p.score = len(words & _words(p.text)) / (len(words) or 1)
        return sorted(passages, key=lambda p: p.score, reverse=True)[:limit]

    def playable(self, release_id: int) -> list[PlayableTrack]:
        return playable(release_id)

    def artist_urls(self, release_id: int) -> list[str]:
        return []

    def web_passages(self, release_id: int) -> list[WebPassage]:
        return WEB.get(release_id, [])

    def add_web_passages(self, release_id: int, passages: list[WebPassage], keep: int) -> None:
        pass

    def mark_used(self, release_id: int, positions: list[int]) -> None:
        pass

    # Session store.

    def save_session(self, session: Session) -> None:
        pass

    def sessions_since(self, since: datetime) -> list[Session]:
        return []


class Player:
    """A player that starts, or fails, as the scenario says, and remembers what it was asked."""

    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.played: list[tuple[int, list[str]]] = []

    def __call__(
        self, release: Release, tracks: list[PlayableTrack]
    ) -> Literal["playing", "failed"]:
        self.played.append((release.id, [t.position for t in tracks]))
        return "playing" if self.works else "failed"


class NoWeb:
    """A web search that finds nothing beyond the stored passages."""

    def search(
        self, query: str, include_domains: list[str], exclude_domains: list[str]
    ) -> list[WebPage]:
        return []


class SharedWords:
    """A reranker that keeps the passages sharing a word with the question."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranked]:
        words = {w for w in _words(query) if len(w) > 3}
        scored = [
            Ranked(index=i, score=0.9 if words & _words(d) else 0.0)
            for i, d in enumerate(documents)
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)[:top_n]


class NoEmbeddings:
    """Embeddings aren't needed: the world matches questions by words."""

    model = "none"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class Scripted:
    """A model that always says the same: for checking the harness without spending."""

    model = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: Any = None
    ) -> Reply:
        if on_text:
            on_text(self.text)
        return Reply(text=self.text)


def build_agent(model: Any) -> tuple[Agent, World]:
    """Return the real agent wired to a fresh world, as the server wires it to the real one."""
    world = World()
    lookup = Lookup(world, NoWeb(), SharedWords())  # type: ignore[arg-type]
    tools = Tools(world, NoEmbeddings(), lookup)  # type: ignore[arg-type]
    return Agent(model, tools, world, now=lambda: NOW), world  # type: ignore[arg-type]


def playable(release_id: int) -> list[PlayableTrack]:
    """Return the record's tracks on the fake music server."""
    release = RELEASES.get(release_id)
    if release is None:
        return []
    positions = TAGS[str(release_id)]["playable"]
    return [
        PlayableTrack(
            position=t.position,
            title=t.title,
            song_id=f"{release_id}-{t.position}",
            duration_seconds=t.duration_seconds,
        )
        for t in release.tracks
        if t.position in positions
    ]


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", fold(text)))


def _describe(release: Release) -> str:
    year = f" ({release.original_year})" if release.original_year else ""
    styles = ", ".join([*release.genres, *release.styles])
    return f"{', '.join(release.artists)} - {release.title}{year}. {styles}. {release.notes or ''}"
