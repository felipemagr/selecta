"""Interfaces the application needs from the outside world."""

from typing import Literal, Protocol

from pydantic import BaseModel

from selecta.domain.collection import Collection, Release, ReleaseSummary
from selecta.domain.enrichment import Energy, Enrichment, Mood


class CollectionQuery(BaseModel):
    """Filters for searching the collection. Unset filters match everything."""

    text: str | None = None
    genre: str | None = None
    style: str | None = None
    label: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    mood: Mood | None = None
    energy: Energy | None = None
    owned: bool = True
    sort: Literal["relevance", "oldest", "newest", "recently_added"] = "relevance"
    limit: int = 10


class ModelOutputError(Exception):
    """A model answered with something that does not match the requested schema."""


ChunkKind = Literal["release", "artist"]


class Chunk(BaseModel):
    """A piece of text about a release or artist, embedded for semantic search."""

    kind: ChunkKind
    ref_id: int
    text: str
    text_hash: str
    embedding: list[float]


class Passage(BaseModel):
    """A chunk returned by semantic search, best match first."""

    kind: ChunkKind
    ref_id: int
    text: str
    score: float


class CollectionSource(Protocol):
    """Where the listener's collection comes from."""

    def load(self) -> Collection:
        """Return the full collection."""
        ...


class CollectionStore(Protocol):
    """Where collections are kept and searched."""

    def replace(self, collection: Collection) -> None:
        """Make the stored collection match `collection` exactly."""
        ...

    def search(self, query: CollectionQuery) -> list[ReleaseSummary]:
        """Return releases matching the query."""
        ...

    def get_release(self, release_id: int) -> Release | None:
        """Return a release with tracks and credits, or None if unknown."""
        ...

    def enrichment_hashes(self) -> dict[int, str]:
        """Return the input hash each enriched release was enriched from."""
        ...

    def save_enrichment(self, release_id: int, enrichment: Enrichment, input_hash: str) -> None:
        """Store a release's enrichment and the hash of the input it came from."""
        ...

    def chunk_hashes(self) -> dict[tuple[ChunkKind, int], str]:
        """Return the text hash of every stored chunk."""
        ...

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks."""
        ...

    def delete_chunks_except(self, keep: set[tuple[ChunkKind, int]]) -> None:
        """Delete every chunk not in `keep`."""
        ...

    def semantic_search(self, text: str, embedding: list[float], limit: int) -> list[Passage]:
        """Return the chunks closest to the query, by meaning and by keywords."""
        ...


class ChatModel(Protocol):
    """An LLM that answers with JSON matching a schema."""

    model: str

    def complete_json[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        """Return the model's answer to `user`, validated against `schema`."""
        ...


class Embedder(Protocol):
    """A text embedding model."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, in order."""
        ...
