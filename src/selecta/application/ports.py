"""Interfaces the application needs from the outside world."""

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from selecta.application.conversation import Message, Reply, Session, ToolSpec
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
    limit: int = Field(default=10, ge=1, le=20)


class ModelError(Exception):
    """A model call failed."""


class ModelOutputError(ModelError):
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


class WebPassage(BaseModel):
    """A passage about a record found on the web, with its source. Stored ones have a position."""

    url: str
    text: str
    position: int | None = None


class WebPage(BaseModel):
    """A web page found for a query, with the passage that matched it."""

    url: str
    title: str
    content: str


class WebSearchError(Exception):
    """A web search failed."""


class Ranked(BaseModel):
    """A document's place in a reranking: its index in the input and how well it answers."""

    index: int
    score: float


class RerankError(Exception):
    """A reranking failed."""


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

    def owned_summaries(
        self, mood: Mood | None = None, energy: Energy | None = None
    ) -> list[ReleaseSummary]:
        """Return every owned release with the given mood and energy, unset meaning any."""
        ...

    def owned_summary(self, release_id: int) -> ReleaseSummary | None:
        """Return an owned release, or None if it isn't owned."""
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

    def artist_urls(self, release_id: int) -> list[str]:
        """Return the links Discogs lists for the release's artists."""
        ...

    def web_passages(self, release_id: int) -> list[WebPassage]:
        """Return every stored web passage about the release."""
        ...

    def add_web_passages(self, release_id: int, passages: list[WebPassage], keep: int) -> None:
        """Store new web passages about a release, then drop the least used beyond `keep`."""
        ...

    def mark_used(self, release_id: int, positions: list[int]) -> None:
        """Count one more use of each stored web passage."""
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


class AgentModel(Protocol):
    """An LLM with tool calling."""

    model: str

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        """Return the next reply."""
        ...


class WebSearch(Protocol):
    """A web search engine."""

    def search(
        self, query: str, include_domains: list[str], exclude_domains: list[str]
    ) -> list[WebPage]:
        """Return the passages that best match `query`. Raises WebSearchError.

        With `include_domains` empty, the whole web is searched, minus `exclude_domains`.
        """
        ...


class Reranker(Protocol):
    """A model that scores how well each document answers a query."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranked]:
        """Return the best `top_n` documents, best first. Raises RerankError."""
        ...


class SessionStore(Protocol):
    """Where conversations are kept."""

    def save_session(self, session: Session) -> None:
        """Insert or replace a session."""
        ...

    def load_session(self, session_id: str) -> Session | None:
        """Return a session, or None if unknown."""
        ...
