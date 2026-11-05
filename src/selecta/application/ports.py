"""Interfaces the application needs from the outside world."""

from typing import Literal, Protocol

from pydantic import BaseModel

from selecta.domain.collection import Collection, Release, ReleaseSummary


class CollectionQuery(BaseModel):
    """Filters for searching a household's records. Unset filters match everything."""

    text: str | None = None
    genre: str | None = None
    style: str | None = None
    label: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    owned: bool = True
    sort: Literal["relevance", "oldest", "newest", "recently_added"] = "relevance"
    limit: int = 10


class CollectionSource(Protocol):
    """Where a household's collection comes from."""

    def load(self) -> Collection:
        """Return the full collection."""
        ...


class CollectionStore(Protocol):
    """Where collections are kept and searched."""

    def replace(self, household_id: str, collection: Collection) -> None:
        """Make the stored collection match `collection` exactly."""
        ...

    def search(self, household_id: str, query: CollectionQuery) -> list[ReleaseSummary]:
        """Return releases matching the query."""
        ...

    def get_release(self, household_id: str, release_id: int) -> Release | None:
        """Return one of the household's releases with tracks and credits, or None."""
        ...
