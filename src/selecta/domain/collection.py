"""Records, artists and the listener's ownership of them."""

from datetime import datetime

from pydantic import BaseModel


class Track(BaseModel):
    """One track on a release."""

    position: str
    title: str
    duration_seconds: int | None = None


class Credit(BaseModel):
    """A person or company credited on a release, optionally on specific tracks."""

    name: str
    role: str
    tracks: str = ""


class Release(BaseModel):
    """A specific pressing of a record."""

    id: int
    title: str
    artists: list[str]
    artist_ids: list[int]
    year: int | None = None
    original_year: int | None = None
    label: str | None = None
    country: str | None = None
    format: str | None = None
    genres: list[str] = []
    styles: list[str] = []
    notes: str | None = None
    cover_url: str | None = None
    tracks: list[Track] = []
    credits: list[Credit] = []


class ReleaseSummary(BaseModel):
    """The fields needed to mention or pick a release."""

    id: int
    title: str
    artists: list[str]
    original_year: int | None
    label: str | None
    format: str | None
    genres: list[str]
    styles: list[str]


class Artist(BaseModel):
    """An artist or group with its biography."""

    id: int
    name: str
    members: list[str] = []
    profile: str | None = None


class CollectionItem(BaseModel):
    """A release the listener owns or wants."""

    release_id: int
    owned: bool
    date_added: datetime | None = None


class Collection(BaseModel):
    """Everything known about the listener's records."""

    releases: list[Release]
    artists: list[Artist]
    items: list[CollectionItem]
