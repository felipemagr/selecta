import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import dict_row

from selecta.application.ports import CollectionQuery
from selecta.domain.collection import Collection, CollectionItem, Credit, Release, Track
from selecta.infrastructure import config
from selecta.infrastructure.integrations.postgres import PostgresStore


@pytest.fixture
def store() -> Iterator[PostgresStore]:
    if not config.DATABASE.URL:
        pytest.skip("DATABASE_URL is not set")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(config.DATABASE.URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("no database, run `make db`")
    admin.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public")
    admin.execute(f"CREATE SCHEMA {schema}")
    conn = psycopg.connect(
        config.DATABASE.URL,
        autocommit=True,
        row_factory=dict_row,
        options=f"-c search_path={schema},public",
    )
    yield PostgresStore(conn)
    conn.close()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def _release(id: int, artist: str, year: int, original_year: int | None = None) -> Release:
    return Release(
        id=id,
        title=f"Album {id}",
        artists=[artist],
        artist_ids=[id],
        year=year,
        original_year=original_year or year,
        genres=["Electronic"],
        tracks=[Track(position="A1", title="Intro", duration_seconds=300)],
        credits=[Credit(name="Someone", role="Drums")],
    )


def _collection(*releases: Release, wanted: Release | None = None) -> Collection:
    items = [CollectionItem(release_id=r.id, owned=True) for r in releases]
    if wanted:
        items.append(CollectionItem(release_id=wanted.id, owned=False))
    all_releases = [*releases, *([wanted] if wanted else [])]
    return Collection(releases=all_releases, artists=[], items=items)


def test_text_search_tolerates_misspelled_names(store: PostgresStore) -> None:
    releases = _release(1, "Kraftwerk", 1978), _release(2, "Dire Straits", 1979)
    store.replace(_collection(*releases))

    hits = store.search(CollectionQuery(text="craftwerk"))

    assert [r.id for r in hits] == [1]


def test_oldest_sorts_by_original_year_not_reissue(store: PostgresStore) -> None:
    reissue = _release(1, "A", year=2020, original_year=1960)
    store.replace(_collection(reissue, _release(2, "B", 1970)))

    hits = store.search(CollectionQuery(sort="oldest", limit=1))

    assert [r.id for r in hits] == [1]


def test_search_excludes_wantlist_unless_asked(store: PostgresStore) -> None:
    store.replace(_collection(_release(1, "A", 1970), wanted=_release(2, "B", 1980)))

    owned = store.search(CollectionQuery())
    wanted = store.search(CollectionQuery(owned=False))

    assert ([r.id for r in owned], [r.id for r in wanted]) == ([1], [2])


def test_replace_removes_records_no_longer_in_collection(store: PostgresStore) -> None:
    store.replace(_collection(_release(1, "A", 1970), _release(2, "B", 1980)))

    store.replace(_collection(_release(1, "A", 1970)))

    assert [r.id for r in store.search(CollectionQuery())] == [1]
    assert store.get_release(2) is None


def test_get_release_returns_tracks_and_credits(store: PostgresStore) -> None:
    release = _release(1, "A", 1970)
    store.replace(_collection(release))

    assert store.get_release(1) == release
