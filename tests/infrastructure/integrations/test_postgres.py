from datetime import UTC, datetime

from selecta.application.conversation import Message, Session
from selecta.application.ports import Chunk, CollectionQuery, WebPassage
from selecta.domain.brief import Answer, Brief
from selecta.domain.collection import Artist, Collection, CollectionItem, Credit, Release, Track
from selecta.domain.enrichment import Enrichment
from selecta.infrastructure.integrations.postgres import PostgresStore


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


def _one_hot(i: int) -> list[float]:
    return [1.0 if d == i else 0.0 for d in range(1024)]


def test_search_filters_by_enriched_mood(store: PostgresStore) -> None:
    store.replace(_collection(_release(1, "A", 1970), _release(2, "B", 1980)))
    store.save_enrichment(1, Enrichment(mood="calm", energy="low"), "h")

    hits = store.search(CollectionQuery(mood="calm"))

    assert [(r.id, r.mood) for r in hits] == [(1, "calm")]


def test_semantic_search_fuses_vector_and_keyword_matches(store: PostgresStore) -> None:
    texts = {1: "a quiet piano record", 2: "late night jazz", 3: "folk with accordion"}
    store.save_chunks(
        [
            Chunk(kind="release", ref_id=i, text=t, text_hash="h", embedding=_one_hot(i))
            for i, t in texts.items()
        ]
    )

    hits = store.semantic_search("accordion", _one_hot(2), limit=3)

    assert {p.ref_id for p in hits[:2]} == {2, 3}


def test_migrations_apply_once(store: PostgresStore) -> None:
    store.replace(_collection(_release(1, "A", 1970)))

    reopened = PostgresStore(store._conn)

    assert [r.id for r in reopened.search(CollectionQuery())] == [1]


def test_session_round_trips(store: PostgresStore) -> None:
    session = Session(
        id="s1",
        started_at=datetime(2026, 11, 5, tzinfo=UTC),
        brief=Brief(mood=Answer(value="calm", evidence="something calm")),
        messages=[Message(role="user", content="something calm")],
    )

    store.save_session(session)

    assert store.load_session("s1") == session


def test_duration_is_known_only_when_every_track_has_one(store: PostgresStore) -> None:
    full = _release(1, "A", 1970)
    tracks = [Track(position="A1", title="x", duration_seconds=60), Track(position="A2", title="y")]
    partial = _release(2, "B", 1970).model_copy(update={"tracks": tracks})
    store.replace(_collection(full, partial))

    durations = {r.id: r.duration_seconds for r in store.search(CollectionQuery())}

    assert durations == {1: 300, 2: None}


def test_owned_summaries_filter_by_mood_and_skip_the_wantlist(store: PostgresStore) -> None:
    store.replace(
        _collection(_release(1, "A", 1970), _release(2, "B", 1970), wanted=_release(3, "C", 1970))
    )
    store.save_enrichment(1, Enrichment(mood="calm", energy="low"), "h")
    store.save_enrichment(3, Enrichment(mood="calm", energy="low"), "h")

    assert [r.id for r in store.owned_summaries(mood="calm")] == [1]
    assert [r.id for r in store.owned_summaries()] == [1, 2]
    assert store.owned_summary(3) is None


def _web(url: str) -> WebPassage:
    return WebPassage(url=url, text=f"Page {url}.")


def test_web_passages_keep_the_most_used_and_skip_duplicates(store: PostgresStore) -> None:
    collection = _collection(_release(1, "A", 1970))
    collection.artists = [Artist(id=1, name="A", urls=["https://a.example"])]
    store.replace(collection)
    store.add_web_passages(1, [_web("u0"), _web("u1"), _web("u2")], keep=3)
    store.mark_used(1, [0, 2])

    store.add_web_passages(1, [_web("u2"), _web("u3")], keep=3)

    assert store.artist_urls(1) == ["https://a.example"]
    assert [p.url for p in store.web_passages(1)] == ["u0", "u2", "u3"]


def test_web_passages_survive_reindex_until_the_record_leaves(store: PostgresStore) -> None:
    store.replace(_collection(_release(1, "A", 1970)))
    store.add_web_passages(1, [_web("u0")], keep=5)

    store.delete_chunks_except(set())
    kept = store.web_passages(1)
    store.replace(_collection(_release(2, "B", 1970)))

    assert len(kept) == 1
    assert store.web_passages(1) == []
