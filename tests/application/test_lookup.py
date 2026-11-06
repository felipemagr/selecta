import pytest

from selecta.application.lookup import MIN_RELEVANCE, REFERENCE_DOMAINS, Lookup, source_domains
from selecta.application.ports import Ranked, RerankError, WebPage, WebPassage, WebSearchError
from selecta.domain.collection import ReleaseSummary

RELEASE = ReleaseSummary(
    id=1,
    title="Album",
    artists=["Band"],
    original_year=1970,
    label=None,
    format=None,
    genres=[],
    styles=[],
)


class FakeStore:
    def __init__(self, *stored: WebPassage) -> None:
        self.stored = list(stored)
        self.added: list[WebPassage] = []
        self.used: list[int] = []

    def web_passages(self, release_id: int) -> list[WebPassage]:
        return self.stored

    def artist_urls(self, release_id: int) -> list[str]:
        return ["https://band.example"]

    def add_web_passages(self, release_id: int, passages: list[WebPassage], keep: int) -> None:
        self.added += passages

    def mark_used(self, release_id: int, positions: list[int]) -> None:
        self.used += positions


class FakeReranker:
    """Scores stored passages with `stored`, fresh web results with `fresh`."""

    def __init__(self, stored: float = 0, fresh: float = 1, fails: bool = False) -> None:
        self.scores = {"stored": stored, "fresh": fresh}
        self.fails = fails

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranked]:
        if self.fails:
            raise RerankError("down")
        scores = [self.scores["stored" if d.startswith("Stored") else "fresh"] for d in documents]
        ranked = sorted(enumerate(scores), key=lambda s: -s[1])[:top_n]
        return [Ranked(index=i, score=s) for i, s in ranked]


class FakeWeb:
    def __init__(self, preferred: list[str], open_web: list[str], *failing: str) -> None:
        self.results = {"preferred": preferred, "open": open_web}
        self.failing = set(failing)
        self.searches = 0

    def search(self, query: str, include: list[str], exclude: list[str]) -> list[WebPage]:
        self.searches += 1
        which = "preferred" if include else "open"
        if which in self.failing:
            raise WebSearchError(which)
        return [WebPage(url=u, title="Fresh", content="text") for u in self.results[which]]


def _stored(position: int) -> WebPassage:
    return WebPassage(url=f"https://s{position}", text="Stored passage.", position=position)


def _look_up(
    store: FakeStore, web: FakeWeb, reranker: FakeReranker
) -> tuple[list[WebPassage], list[str]]:
    waits: list[str] = []
    lookup = Lookup(store, web, reranker)  # type: ignore[arg-type]
    return lookup.passages(RELEASE, "how was it recorded", lambda: waits.append("wait")), waits


def test_relevant_stored_passages_answer_without_searching() -> None:
    store = FakeStore(_stored(0), _stored(1))
    web = FakeWeb(["https://new"], [])

    passages, waits = _look_up(store, web, FakeReranker(stored=MIN_RELEVANCE))

    assert [p.url for p in passages] == ["https://s0", "https://s1"]
    assert store.used == [0, 1]
    assert waits == [] and web.searches == 0


def test_search_when_stored_passages_fall_short_preferred_first() -> None:
    store = FakeStore(_stored(0))
    web = FakeWeb(["https://wiki", "https://both"], ["https://both", "https://blog"])

    passages, waits = _look_up(store, web, FakeReranker(stored=MIN_RELEVANCE - 0.01))

    assert [p.url for p in passages] == ["https://wiki", "https://both", "https://blog"]
    assert waits == ["wait"]
    assert store.added == passages


def test_irrelevant_web_results_are_neither_returned_nor_stored() -> None:
    store = FakeStore()

    passages, _ = _look_up(store, FakeWeb(["https://wiki"], []), FakeReranker(fresh=0.1))

    assert passages == [] and store.added == []


def test_reranker_down_raises_before_searching() -> None:
    web = FakeWeb(["https://wiki"], [])

    with pytest.raises(RerankError):
        _look_up(FakeStore(_stored(0)), web, FakeReranker(fails=True))

    assert web.searches == 0


def test_one_failed_search_still_answers_from_the_other() -> None:
    web = FakeWeb([], ["https://blog"], "preferred")

    passages, _ = _look_up(FakeStore(), web, FakeReranker())

    assert [p.url for p in passages] == ["https://blog"]


def test_every_search_failing_raises() -> None:
    with pytest.raises(WebSearchError):
        _look_up(FakeStore(), FakeWeb([], [], "preferred", "open"), FakeReranker())


def test_artist_sites_come_first_without_social_media_or_duplicates() -> None:
    urls = [
        "https://www.robotsband.com/",
        "https://en.wikipedia.org/wiki/Robotsband",
        "https://www.facebook.com/robotsband",
        "https://m.facebook.com/robotsband",
        "https://open.spotify.com/artist/x",
        "http://robotsband.com:8080/concerts",
        "allmusic.com/artist/robotsband",
    ]

    assert source_domains(urls) == [
        "robotsband.com",
        "allmusic.com",
        "wikipedia.org",
        "bandcamp.com",
        "discogs.com",
    ]


def test_artist_without_links_gets_the_reference_sites() -> None:
    assert source_domains([]) == list(REFERENCE_DOMAINS)
