"""Web lookups about the listener's records: stored passages first, then the web."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from selecta.application.ports import (
    CollectionStore,
    Reranker,
    WebPassage,
    WebSearch,
    WebSearchError,
)
from selecta.domain.collection import ReleaseSummary

logger = logging.getLogger(__name__)

REFERENCE_DOMAINS = ("wikipedia.org", "allmusic.com", "bandcamp.com", "discogs.com")
IGNORED_DOMAINS = frozenset(
    {
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "youtube.com",
        "tiktok.com",
        "myspace.com",
        "soundcloud.com",
        "imdb.com",
        "web.archive.org",
        "linktr.ee",
        "linkedin.com",
        "spotify.com",
        "apple.com",
    }
)
MAX_PASSAGES_PER_RECORD = 30
MAX_PASSAGES = 5
MIN_RELEVANCE = 0.2


def source_domains(artist_urls: list[str]) -> list[str]:
    """Return the artist's own sites first, then the reference sites."""
    domains: list[str] = []
    for url in artist_urls:
        host = urlparse(url if "//" in url else f"//{url}").hostname or ""
        domain = host.removeprefix("www.")
        covered = any(domain.endswith(f".{d}") for d in REFERENCE_DOMAINS)
        if domain and not covered and not _ignored(domain) and domain not in domains:
            domains.append(domain)
    domains += [d for d in REFERENCE_DOMAINS if d not in domains]
    return domains


class Lookup:
    """Answers about a record from stored web passages, searching the web when they fall short."""

    def __init__(self, store: CollectionStore, web: WebSearch, reranker: Reranker) -> None:
        self._store = store
        self._web = web
        self._reranker = reranker

    def passages(
        self, release: ReleaseSummary, question: str, before_search: Callable[[], None]
    ) -> list[WebPassage]:
        """Return passages that answer the question, best first, each with its URL.

        Raises WebSearchError or RerankError when the answer can't be checked.
        """
        stored = self._relevant(question, self._store.web_passages(release.id))
        if stored:
            positions = [p.position for p in stored if p.position is not None]
            self._store.mark_used(release.id, positions)
            return stored
        before_search()
        found = self._relevant(question, self._search(release, question))
        if found:
            self._store.add_web_passages(release.id, found, MAX_PASSAGES_PER_RECORD)
        return found

    def _relevant(self, question: str, passages: list[WebPassage]) -> list[WebPassage]:
        """Return the passages that answer the question, best first. Raises RerankError."""
        if not passages:
            return []
        ranked = self._reranker.rerank(question, [p.text for p in passages], MAX_PASSAGES)
        return [passages[r.index] for r in ranked if r.score >= MIN_RELEVANCE]

    def _search(self, release: ReleaseSummary, question: str) -> list[WebPassage]:
        """Search the preferred sources and the open web at once, preferred results first."""
        year = f" ({release.original_year})" if release.original_year else ""
        query = f"{', '.join(release.artists)} {release.title}{year} {question}"
        preferred = source_domains(self._store.artist_urls(release.id))
        with ThreadPoolExecutor(max_workers=2) as pool:
            searches = [
                pool.submit(self._web.search, query, preferred, []),
                pool.submit(self._web.search, query, [], sorted(IGNORED_DOMAINS)),
            ]
        results, failures = [], 0
        for search in searches:
            try:
                results += search.result()
            except WebSearchError as e:
                logger.warning("one web search failed: %s", e)
                failures += 1
        if failures == len(searches):
            raise WebSearchError("every web search failed")
        passages: list[WebPassage] = []
        for page in results:
            if all(p.url != page.url for p in passages):
                passages.append(WebPassage(url=page.url, text=f"{page.title}. {page.content}"))
        return passages


def _ignored(domain: str) -> bool:
    return any(domain == d or domain.endswith(f".{d}") for d in IGNORED_DOMAINS)
