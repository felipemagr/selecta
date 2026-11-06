"""Tavily web search over the HTTP API."""

import httpx

from selecta.application.ports import WebPage, WebSearchError
from selecta.infrastructure import config


def connect(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return an HTTP client authenticated against the Tavily API."""
    return httpx.Client(
        base_url=config.TAVILY.API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config.TAVILY.TIMEOUT_SECONDS,
        transport=transport,
    )


class TavilySearch:
    """WebSearch on Tavily's search endpoint."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def search(
        self, query: str, include_domains: list[str], exclude_domains: list[str]
    ) -> list[WebPage]:
        """Return the passages that best match `query`, dropping any without a URL."""
        body = {
            "query": query,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "max_results": config.TAVILY.MAX_RESULTS,
            "search_depth": "basic",
        }
        # No retries: the listener is waiting.
        try:
            response = self._http.post("/search", json=body)
            response.raise_for_status()
            results = response.json()["results"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise WebSearchError(f"Tavily search failed: {e}") from e
        return [
            WebPage(
                url=r["url"],
                title=r.get("title") or r["url"],
                content=(r.get("content") or "")[: config.TAVILY.MAX_CONTENT_CHARS],
            )
            for r in results
            if r.get("url")
        ]
