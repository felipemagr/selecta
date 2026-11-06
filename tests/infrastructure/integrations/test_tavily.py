import httpx
import pytest

from selecta.application.ports import WebSearchError
from selecta.infrastructure.integrations import tavily
from selecta.infrastructure.integrations.tavily import TavilySearch


def _search(response: httpx.Response) -> TavilySearch:
    return TavilySearch(tavily.connect("key", httpx.MockTransport(lambda _: response)))


def test_results_without_url_are_dropped() -> None:
    results = [
        {"url": "https://en.wikipedia.org/wiki/X", "title": "X", "content": "Recorded in 1970."},
        {"url": "", "title": "No source", "content": "Unsourced claim."},
    ]

    search = _search(httpx.Response(200, json={"results": results}))

    pages = search.search("x", ["wikipedia.org"], [])

    assert [p.url for p in pages] == ["https://en.wikipedia.org/wiki/X"]


@pytest.mark.parametrize(
    "response", [httpx.Response(429), httpx.Response(200, text="not json")], ids=["429", "bad"]
)
def test_failure_raises_web_search_error(response: httpx.Response) -> None:
    with pytest.raises(WebSearchError):
        _search(response).search("x", [], [])
