import httpx
import pytest

from selecta.application.ports import Ranked, RerankError
from selecta.infrastructure.integrations import cohere
from selecta.infrastructure.integrations.cohere import CohereReranker


def _reranker(response: httpx.Response) -> CohereReranker:
    return CohereReranker(cohere.connect("key", httpx.MockTransport(lambda _: response)))


def test_results_keep_the_input_index_and_score() -> None:
    body = {"results": [{"index": 2, "relevance_score": 0.3}, {"index": 0, "relevance_score": 0.1}]}

    ranked = _reranker(httpx.Response(200, json=body)).rerank("q", ["a", "b", "c"], 2)

    assert ranked == [Ranked(index=2, score=0.3), Ranked(index=0, score=0.1)]


@pytest.mark.parametrize(
    "response", [httpx.Response(429), httpx.Response(200, json={})], ids=["429", "bad"]
)
def test_failure_raises_rerank_error(response: httpx.Response) -> None:
    with pytest.raises(RerankError):
        _reranker(response).rerank("q", ["a"], 1)
