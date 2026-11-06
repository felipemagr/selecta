"""Cohere reranking over the HTTP API."""

import httpx

from selecta.application.ports import Ranked, RerankError
from selecta.infrastructure import config


def connect(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return an HTTP client authenticated against the Cohere API."""
    return httpx.Client(
        base_url=config.COHERE.API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config.COHERE.TIMEOUT_SECONDS,
        transport=transport,
    )


class CohereReranker:
    """Reranker on Cohere's rerank endpoint."""

    def __init__(self, http: httpx.Client, model: str = config.COHERE.RERANK_MODEL) -> None:
        self._http = http
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranked]:
        """Return the best `top_n` documents, best first."""
        body = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        # No retries: the listener is waiting.
        try:
            response = self._http.post("/v2/rerank", json=body)
            response.raise_for_status()
            return [
                Ranked(index=r["index"], score=r["relevance_score"])
                for r in response.json()["results"]
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            raise RerankError(f"Cohere rerank failed: {e}") from e
