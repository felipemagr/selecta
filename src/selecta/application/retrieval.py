"""Searching the collection by meaning."""

from selecta.application.ports import CollectionStore, Embedder, Passage


def semantic_search(
    store: CollectionStore, embedder: Embedder, text: str, limit: int = 5
) -> list[Passage]:
    """Return the passages about releases and artists that best match `text`."""
    [embedding] = embedder.embed([text])
    return store.semantic_search(text, embedding, limit)
