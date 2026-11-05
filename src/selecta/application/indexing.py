"""Loading a collection into the store."""

import logging

from selecta.application.ports import CollectionSource, CollectionStore

logger = logging.getLogger(__name__)


def index_collection(source: CollectionSource, store: CollectionStore) -> None:
    """Replace the stored collection with the source's current one."""
    collection = source.load()
    store.replace(collection)
    logger.info(
        "indexed %d releases and %d artists", len(collection.releases), len(collection.artists)
    )
