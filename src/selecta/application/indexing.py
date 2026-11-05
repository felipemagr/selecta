"""Loading a collection into the store."""

import logging

from selecta.application.ports import CollectionSource, CollectionStore

logger = logging.getLogger(__name__)


def index_collection(source: CollectionSource, store: CollectionStore, household_id: str) -> None:
    """Replace the household's stored collection with the source's current one."""
    collection = source.load()
    store.replace(household_id, collection)
    logger.info(
        "indexed %d releases and %d artists", len(collection.releases), len(collection.artists)
    )
