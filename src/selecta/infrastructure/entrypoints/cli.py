"""Command-line entry points."""

import logging
import sys

from selecta.application.indexing import index_collection
from selecta.infrastructure import config
from selecta.infrastructure.integrations.discogs import DiscogsError, DiscogsExport
from selecta.infrastructure.integrations.postgres import PostgresStore


def index() -> None:
    """Load the Discogs export into the database."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = PostgresStore.connect(config.DATABASE.URL)
    try:
        index_collection(DiscogsExport(config.DISCOGS.OUT_DIR), store, config.HOUSEHOLD.ID)
    except DiscogsError as e:
        sys.exit(str(e))
    finally:
        store.close()
