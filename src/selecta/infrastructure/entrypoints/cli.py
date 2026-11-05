"""The `selecta` command."""

import argparse
import logging
import sys

import psycopg

from selecta.application.indexing import index_collection
from selecta.infrastructure import config
from selecta.infrastructure.integrations.discogs import (
    DiscogsClient,
    DiscogsError,
    DiscogsExport,
    export,
)
from selecta.infrastructure.integrations.postgres import PostgresStore


def _export() -> None:
    if not config.DISCOGS.TOKEN:
        sys.exit("DISCOGS_TOKEN is not set, see .env.example")
    with DiscogsClient(config.DISCOGS.TOKEN) as client:
        export(client, config.DISCOGS.OUT_DIR)


def _index() -> None:
    if not config.DATABASE.URL:
        sys.exit("DATABASE_URL is not set, see .env.example")
    store = PostgresStore.connect(config.DATABASE.URL)
    try:
        index_collection(DiscogsExport(config.DISCOGS.OUT_DIR), store)
    finally:
        store.close()


def _sync() -> None:
    _export()
    _index()


def main() -> None:
    """Run a `selecta` subcommand."""
    parser = argparse.ArgumentParser(prog="selecta")
    commands = parser.add_subparsers(required=True)
    for name, run, text in [
        ("export", _export, "fetch the collection from Discogs into the data directory"),
        ("index", _index, "load the exported collection into the database"),
        ("sync", _sync, "export, then index"),
    ]:
        commands.add_parser(name, help=text).set_defaults(run=run)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        args.run()
    except DiscogsError as e:
        sys.exit(str(e))
    except psycopg.OperationalError as e:
        sys.exit(f"database unavailable: {e}")
