"""The `selecta` command."""

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from selecta.application.agent import Agent
from selecta.application.conversation import Session
from selecta.application.indexing import index_collection
from selecta.application.retrieval import semantic_search
from selecta.application.tools import Tools
from selecta.infrastructure import config
from selecta.infrastructure.integrations import mistral
from selecta.infrastructure.integrations.discogs import (
    DiscogsClient,
    DiscogsError,
    DiscogsExport,
    export,
)
from selecta.infrastructure.integrations.mistral import MistralChat, MistralEmbedder, MistralError
from selecta.infrastructure.integrations.postgres import PostgresStore

logger = logging.getLogger(__name__)


def _require(name: str, value: str | None) -> str:
    if not value:
        sys.exit(f"{name} is not set, see .env.example")
    return value


@contextmanager
def _store() -> Iterator[PostgresStore]:
    store = PostgresStore.connect(_require("DATABASE_URL", config.DATABASE.URL))
    try:
        yield store
    finally:
        store.close()


def _export(_: argparse.Namespace) -> None:
    with DiscogsClient(_require("DISCOGS_TOKEN", config.DISCOGS.TOKEN)) as client:
        export(client, config.DISCOGS.OUT_DIR)


def _index(_: argparse.Namespace) -> None:
    api_key = _require("MISTRAL_API_KEY", config.MISTRAL.API_KEY)
    with _store() as store, mistral.connect(api_key) as http:
        source = DiscogsExport(config.DISCOGS.OUT_DIR)
        index_collection(source, store, MistralChat(http), MistralEmbedder(http))


def _sync(args: argparse.Namespace) -> None:
    _require("MISTRAL_API_KEY", config.MISTRAL.API_KEY)
    _export(args)
    _index(args)


def _search(args: argparse.Namespace) -> None:
    api_key = _require("MISTRAL_API_KEY", config.MISTRAL.API_KEY)
    with _store() as store, mistral.connect(api_key) as http:
        for p in semantic_search(store, MistralEmbedder(http), args.text, args.limit):
            print(f"{p.score:.4f}  {p.kind} {p.ref_id}  {p.text[:160]}")


def _chat(_: argparse.Namespace) -> None:
    api_key = _require("MISTRAL_API_KEY", config.MISTRAL.API_KEY)
    with _store() as store, mistral.connect(api_key) as http:
        model = MistralChat(http, config.MISTRAL.AGENT_MODEL)
        agent = Agent(model, Tools(store, MistralEmbedder(http)), store)
        session = agent.start()
        print(f"session {session.id}, empty line or Ctrl-D to stop")
        try:
            while text := _read_line():
                print(f"selecta> {agent.reply(session, text)}")
        except KeyboardInterrupt:
            print()
        _print_summary(agent, session)


def _print_summary(agent: Agent, session: Session) -> None:
    try:
        summary = agent.finish(session)
    except (Exception, KeyboardInterrupt):
        logger.exception("could not write the summary")
        return
    print(json.dumps(summary.model_dump(), indent=2, ensure_ascii=False))


def _read_line() -> str:
    try:
        return input("you> ").strip()
    except EOFError:
        return ""


def main() -> None:
    """Run a `selecta` subcommand."""
    parser = argparse.ArgumentParser(prog="selecta")
    commands = parser.add_subparsers(required=True)
    for name, run, text in [
        ("export", _export, "fetch the collection from Discogs into the data directory"),
        ("index", _index, "load the export into the database, enrich and embed it"),
        ("sync", _sync, "export, then index"),
        ("search", _search, "search the collection by meaning"),
        ("chat", _chat, "talk to the agent in the terminal"),
    ]:
        commands.add_parser(name, help=text).set_defaults(run=run)
    search = commands.choices["search"]
    search.add_argument("text")
    search.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        args.run(args)
    except (DiscogsError, MistralError) as e:
        sys.exit(str(e))
    except psycopg.OperationalError as e:
        sys.exit(f"database unavailable: {e}")
