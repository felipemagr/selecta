# Contributing

Thanks for wanting to help. Selecta is small on purpose: the shortest code that does the job,
with every decision readable in plain Python.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Docker (for the database).

```bash
make install                 # uv sync
cp .env.example .env         # at least MISTRAL_API_KEY and POSTGRES_PASSWORD
make db                      # Postgres with pgvector, on localhost
uv run --env-file .env selecta index --from examples/collection.json
uv run --env-file .env selecta chat
```

## Before a pull request

- `make test` passes: lint, format, import layers, then pytest. It's free and offline; the
  database tests skip when the database isn't running.
- `make quality` fixes lint and format.
- If you changed a prompt, a tool description or how the agent decides, run `make eval` before
  and after, and put both pass^k numbers in the pull request. It calls the real model and costs
  a little (well under a dollar a run).
- One change per pull request, titled as a [Conventional Commit](https://www.conventionalcommits.org)
  (`feat(tools): ...`, `fix(tv): ...`).

## How the code is laid out

```
src/selecta/
├── domain/          types and rules, no I/O: the brief, the collection, matching, picking
├── application/     the agent loop, its tools, prompts, and ports (the interfaces below)
└── infrastructure/
    ├── config.py      every setting and environment variable
    ├── integrations/  one module per outside service: Mistral, Postgres, Discogs, Navidrome...
    └── entrypoints/   the CLI, the web server, and the TV and phone pages
android/             the Android TV app: a full-screen shell for the /tv page
examples/            the synthetic sample collection
tests/               mirrors src/; tests/evals/ holds the live evals
```

Imports only point inward (`infrastructure` → `application` → `domain`); `make test` enforces it.
A rule goes in `domain/`, coordination in `application/`, anything that talks to a network or
disk in `infrastructure/`.

A few habits that keep it legible:

- **Code decides, the model talks.** If a behaviour has consequences (playing, picking), it's
  checked in code, and the agent only says something happened when the tool's result says so.
- **Data over instructions.** Prefer giving the model facts (in the interview state or a tool
  result) over adding rules to the prompt.
- **Test what can break**, one behaviour per test, and fake only the ports (the model, the music
  server, the web), never the agent loop.
- **No real personal data** in the repo: fixtures and examples are invented.

## Adding a music server

Playback goes through one port, `MusicLibrary` in `application/ports.py`: list the server's
songs (title, artists, duration). Selecta matches them to each record's tracks and stores which
ones can play. The stream itself is opened by `open_stream` in the integration and passed
through the web server, so the TV never needs the server's credentials.

To add, say, Jellyfin:

1. Write `infrastructure/integrations/jellyfin.py` with a class whose `songs()` returns
   `list[Song]`, and an `open_stream(http, song_id, byte_range)` like Navidrome's.
2. Add its settings to `config.py` in their own section.
3. Choose it in `cli.py` where `_library()` builds the Navidrome one.
4. Test it against a fake HTTP transport, like `tests/infrastructure/integrations/test_navidrome.py`.

## Adding a tool

1. Describe it in `SPECS` in `application/tools.py`: what it does and when to use it, in a few
   plain sentences.
2. Route it in `Tools.run`, and return what actually happened, including failures, as text the
   model can act on.
3. If it needs an outside service, add a port in `application/ports.py` and an integration.
4. Add a unit test with a fake port, and an eval scenario in `tests/evals/scenarios.py` if
   choosing the tool is the hard part.

## Evals

`tests/evals/` runs the real agent and model in an in-memory world built from
`examples/collection.json` (the player and the web are fakes). A scenario is a frozen situation,
what the listener says (two ways, in English, Spanish and French) and a check on what the agent
did. `checks.py` holds the rules every run must keep. Add a scenario for a capability, not for one
bad reply: say what must happen, in a way that holds for any good answer.

```bash
make eval                               # every scenario, 5 runs each
make eval ARGS="--only play --k 2"      # some scenarios, fewer runs
make eval ARGS="--simulate"             # whole conversations with simulated listeners
make eval ARGS="--scripted"             # no API calls: checks the harness itself
```

Reports go to `data/evals/`.
