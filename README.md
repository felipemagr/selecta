# selecta

A voice record-picker that lives on a debloated TV. It asks what you want to hear, picks a record from your own collection and plays it.

Work in progress.

```bash
make install
make test
```

## Collection data

Selecta reads your record collection from the [Discogs API](https://www.discogs.com/developers). Nothing is committed: the export lands in `data/discogs/`, which is gitignored.

1. Create a personal access token at <https://www.discogs.com/settings/developers>.
2. `cp .env.example .env` and set `DISCOGS_TOKEN`.
3. `make discogs`

The script finds the token's owner (`/oauth/identity`) and saves:

| File | Endpoint | Contents |
|---|---|---|
| `profile.json` | `/users/{user}` | profile and collection counts |
| `folders.json` | `/users/{user}/collection/folders` | collection folders |
| `fields.json` | `/users/{user}/collection/fields` | custom field definitions (condition, notes) |
| `value.json` | `/users/{user}/collection/value` | estimated collection value |
| `collection.json` | `/users/{user}/collection/folders/0/releases` | every copy owned: folder, rating, date added, notes |
| `wantlist.json` | `/users/{user}/wants` | wantlist |
| `releases/{id}.json` | `/releases/{id}` | tracklist with durations, credits, labels, formats, genres, styles, identifiers, community stats, images |
| `masters/{id}.json` | `/masters/{id}` | original release year and main release |
| `artists/{id}.json` | `/artists/{id}` | artist profile, members, links |

Discogs allows 60 authenticated requests per minute, so a collection of ~100 records takes a few minutes. The script waits when the budget runs out. Release, master and artist files are cached, so rerunning is a quick resync: it fetches only records added since the last run and deletes files for records you removed. Delete `data/discogs/` to fetch everything again.

## Collection database

The agent searches the collection in PostgreSQL (`compose.yaml` runs it locally with Docker).

```bash
make index   # starts the database and loads data/discogs/ into it
```

Rerunning `make index` after `make discogs` makes the database match the collection again: new records are added and removed ones deleted. Text search uses `pg_trgm`, so misspelled names still match. Tests for the database run against it and skip when it isn't running.
