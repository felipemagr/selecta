"""Collection and session store on PostgreSQL."""

from datetime import datetime
from importlib.resources import files
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from selecta.application.conversation import Session
from selecta.application.ports import Chunk, ChunkKind, CollectionQuery, Passage, WebPassage
from selecta.domain.collection import Collection, CollectionItem, Release, ReleaseSummary
from selecta.domain.enrichment import Energy, Enrichment, Mood
from selecta.domain.library import PlayableTrack

# The length is what can be played when every playable track's length is known, else Discogs'.
_SUMMARY_COLUMNS = (
    "id, title, artists, original_year, label, format, genres, styles, mood, energy, "
    "coalesce((SELECT sum(p.duration_seconds) FROM downloaded_tracks p WHERE p.release_id = id "
    "HAVING count(p.duration_seconds) = count(*)), duration_seconds) AS duration_seconds, "
    "jsonb_array_length(tracks) AS track_count, "
    "(SELECT count(*) FROM downloaded_tracks p WHERE p.release_id = id) AS playable_count"
)
_COLUMNS = [
    "id",
    "owned",
    "date_added",
    "title",
    "artists",
    "artist_ids",
    "year",
    "original_year",
    "label",
    "country",
    "format",
    "genres",
    "styles",
    "notes",
    "cover_url",
    "tracks",
    "credits",
    "search_text",
    "duration_seconds",
]
_SORTS = {
    "oldest": "coalesce(original_year, year) ASC NULLS LAST",
    "newest": "coalesce(original_year, year) DESC NULLS LAST",
    "recently_added": "date_added DESC NULLS LAST",
}
# Measured on real data: one-letter misspellings ("craftwerk") score 0.70, unrelated hits <= 0.64.
FUZZY_MATCH_THRESHOLD = 0.65
# Reciprocal rank fusion constant, the usual default.
RRF_K = 60
SEMANTIC_CANDIDATES = 50


class PostgresStore:
    """CollectionStore and SessionStore backed by PostgreSQL."""

    def __init__(self, conn: psycopg.Connection) -> None:
        """Wrap an autocommit dict-row connection and make sure the schema exists."""
        self._conn = conn
        _migrate(conn)
        conn.execute(f"SET pg_trgm.word_similarity_threshold = {FUZZY_MATCH_THRESHOLD}")

    @classmethod
    def connect(cls, url: str) -> "PostgresStore":
        """Open a connection to `url`."""
        return cls(psycopg.connect(url, autocommit=True, row_factory=dict_row))

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def replace(self, collection: Collection) -> None:
        """Make the stored collection match `collection` exactly, in one transaction."""
        items = {i.release_id: i for i in collection.items}
        rows = [_release_row(r, items[r.id]) for r in collection.releases]
        with self._conn.transaction():
            self._conn.cursor().executemany(
                f"""
                INSERT INTO releases ({", ".join(_COLUMNS)})
                VALUES ({", ".join(f"%({c})s" for c in _COLUMNS)})
                ON CONFLICT (id) DO UPDATE SET ({", ".join(_COLUMNS[1:])})
                    = ({", ".join(f"EXCLUDED.{c}" for c in _COLUMNS[1:])})
                """,
                rows,
            )
            self._conn.cursor().executemany(
                """
                INSERT INTO artists (id, name, members, profile, urls)
                VALUES (%(id)s, %(name)s, %(members)s, %(profile)s, %(urls)s)
                ON CONFLICT (id) DO UPDATE SET (name, members, profile, urls)
                    = (EXCLUDED.name, EXCLUDED.members, EXCLUDED.profile, EXCLUDED.urls)
                """,
                [a.model_dump(mode="json") for a in collection.artists],
            )
            self._conn.execute(
                "DELETE FROM releases WHERE NOT (id = ANY(%s))", [[r["id"] for r in rows]]
            )
            self._conn.execute(
                "DELETE FROM artists WHERE id NOT IN (SELECT unnest(artist_ids) FROM releases)"
            )
            self._conn.execute(
                "DELETE FROM chunks WHERE (kind IN ('release', 'web') AND ref_id NOT IN "
                "(SELECT id FROM releases WHERE owned)) "
                "OR (kind = 'artist' AND ref_id NOT IN (SELECT id FROM artists))"
            )

    def search(self, query: CollectionQuery) -> list[ReleaseSummary]:
        """Return releases matching every set filter."""
        where = ["owned = %(owned)s"]
        if query.text:
            where.append("%(text)s <%% search_text")
        if query.genre:
            where.append("lower(%(genre)s) = ANY (SELECT lower(g) FROM unnest(genres) g)")
        if query.style:
            where.append("lower(%(style)s) = ANY (SELECT lower(s) FROM unnest(styles) s)")
        if query.label:
            where.append("strpos(lower(label), lower(%(label)s)) > 0")
        if query.year_from:
            where.append("coalesce(original_year, year) >= %(year_from)s")
        if query.year_to:
            where.append("coalesce(original_year, year) <= %(year_to)s")
        if query.mood:
            where.append("mood = %(mood)s")
        if query.energy:
            where.append("energy = %(energy)s")
        order = _SORTS.get(query.sort)
        if order is None:
            order = "word_similarity(%(text)s, search_text) DESC" if query.text else "title"
        rows = self._conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM releases "
            f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s",
            query.model_dump(),
        ).fetchall()
        return [ReleaseSummary.model_validate(row) for row in rows]

    def owned_summaries(
        self, mood: Mood | None = None, energy: Energy | None = None, playable: bool = False
    ) -> list[ReleaseSummary]:
        """Return every owned release with the given mood and energy, unset meaning any.

        With `playable`, only those with at least one track on the music server.
        """
        rows = self._conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM releases WHERE owned "
            "AND (%(mood)s::text IS NULL OR mood = %(mood)s) "
            "AND (%(energy)s::text IS NULL OR energy = %(energy)s) "
            "AND (NOT %(playable)s OR EXISTS "
            "(SELECT 1 FROM downloaded_tracks p WHERE p.release_id = id)) ORDER BY id",
            {"mood": mood, "energy": energy, "playable": playable},
        ).fetchall()
        return [ReleaseSummary.model_validate(row) for row in rows]

    def owned_summary(self, release_id: int) -> ReleaseSummary | None:
        """Return an owned release, or None if it isn't owned."""
        row = self._conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM releases WHERE owned AND id = %s", [release_id]
        ).fetchone()
        return ReleaseSummary.model_validate(row) if row else None

    def get_release(self, release_id: int) -> Release | None:
        """Return a release with tracks and credits, or None if unknown."""
        row = self._conn.execute("SELECT * FROM releases WHERE id = %s", [release_id]).fetchone()
        return Release.model_validate(row) if row else None

    def enrichment_hashes(self) -> dict[int, str]:
        """Return the input hash each enriched release was enriched from."""
        rows = self._conn.execute(
            "SELECT id, enrichment_hash FROM releases WHERE enrichment_hash IS NOT NULL"
        ).fetchall()
        return {r["id"]: r["enrichment_hash"] for r in rows}

    def save_enrichment(self, release_id: int, enrichment: Enrichment, input_hash: str) -> None:
        """Store a release's enrichment and the hash of the input it came from."""
        self._conn.execute(
            "UPDATE releases SET (mood, energy, enrichment_hash) "
            "= (%(mood)s, %(energy)s, %(hash)s) WHERE id = %(id)s",
            {**enrichment.model_dump(), "hash": input_hash, "id": release_id},
        )

    def chunk_hashes(self) -> dict[tuple[ChunkKind, int], str]:
        """Return the text hash of every stored chunk."""
        rows = self._conn.execute(
            "SELECT kind, ref_id, text_hash FROM chunks WHERE kind <> 'web'"
        ).fetchall()
        return {(r["kind"], r["ref_id"]): r["text_hash"] for r in rows}

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks."""
        self._conn.cursor().executemany(
            """
            INSERT INTO chunks (kind, ref_id, text, text_hash, embedding)
            VALUES (%(kind)s, %(ref_id)s, %(text)s, %(text_hash)s, %(embedding)s::vector)
            ON CONFLICT (kind, ref_id, position) DO UPDATE SET (text, text_hash, embedding)
                = (EXCLUDED.text, EXCLUDED.text_hash, EXCLUDED.embedding)
            """,
            [{**c.model_dump(), "embedding": str(c.embedding)} for c in chunks],
        )

    def delete_chunks_except(self, keep: set[tuple[ChunkKind, int]]) -> None:
        """Delete every chunk not in `keep`."""
        kinds, ref_ids = zip(*keep, strict=True) if keep else ((), ())
        self._conn.execute(
            "DELETE FROM chunks WHERE kind <> 'web' AND (kind, ref_id) NOT IN "
            "(SELECT * FROM unnest(%s::text[], %s::bigint[]))",
            [list(kinds), list(ref_ids)],
        )

    def semantic_search(self, text: str, embedding: list[float], limit: int) -> list[Passage]:
        """Return the chunks closest to the query, fusing vector and keyword ranks (RRF)."""
        rows = self._conn.execute(
            """
            WITH by_vector AS (
                SELECT kind, ref_id, position, row_number() OVER (ORDER BY distance) AS rank
                FROM (
                    SELECT kind, ref_id, position, embedding <=> %(embedding)s::vector AS distance
                    FROM chunks WHERE kind <> 'web'
                    ORDER BY distance LIMIT %(candidates)s
                ) nearest
            ), by_keyword AS (
                SELECT kind, ref_id, position, row_number() OVER (ORDER BY relevance DESC) AS rank
                FROM (
                    SELECT kind, ref_id, position, ts_rank_cd(tsv, q) AS relevance
                    FROM chunks, websearch_to_tsquery('english', %(text)s) q
                    WHERE kind <> 'web' AND tsv @@ q ORDER BY relevance DESC LIMIT %(candidates)s
                ) matching
            )
            SELECT kind, ref_id, text, sum(1.0 / (%(k)s + rank)) AS score
            FROM (SELECT * FROM by_vector UNION ALL SELECT * FROM by_keyword) ranked
            JOIN chunks USING (kind, ref_id, position)
            GROUP BY kind, ref_id, position, text
            ORDER BY score DESC LIMIT %(limit)s
            """,
            {
                "text": text,
                "embedding": str(embedding),
                "candidates": SEMANTIC_CANDIDATES,
                "k": RRF_K,
                "limit": limit,
            },
        ).fetchall()
        return [Passage.model_validate(row) for row in rows]

    def artist_urls(self, release_id: int) -> list[str]:
        """Return the links Discogs lists for the release's artists."""
        rows = self._conn.execute(
            "SELECT unnest(a.urls) AS url FROM releases r "
            "JOIN artists a ON a.id = ANY (r.artist_ids) WHERE r.id = %s",
            [release_id],
        ).fetchall()
        return [r["url"] for r in rows]

    def web_passages(self, release_id: int) -> list[WebPassage]:
        """Return every stored web passage about the release."""
        rows = self._conn.execute(
            "SELECT url, text, position FROM chunks WHERE kind = 'web' AND ref_id = %s "
            "ORDER BY position",
            [release_id],
        ).fetchall()
        return [WebPassage.model_validate(row) for row in rows]

    def add_web_passages(self, release_id: int, passages: list[WebPassage], keep: int) -> None:
        """Store passages that just answered a question, then drop the least used beyond `keep`.

        The passages just added count as used once and are never the ones dropped.
        """
        with self._conn.transaction():
            row = self._conn.execute(
                "SELECT coalesce(max(position) + 1, 0) AS next FROM chunks "
                "WHERE kind = 'web' AND ref_id = %s",
                [release_id],
            ).fetchone()
            first_new = row["next"] if row else 0
            self._conn.cursor().executemany(
                """
                INSERT INTO chunks (kind, ref_id, position, url, text, text_hash, uses)
                SELECT 'web', %(id)s, %(position)s, %(url)s, %(text)s, md5(%(url)s || %(text)s), 1
                WHERE NOT EXISTS (
                    SELECT 1 FROM chunks WHERE kind = 'web' AND ref_id = %(id)s
                    AND text_hash = md5(%(url)s || %(text)s)
                )
                """,
                [
                    {"id": release_id, "position": first_new + i, "url": p.url, "text": p.text}
                    for i, p in enumerate(passages)
                ],
            )
            self._conn.execute(
                """
                DELETE FROM chunks WHERE kind = 'web' AND ref_id = %(id)s AND position IN (
                    SELECT position FROM chunks
                    WHERE kind = 'web' AND ref_id = %(id)s AND position < %(first_new)s
                    ORDER BY uses, position
                    LIMIT greatest(
                        (SELECT count(*) FROM chunks WHERE kind = 'web' AND ref_id = %(id)s)
                        - %(keep)s,
                        0
                    )
                )
                """,
                {"id": release_id, "first_new": first_new, "keep": keep},
            )

    def mark_used(self, release_id: int, positions: list[int]) -> None:
        """Count one more use of each stored web passage."""
        self._conn.execute(
            "UPDATE chunks SET uses = uses + 1 "
            "WHERE kind = 'web' AND ref_id = %s AND position = ANY (%s)",
            [release_id, positions],
        )

    def save_playable(self, release_id: int, tracks: list[PlayableTrack]) -> None:
        """Replace the release's playable tracks."""
        with self._conn.transaction():
            self._conn.execute("DELETE FROM downloaded_tracks WHERE release_id = %s", [release_id])
            self._conn.cursor().executemany(
                "INSERT INTO downloaded_tracks VALUES (%(release_id)s, %(ord)s, %(position)s, "
                "%(title)s, %(song_id)s, %(duration_seconds)s)",
                [
                    {**t.model_dump(), "release_id": release_id, "ord": i}
                    for i, t in enumerate(tracks)
                ],
            )

    def has_song(self, song_id: str) -> bool:
        """Return whether the song plays a track of one of the listener's records."""
        row = self._conn.execute(
            "SELECT EXISTS (SELECT 1 FROM downloaded_tracks WHERE song_id = %s) AS found", [song_id]
        ).fetchone()
        return bool(row and row["found"])

    def playable(self, release_id: int) -> list[PlayableTrack]:
        """Return the release's playable tracks, in the record's order."""
        rows = self._conn.execute(
            "SELECT position, title, song_id, duration_seconds FROM downloaded_tracks "
            "WHERE release_id = %s ORDER BY ord",
            [release_id],
        ).fetchall()
        return [PlayableTrack.model_validate(row) for row in rows]

    def save_session(self, session: Session) -> None:
        """Insert or replace a session."""
        self._conn.execute(
            "INSERT INTO sessions (id, started_at, document) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document, updated_at = now()",
            [session.id, session.started_at, Jsonb(session.model_dump(mode="json"))],
        )

    def load_session(self, session_id: str) -> Session | None:
        """Return a session, or None if unknown."""
        row = self._conn.execute(
            "SELECT document FROM sessions WHERE id = %s", [session_id]
        ).fetchone()
        return Session.model_validate(row["document"]) if row else None

    def sessions_since(self, since: datetime) -> list[Session]:
        """Return the sessions started at or after `since`, oldest first, without their turns."""
        rows = self._conn.execute(
            "SELECT document - 'messages' - 'turns' AS document FROM sessions "
            "WHERE started_at >= %s ORDER BY started_at",
            [since],
        ).fetchall()
        return [Session.model_validate(r["document"]) for r in rows]


def _migrate(conn: psycopg.Connection) -> None:
    """Apply the SQL files in `migrations/` that have not run yet, in name order."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY)")
    done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
    migrations = files(__package__).joinpath("migrations")
    for name in sorted(f.name for f in migrations.iterdir() if f.name.endswith(".sql")):
        if name in done:
            continue
        with conn.transaction():
            conn.execute(migrations.joinpath(name).read_text())
            conn.execute("INSERT INTO schema_migrations VALUES (%s)", [name])


def _release_row(release: Release, item: CollectionItem) -> dict[str, Any]:
    row = {**release.model_dump(mode="json"), "owned": item.owned, "date_added": item.date_added}
    row["tracks"] = Jsonb(row["tracks"])
    row["credits"] = Jsonb(row["credits"])
    row["duration_seconds"] = release.total_seconds()
    row["search_text"] = " ".join(
        [
            release.title,
            *release.artists,
            release.label or "",
            release.notes or "",
            *release.genres,
            *release.styles,
            *(t.title for t in release.tracks),
            *(c.name for c in release.credits),
        ]
    )
    return row
