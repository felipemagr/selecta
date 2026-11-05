"""Collection store on PostgreSQL: fuzzy text via pg_trgm, semantic search via pgvector."""

from importlib.resources import files
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from selecta.application.ports import Chunk, ChunkKind, CollectionQuery, Passage
from selecta.domain.collection import Collection, CollectionItem, Release, ReleaseSummary
from selecta.domain.enrichment import Enrichment

_SUMMARY_COLUMNS = "id, title, artists, original_year, label, format, genres, styles, mood, energy"
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
    """CollectionStore backed by PostgreSQL."""

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
                INSERT INTO artists VALUES (%(id)s, %(name)s, %(members)s, %(profile)s)
                ON CONFLICT (id) DO UPDATE SET (name, members, profile)
                    = (EXCLUDED.name, EXCLUDED.members, EXCLUDED.profile)
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
                "DELETE FROM chunks WHERE (kind = 'release' AND ref_id NOT IN "
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
        rows = self._conn.execute("SELECT kind, ref_id, text_hash FROM chunks").fetchall()
        return {(r["kind"], r["ref_id"]): r["text_hash"] for r in rows}

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks."""
        self._conn.cursor().executemany(
            """
            INSERT INTO chunks (kind, ref_id, text, text_hash, embedding)
            VALUES (%(kind)s, %(ref_id)s, %(text)s, %(text_hash)s, %(embedding)s::vector)
            ON CONFLICT (kind, ref_id) DO UPDATE SET (text, text_hash, embedding)
                = (EXCLUDED.text, EXCLUDED.text_hash, EXCLUDED.embedding)
            """,
            [{**c.model_dump(), "embedding": str(c.embedding)} for c in chunks],
        )

    def delete_chunks_except(self, keep: set[tuple[ChunkKind, int]]) -> None:
        """Delete every chunk not in `keep`."""
        kinds, ref_ids = zip(*keep, strict=True) if keep else ((), ())
        self._conn.execute(
            "DELETE FROM chunks WHERE (kind, ref_id) NOT IN "
            "(SELECT * FROM unnest(%s::text[], %s::bigint[]))",
            [list(kinds), list(ref_ids)],
        )

    def semantic_search(self, text: str, embedding: list[float], limit: int) -> list[Passage]:
        """Return the chunks closest to the query, fusing vector and keyword ranks (RRF)."""
        rows = self._conn.execute(
            """
            WITH by_vector AS (
                SELECT kind, ref_id, row_number() OVER (ORDER BY distance) AS rank
                FROM (
                    SELECT kind, ref_id, embedding <=> %(embedding)s::vector AS distance
                    FROM chunks ORDER BY distance LIMIT %(candidates)s
                ) nearest
            ), by_keyword AS (
                SELECT kind, ref_id, row_number() OVER (ORDER BY relevance DESC) AS rank
                FROM (
                    SELECT kind, ref_id, ts_rank_cd(tsv, q) AS relevance
                    FROM chunks, websearch_to_tsquery('english', %(text)s) q
                    WHERE tsv @@ q ORDER BY relevance DESC LIMIT %(candidates)s
                ) matching
            )
            SELECT kind, ref_id, text, sum(1.0 / (%(k)s + rank)) AS score
            FROM (SELECT * FROM by_vector UNION ALL SELECT * FROM by_keyword) ranked
            JOIN chunks USING (kind, ref_id)
            GROUP BY kind, ref_id, text
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
