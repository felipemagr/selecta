"""Collection store on PostgreSQL, with fuzzy text search via pg_trgm."""

from importlib.resources import files
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from selecta.application.ports import CollectionQuery
from selecta.domain.collection import Collection, Release, ReleaseSummary

_SUMMARY_COLUMNS = (
    "r.id, r.title, r.artists, r.original_year, r.label, r.format, r.genres, r.styles"
)
_SORTS = {
    "oldest": "coalesce(r.original_year, r.year) ASC NULLS LAST",
    "newest": "coalesce(r.original_year, r.year) DESC NULLS LAST",
    "recently_added": "c.date_added DESC NULLS LAST",
}
# Measured on real data: one-letter misspellings ("craftwerk") score 0.70, unrelated hits <= 0.64.
FUZZY_MATCH_THRESHOLD = 0.65


class PostgresStore:
    """CollectionStore backed by PostgreSQL."""

    def __init__(self, conn: psycopg.Connection) -> None:
        """Wrap an autocommit dict-row connection and make sure the schema exists."""
        self._conn = conn
        conn.execute(files(__package__).joinpath("schema.sql").read_text())
        conn.execute(f"SET pg_trgm.word_similarity_threshold = {FUZZY_MATCH_THRESHOLD}")

    @classmethod
    def connect(cls, url: str) -> "PostgresStore":
        """Open a connection to `url`."""
        return cls(psycopg.connect(url, autocommit=True, row_factory=dict_row))

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def replace(self, household_id: str, collection: Collection) -> None:
        """Make the stored collection match `collection` exactly, in one transaction."""
        owned = {i.release_id for i in collection.items if i.owned}
        with self._conn.transaction():
            self._conn.cursor().executemany(
                """
                INSERT INTO releases VALUES (
                    %(id)s, %(title)s, %(artists)s, %(artist_ids)s, %(year)s, %(original_year)s,
                    %(label)s, %(country)s, %(format)s, %(genres)s, %(styles)s, %(notes)s,
                    %(cover_url)s, %(tracks)s, %(credits)s, %(search_text)s)
                ON CONFLICT (id) DO UPDATE SET (
                    title, artists, artist_ids, year, original_year, label, country, format,
                    genres, styles, notes, cover_url, tracks, credits, search_text
                ) = (
                    EXCLUDED.title, EXCLUDED.artists, EXCLUDED.artist_ids, EXCLUDED.year,
                    EXCLUDED.original_year, EXCLUDED.label, EXCLUDED.country, EXCLUDED.format,
                    EXCLUDED.genres, EXCLUDED.styles, EXCLUDED.notes, EXCLUDED.cover_url,
                    EXCLUDED.tracks, EXCLUDED.credits, EXCLUDED.search_text)
                WHERE %(full)s
                """,
                # Wantlist rows are summaries, so they never overwrite a full release.
                [{**_release_row(r), "full": r.id in owned} for r in collection.releases],
            )
            self._conn.execute("DELETE FROM collection WHERE household_id = %s", [household_id])
            self._conn.cursor().executemany(
                "INSERT INTO collection VALUES (%s, %s, %s, %s)",
                [(household_id, i.release_id, i.owned, i.date_added) for i in collection.items],
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
                "DELETE FROM releases WHERE id NOT IN (SELECT release_id FROM collection)"
            )
            self._conn.execute(
                "DELETE FROM artists WHERE id NOT IN (SELECT unnest(artist_ids) FROM releases)"
            )

    def search(self, household_id: str, query: CollectionQuery) -> list[ReleaseSummary]:
        """Return the household's releases matching every set filter."""
        where = ["c.household_id = %(household)s", "c.owned = %(owned)s"]
        params: dict[str, Any] = {"household": household_id, **query.model_dump()}
        if query.text:
            where.append("%(text)s <%% r.search_text")
        if query.genre:
            where.append("lower(%(genre)s) = ANY (SELECT lower(g) FROM unnest(r.genres) g)")
        if query.style:
            where.append("lower(%(style)s) = ANY (SELECT lower(s) FROM unnest(r.styles) s)")
        if query.label:
            where.append("strpos(lower(r.label), lower(%(label)s)) > 0")
        if query.year_from:
            where.append("coalesce(r.original_year, r.year) >= %(year_from)s")
        if query.year_to:
            where.append("coalesce(r.original_year, r.year) <= %(year_to)s")
        order = _SORTS.get(query.sort)
        if order is None:
            order = "word_similarity(%(text)s, r.search_text) DESC" if query.text else "r.title"
        rows = self._conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM releases r JOIN collection c ON c.release_id = r.id "
            f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(limit)s",
            params,
        ).fetchall()
        return [ReleaseSummary.model_validate(row) for row in rows]

    def get_release(self, household_id: str, release_id: int) -> Release | None:
        """Return one of the household's releases with tracks and credits, or None."""
        row = self._conn.execute(
            "SELECT r.* FROM releases r JOIN collection c ON c.release_id = r.id "
            "WHERE c.household_id = %s AND r.id = %s",
            [household_id, release_id],
        ).fetchone()
        return Release.model_validate(row) if row else None


def _release_row(release: Release) -> dict[str, Any]:
    row = release.model_dump(mode="json")
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
