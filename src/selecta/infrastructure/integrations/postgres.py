"""Collection store on PostgreSQL, with fuzzy text search via pg_trgm."""

from importlib.resources import files
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from selecta.application.ports import CollectionQuery
from selecta.domain.collection import Collection, CollectionItem, Release, ReleaseSummary

_SUMMARY_COLUMNS = "id, title, artists, original_year, label, format, genres, styles"
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
