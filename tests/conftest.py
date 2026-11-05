import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import dict_row

from selecta.infrastructure import config
from selecta.infrastructure.integrations.postgres import PostgresStore


@pytest.fixture
def store() -> Iterator[PostgresStore]:
    if not config.DATABASE.URL:
        pytest.skip("DATABASE_URL is not set")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(config.DATABASE.URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("no database, run `make db`")
    admin.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public")
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    admin.execute(f"CREATE SCHEMA {schema}")
    conn = psycopg.connect(
        config.DATABASE.URL,
        autocommit=True,
        row_factory=dict_row,
        options=f"-c search_path={schema},public",
    )
    yield PostgresStore(conn)
    conn.close()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()
