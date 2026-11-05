CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS releases (
    id bigint PRIMARY KEY,
    owned boolean NOT NULL,
    date_added timestamptz,
    title text NOT NULL,
    artists text[] NOT NULL,
    artist_ids bigint[] NOT NULL,
    year int,
    original_year int,
    label text,
    country text,
    format text,
    genres text[] NOT NULL,
    styles text[] NOT NULL,
    notes text,
    cover_url text,
    tracks jsonb NOT NULL,
    credits jsonb NOT NULL,
    search_text text NOT NULL
);

CREATE INDEX IF NOT EXISTS releases_search_text_idx ON releases USING gin (search_text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS artists (
    id bigint PRIMARY KEY,
    name text NOT NULL,
    members text[] NOT NULL,
    profile text
);
