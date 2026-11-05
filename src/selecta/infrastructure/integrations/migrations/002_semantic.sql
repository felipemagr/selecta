CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE releases
    ADD COLUMN mood text,
    ADD COLUMN energy text,
    ADD COLUMN enrichment_hash text;

CREATE TABLE chunks (
    kind text NOT NULL,
    ref_id bigint NOT NULL,
    text text NOT NULL,
    text_hash text NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (kind, ref_id)
);

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
