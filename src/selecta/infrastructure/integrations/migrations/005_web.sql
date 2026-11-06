ALTER TABLE artists ADD COLUMN urls text[] NOT NULL DEFAULT '{}';

-- Web passages live next to the Discogs chunks: several per record, with their source and how
-- often they answered a question. They are reranked.
ALTER TABLE chunks
    ADD COLUMN position int NOT NULL DEFAULT 0,
    ADD COLUMN url text,
    ADD COLUMN uses int NOT NULL DEFAULT 0,
    ALTER COLUMN embedding DROP NOT NULL,
    DROP CONSTRAINT chunks_pkey,
    ADD PRIMARY KEY (kind, ref_id, position);
