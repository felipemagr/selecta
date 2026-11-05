CREATE TABLE sessions (
    id text PRIMARY KEY,
    started_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    document jsonb NOT NULL
);
