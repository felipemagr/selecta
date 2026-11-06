-- The tracks of each record that have a song on the music server, in the record's order.
CREATE TABLE downloaded_tracks (
    release_id bigint NOT NULL REFERENCES releases (id) ON DELETE CASCADE,
    ord int NOT NULL,
    position text NOT NULL,
    title text NOT NULL,
    song_id text NOT NULL,
    duration_seconds int,
    PRIMARY KEY (release_id, ord)
);
