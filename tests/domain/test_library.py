import pytest

from selecta.domain.collection import Release, Track
from selecta.domain.library import Song, SongIndex, norm, playable_tracks


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Quiet Water (Remastered 2009)", "quiet water"),
        ("Canción - 2017 Remaster", "cancion"),
        ("Morning Train", "morning train"),
        ("The Long Road", "long road"),
        ("Rose & Thorn", "rose and thorn"),
        ("Part 1 - Olive Grove", "part 1 olive grove"),
    ],
)
def test_norm_keeps_what_two_spellings_share(text: str, expected: str) -> None:
    assert norm(text) == expected


def _release(*tracks: Track, artists: list[str] | None = None) -> Release:
    artists = artists or ["Various"]
    return Release(id=1, title="R", artists=artists, artist_ids=[], tracks=list(tracks))


def test_compilation_tracks_match_songs_filed_under_other_albums() -> None:
    release = _release(
        Track(position="A1", title="Night Drive", artists=["Unit Seven"]),
        Track(position="A2", title="Low Tide", artists=["K. Marlo"]),
        Track(position="B1", title="Missing", artists=["Nobody"]),
    )
    songs = [
        Song(id="s2", title="Low Tide", artists=["K. Marlo"], duration_seconds=300),
        Song(id="s1", title="Night Drive (Original Mix)", artists=["Unit Seven"]),
    ]

    found = playable_tracks(release, SongIndex(songs))

    assert [(t.position, t.song_id) for t in found] == [("A1", "s1"), ("A2", "s2")]


@pytest.mark.parametrize(
    ("artist", "title", "song_artist", "song_title"),
    [
        ("Juan Ruiz Cantaor", "Reencuentro", "El Cantaor", "Reencuentro"),
        ("Hum", "Circles", "Hum", "Circles (2003 Remaster)"),
    ],
)
def test_spelling_differences_still_match(
    artist: str, title: str, song_artist: str, song_title: str
) -> None:
    release = _release(Track(position="A1", title=title), artists=[artist])

    index = SongIndex([Song(id="s", title=song_title, artists=[song_artist])])

    found = playable_tracks(release, index)

    assert [t.song_id for t in found] == ["s"]


@pytest.mark.parametrize(
    ("artist", "other"),
    [
        ("North, South & West", "The Strangers"),
        ("Maria Lopez", "Pedro Lopez Group"),
        ("Little Feat", "Little Johnny"),
    ],
)
def test_same_title_by_another_artist_does_not_match(artist: str, other: str) -> None:
    release = _release(Track(position="A1", title="Evening Song"), artists=[artist])

    index = SongIndex([Song(id="s", title="Evening Song", artists=[other])])

    assert playable_tracks(release, index) == []


def test_titles_with_no_latin_letters_match_only_themselves() -> None:
    release = _release(Track(position="A1", title="朝の光"), artists=["青い空"])
    songs = [
        Song(id="other", title="夜の雨", artists=["青い空"]),
        Song(id="same", title="朝の光", artists=["青い空"]),
    ]

    assert [t.song_id for t in playable_tracks(release, SongIndex(songs))] == ["same"]


@pytest.mark.parametrize(
    ("title", "artist", "song_title", "song_artist"),
    [
        ("Comienzos = Beginnings", "Northbound", "Beginnings", "Northbound"),
        ("Morning Train = Tren De La Mañana", "Northbound", "Morning Train", "Northbound"),
        ("Shining Day", "Fire, Water & Stone", "Shining Day", "Some Producer, Fire"),
    ],
)
def test_two_language_titles_and_split_band_names_match(
    title: str, artist: str, song_title: str, song_artist: str
) -> None:
    release = _release(Track(position="A1", title=title), artists=[artist])
    index = SongIndex([Song(id="s", title=song_title, artists=[song_artist])])

    assert [t.song_id for t in playable_tracks(release, index)] == ["s"]
