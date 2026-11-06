"""Matching the listener's records to the songs on their music server."""

import re
import unicodedata

from pydantic import BaseModel

from selecta.domain.collection import Release

# Version notes that differ between a vinyl's tracklist and a download of the same song.
_VERSION = re.compile(
    r"\s-\s.*\b(remaster(ed)?|version|mono|stereo|edit|live|mix|bonus)\b.*$", re.I
)
_BRACKETS = re.compile(r"[(\[].*?[)\]]")
# Some pressings print the title in two languages, joined by " = ".
_TRANSLATION = re.compile(r"\s=\s")
_FEATURING = re.compile(r"\b(feat\.|ft\.|featuring\s).*$")
_NOT_WORD = re.compile(r"[^\w]+")


class Song(BaseModel):
    """A song on the music server."""

    id: str
    title: str
    artists: list[str]
    duration_seconds: int | None = None


class PlayableTrack(BaseModel):
    """A track of a record that has a song on the music server."""

    position: str
    title: str
    song_id: str
    duration_seconds: int | None = None


def norm(text: str) -> str:
    """Return text reduced to what two spellings of the same title or name share."""
    text = _BRACKETS.sub(" ", _VERSION.sub("", text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    text = _FEATURING.sub("", text.replace("&", " and "))
    return " ".join(_NOT_WORD.sub(" ", text).split()).removeprefix("the ")


class SongIndex:
    """The server's songs by normalised title, built once for matching many records."""

    def __init__(self, songs: list[Song]) -> None:
        self._by_title: dict[str, list[tuple[Song, list[set[str]]]]] = {}
        for song in songs:
            # Servers can cut a band name at its comma, so each part is tried too.
            names = [n for a in song.artists for n in (a, *a.split(","))]
            words = [_words(norm(n)) for n in names]
            self._by_title.setdefault(norm(song.title), []).append((song, words))

    def find(self, title: str, artists: set[str], taken: set[str]) -> Song | None:
        """Return a song with this normalised title by one of these artists, not yet taken."""
        for song, song_words in self._by_title.get(title, []):
            if song.id not in taken and _same_artist(artists, song_words):
                return song
        return None


def playable_tracks(release: Release, index: SongIndex) -> list[PlayableTrack]:
    """Return the release's tracks found on the server, in the record's order.

    A track matches a song with the same title by one of the same artists, whatever album the
    song was filed under: downloads of a compilation keep each song's original album.
    """
    found: list[PlayableTrack] = []
    taken: set[str] = set()
    for track in release.tracks:
        artists = {norm(a) for a in track.artists or release.artists}
        # A two-language title can be downloaded under either one.
        titles = [t for t in map(norm, _TRANSLATION.split(track.title)) if t]
        if song := next((s for t in titles if (s := index.find(t, artists, taken))), None):
            taken.add(song.id)
            found.append(
                PlayableTrack(
                    position=track.position,
                    title=track.title,
                    song_id=song.id,
                    duration_seconds=song.duration_seconds,
                )
            )
    return found


def _same_artist(wanted: set[str], song_words: list[set[str]]) -> bool:
    """Return whether one name's distinctive words are all in the other."""
    for name in wanted:
        for words in song_words:
            short, long = sorted((_words(name), words), key=len)
            if short and short <= long:
                return True
    return False


def _words(name: str) -> set[str]:
    """Return a name's distinctive words: the long ones, or the whole name if it's short."""
    return {w for w in name.split() if len(w) > 3} or ({name} if name else set())
