import httpx
import pytest

from selecta.application.ports import MusicLibraryError
from selecta.infrastructure import config
from selecta.infrastructure.integrations import navidrome
from selecta.infrastructure.integrations.navidrome import NavidromeLibrary


@pytest.fixture(autouse=True)
def server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.NAVIDROME, "URL", "http://music")
    monkeypatch.setattr(config.NAVIDROME, "USER", "selecta")
    monkeypatch.setattr(config.NAVIDROME, "PASSWORD", "secret")


def _song(i: int) -> dict[str, object]:
    return {"id": f"s{i}", "title": f"T{i}", "artists": [{"name": "A"}], "duration": 60}


def test_every_page_of_songs_is_read_without_sending_the_password() -> None:
    pages = [[_song(i) for i in range(config.NAVIDROME.PAGE_SIZE)], [_song(999)]]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = {"status": "ok", "searchResult3": {"song": pages.pop(0)}}
        return httpx.Response(200, json={"subsonic-response": body})

    songs = NavidromeLibrary(navidrome.connect(httpx.MockTransport(handler))).songs()

    assert len(songs) == config.NAVIDROME.PAGE_SIZE + 1
    assert all(b"secret" not in r.url.query for r in requests)


def test_refused_request_raises() -> None:
    body = {"subsonic-response": {"status": "failed", "error": {"message": "Wrong username"}}}
    http = navidrome.connect(httpx.MockTransport(lambda _: httpx.Response(200, json=body)))

    with pytest.raises(MusicLibraryError):
        NavidromeLibrary(http).songs()
