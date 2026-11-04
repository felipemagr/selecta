import json
from pathlib import Path

import httpx
import pytest

from selecta.infrastructure.integrations import discogs
from selecta.infrastructure.integrations.discogs import DiscogsClient, export


def _page(key: str, items: list[dict], page: int, pages: int) -> dict:
    return {key: items, "pagination": {"page": page, "pages": pages}}


class FakeDiscogs:
    """Two collection pages, one release with a master and an artist."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.rate_limit_once = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if self.rate_limit_once:
            self.rate_limit_once = False
            return httpx.Response(429)
        page = int(request.url.params.get("page", 1))
        match path:
            case "/oauth/identity":
                body = {"username": "listener"}
            case "/users/listener/collection/folders/0/releases":
                body = _page("releases", [{"id": page}], page, 2)
            case "/users/listener/wants":
                body = _page("wants", [], 1, 1)
            case "/releases/1" | "/releases/2":
                body = {"id": 1, "master_id": 10, "artists": [{"id": 100}]}
            case "/artists/100":
                return httpx.Response(404)
            case _:
                body = {}
        return httpx.Response(200, json=body)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeDiscogs:
    monkeypatch.setattr(discogs.time, "sleep", lambda _: None)
    return FakeDiscogs()


def _export(fake: FakeDiscogs, out: Path) -> None:
    with DiscogsClient("token", transport=httpx.MockTransport(fake)) as client:
        export(client, out)


def test_export_fetches_every_collection_page(fake: FakeDiscogs, tmp_path: Path) -> None:
    _export(fake, tmp_path)

    collection = json.loads((tmp_path / "collection.json").read_text())
    assert [r["id"] for r in collection] == [1, 2]
    assert (tmp_path / "releases" / "2.json").exists()
    assert (tmp_path / "masters" / "10.json").exists()
    assert not (tmp_path / "artists" / "100.json").exists()


def test_export_resumes_from_cached_details(fake: FakeDiscogs, tmp_path: Path) -> None:
    _export(fake, tmp_path)
    fake.paths.clear()

    _export(fake, tmp_path)

    assert not {"/releases/1", "/releases/2", "/masters/10"} & set(fake.paths)


def test_export_removes_details_no_longer_in_collection(fake: FakeDiscogs, tmp_path: Path) -> None:
    for kind in ("releases", "masters"):
        (tmp_path / kind).mkdir()
        (tmp_path / kind / "99.json").write_text("{}")

    _export(fake, tmp_path)

    assert sorted(p.name for p in (tmp_path / "releases").iterdir()) == ["1.json", "2.json"]
    assert not (tmp_path / "masters" / "99.json").exists()


def test_rate_limited_request_is_retried(fake: FakeDiscogs, tmp_path: Path) -> None:
    fake.rate_limit_once = True

    _export(fake, tmp_path)

    assert fake.paths[:2] == ["/oauth/identity", "/oauth/identity"]
