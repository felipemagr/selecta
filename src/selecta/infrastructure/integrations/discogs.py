"""Dump everything the Discogs API exposes about the token owner's collection to disk."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from selecta.domain.collection import Artist, Collection, CollectionItem, Credit, Release, Track
from selecta.infrastructure import config

logger = logging.getLogger(__name__)

# Discogs disambiguates names as "Name (2)" and marks up profiles as [a=Name], [a123], [b]...[/b].
_SUFFIX = re.compile(r" \(\d+\)$")
_MARKUP_LINK = re.compile(r"\[[alm]=([^\]]+)\]")
_MARKUP_TAG = re.compile(r"\[/?[a-z]+\d*[^\]]*\]")


class DiscogsError(Exception):
    """The Discogs API failed in a way a retry did not fix."""


class DiscogsClient:
    """Authenticated Discogs API client that retries rate limits and transient errors."""

    def __init__(self, token: str, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(
            base_url=config.DISCOGS.API_URL,
            headers={
                "Authorization": f"Discogs token={token}",
                "User-Agent": config.DISCOGS.USER_AGENT,
            },
            timeout=config.DISCOGS.TIMEOUT_SECONDS,
            transport=transport,
        )

    def __enter__(self) -> "DiscogsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self._http.close()

    def get(self, path: str, **params: Any) -> dict[str, Any] | None:
        """Return a JSON resource, or None if it does not exist."""
        for _ in range(config.DISCOGS.MAX_RETRIES):
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as e:
                logger.warning("%s on %s, retrying", type(e).__name__, path)
                time.sleep(config.DISCOGS.RETRY_WAIT_SECONDS)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                logger.warning("%s on %s, retrying", response.status_code, path)
                time.sleep(config.DISCOGS.RETRY_WAIT_SECONDS)
                continue
            if response.status_code == 404:
                logger.warning("not found: %s", path)
                return None
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DiscogsError(f"GET {path} failed: {response.status_code}") from e
            if response.headers.get("X-Discogs-Ratelimit-Remaining") == "0":
                time.sleep(config.DISCOGS.RATE_LIMIT_WAIT_SECONDS)
            return response.json()
        raise DiscogsError(f"GET {path} failed after {config.DISCOGS.MAX_RETRIES} tries")

    def require(self, path: str, **params: Any) -> dict[str, Any]:
        """Return a JSON resource, raising DiscogsError if it does not exist."""
        data = self.get(path, **params)
        if data is None:
            raise DiscogsError(f"GET {path} not found")
        return data

    def get_all(self, path: str, key: str) -> list[dict[str, Any]]:
        """Return the items under `key` from every page of a paginated list."""
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.require(path, page=page, per_page=config.DISCOGS.PAGE_SIZE)
            items.extend(data[key])
            if page >= data["pagination"]["pages"]:
                return items
            page += 1


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _fetch_details(client: DiscogsClient, kind: str, ids: set[int], out_dir: Path) -> list[dict]:
    """Fetch each resource once and delete files no longer referenced.

    Files on disk are reused, so a rerun resumes and only fetches new ids.
    """
    for path in (out_dir / kind).glob("*.json"):
        if int(path.stem) not in ids:
            logger.info("removing %s %s", kind, path.stem)
            path.unlink()
    results = []
    for i, item_id in enumerate(sorted(ids), 1):
        path = out_dir / kind / f"{item_id}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        logger.info("%s %d/%d", kind, i, len(ids))
        data = client.get(f"/{kind}/{item_id}")
        if data is None:
            continue
        _write(path, data)
        results.append(data)
    return results


def export(client: DiscogsClient, out_dir: Path) -> None:
    """Save the token owner's account, collection and its release, master and artist details."""
    user = f"/users/{client.require('/oauth/identity')['username']}"

    _write(out_dir / "profile.json", client.require(user))
    _write(out_dir / "folders.json", client.require(f"{user}/collection/folders"))
    _write(out_dir / "fields.json", client.require(f"{user}/collection/fields"))
    _write(out_dir / "value.json", client.require(f"{user}/collection/value"))
    _write(out_dir / "wantlist.json", client.get_all(f"{user}/wants", "wants"))
    collection = client.get_all(f"{user}/collection/folders/0/releases", "releases")
    _write(out_dir / "collection.json", collection)

    releases = _fetch_details(client, "releases", {r["id"] for r in collection}, out_dir)
    master_ids = {r["master_id"] for r in releases if r.get("master_id")}
    artist_ids = {a["id"] for r in releases for a in r.get("artists", []) if a.get("id")}
    _fetch_details(client, "masters", master_ids, out_dir)
    _fetch_details(client, "artists", artist_ids, out_dir)
    logger.info("exported %d collection items to %s", len(collection), out_dir)


class DiscogsExport:
    """Collection source that reads the files written by `export`."""

    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir

    def load(self) -> Collection:
        """Return the exported collection, with wantlist items as not owned."""
        if not (self._dir / "collection.json").exists():
            raise DiscogsError(f"no export in {self._dir}, run `selecta export` first")
        releases: dict[int, Release] = {}
        items: dict[int, CollectionItem] = {}
        for item in _read(self._dir / "collection.json"):
            # Releases Discogs couldn't return fall back to the list item's summary.
            data = _read(self._dir / "releases" / f"{item['id']}.json") or item["basic_information"]
            master = _read(self._dir / "masters" / f"{data.get('master_id')}.json")
            releases[item["id"]] = _release(data, master.get("year") or None)
            items.setdefault(item["id"], _item(item, owned=True))
        for item in _read(self._dir / "wantlist.json"):
            if item["id"] not in items:
                releases[item["id"]] = _release(item["basic_information"], None)
                items[item["id"]] = _item(item, owned=False)
        artists = [_artist(_read(p)) for p in sorted((self._dir / "artists").glob("*.json"))]
        return Collection(
            releases=list(releases.values()), artists=artists, items=list(items.values())
        )


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def _name(artist: dict[str, Any]) -> str:
    return _SUFFIX.sub("", artist.get("anv") or artist["name"])


def _seconds(duration: str) -> int | None:
    try:
        return sum(int(p) * 60**i for i, p in enumerate(reversed(duration.split(":"))))
    except ValueError:
        return None


def _release(data: dict[str, Any], original_year: int | None) -> Release:
    """Map a full release or a list item's `basic_information`, which lacks tracks and credits."""
    year = data.get("year") or None
    fmt = (data.get("formats") or [{}])[0]
    images = data.get("images") or []
    cover = next((i["uri"] for i in images if i.get("type") == "primary"), None)
    entries = [e for t in data.get("tracklist", []) for e in [t, *t.get("sub_tracks", [])]]
    tracks = [t for t in entries if t.get("type_") == "track"]
    credits = [*data.get("extraartists", [])]
    for t in tracks:
        credits += [
            {**c, "tracks": c.get("tracks") or t["position"]} for c in t.get("extraartists", [])
        ]
    return Release(
        id=data["id"],
        title=data["title"],
        artists=[_name(a) for a in data.get("artists", [])],
        artist_ids=[a["id"] for a in data.get("artists", [])],
        year=year,
        original_year=original_year or year,
        label=next((_SUFFIX.sub("", lb["name"]) for lb in data.get("labels", [])), None),
        country=data.get("country") or None,
        format=", ".join([fmt["name"], *fmt.get("descriptions", [])]) if fmt else None,
        genres=data.get("genres", []),
        styles=data.get("styles", []),
        notes=data.get("notes") or None,
        cover_url=cover or data.get("cover_image"),
        tracks=[
            Track(
                position=t["position"],
                title=t["title"],
                duration_seconds=_seconds(t["duration"]) if t.get("duration") else None,
            )
            for t in tracks
        ],
        credits=[
            Credit(name=_name(c), role=c["role"], tracks=c.get("tracks", "")) for c in credits
        ],
    )


def _item(item: dict[str, Any], owned: bool) -> CollectionItem:
    return CollectionItem(release_id=item["id"], owned=owned, date_added=item.get("date_added"))


def _artist(data: dict[str, Any]) -> Artist:
    profile = _MARKUP_LINK.sub(r"\1", data.get("profile") or "")
    profile = _MARKUP_TAG.sub("", profile).strip()
    return Artist(
        id=data["id"],
        name=_SUFFIX.sub("", data["name"]),
        members=[_SUFFIX.sub("", m["name"]) for m in data.get("members", [])],
        profile=profile or None,
        urls=data.get("urls") or [],
    )
