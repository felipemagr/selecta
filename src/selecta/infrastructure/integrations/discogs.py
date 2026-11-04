"""Dump everything the Discogs API exposes about the token owner's collection to disk."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from selecta.infrastructure import config

logger = logging.getLogger(__name__)


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
    """Fetch each resource once. Files on disk are reused, so a rerun resumes."""
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


def main() -> None:
    """Run the export with the token from the environment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not config.DISCOGS.TOKEN:
        sys.exit("DISCOGS_TOKEN is not set, see .env.example")
    try:
        with DiscogsClient(config.DISCOGS.TOKEN) as client:
            export(client, config.DISCOGS.OUT_DIR)
    except DiscogsError as e:
        sys.exit(str(e))
