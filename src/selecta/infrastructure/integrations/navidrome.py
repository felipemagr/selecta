"""Navidrome over the Subsonic API: the song list, and streaming one song."""

import hashlib
import secrets
from typing import Any

import httpx

from selecta.application.ports import MusicLibraryError
from selecta.domain.library import Song
from selecta.infrastructure import config


class NavidromeError(MusicLibraryError):
    """Navidrome could not be reached or refused the request."""


def connect(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return an HTTP client for the Subsonic API."""
    return httpx.Client(
        base_url=f"{config.NAVIDROME.URL}/rest",
        timeout=config.NAVIDROME.TIMEOUT_SECONDS,
        transport=transport,
    )


class NavidromeLibrary:
    """MusicLibrary on Navidrome."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def songs(self) -> list[Song]:
        """Return every song on the server."""
        size, found = config.NAVIDROME.PAGE_SIZE, []
        while True:
            params = {"query": "", "songCount": size, "songOffset": len(found)}
            body = self._get("/search3", {**params, "albumCount": 0, "artistCount": 0})
            page = body.get("searchResult3", {}).get("song", [])
            found += [
                Song(
                    id=s["id"],
                    title=s["title"],
                    artists=[a["name"] for a in s.get("artists", [])] or [s.get("artist", "")],
                    duration_seconds=s.get("duration"),
                )
                for s in page
            ]
            if len(page) < size:
                return found

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.get(path, params={**auth(), **params})
            response.raise_for_status()
            body = response.json()["subsonic-response"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            # The error's URL carries a reusable token, so only its kind is reported.
            raise NavidromeError(f"GET {path} failed: {type(e).__name__}") from e
        if body.get("status") != "ok":
            raise NavidromeError(f"GET {path} refused: {body.get('error')}")
        return body


async def open_stream(
    http: httpx.AsyncClient, song_id: str, byte_range: str | None = None
) -> httpx.Response:
    """Return the streaming response for a song's original file; the caller closes it.

    `format=raw` asks for the file as stored, so the server's old CPU never transcodes.
    `byte_range` is the player's Range header, so it can seek.
    """
    params = {**auth(), "id": song_id, "format": "raw"}
    headers = {"Range": byte_range} if byte_range else {}
    url = f"{config.NAVIDROME.URL}/rest/stream"
    request = http.build_request("GET", url, params=params, headers=headers)
    try:
        response = await http.send(request, stream=True)
    except httpx.HTTPError as e:
        raise NavidromeError(f"stream {song_id} failed: {type(e).__name__}") from e
    # Subsonic reports errors as a JSON or XML body with status 200.
    if response.is_error or "json" in response.headers.get("content-type", ""):
        await response.aclose()
        raise NavidromeError(f"stream {song_id} failed: {response.status_code}")
    return response


def auth() -> dict[str, str]:
    """Return Subsonic token auth: md5(password + salt), so the password never travels."""
    salt = secrets.token_hex(6)
    token = hashlib.md5(((config.NAVIDROME.PASSWORD or "") + salt).encode()).hexdigest()
    return {
        "u": config.NAVIDROME.USER or "",
        "t": token,
        "s": salt,
        "v": "1.16.1",
        "c": "selecta",
        "f": "json",
    }
