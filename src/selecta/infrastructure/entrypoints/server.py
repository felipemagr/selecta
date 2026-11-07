"""The voice server: phones are push-to-talk remotes, the TV shows the conversation and plays."""

import asyncio
import contextlib
import json
import logging
import math
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from selecta.application.ports import MusicLibraryError, PlaybackResult
from selecta.application.voice import (
    Audio,
    Done,
    Event,
    Line,
    Play,
    Problem,
    StopMusic,
    Transcript,
    VoiceSession,
)
from selecta.infrastructure import config

logger = logging.getLogger(__name__)

_STATIC = files(__package__).joinpath("static")


def create_app(
    open_session: Callable[[], Awaitable[VoiceSession]],
    open_song: Callable[[str, str | None], Awaitable[httpx.Response]] | None = None,
    open_cover: Callable[[int], Awaitable[httpx.Response | None]] | None = None,
) -> FastAPI:
    """Return the app. `open_session` starts a new spoken conversation.

    `open_song` opens a song's audio on the music server, which /stream/{song_id} passes through,
    so the player never needs the server's credentials. It only opens songs of the collection.
    `open_cover` opens a record's cover, served from this server so the TV can read its pixels.

    Every phone and screen shares one conversation, which closes after an idle hour.

    Over /talk, a phone sends {"type": "start"} when the button is pressed, binary 16 kHz
    16-bit PCM while it is held, and {"type": "stop"} on release. Over /screen, the TV receives
    {"type": "started", "turn": n}, then the turn's events (transcript, line, problem, play,
    stop_music, done) and each line's audio as binary float32 PCM at 24 kHz; phones get the
    same JSON events. While a phone talks, screens get its "level". The TV answers "play" with
    {"type": "play_started"} or {"type": "play_failed"}, and reports
    {"type": "heard", "lines": n, "turn": n} as lines finish playing.
    """
    room = _Room(open_session)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await room.close()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    async def remote() -> HTMLResponse:
        return HTMLResponse(_STATIC.joinpath("talk.html").read_text())

    @app.get("/tv")
    async def tv() -> HTMLResponse:
        return HTMLResponse(_STATIC.joinpath("tv.html").read_text())

    @app.get("/stream/{song_id}")
    async def song(song_id: str, request: Request) -> StreamingResponse:
        if open_song is None:
            raise HTTPException(404, "no music server")
        try:
            upstream = await open_song(song_id, request.headers.get("range"))
        except MusicLibraryError as e:
            raise HTTPException(502, str(e)) from e
        # Passed through as is, so the player can seek.
        kept = ("content-length", "content-range", "accept-ranges")
        return StreamingResponse(
            _relay(upstream),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={k: v for k, v in upstream.headers.items() if k.lower() in kept},
        )

    @app.get("/cover/{release_id}")
    async def cover(release_id: int) -> StreamingResponse:
        upstream = await open_cover(release_id) if open_cover else None
        if upstream is None:
            raise HTTPException(404, "no cover")
        return StreamingResponse(
            _relay(upstream), media_type=upstream.headers.get("content-type", "image/jpeg")
        )

    @app.websocket("/talk")
    async def talk(ws: WebSocket) -> None:
        await ws.accept()
        room.remotes.add(ws)
        try:
            await _send([ws], room.screen_state())
            async for message in _receive(ws):
                if isinstance(message, bytes):
                    await room.hear(ws, message)
                elif message.get("type") == "start":
                    await room.start(ws)
                elif message.get("type") == "stop":
                    room.stop(ws)
        finally:
            room.remotes.discard(ws)
            room.stop(ws)

    @app.websocket("/screen")
    async def screen(ws: WebSocket) -> None:
        await ws.accept()
        room.screens.add(ws)
        await _send(room.remotes, room.screen_state())
        try:
            async for message in _receive(ws):
                if isinstance(message, bytes):
                    continue
                kind = message.get("type")
                if kind in ("play_started", "play_failed"):
                    room.player_said("playing" if kind == "play_started" else "failed")
                elif kind == "heard":
                    room.heard(int(message["lines"]), int(message["turn"]))
        finally:
            room.screens.discard(ws)
            await _send(room.remotes, room.screen_state())

    return app


class _Room:
    """The household's one conversation: phones talk into it, screens show and play it."""

    def __init__(self, open_session: Callable[[], Awaitable[VoiceSession]]) -> None:
        self._open_session = open_session
        self._voice: VoiceSession | None = None
        self.remotes: set[WebSocket] = set()
        self.screens: set[WebSocket] = set()
        self._turn: asyncio.Task[None] | None = None
        self._audio: asyncio.Queue[bytes | None] | None = None
        # The phone whose press is being recorded: only its audio and release count.
        self._speaker: WebSocket | None = None
        self._opening = asyncio.Lock()
        self._idle: asyncio.Task[None] | None = None
        # The current turn's number, its lines sent to the screens, and how many finished
        # playing there. Screens report with the number, so a late report can't count.
        self._turns = 0
        self._sent = 0
        self._heard = 0

    def screen_state(self) -> dict[str, Any]:
        return {"type": "screen", "on": bool(self.screens)}

    async def start(self, remote: WebSocket) -> None:
        """Start a turn, cutting off the last reply if it is still being made or spoken."""
        if not self.screens:
            await _send([remote], {"type": "problem", "text": "Open Selecta on the TV first."})
            return
        if self._idle:
            self._idle.cancel()
        self._audio, self._speaker = asyncio.Queue(), remote
        audio = self._audio
        async with self._opening:
            if self._voice is None:
                self._voice = await self._open_session()
        voice = self._voice
        running = self._turn is not None and not self._turn.done()
        if self._turn and running:
            self._turn.cancel()
        heard = self._heard if running or self._heard < self._sent else None
        self._sent = self._heard = 0
        self._turns += 1
        # Everything sent before this belongs to an older turn; the pages drop it.
        await self._broadcast({"type": "started", "turn": self._turns})
        self._turn = asyncio.create_task(self._run(voice.turn(_drain(audio), self._emit, heard)))

    async def hear(self, remote: WebSocket, pcm: bytes) -> None:
        if self._audio and remote is self._speaker:
            self._audio.put_nowait(pcm)
            await _send(self.screens, {"type": "level", "value": _level(pcm)})

    def stop(self, remote: WebSocket) -> None:
        if self._audio and remote is self._speaker:
            self._audio.put_nowait(None)
            self._audio = self._speaker = None

    def heard(self, lines: int, turn: int) -> None:
        if turn == self._turns:
            self._heard = lines

    def player_said(self, result: PlaybackResult) -> None:
        if self._voice:
            self._voice.player_said(result)

    async def close(self) -> None:
        """End the conversation, writing its summary."""
        if self._idle:
            self._idle.cancel()
        if self._turn and not self._turn.done():
            self._turn.cancel()
        voice, self._voice = self._voice, None
        if voice:
            await voice.close()

    async def _run(self, turn: Awaitable[None]) -> None:
        try:
            await turn
        except Exception:
            logger.exception("voice turn failed")
            await self._emit(Problem("Something went wrong on my side. Try again?"))
            await self._emit(Done())
        self._idle = asyncio.create_task(self._close_when_idle())

    async def _close_when_idle(self) -> None:
        await asyncio.sleep(config.SERVER.SESSION_IDLE_MINUTES * 60)
        # A press during the summary must not interrupt it.
        await asyncio.shield(self.close())

    async def _emit(self, event: Event) -> None:
        if isinstance(event, Audio):
            await _send(self.screens, event.pcm)
            return
        if isinstance(event, Line):
            self._sent = event.index + 1
        await self._broadcast(_wire(event))

    async def _broadcast(self, message: dict[str, Any]) -> None:
        await _send([*self.screens, *self.remotes], message)


async def _send(sockets: Iterable[WebSocket], message: dict[str, Any] | bytes) -> None:
    for ws in list(sockets):
        # A page that just left must not break the turn.
        with contextlib.suppress(Exception):
            if isinstance(message, bytes):
                await ws.send_bytes(message)
            else:
                await ws.send_text(json.dumps(message))


async def _receive(ws: WebSocket) -> AsyncIterator[dict[str, Any] | bytes]:
    while (message := await ws.receive())["type"] != "websocket.disconnect":
        if (data := message.get("bytes")) is not None:
            yield data
        else:
            yield json.loads(message.get("text") or "{}")


async def _relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Yield the upstream body, closing it even when the player hangs up (on a seek or skip)."""
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()


async def _drain(queue: "asyncio.Queue[bytes | None]") -> AsyncIterator[bytes]:
    """Yield the utterance's audio until the button is released or the audio stops coming."""
    while True:
        try:
            chunk = await asyncio.wait_for(queue.get(), config.VOICE.MAX_SILENCE_SECONDS)
        except TimeoutError:
            return
        if chunk is None:
            return
        yield chunk


def _level(pcm: bytes) -> float:
    """Return how loud 16-bit PCM is, from 0 to 1, for the screen to show the voice."""
    samples = array("h", pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
    return round(min(1.0, rms * 5), 2)


def _wire(event: Event) -> dict[str, Any]:
    match event:
        case Transcript(text=text):
            return {"type": "transcript", "text": text}
        case Line(index=index, text=text):
            return {"type": "line", "index": index, "text": text}
        case Problem(text=text):
            return {"type": "problem", "text": text}
        case Play(release=release, tracks=tracks):
            record = {
                "title": release.title,
                "artists": release.artists,
                "year": release.original_year or release.year,
                "label": release.label,
                "cover": f"/cover/{release.id}" if release.cover_url else None,
            }
            playlist = [
                {
                    "url": f"/stream/{t.song_id}",
                    "position": t.position,
                    "title": t.title,
                    "seconds": t.duration_seconds,
                }
                for t in tracks
            ]
            return {"type": "play", "record": record, "tracks": playlist}
        case StopMusic():
            return {"type": "stop_music"}
        case _:
            return {"type": "done"}
