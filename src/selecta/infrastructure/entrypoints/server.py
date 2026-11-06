"""The voice server: a push-to-talk page and the WebSocket it talks over."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib.resources import files

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from selecta.application.ports import MusicLibraryError
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

_PAGE = files(__package__).joinpath("static", "talk.html").read_text()


def create_app(
    open_session: Callable[[], Awaitable[VoiceSession]],
    open_song: Callable[[str, str | None], Awaitable[httpx.Response]] | None = None,
) -> FastAPI:
    """Return the app. `open_session` starts a new spoken conversation for each connection.

    `open_song` opens a song's audio on the music server, which /stream/{song_id} passes through,
    so the player never needs the server's credentials. It only opens songs of the collection.

    Protocol over /talk: the phone sends {"type": "start", "heard_lines": n | null} when the button
    is pressed (with `heard_lines` set if that cut off the agent), binary 16 kHz 16-bit PCM while it
    is held, and {"type": "stop"} on release. The server answers "start" with {"type": "started"},
    then sends JSON events (transcript, line, problem, play, done) and the audio of
    each line as binary float32 PCM at 24 kHz. After "play", the page answers
    {"type": "play_started"} or {"type": "play_failed"}.
    """
    app = FastAPI()

    @app.get("/")
    async def page() -> HTMLResponse:
        return HTMLResponse(_PAGE)

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

    @app.websocket("/talk")
    async def talk(ws: WebSocket) -> None:
        await ws.accept()
        voice = await open_session()
        current: asyncio.Task[None] | None = None
        audio: asyncio.Queue[bytes | None] | None = None

        async def emit(event: Event) -> None:
            if isinstance(event, Audio):
                await ws.send_bytes(event.pcm)
            else:
                await ws.send_text(json.dumps(_wire(event)))

        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (chunk := message.get("bytes")) is not None:
                    if audio:
                        audio.put_nowait(chunk)
                    continue
                command = json.loads(message.get("text") or "{}")
                if command.get("type") == "start":
                    if current and not current.done():
                        current.cancel()
                    heard = command.get("heard_lines")
                    audio = asyncio.Queue()
                    # Everything sent before this belongs to an older turn; the page drops it.
                    await ws.send_text(json.dumps({"type": "started"}))
                    turn = voice.turn(_drain(audio), emit, None if heard is None else int(heard))
                    current = asyncio.create_task(_run(turn, emit))
                elif command.get("type") == "stop" and audio:
                    audio.put_nowait(None)
                elif command.get("type") in ("play_started", "play_failed"):
                    started = command["type"] == "play_started"
                    voice.player_said("playing" if started else "failed")
        except WebSocketDisconnect:
            pass
        finally:
            if current and not current.done():
                current.cancel()
            await voice.close()

    return app


async def _run(turn: Awaitable[None], emit: Callable[[Event], Awaitable[None]]) -> None:
    try:
        await turn
    except Exception:
        logger.exception("voice turn failed")
        with contextlib.suppress(Exception):
            await emit(Problem("Something went wrong on my side. Try again?"))
            await emit(Done())


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


def _wire(event: Event) -> dict[str, object]:
    match event:
        case Transcript(text=text):
            return {"type": "transcript", "text": text}
        case Line(index=index, text=text):
            return {"type": "line", "index": index, "text": text}
        case Problem(text=text):
            return {"type": "problem", "text": text}
        case Play(song_ids=song_ids):
            return {"type": "play", "tracks": [f"/stream/{song_id}" for song_id in song_ids]}
        case StopMusic():
            return {"type": "stop_music"}
        case _:
            return {"type": "done"}
