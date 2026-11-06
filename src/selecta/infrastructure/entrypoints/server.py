"""The voice server: a push-to-talk page and the WebSocket it talks over."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib.resources import files

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from selecta.application.voice import (
    Audio,
    Done,
    Event,
    Line,
    Problem,
    Transcript,
    VoiceSession,
)
from selecta.infrastructure import config

logger = logging.getLogger(__name__)

_PAGE = files(__package__).joinpath("static", "talk.html").read_text()


def create_app(open_session: Callable[[], Awaitable[VoiceSession]]) -> FastAPI:
    """Return the app. `open_session` starts a new spoken conversation for each connection.

    Protocol over /talk: the phone sends {"type": "start", "heard_lines": n | null} when the button
    is pressed (with `heard_lines` set if that cut off the agent), binary 16 kHz 16-bit PCM while it
    is held, and {"type": "stop"} on release. The server answers "start" with {"type": "started"},
    then sends JSON events (transcript, line, problem, done) and the audio of
    each line as binary float32 PCM at 24 kHz.
    """
    app = FastAPI()

    @app.get("/")
    async def page() -> HTMLResponse:
        return HTMLResponse(_PAGE)

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
        case _:
            return {"type": "done"}
