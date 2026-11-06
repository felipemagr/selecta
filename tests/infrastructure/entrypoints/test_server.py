import json
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
from fastapi.testclient import TestClient

from selecta.application.voice import Audio, Done, Event, Line, Transcript
from selecta.infrastructure.entrypoints.server import create_app


class FakeVoice:
    def __init__(self) -> None:
        self.heard: list[bytes] = []
        self.cut_offs: list[int | None] = []
        self.closed = False
        self.player: list[str] = []

    def player_said(self, result: str) -> None:
        self.player.append(result)

    async def turn(
        self,
        audio: AsyncIterator[bytes],
        emit: Callable[[Event], Awaitable[None]],
        heard_lines: int | None = None,
    ) -> None:
        self.cut_offs.append(heard_lines)
        self.heard += [chunk async for chunk in audio]
        await emit(Transcript("hi"))
        await emit(Line(0, "Hello."))
        await emit(Audio(b"\x00\x00\x80\x3f"))
        await emit(Done())

    async def close(self) -> None:
        self.closed = True


def _client(voice: FakeVoice) -> TestClient:
    async def open_session() -> FakeVoice:
        return voice

    async def open_song(song_id: str, byte_range: str | None) -> httpx.Response:
        headers = {"content-type": "audio/flac"}
        body = httpx.ByteStream(b"flac " + song_id.encode())
        return httpx.Response(200, stream=body, headers=headers)

    return TestClient(create_app(open_session, open_song))  # type: ignore[arg-type]


def test_push_to_talk_round_trip() -> None:
    voice = FakeVoice()
    with _client(voice).websocket_connect("/talk") as ws:
        ws.send_text(json.dumps({"type": "start", "heard_lines": None}))
        ws.send_bytes(b"\x01\x00")
        ws.send_text(json.dumps({"type": "stop"}))

        received = [ws.receive_json() for _ in range(3)] + [ws.receive_bytes(), ws.receive_json()]

    assert received == [
        {"type": "started"},
        {"type": "transcript", "text": "hi"},
        {"type": "line", "index": 0, "text": "Hello."},
        b"\x00\x00\x80\x3f",
        {"type": "done"},
    ]
    assert voice.heard == [b"\x01\x00"] and voice.cut_offs == [None] and voice.closed


def test_pressing_while_the_agent_speaks_records_what_was_heard() -> None:
    voice = FakeVoice()
    with _client(voice).websocket_connect("/talk") as ws:
        ws.send_text(json.dumps({"type": "start", "heard_lines": 2}))
        ws.send_text(json.dumps({"type": "stop"}))
        ws.receive_json()
        ws.receive_json()

    assert voice.cut_offs == [2]


def test_page_is_served() -> None:
    assert "Hold to talk" in _client(FakeVoice()).get("/").text


def test_songs_stream_through_the_server() -> None:
    response = _client(FakeVoice()).get("/stream/s1")

    assert response.headers["content-type"] == "audio/flac" and response.content == b"flac s1"


def test_player_reports_reach_the_session() -> None:
    voice = FakeVoice()
    with _client(voice).websocket_connect("/talk") as ws:
        ws.send_text(json.dumps({"type": "play_started"}))
        ws.send_text(json.dumps({"type": "play_failed"}))

    assert voice.player == ["playing", "failed"]
