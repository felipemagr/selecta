import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi.testclient import TestClient

from selecta.application.voice import Audio, Done, Event, Line, Transcript
from selecta.infrastructure.entrypoints.server import create_app


class FakeVoice:
    def __init__(self) -> None:
        self.heard: list[bytes] = []
        self.cut_offs: list[int | None] = []
        self.closed = False

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

    return TestClient(create_app(open_session))  # type: ignore[arg-type]


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
