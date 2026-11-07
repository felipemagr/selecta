import json
import time
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

    async def open_cover(release_id: int) -> httpx.Response | None:
        if release_id != 1:
            return None
        body = httpx.ByteStream(b"jpeg")
        return httpx.Response(200, stream=body, headers={"content-type": "image/jpeg"})

    return TestClient(create_app(open_session, open_song, open_cover))  # type: ignore[arg-type]


def _press(phone: object, *chunks: bytes) -> None:
    phone.send_text(json.dumps({"type": "start"}))  # type: ignore[attr-defined]
    for chunk in chunks:
        phone.send_bytes(chunk)  # type: ignore[attr-defined]
    phone.send_text(json.dumps({"type": "stop"}))  # type: ignore[attr-defined]


def _until_done(tv: object) -> None:
    while tv.receive().get("text") != json.dumps({"type": "done"}):  # type: ignore[attr-defined]
        pass


def _report_heard(tv: object, lines: int, turn: int) -> None:
    tv.send_text(json.dumps({"type": "heard", "lines": lines, "turn": turn}))  # type: ignore[attr-defined]
    # The TV's and the phone's sockets have no order between them: let the report land first.
    time.sleep(0.1)


def test_the_phone_talks_and_the_tv_shows_and_speaks() -> None:
    voice = FakeVoice()
    with (
        _client(voice) as client,
        client.websocket_connect("/screen") as tv,
        client.websocket_connect("/talk") as phone,
    ):
        assert phone.receive_json() == {"type": "screen", "on": True}
        _press(phone, b"\x00\x40")

        on_tv = [tv.receive_json() for _ in range(4)] + [tv.receive_bytes(), tv.receive_json()]
        on_phone = [phone.receive_json() for _ in range(4)]

    assert on_tv == [
        {"type": "started", "turn": 1},
        {"type": "level", "value": 1.0},
        {"type": "transcript", "text": "hi"},
        {"type": "line", "index": 0, "text": "Hello."},
        b"\x00\x00\x80\x3f",
        {"type": "done"},
    ]
    assert on_phone == [
        {"type": "started", "turn": 1},
        {"type": "transcript", "text": "hi"},
        {"type": "line", "index": 0, "text": "Hello."},
        {"type": "done"},
    ]
    assert voice.heard == [b"\x00\x40"] and voice.cut_offs == [None]


def test_pressing_before_the_tv_heard_everything_records_what_it_heard() -> None:
    voice = FakeVoice()
    with (
        _client(voice) as client,
        client.websocket_connect("/screen") as tv,
        client.websocket_connect("/talk") as phone,
    ):
        _press(phone)
        _until_done(tv)
        _press(phone)
        _until_done(tv)
        _report_heard(tv, lines=1, turn=2)
        _press(phone)
        _until_done(tv)

    assert voice.cut_offs == [None, 0, None]


def test_another_phone_leaving_does_not_end_the_recording() -> None:
    voice = FakeVoice()
    with (
        _client(voice) as client,
        client.websocket_connect("/screen") as tv,
        client.websocket_connect("/talk") as phone,
    ):
        phone.send_text(json.dumps({"type": "start"}))
        phone.send_bytes(b"\x01\x00")
        with client.websocket_connect("/talk"):
            pass
        time.sleep(0.1)
        phone.send_bytes(b"\x02\x00")
        phone.send_text(json.dumps({"type": "stop"}))
        _until_done(tv)

    assert voice.heard == [b"\x01\x00", b"\x02\x00"]


def test_without_a_tv_the_phone_is_told_and_nothing_starts() -> None:
    voice = FakeVoice()
    with _client(voice) as client, client.websocket_connect("/talk") as phone:
        assert phone.receive_json() == {"type": "screen", "on": False}
        _press(phone)

        assert phone.receive_json()["type"] == "problem"
    assert voice.cut_offs == []


def test_player_reports_reach_the_session_and_it_closes_on_shutdown() -> None:
    voice = FakeVoice()
    with (
        _client(voice) as client,
        client.websocket_connect("/screen") as tv,
        client.websocket_connect("/talk") as phone,
    ):
        _press(phone)
        _until_done(tv)
        tv.send_text(json.dumps({"type": "play_started"}))
        tv.send_text(json.dumps({"type": "play_failed"}))
        _report_heard(tv, lines=1, turn=1)
        _press(phone)
        _until_done(tv)

    assert voice.player == ["playing", "failed"] and voice.closed


def test_pages_are_served() -> None:
    client = _client(FakeVoice())

    assert all(client.get(path).status_code == 200 for path in ("/", "/tv"))


def test_songs_stream_through_the_server() -> None:
    response = _client(FakeVoice()).get("/stream/s1")

    assert response.headers["content-type"] == "audio/flac" and response.content == b"flac s1"


def test_covers_are_served_and_missing_ones_are_not_found() -> None:
    client = _client(FakeVoice())

    assert client.get("/cover/1").content == b"jpeg" and client.get("/cover/2").status_code == 404
