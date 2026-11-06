import asyncio
import base64
import json

import httpx
import pytest

from selecta.application.ports import SpeechError
from selecta.infrastructure.integrations import voxtral
from selecta.infrastructure.integrations.voxtral import VoxtralSpeaker, VoxtralTranscriber


def _http(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return voxtral.connect("key", handler)


def _sse(*events: dict[str, object]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_speech_yields_whole_samples_across_deltas() -> None:
    body = _sse(
        {"type": "speech.audio.delta", "audio_data": base64.b64encode(b"abc").decode()},
        {"type": "speech.audio.delta", "audio_data": base64.b64encode(b"defgh").decode()},
        {"type": "speech.audio.done"},
    )
    http = _http(httpx.MockTransport(lambda _: httpx.Response(200, text=body)))

    async def collect() -> list[bytes]:
        return [chunk async for chunk in VoxtralSpeaker(http).speak("Hi.", "en")]

    assert asyncio.run(collect()) == [b"abcdefgh"]


def test_speech_is_retried_once_before_any_audio() -> None:
    body = _sse(
        {"type": "speech.audio.delta", "audio_data": base64.b64encode(b"abcd").decode()},
        {"type": "speech.audio.done"},
    )
    responses = [httpx.Response(500), httpx.Response(200, text=body)]
    http = _http(httpx.MockTransport(lambda _: responses.pop(0)))

    async def collect() -> list[bytes]:
        return [chunk async for chunk in VoxtralSpeaker(http).speak("Hi.", "en")]

    assert asyncio.run(collect()) == [b"abcd"]


def test_speech_failure_raises_speech_error() -> None:
    http = _http(httpx.MockTransport(lambda _: httpx.Response(429)))

    async def collect() -> list[bytes]:
        return [chunk async for chunk in VoxtralSpeaker(http).speak("Hi.", "en")]

    with pytest.raises(SpeechError):
        asyncio.run(collect())


def _transcribe(response: httpx.Response) -> tuple[str, list[bytes]]:
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return response

    transcriber = VoxtralTranscriber(_http(httpx.MockTransport(handler)), ["Robotsband"])

    return asyncio.run(transcriber.transcribe(b"\x01\x00\x02\x00", "en")), sent


def test_whole_utterance_is_sent_with_the_vocabulary() -> None:
    text, sent = _transcribe(httpx.Response(200, json={"text": " Robotsband, please. "}))

    assert text == "Robotsband, please."
    assert b"\x01\x00\x02\x00" in sent[0] and b"Robotsband" in sent[0]


def test_failed_transcription_raises_speech_error() -> None:
    with pytest.raises(SpeechError):
        _transcribe(httpx.Response(500))
