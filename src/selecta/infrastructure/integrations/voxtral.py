"""Mistral Voxtral transcription and streamed speech."""

import base64
import io
import json
import wave
from collections.abc import AsyncIterator

import httpx

from selecta.application.ports import SpeechError
from selecta.infrastructure import config


class VoxtralTranscriber:
    """SpeechToText on Voxtral's batch transcription: the whole utterance, once it ends."""

    def __init__(self, http: httpx.AsyncClient, vocabulary: list[str]) -> None:
        """Wire the transcriber. `vocabulary` are names to expect, like the listener's artists."""
        self._http = http
        # The endpoint takes the names comma-separated.
        self._vocabulary = ",".join(name.replace(",", " ") for name in vocabulary)

    async def transcribe(self, pcm: bytes, language: str | None) -> str:
        """Return the transcript of the utterance."""
        data = {"model": config.VOICE.STT_MODEL, "context_bias": self._vocabulary}
        if language:
            data["language"] = language
        files = {"file": ("utterance.wav", _wav(pcm), "audio/wav")}
        try:
            response = await self._http.post("/audio/transcriptions", data=data, files=files)
            response.raise_for_status()
            return response.json()["text"].strip()
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise SpeechError(f"Voxtral transcription failed: {e}") from e


class VoxtralSpeaker:
    """TextToSpeech on Voxtral's speech endpoint, streamed as raw float32 PCM."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def speak(self, text: str, language: str) -> AsyncIterator[bytes]:
        """Yield the spoken text as float32 PCM chunks at 24 kHz, as Voxtral returns it."""
        body = {
            "model": config.VOICE.TTS_MODEL,
            "input": text,
            "voice_id": config.VOICE.VOICES[language],
            "response_format": "pcm",
            "stream": True,
        }
        for attempt in range(config.VOICE.TTS_ATTEMPTS):
            started = False
            try:
                async for pcm in self._stream(body):
                    started = True
                    yield pcm
                return
            except (httpx.HTTPError, ValueError, KeyError) as e:
                # Retried only before any audio, so a line is never heard twice.
                if started or attempt == config.VOICE.TTS_ATTEMPTS - 1:
                    raise SpeechError(f"Voxtral speech failed: {e}") from e

    async def _stream(self, body: dict[str, object]) -> AsyncIterator[bytes]:
        # Chunks can split a 4-byte sample; the remainder waits for the next chunk.
        leftover = b""
        async with self._http.stream("POST", "/audio/speech", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line.removeprefix("data:"))
                if event.get("type") == "speech.audio.delta":
                    data = leftover + base64.b64decode(event["audio_data"])
                    whole = len(data) - len(data) % 4
                    leftover = data[whole:]
                    if whole:
                        yield data[:whole]
                elif event.get("type") == "speech.audio.done":
                    return


def _wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.VOICE.STT_SAMPLE_RATE)
        w.writeframes(pcm)
    return buffer.getvalue()


def connect(api_key: str, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """Return an async HTTP client authenticated against the Mistral API."""
    return httpx.AsyncClient(
        base_url=config.MISTRAL.API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config.VOICE.TIMEOUT_SECONDS,
        transport=transport,
    )
