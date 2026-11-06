"""Spoken turns: the listener's speech in, the agent's reply out as speech, interruptible."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from selecta.application.agent import Agent
from selecta.application.conversation import Session, Spoken
from selecta.application.ports import SessionStore, SpeechError, SpeechToText, TextToSpeech

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    """The listener's whole utterance."""

    text: str


@dataclass
class Line:
    """A line the agent is about to speak. The audio that follows belongs to it."""

    index: int
    text: str


@dataclass
class Audio:
    """A chunk of the current line's audio."""

    pcm: bytes


@dataclass
class Problem:
    """Something the listener should know went wrong."""

    text: str


@dataclass
class Done:
    """The turn is over."""


Event = Transcript | Line | Audio | Problem | Done


class VoiceSession:
    """One spoken conversation. Turns run one at a time; a new one can cut off the last."""

    def __init__(
        self,
        agent: Agent,
        stt: SpeechToText,
        tts: TextToSpeech,
        store: SessionStore,
        session: Session,
    ) -> None:
        self._agent = agent
        self._stt = stt
        self._tts = tts
        self._store = store
        self.session = session
        # The agent blocks on network calls, so it runs in a thread that outlives a cancelled turn.
        self._thread = ThreadPoolExecutor(max_workers=1)
        self._job: Future[str] | None = None
        self._lines: list[str] = []
        self._first_audio_ms: int | None = None
        self._spoken_turn: int | None = None

    async def turn(
        self,
        audio: AsyncIterator[bytes],
        emit: Callable[[Event], Awaitable[None]],
        heard_lines: int | None = None,
    ) -> None:
        """Transcribe the utterance, run the agent and speak its reply, emitting each step.

        `heard_lines` is set when this press cut off the previous reply: how many of its lines
        finished playing. The previous agent call may still be running; it is awaited after the
        utterance is recorded and transcribed.
        """
        language = self.session.language if self.session.turns else None
        pcm = b"".join([chunk async for chunk in audio])
        try:
            text = await self._stt.transcribe(pcm, language)
        except SpeechError as e:
            logger.warning("transcription failed: %s", e)
            text = None
        await self._settle()
        if heard_lines is not None:
            await self._cut_off(heard_lines)
        if text is None:
            await emit(Problem("I didn't catch that. Try again?"))
            await emit(Done())
            return
        heard_at = time.monotonic()
        await emit(Transcript(text))
        if not text:
            await emit(Done())
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def say(line: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, line)

        def run() -> str:
            try:
                return self._agent.reply(self.session, text, say=say)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        self._lines, self._first_audio_ms = [], None
        self._spoken_turn = len(self.session.turns)
        self._job = self._thread.submit(run)
        await self._speak(queue, emit, heard_at)
        await asyncio.wrap_future(self._job)
        self.session.turns[self._spoken_turn].spoken = Spoken(
            lines=self._lines, first_audio_ms=self._first_audio_ms
        )
        await self._save()
        await emit(Done())

    async def _cut_off(self, heard_lines: int) -> None:
        """Record that the listener cut off the last reply after `heard_lines` lines."""
        index, self._spoken_turn = self._spoken_turn, None
        if index is None or index >= len(self.session.turns):
            return
        turn = self.session.turns[index]
        heard = self._lines[:heard_lines]
        turn.spoken = Spoken(lines=self._lines, heard=heard, first_audio_ms=self._first_audio_ms)
        # The model's history should hold what the listener heard, not what was planned: each of
        # this turn's assistant messages keeps only its lines that were heard.
        start = max(i for i, m in enumerate(self.session.messages) if m.role == "user")
        left = list(heard)
        for message in self.session.messages[start:]:
            if message.role != "assistant" or not message.content:
                continue
            # A message holds some of the turn's lines, joined: keep the heard ones at its start.
            kept: list[str] = []
            while left and message.content.startswith(" ".join([*kept, left[0]])):
                kept.append(left.pop(0))
            message.content = " ".join(kept) or None
        final = self.session.messages[-1]
        if final.role == "assistant" and not final.tool_calls and not final.content:
            final.content = "(The listener cut this reply off unheard.)"
        await self._save()

    async def close(self) -> None:
        """Wait for the last turn, then write the session summary."""
        await self._settle()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._thread, self._agent.finish, self.session)
        self._thread.shutdown()

    async def _speak(
        self,
        queue: "asyncio.Queue[str | None]",
        emit: Callable[[Event], Awaitable[None]],
        heard_at: float,
    ) -> None:
        while (line := await queue.get()) is not None:
            await emit(Line(len(self._lines), line))
            self._lines.append(line)
            try:
                async for pcm in self._tts.speak(line, self.session.language):
                    if self._first_audio_ms is None:
                        self._first_audio_ms = round((time.monotonic() - heard_at) * 1000)
                    await emit(Audio(pcm))
            except SpeechError as e:
                logger.warning("speech failed, the line is shown only: %s", e)

    async def _settle(self) -> None:
        """Wait for a turn's agent call that is still running, even if its turn was cancelled."""
        if self._job and not self._job.done():
            await asyncio.wrap_future(self._job)

    async def _save(self) -> None:
        # In the session's thread: the store is synchronous and the agent writes there too.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._thread, self._store.save_session, self.session)
