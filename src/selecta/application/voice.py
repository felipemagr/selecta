"""Spoken turns: the listener's speech in, the agent's reply out as speech, interruptible."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from selecta.application.agent import Agent
from selecta.application.conversation import Session, Spoken
from selecta.application.ports import (
    PlaybackResult,
    SessionStore,
    SpeechError,
    SpeechToText,
    TextToSpeech,
)
from selecta.domain.library import PlayableTrack

logger = logging.getLogger(__name__)

# How long the player has to start a record before the agent says it can't.
PLAYER_TIMEOUT_SECONDS = 10


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
class Play:
    """Tracks for the player to play in order, each by its song id."""

    song_ids: list[str]


@dataclass
class StopMusic:
    """The player should stop the record."""


@dataclass
class Done:
    """The turn is over."""


Event = Transcript | Line | Audio | Problem | Play | StopMusic | Done


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
        self._player: asyncio.Future[PlaybackResult] | None = None
        self._turns = 0

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
        self._turns += 1
        this_turn = self._turns
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

        def play(tracks: list[PlayableTrack]) -> PlaybackResult:
            # A turn the listener cut off must not start its record.
            if this_turn != self._turns:
                return "cancelled"
            started = asyncio.run_coroutine_threadsafe(self._play(tracks, emit), loop)
            return started.result()

        def run() -> str:
            try:
                return self._agent.reply(self.session, text, say=say, play=play)
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

    def player_said(self, result: PlaybackResult) -> None:
        """Pass on what the player reported after a Play: it started, or it failed."""
        if self._player and not self._player.done():
            self._player.set_result(result)

    async def _play(
        self, tracks: list[PlayableTrack], emit: Callable[[Event], Awaitable[None]]
    ) -> PlaybackResult:
        """Send the tracks to the player and wait until it says whether it started."""
        self._player = asyncio.get_running_loop().create_future()
        await emit(Play([t.song_id for t in tracks]))
        try:
            return await asyncio.wait_for(self._player, PLAYER_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("the player didn't answer")
            # It may still start later, after the agent said it couldn't.
            await emit(StopMusic())
            return "failed"

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
