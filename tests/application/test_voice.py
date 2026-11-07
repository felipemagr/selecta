import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest

import selecta.application.voice as voice_module
from selecta.application.agent import Agent
from selecta.application.conversation import Message, Reply, Session, ToolSpec
from selecta.application.ports import SpeechError
from selecta.application.tools import Tools
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
from selecta.domain.collection import Release
from selecta.domain.library import PlayableTrack


class ScriptedModel:
    model = "scripted"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: Callable[[str], None] | None = None,
    ) -> Reply:
        text = self.texts.pop(0)
        if on_text:
            on_text(text)
        return Reply(text=text)


class Sessions:
    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save_session(self, session: Session) -> None:
        self.saved.append(session.model_copy(deep=True))

    def sessions_since(self, since: datetime) -> list[Session]:
        return []


class Ears:
    def __init__(self, text: str = "", fails: bool = False) -> None:
        self.text = text
        self.fails = fails
        self.heard = b""

    async def transcribe(self, pcm: bytes, language: str | None) -> str:
        self.heard += pcm
        if self.fails:
            raise SpeechError("down")
        return self.text


class Mouth:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails

    async def speak(self, text: str, language: str) -> AsyncIterator[bytes]:
        if self.fails:
            raise SpeechError("down")
        yield text.encode()


def _voice(ears: Ears, *replies: str, mouth: Mouth | None = None) -> VoiceSession:
    store = Sessions()
    agent = Agent(ScriptedModel(*replies), Tools(store, embedder=None), store)  # type: ignore[arg-type]
    session = Session(id="s", started_at=datetime(2026, 11, 6, tzinfo=UTC))
    return VoiceSession(agent, ears, mouth or Mouth(), store, session)  # type: ignore[arg-type]


def _turn(voice: VoiceSession, heard_lines: int | None = None) -> list[Event]:
    events: list[Event] = []

    async def audio() -> AsyncIterator[bytes]:
        yield b"pcm"

    async def emit(event: Event) -> None:
        events.append(event)

    asyncio.run(voice.turn(audio(), emit, heard_lines))
    return events


def test_reply_is_spoken_sentence_by_sentence_and_traced() -> None:
    voice = _voice(Ears("something calm"), "Calm it is. How long do you have?")

    events = _turn(voice)

    assert [e for e in events if not isinstance(e, Audio)] == [
        Transcript("something calm"),
        Line(0, "Calm it is."),
        Line(1, "How long do you have?"),
        Done(),
    ]
    spoken = voice.session.turns[0].spoken
    assert spoken is not None and spoken.lines == ["Calm it is.", "How long do you have?"]
    assert spoken.first_audio_ms is not None


def test_cutting_off_a_reply_keeps_only_what_was_heard() -> None:
    voice = _voice(Ears("tell me everything"), "First part. Second part. Third part.", "Sure.")
    _turn(voice)

    _turn(voice, heard_lines=1)

    spoken = voice.session.turns[0].spoken
    assert spoken is not None and spoken.heard == ["First part."]
    replies = [m.content for m in voice.session.messages if m.role == "assistant"]
    assert replies == ["First part.", "Sure."]


def test_cutting_off_before_any_reply_leaves_history_alone() -> None:
    voice = _voice(Ears("hi"), "Hello there.")

    _turn(voice, heard_lines=0)

    assert voice.session.turns[0].spoken is not None
    assert voice.session.turns[0].spoken.heard is None


def test_a_turn_dropped_before_the_agent_leaves_the_reply_before_it_heard() -> None:
    voice = _voice(Ears("hi"), "First part. Second part.", "Sure.")
    _turn(voice)

    async def dropped() -> None:
        async def endless() -> AsyncIterator[bytes]:
            await asyncio.Event().wait()
            yield b""

        async def emit(event: Event) -> None:
            pass

        task = asyncio.create_task(voice.turn(endless(), emit))
        await asyncio.sleep(0)
        task.cancel()

    asyncio.run(dropped())
    _turn(voice, heard_lines=0)

    replies = [m.content for m in voice.session.messages if m.role == "assistant"]
    assert replies == ["First part. Second part.", "Sure."]


def test_failed_transcription_asks_again_without_calling_the_agent() -> None:
    voice = _voice(Ears(fails=True))

    events = _turn(voice)

    assert isinstance(events[0], Problem) and events[-1] == Done()
    assert voice.session.turns == []


def test_failed_speech_still_shows_the_lines() -> None:
    voice = _voice(Ears("hi"), "Hello there.", mouth=Mouth(fails=True))

    events = _turn(voice)

    assert Line(0, "Hello there.") in events and events[-1] == Done()


def test_cut_off_keeps_the_heard_part_of_each_message() -> None:
    voice = _voice(Ears("hi"), "One. Two. Three.", "Okay.")
    _turn(voice)

    _turn(voice, heard_lines=2)

    replies = [m.content for m in voice.session.messages if m.role == "assistant"]
    assert replies == ["One. Two.", "Okay."]


def test_cut_off_matches_lines_that_start_with_a_number() -> None:
    voice = _voice(Ears("hi"), "Here's Album 1! 2 sides, 40 minutes. Enjoy.", "Okay.")
    _turn(voice)

    _turn(voice, heard_lines=1)

    replies = [m.content for m in voice.session.messages if m.role == "assistant"]
    assert replies == ["Here's Album 1! 2 sides, 40 minutes.", "Okay."]


def test_player_answer_is_passed_back_and_silence_means_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice = _voice(Ears("hi"))
    record = Release(id=1, title="Album", artists=["Someone"], artist_ids=[])
    track = PlayableTrack(position="A1", title="T", song_id="s1")
    sent: list[Event] = []

    async def emit(event: Event) -> None:
        sent.append(event)

    async def scenario() -> tuple[str, str]:
        answered = asyncio.create_task(voice._play(record, [track], emit))
        await asyncio.sleep(0)
        voice.player_said("playing")
        monkeypatch.setattr(voice_module, "PLAYER_TIMEOUT_SECONDS", 0.01)
        silent = await voice._play(record, [track], emit)
        return await answered, silent

    assert asyncio.run(scenario()) == ("playing", "failed")
    assert sent == [Play(record, [track]), Play(record, [track]), StopMusic()]
