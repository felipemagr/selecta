import json
from datetime import UTC, datetime

import pytest

from selecta.application.agent import MAX_MODEL_ROUNDS, Agent
from selecta.application.conversation import (
    Message,
    Pick,
    Rejection,
    Reply,
    Session,
    ToolCall,
    ToolSpec,
)
from selecta.application.ports import ModelError
from selecta.application.prompts import FALLBACK_REPLIES
from selecta.application.tools import Tools
from selecta.domain.brief import Answer, Brief
from selecta.domain.collection import ReleaseSummary

NOW = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)


class ScriptedModel:
    model = "scripted"

    def __init__(self, *replies: Reply | Exception) -> None:
        self.replies = list(replies)
        self.systems: list[str] = []

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.systems.append(system)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class MemoryStore:
    def __init__(self, *releases: ReleaseSummary) -> None:
        self.sessions: dict[str, Session] = {}
        self.releases = list(releases)

    def owned_summaries(
        self, mood: str | None = None, energy: str | None = None
    ) -> list[ReleaseSummary]:
        return [r for r in self.releases if mood in (None, r.mood) and energy in (None, r.energy)]

    def owned_summary(self, release_id: int) -> ReleaseSummary | None:
        return next((r for r in self.releases if r.id == release_id), None)

    def save_session(self, session: Session) -> None:
        self.sessions[session.id] = session.model_copy(deep=True)

    def load_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


def _update(**slots: object) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name="update_brief", arguments=slots)])


def _agent(model: ScriptedModel, *releases: ReleaseSummary) -> tuple[Agent, MemoryStore]:
    store = MemoryStore(*releases)
    tools = Tools(store, embedder=None)  # type: ignore[arg-type]
    return Agent(model, tools, store, now=lambda: NOW), store  # type: ignore[arg-type]


def test_answer_is_saved_to_brief_and_traced() -> None:
    model = ScriptedModel(
        _update(energy={"value": "low", "evidence": "something chill"}),
        Reply(text="Chill it is. How long do you have?"),
    )
    agent, store = _agent(model)
    session = agent.start()

    reply = agent.reply(session, "something chill")

    saved = store.sessions[session.id]
    assert reply == "Chill it is. How long do you have?"
    assert saved.brief.energy is not None and saved.brief.energy.value == "low"
    assert [r.call.name for r in saved.turns[0].tool_runs] == ["update_brief"]
    assert "Next to ask: mood" in model.systems[1]


def test_invalid_value_is_rejected_and_the_rest_saved() -> None:
    model = ScriptedModel(
        _update(
            minutes={"value": 1000, "evidence": "a thousand minutes"},
            occasion={"value": "dinner", "evidence": "dinner with friends"},
        ),
        Reply(text="That's a long dinner. How many minutes, really?"),
    )
    agent, _ = _agent(model)
    session = agent.start()

    agent.reply(session, "a thousand minutes, dinner with friends")

    result = json.loads(session.turns[0].tool_runs[0].result)
    assert session.brief.minutes is None and session.brief.occasion is not None
    assert result["saved"] == ["occasion"] and "minutes" in result["rejected"]


def test_model_that_never_answers_gets_the_fallback() -> None:
    model = ScriptedModel(*[_update() for _ in range(MAX_MODEL_ROUNDS)])
    agent, _ = _agent(model)

    assert agent.reply(agent.start(), "hi") == FALLBACK_REPLIES["en"]


def test_model_failure_gets_the_fallback_and_is_traced() -> None:
    agent, store = _agent(ScriptedModel(ModelError("503")))
    session = agent.start()

    reply = agent.reply(session, "hi")

    assert reply == FALLBACK_REPLIES["en"]
    assert store.sessions[session.id].turns[0].error == "503"


class BrokenStore(MemoryStore):
    def search(self, query: object) -> list[object]:
        raise RuntimeError("database gone")


def test_failing_tool_is_reported_to_the_model_not_raised() -> None:
    call = ToolCall(id="c1", name="search_collection", arguments={"text": "kraftwerk"})
    model = ScriptedModel(Reply(tool_calls=[call]), Reply(text="I can't check right now."))
    store = BrokenStore()
    agent = Agent(model, Tools(store, embedder=None), store, now=lambda: NOW)  # type: ignore[arg-type]
    session = agent.start()

    reply = agent.reply(session, "do I have kraftwerk?")

    assert reply == "I can't check right now."
    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "assistant"]


def test_null_slot_is_ignored() -> None:
    model = ScriptedModel(_update(minutes=None), Reply(text="How long do you have?"))
    agent, _ = _agent(model)
    session = agent.start()

    agent.reply(session, "hmm")

    assert session.brief.minutes is None


def test_text_sent_with_tool_calls_is_spoken() -> None:
    first = Reply(text="Let me look.", tool_calls=_update().tool_calls)
    agent, _ = _agent(ScriptedModel(first, Reply(text="What mood are you in?")))

    assert agent.reply(agent.start(), "hi") == "Let me look. What mood are you in?"


@pytest.mark.parametrize("language", ["en", "es", "fr"])
def test_fallback_is_in_the_listener_language(language: str) -> None:
    model = ScriptedModel(_update(language=language), ModelError("503"))
    agent, _ = _agent(model)

    assert agent.reply(agent.start(), "hi") == FALLBACK_REPLIES[language]


def _record(id: int, mood: str = "calm", minutes: int | None = 40) -> ReleaseSummary:
    return ReleaseSummary(
        id=id,
        title=f"Album {id}",
        artists=["Someone"],
        original_year=1970,
        label=None,
        format=None,
        genres=[],
        styles=[],
        mood=mood,
        energy="low",
        duration_seconds=minutes * 60 if minutes else None,
    )


def _complete_brief() -> Brief:
    return Brief(
        mood=Answer(value="calm", evidence="calm"),
        energy=Answer(value="low", evidence="chill"),
        minutes=Answer(value=45, evidence="45 minutes"),
        occasion=Answer(value="solo", evidence="just me"),
    )


def _pick(release_id: int | str) -> Reply:
    args = {"release_id": release_id, "reason": "x"}
    return Reply(tool_calls=[ToolCall(id="p1", name="pick_record", arguments=args)])


def _offered(*records: ReleaseSummary) -> str:
    model = ScriptedModel(Reply(text="Let me think."))
    agent, _ = _agent(model, *records)
    session = agent.start()
    session.brief = _complete_brief()
    agent.reply(session, "that's all")
    return model.systems[0]


def test_complete_brief_puts_candidates_in_the_prompt() -> None:
    system = _offered(_record(1), _record(2), _record(3))

    assert "Pick one with pick_record" in system and "- 1: Someone - Album 1" in system


@pytest.mark.parametrize(
    ("slot", "value", "dropped"),
    [("mood", "upbeat", True), ("minutes", 20, True), ("mood", "calm", False)],
)
def test_changing_the_brief_after_a_pick_drops_it(slot: str, value: object, dropped: bool) -> None:
    change = _update(language="en", **{slot: {"value": value, "evidence": "actually"}})
    agent, _ = _agent(ScriptedModel(change, Reply(text="Sure.")), _record(1))
    session = agent.start()
    session.brief = _complete_brief()
    session.pick = Pick(release_id=1, title="Album 1", artists=["Someone"], reason="x")

    agent.reply(session, "actually")

    assert (session.pick is None) == dropped


def test_rejected_record_can_be_picked_when_asked_by_name() -> None:
    args = {"release_id": 1, "reason": "they asked", "asked_by_name": True}
    pick = Reply(tool_calls=[ToolCall(id="p1", name="pick_record", arguments=args)])
    agent, _ = _agent(ScriptedModel(pick, Reply(text="Album 1 it is.")), _record(1))
    session = agent.start()
    session.rejections = [Rejection(release_id=1, reason="no")]

    agent.reply(session, "actually, put Album 1 on")

    assert session.pick is not None and session.rejected_ids() == [1]


@pytest.mark.parametrize(
    ("release_id", "asked_by_name", "complete", "refusal"),
    [
        (3, False, True, "Pick from the candidates"),
        (1, False, True, "The listener already rejected that record"),
        (2, False, False, "Finish the interview first"),
        (99, True, True, "That record isn't in the listener's collection"),
    ],
)
def test_pick_is_refused(release_id: int, asked_by_name: bool, complete: bool, refusal: str):
    args = {"release_id": release_id, "reason": "x", "asked_by_name": asked_by_name}
    pick = Reply(tool_calls=[ToolCall(id="p", name="pick_record", arguments=args)])
    records = _record(1), _record(2), _record(4), _record(5), _record(3, mood="dark")
    agent, _ = _agent(ScriptedModel(pick, Reply(text="Hmm.")), *records)
    session = agent.start()
    session.brief = _complete_brief() if complete else Brief()
    session.rejections = [Rejection(release_id=1, reason="too sad")]

    agent.reply(session, "go")

    assert session.pick is None
    assert session.turns[0].tool_runs[0].result.startswith(refusal)


def test_text_sent_with_a_refused_pick_is_not_spoken() -> None:
    announce = Reply(text="Here's Album 9!", tool_calls=_pick(9).tool_calls)
    agent, _ = _agent(ScriptedModel(announce, Reply(text="Let me find another.")), _record(1))
    session = agent.start()
    session.brief = _complete_brief()

    assert agent.reply(session, "go") == "Let me find another."


def test_text_sent_with_a_pick_dropped_in_the_same_round_is_not_spoken() -> None:
    change = ToolCall(
        id="u",
        name="update_brief",
        arguments={"language": "en", "mood": {"value": "dark", "evidence": "darker"}},
    )
    both = Reply(text="Here's Album 1!", tool_calls=[*_pick(1).tool_calls, change])
    agent, _ = _agent(ScriptedModel(both, Reply(text="Darker, then.")), _record(1))
    session = agent.start()
    session.brief = _complete_brief()

    assert agent.reply(session, "something darker") == "Darker, then."


def test_pick_made_mid_interview_survives_answering_a_slot() -> None:
    by_name = {"release_id": 1, "reason": "asked", "asked_by_name": True}
    pick = Reply(tool_calls=[ToolCall(id="p", name="pick_record", arguments=by_name)])
    answer = _update(language="en", mood={"value": "calm", "evidence": "calm"})
    model = ScriptedModel(pick, Reply(text="Album 1."), answer, Reply(text="Noted."))
    agent, _ = _agent(model, _record(1))
    session = agent.start()

    agent.reply(session, "put Album 1 on")
    agent.reply(session, "something calm")

    assert session.pick is not None
    assert "Current pick: Someone - Album 1" in model.systems[-1]


@pytest.mark.parametrize(("minutes", "note"), [(40, "fits"), (None, "unknown"), (90, "longer")])
def test_pick_result_says_how_the_length_fits(minutes: int | None, note: str) -> None:
    model = ScriptedModel(_pick(1), Reply(text="Here you go."))
    agent, _ = _agent(model, _record(1, minutes=minutes))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go")

    assert session.pick is not None and note in session.turns[0].tool_runs[0].result


def test_rejected_pick_leaves_the_candidates() -> None:
    reject = Reply(tool_calls=[ToolCall(id="r1", name="reject_pick", arguments={})])
    model = ScriptedModel(_pick(1), Reply(text="Album 1?"), reject, Reply(text="Another one?"))
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go")
    agent.reply(session, "not that one")

    assert session.pick is None
    assert session.rejections == [Rejection(release_id=1, reason="not that one")]
    assert "- 1:" not in model.systems[-1]


def test_summary_keeps_key_points_when_the_model_fails() -> None:
    model = ScriptedModel(Reply(text="Hi."), ModelError("503"))
    agent, store = _agent(model)
    session = agent.start()
    agent.reply(session, "hi")

    summary = agent.finish(session)

    assert summary.text is None and summary.key_points["turns"] == 1
    assert store.sessions[session.id].summary == summary
