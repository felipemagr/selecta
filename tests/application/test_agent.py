import json
from datetime import UTC, datetime

import pytest

from selecta.application.agent import MAX_MODEL_ROUNDS, Agent
from selecta.application.conversation import Message, Reply, Session, ToolCall, ToolSpec
from selecta.application.ports import ModelError
from selecta.application.prompts import FALLBACK_REPLIES
from selecta.application.tools import Tools

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
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def save_session(self, session: Session) -> None:
        self.sessions[session.id] = session.model_copy(deep=True)

    def load_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)


def _update(**slots: object) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name="update_brief", arguments=slots)])


def _agent(model: ScriptedModel) -> tuple[Agent, MemoryStore]:
    store = MemoryStore()
    return Agent(model, Tools(store, embedder=None), store, now=lambda: NOW), store  # type: ignore[arg-type]


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
