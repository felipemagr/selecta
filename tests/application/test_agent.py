import json
from collections.abc import Callable
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
from selecta.application.lookup import Lookup
from selecta.application.ports import ModelError, Ranked, WebPage, WebPassage, WebSearchError
from selecta.application.prompts import FALLBACK_REPLIES, WAIT_LINES
from selecta.application.tools import Tools
from selecta.domain.brief import Answer, Brief
from selecta.domain.collection import ReleaseSummary

NOW = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)


class ScriptedModel:
    model = "scripted"

    def __init__(self, *replies: Reply | Exception) -> None:
        self.replies = list(replies)
        self.systems: list[str] = []

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: Callable[[str], None] | None = None,
    ) -> Reply:
        self.systems.append(system)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if on_text and reply.text:
            on_text(reply.text)
        return reply


class MemoryStore:
    def __init__(self, *releases: ReleaseSummary) -> None:
        self.sessions: dict[str, Session] = {}
        self.releases = list(releases)
        self.web: list[WebPassage] = []

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

    def sessions_since(self, since: datetime) -> list[Session]:
        found = [s for s in self.sessions.values() if s.started_at >= since]
        return sorted(found, key=lambda s: s.started_at)

    def artist_urls(self, release_id: int) -> list[str]:
        return []

    def web_passages(self, release_id: int) -> list[WebPassage]:
        return []

    def add_web_passages(self, release_id: int, passages: list[WebPassage], keep: int) -> None:
        self.web += passages

    def mark_used(self, release_id: int, positions: list[int]) -> None:
        pass


class AnswersEverything:
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[Ranked]:
        return [Ranked(index=i, score=1) for i in range(len(documents))][:top_n]


class FakeWeb:
    def __init__(self, *pages: WebPage, fails: bool = False) -> None:
        self.pages = list(pages)
        self.fails = fails

    def search(
        self, query: str, include_domains: list[str], exclude_domains: list[str]
    ) -> list[WebPage]:
        if self.fails:
            raise WebSearchError("down")
        return self.pages


def _update(**slots: object) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name="update_brief", arguments=slots)])


def _agent(
    model: ScriptedModel, *releases: ReleaseSummary, web: FakeWeb | None = None
) -> tuple[Agent, MemoryStore]:
    store = MemoryStore(*releases)
    lookup = Lookup(store, web, AnswersEverything()) if web else None  # type: ignore[arg-type]
    tools = Tools(store, embedder=None, lookup=lookup)  # type: ignore[arg-type]
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


def test_saving_an_answer_and_asking_the_next_question_takes_one_model_call() -> None:
    saved = _update(mood={"value": "calm", "evidence": "calm"})
    first = Reply(text="Calm it is. How much energy?", tool_calls=saved.tool_calls)
    agent, _ = _agent(ScriptedModel(first))
    session = agent.start()

    assert agent.reply(session, "something calm") == "Calm it is. How much energy?"
    assert session.turns[0].model_rounds == 1
    roles = [m.role for m in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.messages[-1].content == "Calm it is. How much energy?"


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


@pytest.mark.parametrize(
    ("minutes", "mentioned"), [(40, False), (None, False), (60, False), (90, True)]
)
def test_length_is_mentioned_only_when_much_longer(minutes: int | None, mentioned: bool) -> None:
    model = ScriptedModel(_pick(1), Reply(text="Here you go."))
    agent, _ = _agent(model, _record(1, minutes=minutes))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go")

    assert session.pick is not None
    assert ("one side is fine" in session.turns[0].tool_runs[0].result) is mentioned


def test_rejected_pick_leaves_the_candidates() -> None:
    reject = Reply(tool_calls=[ToolCall(id="r1", name="reject_pick", arguments={})])
    model = ScriptedModel(_pick(1), Reply(text="Album 1?"), reject, Reply(text="Another one?"))
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go")
    agent.reply(session, "not that one")

    assert session.pick is None
    assert [(r.release_id, r.reason) for r in session.rejections] == [(1, "not that one")]
    assert "- 1:" not in model.systems[-1]


def test_summary_keeps_key_points_when_the_model_fails() -> None:
    model = ScriptedModel(Reply(text="Hi."), ModelError("503"))
    agent, store = _agent(model)
    session = agent.start()
    agent.reply(session, "hi")

    summary = agent.finish(session)

    assert summary.text is None and summary.key_points["turns"] == 1
    assert store.sessions[session.id].summary == summary


def _look_up(question: str = "who produced it", text: str | None = None) -> Reply:
    args = {"release_id": 1, "question": question}
    return Reply(text=text, tool_calls=[ToolCall(id="w1", name="look_up", arguments=args)])


def test_web_search_says_a_wait_line_and_stores_what_it_found() -> None:
    page = WebPage(url="https://en.wikipedia.org/wiki/A", title="A", content="Recorded in 1970.")
    model = ScriptedModel(
        _look_up("how was it recorded"), Reply(text="In 1970."), _look_up("who"), Reply(text="B.")
    )
    agent, store = _agent(model, _record(1), web=FakeWeb(page))
    session = agent.start()

    agent.reply(session, "how was Album 1 recorded?")
    agent.reply(session, "and who played on it?")

    first, second = (t.said_while_waiting for t in session.turns)
    assert first == WAIT_LINES["en"][0] and second == WAIT_LINES["en"][1]
    assert "https://en.wikipedia.org/wiki/A" in session.turns[0].tool_runs[0].result
    assert {p.url for p in store.web} == {"https://en.wikipedia.org/wiki/A"}


def test_web_failure_is_told_to_the_model_not_raised() -> None:
    model = ScriptedModel(_look_up(), Reply(text="I can't check that right now."))
    agent, _ = _agent(model, _record(1), web=FakeWeb(fails=True))
    session = agent.start()

    reply = agent.reply(session, "who produced Album 1?")

    assert reply == "Let me look that up. I can't check that right now."
    assert "can't check" in session.turns[0].tool_runs[0].result


def test_model_text_sent_with_a_search_is_the_wait_line_and_said_once() -> None:
    said: list[str] = []
    page = WebPage(url="https://a.example", title="A", content="Produced by B.")
    model = ScriptedModel(_look_up(text="Let me check."), Reply(text="B produced it."))
    agent, store = _agent(model, _record(1), web=FakeWeb(page))
    session = agent.start()

    reply = agent.reply(session, "who produced Album 1?", say=said.append)

    assert said == ["Let me check.", "B produced it."]
    assert reply == "Let me check. B produced it."


def _past_session(started_at: datetime, picked: int | None, rejected: dict[int, str]) -> Session:
    pick = (
        Pick(release_id=picked, title=f"Album {picked}", artists=["Someone"], reason="x")
        if picked
        else None
    )
    return Session(
        id=f"past-{started_at:%d}",
        started_at=started_at,
        pick=pick,
        rejections=[
            Rejection(release_id=i, reason=why, title=f"Album {i}", artists=["Someone"])
            for i, why in rejected.items()
        ],
    )


def test_past_sessions_put_remembered_records_after_fresh_ones_with_their_history() -> None:
    model = ScriptedModel(Reply(text="Something calm, then?"))
    agent, store = _agent(model, _record(1), _record(2), _record(3))
    store.save_session(_past_session(datetime(2026, 11, 1, tzinfo=UTC), 1, {2: "too sad tonight"}))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go")

    state = model.systems[0]
    last = 'picked Someone - Album 1; they turned down Someone - Album 2 ("too sad tonight")'
    assert f"rejection (Nov 1): {last}." in state
    assert state.index("- 3:") < state.index("- 1:") and state.index("- 3:") < state.index("- 2:")
    assert 'They turned it down on Nov 1 ("too sad tonight")' in state


def test_sessions_older_than_the_memory_window_are_forgotten() -> None:
    model = ScriptedModel(Reply(text="Hi."))
    agent, store = _agent(model)
    store.save_session(_past_session(datetime(2026, 9, 1, tzinfo=UTC), 1, {}))

    assert agent.start().memory == []


def test_skipping_questions_completes_the_brief_and_shortens_replies() -> None:
    call = ToolCall(id="s1", name="skip_questions", arguments={"evidence": "just pick"})
    skip = Reply(tool_calls=[call])
    model = ScriptedModel(skip, Reply(text="Album 1, then."))
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = Brief(mood=Answer(value="calm", evidence="calm"))

    agent.reply(session, "ugh, just pick something")

    assert session.brief.next_slot() is None
    assert session.brief.mood is not None and session.brief.mood.value == "calm"
    assert session.brief.minutes is not None and session.brief.minutes.evidence == "just pick"
    assert "in a hurry" in model.systems[1] and "Pick one with pick_record" in model.systems[1]


@pytest.mark.parametrize(
    ("hurried", "said", "asked"),
    [(False, "nope", True), (True, "nope", False), (False, "hm", False)],
)
def test_two_rejections_ask_what_is_not_working_once(hurried: bool, said: str, asked: bool) -> None:
    model = ScriptedModel(Reply(text="What's not working?"))
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = _complete_brief()
    session.hurried = hurried
    session.rejections = [Rejection(release_id=i, reason="nope") for i in (1, 2)]

    agent.reply(session, said)

    assert ("ask what isn't working" in model.systems[0]) is asked


def test_the_latest_outcome_of_a_remembered_record_wins() -> None:
    agent, store = _agent(ScriptedModel())
    store.save_session(_past_session(datetime(2026, 11, 1, tzinfo=UTC), None, {1: "not tonight"}))
    store.save_session(_past_session(datetime(2026, 11, 3, tzinfo=UTC), 1, {1: "no"}))

    [recalled] = agent.start().memory

    assert (recalled.outcome, recalled.on.day) == ("picked", 3)


def test_reply_is_spoken_sentence_by_sentence_across_rounds() -> None:
    said: list[str] = []
    announce = Reply(text="Here's Album 1! It's short.", tool_calls=_pick(1).tool_calls)
    model = ScriptedModel(announce, Reply(text="Enjoy."))
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = _complete_brief()

    agent.reply(session, "go", say=said.append)

    assert session.pick is not None
    assert said == ["Here's Album 1!", "It's short.", "Enjoy."]


def test_saving_an_answer_without_a_question_asks_in_another_call() -> None:
    saved = _update(mood={"value": "calm", "evidence": "calm"})
    model = ScriptedModel(
        Reply(text="Got it.", tool_calls=saved.tool_calls), Reply(text="How much energy?")
    )
    agent, _ = _agent(model)
    session = agent.start()

    assert agent.reply(session, "calm") == "Got it. How much energy?"


def test_empty_last_reply_after_speaking_ends_the_turn_without_the_fallback() -> None:
    announce = Reply(text="Here's Album 1! It's short.", tool_calls=_pick(1).tool_calls)
    model = ScriptedModel(announce, Reply())
    agent, _ = _agent(model, _record(1), _record(2), _record(3))
    session = agent.start()
    session.brief = _complete_brief()

    reply = agent.reply(session, "go")

    assert reply == "Here's Album 1! It's short."
    assert [m.role for m in session.messages][-2:] == ["tool", "assistant"]
    assert session.messages[-1].content == reply
