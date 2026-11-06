"""The agent loop."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from selecta.application.conversation import Message, Session, Summary, ToolRun, Turn
from selecta.application.memory import last_session, recall
from selecta.application.picking import describe
from selecta.application.ports import AgentModel, ModelError, SessionStore
from selecta.application.prompts import FALLBACK_REPLIES, PERSONA, SUMMARY, WAIT_LINES
from selecta.application.speech import SentenceStream
from selecta.application.tools import Play, Tools
from selecta.domain.brief import SLOT_ORDER

logger = logging.getLogger(__name__)

MAX_MODEL_ROUNDS = 4


class Agent:
    """Conversation runner."""

    def __init__(
        self,
        model: AgentModel,
        tools: Tools,
        store: SessionStore,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._model = model
        self._tools = tools
        self._store = store
        self._now = now

    def start(self) -> Session:
        """Open a new session."""
        now = self._now()
        session = Session(id=uuid.uuid4().hex, started_at=now, memory=recall(self._store, now))
        self._store.save_session(session)
        return session

    def reply(
        self,
        session: Session,
        text: str,
        say: Callable[[str], None] = lambda _: None,
        play: Play | None = None,
    ) -> str:
        """Return the reply to one listener turn, and save the session."""
        started = time.monotonic()
        turn = Turn(
            started_at=self._now(),
            user=text,
            reply="",
            brief_before=session.brief.model_copy(deep=True),
            brief_after=session.brief,
            pick_before=session.pick,
        )
        session.messages.append(Message(role="user", content=text))
        turn.reply = self._run_rounds(session, turn, say, play)
        turn.brief_after = session.brief.model_copy(deep=True)
        turn.pick_after = session.pick
        turn.latency_ms = round((time.monotonic() - started) * 1000)
        session.turns.append(turn)
        self._store.save_session(session)
        return turn.reply

    def _run_rounds(
        self, session: Session, turn: Turn, say: Callable[[str], None], play: Play | None
    ) -> str:
        spoken: list[str] = []

        def speak(line: str) -> None:
            spoken.append(line)
            say(line)

        # The state only changes when a tool changes the session, so it's rebuilt only then.
        state, key = "", None
        for _ in range(MAX_MODEL_ROUNDS):
            if key != (new_key := _state_key(session)):
                state, key = self._state(session), new_key
            turn.contexts.append(state)
            stream = SentenceStream(speak)
            system, specs = f"{PERSONA}\n\n{state}", self._tools.specs(can_play=play is not None)
            try:
                reply = self._model.chat(system, session.messages, specs, stream.feed)
            except ModelError as e:
                logger.warning("model failed, answering with the fallback: %s", e)
                turn.error = str(e)
                # Sentences streamed before the failure were heard, so they stay in the history.
                return self._fall_back(session, speak, spoken, heard=tuple(stream.spoken))
            stream.close()
            turn.model_rounds += 1
            turn.input_tokens += reply.input_tokens
            turn.output_tokens += reply.output_tokens
            message = Message(role="assistant", tool_calls=reply.tool_calls)
            session.messages.append(message)
            if not reply.tool_calls:
                if stream.spoken:
                    message.content = " ".join(stream.spoken)
                    return " ".join(spoken)
                session.messages.pop()
                if not spoken:
                    return self._fall_back(session, speak, spoken)
                # Nothing more to say: what this turn already said becomes its closing message.
                _close_turn(session, spoken)
                return " ".join(spoken)

            def before_search(stream: SentenceStream = stream) -> None:
                # What the model said with the call already covers the wait.
                if turn.said_while_waiting or stream.spoken:
                    return
                turn.said_while_waiting = _wait_line(session)
                stream.spoken.append(turn.said_while_waiting)
                speak(turn.said_while_waiting)

            runs = [
                ToolRun(call=call, result=self._tools.run(call, session, before_search, play))
                for call in reply.tool_calls
            ]
            turn.tool_runs += runs
            session.messages += [
                Message(role="tool", content=r.result, tool_call_id=r.call.id, name=r.call.name)
                for r in runs
            ]
            if stream.spoken and _brief_saved_and_asked(runs, stream.spoken, session):
                # The question goes after the tool results: a turn must end on the assistant.
                session.messages.append(Message(role="assistant", content=" ".join(stream.spoken)))
                return " ".join(spoken)
            # The history holds what the listener heard, not what the model planned to say.
            message.content = " ".join(stream.spoken) or None
        logger.warning("no answer after %d model rounds", MAX_MODEL_ROUNDS)
        return self._fall_back(session, speak, spoken)

    def _fall_back(
        self,
        session: Session,
        speak: Callable[[str], None],
        spoken: list[str],
        heard: tuple[str, ...] = (),
    ) -> str:
        """Speak the fallback reply and record it, after anything already heard this round."""
        fallback = FALLBACK_REPLIES[session.language]
        session.messages.append(Message(role="assistant", content=" ".join([*heard, fallback])))
        speak(fallback)
        return " ".join(spoken)

    def finish(self, session: Session) -> Summary:
        """Write the session summary: key points from code, a short text from the model."""
        key_points = {
            "language": session.language,
            "brief": session.brief.values(),
            "pick": session.pick.model_dump() if session.pick else None,
            "rejections": [r.model_dump() for r in session.rejections],
            "turns": len(session.turns),
            "tokens": sum(t.input_tokens + t.output_tokens for t in session.turns),
        }
        session.summary = Summary(key_points=key_points)
        if session.turns:
            try:
                transcript = "\n".join(
                    f"{m.role}: {m.content}"
                    for m in session.messages
                    if m.role in ("user", "assistant") and m.content
                )
                facts = json.dumps(key_points, ensure_ascii=False)
                content = f"Facts (authoritative): {facts}\n\nTranscript:\n{transcript}"
                request = [Message(role="user", content=content)]
                reply = self._model.chat(SUMMARY, request, [])
                session.summary.text = reply.text
                session.summary.input_tokens = reply.input_tokens
                session.summary.output_tokens = reply.output_tokens
            except ModelError as e:
                logger.warning("summary failed, keeping the key points only: %s", e)
        self._store.save_session(session)
        return session.summary

    def _state(self, session: Session) -> str:
        """Return the interview state for the system prompt."""
        brief = session.brief
        lines = [line] if (line := last_session(session)) else []
        lines.append("Interview state (authoritative):")
        for slot in SLOT_ORDER:
            answer = getattr(brief, slot)
            value = "not asked yet" if answer is None else f'{answer.value} ("{answer.evidence}")'
            lines.append(f"- {slot}: {value}")
        if whole := brief.whole_album:
            lines.append(f'- whole album, not one side: {whole.value} ("{whole.evidence}")')
        if session.pick:
            pick = session.pick
            lines.append(
                f"Current pick: {', '.join(pick.artists)} - {pick.title}. "
                "If they turn it down, call reject_pick and pick another."
            )
        if session.hurried:
            lines.append("The listener is in a hurry: no more questions, one-sentence replies.")
        slot = brief.next_slot()
        if slot and not session.pick:
            lines.append(f"Next to ask: {slot}.")
        elif not session.pick:
            lines.append("The interview is complete.")
            if _just_rejected_again(session):
                lines.append(
                    f"They turned down {len(session.rejections)} records: before offering "
                    "another, ask what isn't working."
                )
            try:
                candidates = describe(self._tools.candidates(session), session)
            except Exception:
                logger.exception("candidate lookup failed")
                candidates = "The collection can't be reached right now: tell the listener."
            lines.append(candidates)
        return "\n".join(lines)


def _state_key(session: Session) -> str:
    brief = session.brief.model_dump_json()
    return f"{brief}|{session.pick}|{session.rejected_ids()}|{session.hurried}"


def _wait_line(session: Session) -> str:
    """Return the next wait line in turn, so the same one is never said twice in a row."""
    lines = WAIT_LINES[session.language]
    said = sum(1 for t in session.turns if t.said_while_waiting)
    return lines[said % len(lines)]


def _just_rejected_again(session: Session) -> bool:
    """Return whether the listener just turned down a second record, and isn't in a hurry."""
    said = next((m.content for m in reversed(session.messages) if m.role == "user"), None)
    rejections = session.rejections
    return len(rejections) >= 2 and rejections[-1].reason == said and not session.hurried


def _brief_saved_and_asked(runs: list[ToolRun], spoken: list[str], session: Session) -> bool:
    """Return whether the round only saved valid answers and already asked the next question.

    Then the interview goes on without another model call.
    """
    asked = any(s.rstrip("\"”’)' ").endswith("?") for s in spoken)
    if not asked or not all(r.call.name == "update_brief" for r in runs):
        return False
    if session.brief.next_slot() is None:
        return False
    try:
        return not any(json.loads(r.result).get("rejected") for r in runs)
    except (json.JSONDecodeError, AttributeError):
        return False


def _close_turn(session: Session, spoken: list[str]) -> None:
    """End the turn on one assistant message holding everything it said, as the API requires."""
    start = max(i for i, m in enumerate(session.messages) if m.role == "user")
    for message in session.messages[start:]:
        if message.role == "assistant":
            message.content = None
    session.messages.append(Message(role="assistant", content=" ".join(spoken)))
