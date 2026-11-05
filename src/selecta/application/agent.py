"""The agent loop."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from selecta.application.conversation import Message, Session, Summary, ToolRun, Turn
from selecta.application.picking import describe
from selecta.application.ports import AgentModel, ModelError, SessionStore
from selecta.application.prompts import FALLBACK_REPLIES, PERSONA, SUMMARY
from selecta.application.tools import SPECS, Tools
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
        session = Session(id=uuid.uuid4().hex, started_at=self._now())
        self._store.save_session(session)
        return session

    def reply(self, session: Session, text: str) -> str:
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
        try:
            turn.reply = self._run_rounds(session, turn)
        except ModelError as e:
            logger.warning("model failed, answering with the fallback: %s", e)
            turn.error = str(e)
            turn.reply = FALLBACK_REPLIES[session.language]
            session.messages.append(Message(role="assistant", content=turn.reply))
        turn.brief_after = session.brief.model_copy(deep=True)
        turn.pick_after = session.pick
        turn.latency_ms = round((time.monotonic() - started) * 1000)
        session.turns.append(turn)
        self._store.save_session(session)
        return turn.reply

    def _run_rounds(self, session: Session, turn: Turn) -> str:
        spoken: list[str] = []
        # The state only changes when a tool changes the session, so it's rebuilt only then.
        state, key = "", None
        for _ in range(MAX_MODEL_ROUNDS):
            if key != (new_key := _state_key(session)):
                state, key = self._state(session), new_key
            turn.contexts.append(state)
            reply = self._model.chat(f"{PERSONA}\n\n{state}", session.messages, SPECS)
            turn.model_rounds += 1
            turn.input_tokens += reply.input_tokens
            turn.output_tokens += reply.output_tokens
            message = Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls)
            session.messages.append(message)
            if not reply.tool_calls:
                spoken += [reply.text] if reply.text else []
                return " ".join(spoken) or FALLBACK_REPLIES[session.language]
            pick_before = session.pick
            runs = [
                ToolRun(call=call, result=self._tools.run(call, session))
                for call in reply.tool_calls
            ]
            turn.tool_runs += runs
            session.messages += [
                Message(role="tool", content=r.result, tool_call_id=r.call.id, name=r.call.name)
                for r in runs
            ]
            # Text sent with a pick is spoken only if this round's pick was accepted.
            tried = any(r.call.name == "pick_record" for r in runs)
            if tried and (session.pick is None or session.pick == pick_before):
                message.content = None
            elif reply.text:
                spoken.append(reply.text)
        logger.warning("no answer after %d model rounds", MAX_MODEL_ROUNDS)
        fallback = FALLBACK_REPLIES[session.language]
        session.messages.append(Message(role="assistant", content=fallback))
        return fallback

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
        lines = ["Interview state (authoritative):"]
        for slot in SLOT_ORDER:
            answer = getattr(brief, slot)
            value = "not asked yet" if answer is None else f'{answer.value} ("{answer.evidence}")'
            lines.append(f"- {slot}: {value}")
        if session.pick:
            pick = session.pick
            lines.append(
                f"Current pick: {', '.join(pick.artists)} - {pick.title}. "
                "If they turn it down, call reject_pick and pick another."
            )
        if slot := brief.next_slot():
            lines.append(f"Next to ask: {slot}.")
        elif not session.pick:
            lines.append("The interview is complete.")
            try:
                candidates = describe(self._tools.candidates(session), brief.listening_minutes())
            except Exception:
                logger.exception("candidate lookup failed")
                candidates = "The collection can't be reached right now: tell the listener."
            lines.append(candidates)
        return "\n".join(lines)


def _state_key(session: Session) -> str:
    return f"{session.brief.model_dump_json()}|{session.pick}|{session.rejected_ids()}"
