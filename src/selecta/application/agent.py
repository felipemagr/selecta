"""The agent loop."""

import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from selecta.application.conversation import Message, Session, ToolRun, Turn
from selecta.application.ports import AgentModel, ModelError, SessionStore
from selecta.application.prompts import FALLBACK_REPLIES, PERSONA
from selecta.application.tools import SPECS, Tools
from selecta.domain.brief import SLOT_ORDER, Brief

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
        turn.latency_ms = round((time.monotonic() - started) * 1000)
        session.turns.append(turn)
        self._store.save_session(session)
        return turn.reply

    def _run_rounds(self, session: Session, turn: Turn) -> str:
        spoken: list[str] = []
        for _ in range(MAX_MODEL_ROUNDS):
            state = _interview_state(session.brief)
            turn.contexts.append(state)
            reply = self._model.chat(f"{PERSONA}\n\n{state}", session.messages, SPECS)
            turn.model_rounds += 1
            turn.input_tokens += reply.input_tokens
            turn.output_tokens += reply.output_tokens
            session.messages.append(
                Message(role="assistant", content=reply.text, tool_calls=reply.tool_calls)
            )
            if reply.text:
                spoken.append(reply.text)
            if not reply.tool_calls:
                return " ".join(spoken) or FALLBACK_REPLIES[session.language]
            for call in reply.tool_calls:
                result = self._tools.run(call, session)
                turn.tool_runs.append(ToolRun(call=call, result=result))
                session.messages.append(
                    Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
                )
        logger.warning("no answer after %d model rounds", MAX_MODEL_ROUNDS)
        fallback = FALLBACK_REPLIES[session.language]
        session.messages.append(Message(role="assistant", content=fallback))
        return fallback


def _interview_state(brief: Brief) -> str:
    """Return the interview state for the system prompt."""
    lines = []
    for slot in SLOT_ORDER:
        answer = getattr(brief, slot)
        state = "not asked yet" if answer is None else f'{answer.value} ("{answer.evidence}")'
        lines.append(f"- {slot}: {state}")
    next_slot = brief.next_slot()
    lines.append(
        f"Next to ask: {next_slot}."
        if next_slot
        else "The interview is complete: sum up in one sentence what they want to hear."
    )
    return "Interview state (authoritative):\n" + "\n".join(lines)
