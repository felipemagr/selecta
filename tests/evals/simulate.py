"""Whole conversations with simulated listeners: another model plays a listener with a persona.

Each persona wants something and behaves a certain way. A conversation passes when a record
starts playing within MAX_TURNS, the persona's own goal is met, and no turn breaks an invariant.

    make eval ARGS="--simulate"
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from selecta.application.conversation import Message, Session
from tests.evals.checks import invariants
from tests.evals.scenarios import Outcome
from tests.evals.world import Player, build_agent

MAX_TURNS = 10
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French"}
QUESTION_TOOLS = {"get_release", "look_up", "semantic_search", "search_collection"}

LISTENER = """\
You are role-playing a person at home talking to Selecta, a voice assistant that picks a record \
from your own vinyl collection and plays it. Speak {language}. {persona}

Reply with only what you would say out loud: one or two short sentences, casual, no stage \
directions. When the music has started and you're happy, say goodbye."""


@dataclass
class Persona:
    """A kind of listener, and what must have happened for its conversation to pass."""

    name: str
    brief: str
    goal: Callable[[Session], str | None] = lambda session: None
    # A listener who changes language: to which, and after how many of their replies.
    switch_to: str | None = None
    switch_after: int = 0

    def language(self, base: str, turn: int) -> str:
        """Return the language the listener speaks on this turn, counting from 1."""
        return self.switch_to if self.switch_to and turn > self.switch_after else base


def _turned_one_down(session: Session) -> str | None:
    return None if session.rejections else "never recorded the rejection"


def _answered_a_question(session: Session) -> str | None:
    used = {r.call.name for t in session.turns for r in t.tool_runs}
    return None if used & QUESTION_TOOLS else "answered the question without looking it up"


PERSONAS = [
    Persona(
        "cooperative",
        "You want something warm and relaxed for about forty minutes, alone after work. Answer "
        "each question simply, and say yes when a record is offered.",
    ),
    Persona(
        "vague",
        "You don't really know what you want. Answer with things like 'whatever' or 'not sure', "
        "but if pushed, say something calm. Accept the first record offered.",
    ),
    Persona(
        "hurried",
        "You have friends arriving and no patience. After the first question, ask it to just "
        "pick something upbeat and put it on now.",
    ),
    Persona(
        "picky",
        "You want something energetic for a small party. Turn down the first record offered, "
        "saying it isn't your thing, then accept the second.",
        goal=_turned_one_down,
    ),
    Persona(
        "curious",
        "You want something calm for reading, about half an hour. Before choosing, ask one "
        "question about a record it mentions, such as who plays on it, then accept it.",
        goal=_answered_a_question,
    ),
    Persona(
        "language switcher",
        "Start in {language}, but after two replies switch to English for the rest. You want "
        "something upbeat, around thirty minutes, just for you.",
        switch_to="en",
        switch_after=2,
    ),
]


def converse(
    agent_model: Any, listener_model: Any, persona: Persona, language: str
) -> dict[str, Any]:
    """Run one whole conversation and return its outcome, or the model error that stopped it."""
    agent, world = build_agent(agent_model)
    player = Player()
    session = agent.start()
    names = LANGUAGE_NAMES[language]
    system = LISTENER.format(language=names, persona=persona.brief.format(language=names))
    # The listener's own view: Selecta's lines are the other side of the conversation.
    heard: list[Message] = [Message(role="user", content="(Selecta is listening.)")]
    failures: list[str] = []
    started = time.monotonic()
    for _ in range(MAX_TURNS):
        said = listener_model.chat(system, heard, [])
        text = (said.text or "").strip()
        heard.append(Message(role="assistant", content=text))
        pick_before = session.pick
        agent.reply(session, text, play=player)  # type: ignore[arg-type]
        turn = session.turns[-1]
        if turn.error:
            return {"error": turn.error}
        spoken = persona.language(language, len(session.turns))
        outcome = Outcome(spoken, session, turn, world, player, pick_before)
        failures += [f"turn {len(session.turns)}: {f}" for f in invariants(outcome)]
        heard.append(Message(role="user", content=turn.reply))
        if player.played:
            break
    if not player.played:
        failures.append(f"no record playing after {MAX_TURNS} turns")
    if missed := persona.goal(session):
        failures.append(missed)
    return {
        "failures": failures,
        "turns": len(session.turns),
        "tokens": sum(t.input_tokens + t.output_tokens for t in session.turns),
        "seconds": time.monotonic() - started,
        "transcript": [(m.role, m.content) for m in session.messages if m.content],
    }
