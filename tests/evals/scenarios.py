"""Eval scenarios: one listener turn after a frozen situation, in three languages.

Each scenario tests a capability, says it two ways per language, and checks what the agent did
(tools, arguments, session state), not its wording. Invariants in checks.py apply to every run.
"""

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from selecta.application.conversation import Message, Pick, Session, ToolCall, ToolRun, Turn
from selecta.domain.brief import Answer, Brief
from tests.evals.world import NOW, RELEASES, TAGS, Player, World, fold, playable

LANGUAGES = ("en", "es", "fr")

# Records of the sample collection the scenarios use.
HARBOUR_LIGHTS = 1001  # has a web passage about its reception
NIGHT_SHIFT = 1003  # six tracks, all playable
LATE_SET = 1007  # credits a drummer
HOLLOW_SEASON = 1008  # vinyl only
MARKET_DAY = 1009
VELVET_HOURS = 1013  # soul, warm, no web passages
DRUMMER = next(c.name for c in RELEASES[LATE_SET].credits if c.role == "Drums").split()[-1]


@dataclass
class Outcome:
    """What one run left behind."""

    language: str
    session: Session
    turn: Turn
    world: World
    player: Player
    pick_before: Pick | None


Check = Callable[[Outcome], list[str]]


@dataclass
class Scenario:
    """A capability: the situation, what the listener says, and what must happen."""

    name: str
    capability: str
    setup: Callable[[str], Session]
    says: dict[str, tuple[str, str]]
    check: Check
    player_works: bool = True


HI = {"en": "Hi.", "es": "Hola.", "fr": "Salut."}
ASK = {
    "mood": {
        "en": "Hi! What are you in the mood for?",
        "es": "¡Hola! ¿Qué te apetece escuchar?",
        "fr": "Salut ! Tu as envie de quoi ?",
    },
    "energy": {
        "en": "Nice. How much energy do you want?",
        "es": "Bien. ¿Con cuánta energía?",
        "fr": "D'accord. Avec quelle énergie ?",
    },
    "minutes": {
        "en": "Got it. How long do you have?",
        "es": "Vale. ¿Cuánto tiempo tienes?",
        "fr": "Ok. Tu as combien de temps ?",
    },
    "occasion": {
        "en": "And is it just you, or is there company?",
        "es": "¿Y estás solo o con gente?",
        "fr": "Et tu es seul ou accompagné ?",
    },
}
ANSWER = {
    "mood": {"en": "Something warm.", "es": "Algo cálido.", "fr": "Quelque chose de chaleureux."},
    "energy": {"en": "Medium.", "es": "Media.", "fr": "Moyenne."},
    "minutes": {
        "en": "About forty minutes.",
        "es": "Unos cuarenta minutos.",
        "fr": "Une quarantaine de minutes.",
    },
    "occasion": {"en": "Just me.", "es": "Solo yo.", "fr": "Juste moi."},
}
VALUES = {"mood": "warm", "energy": "medium", "minutes": 40, "occasion": "solo"}
OFFER = {
    "en": "I'd go with {title} by {artist}: {why} Shall I put it on?",
    "es": "Yo pondría {title} de {artist}: {why} ¿Lo pongo?",
    "fr": "Je mettrais {title} de {artist} : {why} Je le lance ?",
}
WHY = {
    "en": "it fits the evening.",
    "es": "encaja con la noche.",
    "fr": "ça colle à la soirée.",
}
YES = {"en": "Yes, put it on.", "es": "Sí, ponlo.", "fr": "Oui, mets-le."}
PLAYING = {"en": "It's playing.", "es": "Ya suena.", "fr": "C'est lancé."}


# The language switch scenario starts each session in another language.
SWITCH_FROM = {"en": "es", "es": "fr", "fr": "en"}


def _session(
    language: str, answered: int, pick: int | None = None, playing: bool = False
) -> Session:
    """Return a session `answered` slots into the interview, optionally with a pick or playing.

    The history is what the real agent leaves: each answer saved with update_brief, the pick
    made with pick_record, playback started with play_record, each with its result.
    """
    slots = list(ASK)
    brief = Brief(
        **{
            slot: Answer(value=VALUES[slot], evidence=ANSWER[slot][language])
            for slot in slots[:answered]
        }
    )
    messages = [
        Message(role="user", content=HI[language]),
        Message(role="assistant", content=ASK["mood"][language]),
    ]
    for i, slot in enumerate(slots[:answered]):
        following = slots[i + 1] if i + 1 < len(slots) else None
        saved = {"value": VALUES[slot], "evidence": ANSWER[slot][language]}
        result = {"saved": [slot], "rejected": {}, "next_slot": following}
        messages += [
            Message(role="user", content=ANSWER[slot][language]),
            *_tool_round(f"brief{i}", "update_brief", {slot: saved, "language": language}, result),
        ]
        if following:
            messages.append(Message(role="assistant", content=ASK[following][language]))
    # Each run gets its own id: candidates are shuffled by it, as each real session's are.
    session = Session(id=uuid.uuid4().hex, started_at=NOW, language=language, brief=brief)
    if pick is not None:
        release = RELEASES[pick]
        session.pick = Pick(
            release_id=pick, title=release.title, artists=release.artists, reason=WHY[language]
        )
        args = {"release_id": pick, "reason": WHY[language]}
        offer = OFFER[language].format(
            title=release.title, artist=release.artists[0], why=WHY[language]
        )
        messages += [
            *_tool_round("pick1", "pick_record", args, "Picked, not playing yet."),
            Message(role="assistant", content=offer),
        ]
    if playing:
        tracks = playable(pick or 0)
        named = [f"{t.position} {t.title}" for t in tracks]
        following = f"; next is {named[1]}" if len(named) > 1 else ""
        result = f"Playing: {len(tracks)} tracks, from {named[0]}{following}."
        messages += [
            Message(role="user", content=YES[language]),
            *_tool_round("play1", "play_record", {"release_id": pick}, result),
            Message(role="assistant", content=PLAYING[language]),
        ]
    session.messages = messages
    return session


def _tool_round(id: str, name: str, arguments: dict, result: object) -> list[Message]:
    """Return a tool call the agent made and the result it got, as the history keeps them."""
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return [
        Message(role="assistant", tool_calls=[ToolCall(id=id, name=name, arguments=arguments)]),
        Message(role="tool", content=content, tool_call_id=id, name=name),
    ]


def _runs(o: Outcome, name: str) -> list[ToolRun]:
    return [r for r in o.turn.tool_runs if r.call.name == name]


def _need(ok: bool, reason: str) -> list[str]:
    return [] if ok else [reason]


def _played(o: Outcome, release_id: int) -> bool:
    return any(r == release_id for r, _ in o.player.played)


def _played_first(o: Outcome) -> str | None:
    return o.player.played[-1][1][0] if o.player.played and o.player.played[-1][1] else None


def _starts_at(o: Outcome, index: int) -> list[str]:
    expected = playable(NIGHT_SHIFT)[index].position
    return _need(_played_first(o) == expected, f"didn't start from track {expected}")


PLAY_IT = {
    "en": ("Yes, put it on.", "Sounds good, play it."),
    "es": ("Sí, ponlo.", "Dale, que suene."),
    "fr": ("Oui, mets-le.", "Vas-y, lance-le."),
}


DONT_KNOW = re.compile(
    r"don.?t know|not sure|couldn.?t find|can.?t find|no information|nothing (on|about)|"
    r"no (lo )?s[eé]|no tengo|no (he )?encontr|no encuentro|no sabr|no hay informaci|"
    r"je ne sais|pas d.?information|rien trouv|je n.?ai pas trouv|je ne trouve|aucune information"
)


def _says_it_does_not_know(o: Outcome) -> list[str]:
    return _need(bool(DONT_KNOW.search(fold(o.turn.reply))), "didn't say it doesn't know")


SCENARIOS = [
    Scenario(
        name="saves the first answer and asks the next",
        capability="interview",
        setup=lambda lang: _session(lang, 0),
        says={
            "en": ("Something mellow and cosy tonight.", "I want something calm, nothing loud."),
            "es": ("Algo tranquilo y acogedor esta noche.", "Quiero algo calmado, nada ruidoso."),
            "fr": ("Quelque chose de doux et calme ce soir.", "Un truc tranquille, rien de fort."),
        },
        check=lambda o: (
            _need(o.session.brief.mood is not None, "mood not saved")
            + _need(o.turn.reply.count("?") in (1, 2), "didn't ask one question")
        ),
    ),
    Scenario(
        name="saves UNKNOWN when they can't say",
        capability="interview",
        setup=lambda lang: _session(lang, 1),
        says={
            "en": ("No idea, really.", "I don't know, honestly."),
            "es": ("Ni idea.", "No lo sé, la verdad."),
            "fr": ("Aucune idée.", "Je ne sais pas, franchement."),
        },
        check=lambda o: _need(
            o.session.brief.energy is not None and o.session.brief.energy.value == "UNKNOWN",
            "energy not saved as UNKNOWN",
        ),
    ),
    Scenario(
        name="turns a spoken duration into minutes",
        capability="interview",
        setup=lambda lang: _session(lang, 2),
        says={
            "en": ("About half an hour.", "Maybe thirty minutes or so."),
            "es": ("Media hora más o menos.", "Unos treinta minutos."),
            "fr": ("Une demi-heure environ.", "À peu près trente minutes."),
        },
        check=lambda o: _need(
            o.session.brief.listening_minutes() in range(25, 36), "minutes not about 30"
        ),
    ),
    Scenario(
        name="stops asking and picks when they're in a hurry",
        capability="hurry",
        setup=lambda lang: _session(lang, 1),
        says={
            "en": ("Just pick something, no more questions.", "Whatever, surprise me."),
            "es": ("Elige tú ya, no más preguntas.", "Lo que sea, sorpréndeme."),
            "fr": ("Choisis, pas plus de questions.", "Peu importe, surprends-moi."),
        },
        check=lambda o: (
            _need(o.session.hurried, "didn't stop the interview")
            + _need(o.session.pick is not None, "didn't pick")
        ),
    ),
    Scenario(
        name="picks a record that fits the answers",
        capability="pick",
        setup=lambda lang: _session(lang, 3),
        says={
            "en": ("Just me, relaxing at home.", "On my own tonight."),
            "es": ("Solo yo, en casa.", "Estoy solo esta noche."),
            "fr": ("Juste moi, à la maison.", "Seul ce soir."),
        },
        check=lambda o: (
            _need(o.session.pick is not None, "didn't pick")
            + _need(
                o.session.pick is None or TAGS[str(o.session.pick.release_id)]["mood"] == "warm",
                "pick doesn't match the mood",
            )
        ),
    ),
    Scenario(
        name="plays the pick when asked",
        capability="play",
        setup=lambda lang: _session(lang, 4, pick=VELVET_HOURS),
        says=PLAY_IT,
        check=lambda o: _need(_played(o, VELVET_HOURS), "didn't play the pick"),
    ),
    Scenario(
        name="tries to play, and the player fails",
        capability="play",
        setup=lambda lang: _session(lang, 4, pick=VELVET_HOURS),
        says=PLAY_IT,
        check=lambda o: _need(bool(_runs(o, "play_record")), "didn't try to play"),
        player_works=False,
    ),
    Scenario(
        name="a vinyl-only record isn't played",
        capability="play",
        setup=lambda lang: _session(lang, 4, pick=HOLLOW_SEASON),
        says={
            "en": ("Great, play it.", "Put it on, please."),
            "es": ("Genial, ponlo.", "Ponlo, por favor."),
            "fr": ("Super, mets-le.", "Lance-le, s'il te plaît."),
        },
        check=lambda o: (
            _need(bool(_runs(o, "play_record")), "didn't try to play")
            + _need(not o.player.played, "played a vinyl-only record")
        ),
    ),
    Scenario(
        name="plays a record named by the listener",
        capability="play",
        setup=lambda lang: _session(lang, 0),
        says={
            "en": ("Play Market Day by Kalimba Collective.", "Put on Kalimba Collective."),
            "es": ("Pon Market Day de Kalimba Collective.", "Ponme Kalimba Collective."),
            "fr": ("Mets Market Day de Kalimba Collective.", "Lance Kalimba Collective."),
        },
        check=lambda o: _need(_played(o, MARKET_DAY), "didn't play the named record"),
    ),
    Scenario(
        name="skips to the next track",
        capability="navigate",
        setup=lambda lang: _session(lang, 4, pick=NIGHT_SHIFT, playing=True),
        says={
            "en": ("Skip to the next song.", "Next track, please."),
            "es": ("Pasa a la siguiente.", "Siguiente canción, por favor."),
            "fr": ("Passe à la suivante.", "Chanson suivante, s'il te plaît."),
        },
        check=lambda o: _starts_at(o, 1),
    ),
    Scenario(
        name="starts from a given track",
        capability="navigate",
        setup=lambda lang: _session(lang, 4, pick=NIGHT_SHIFT, playing=True),
        says={
            "en": ("Start it from the third song.", "Play it from track three."),
            "es": ("Empiézalo por la tercera canción.", "Ponlo desde la pista tres."),
            "fr": ("Commence à la troisième chanson.", "Mets-le à partir de la piste trois."),
        },
        check=lambda o: _starts_at(o, 2),
    ),
    Scenario(
        name="takes a rejection and offers something else",
        capability="reject",
        setup=lambda lang: _session(lang, 4, pick=VELVET_HOURS),
        says={
            "en": ("Not that one, something else.", "Hmm, no, I don't feel like soul."),
            "es": ("Ese no, otra cosa.", "Mmm, no me apetece soul."),
            "fr": ("Pas celui-là, autre chose.", "Bof, pas envie de soul."),
        },
        check=lambda o: (
            _need(VELVET_HOURS in o.session.rejected_ids(), "didn't record the rejection")
            + _need(
                o.session.pick is None or o.session.pick.release_id != VELVET_HOURS,
                "kept the same record",
            )
        ),
    ),
    Scenario(
        name="answers who plays on the record",
        capability="questions",
        setup=lambda lang: _session(lang, 4, pick=LATE_SET),
        says={
            "en": ("Who plays the drums on this one?", "Who's the drummer here?"),
            "es": ("¿Quién toca la batería en este disco?", "¿Quién es el batería?"),
            "fr": ("Qui joue de la batterie sur ce disque ?", "C'est qui le batteur ?"),
        },
        check=lambda o: _need(fold(DRUMMER) in fold(o.turn.reply), "didn't name the drummer"),
    ),
    Scenario(
        name="names the source of a web fact",
        capability="questions",
        setup=lambda lang: _session(lang, 4, pick=HARBOUR_LIGHTS),
        says={
            "en": ("What did critics say about it?", "How was it received by critics?"),
            "es": ("¿Qué dijo la crítica de este disco?", "¿Cómo lo recibieron los críticos?"),
            "fr": ("Qu'en a dit la critique ?", "Comment les critiques l'ont reçu ?"),
        },
        check=lambda o: _need(
            "folk-archive" in fold(o.turn.reply) or "folk archive" in fold(o.turn.reply),
            "didn't name the source",
        ),
    ),
    Scenario(
        name="says it doesn't know what isn't found",
        capability="questions",
        setup=lambda lang: _session(lang, 4, pick=VELVET_HOURS),
        says={
            "en": ("What did critics say about it?", "Was it a hit when it came out?"),
            "es": ("¿Qué dijo la crítica de este disco?", "¿Tuvo éxito cuando salió?"),
            "fr": ("Qu'en a dit la critique ?", "Il a eu du succès à sa sortie ?"),
        },
        check=_says_it_does_not_know,
    ),
    Scenario(
        name="doesn't play a record the listener doesn't own",
        capability="collection",
        setup=lambda lang: _session(lang, 0),
        says={
            "en": ("Play Abbey Road.", "Put on some Beatles."),
            "es": ("Pon Abbey Road.", "Ponme algo de los Beatles."),
            "fr": ("Mets Abbey Road.", "Mets-moi les Beatles."),
        },
        check=lambda o: (
            _need(not o.player.played, "played something")
            + _need(o.session.pick is None, "picked something")
        ),
    ),
    Scenario(
        name="follows the listener into another language",
        capability="language",
        setup=lambda lang: _session(SWITCH_FROM[lang], 1),
        says={
            "en": ("Let's switch to English. Something lively.", "In English: high energy."),
            "es": ("Mejor en español. Con mucha energía.", "Hablemos en español: energía alta."),
            "fr": ("Parlons français. Plein d'énergie.", "En français : énergique."),
        },
        check=lambda o: _need(o.session.language == o.language, "session language not updated"),
    ),
]
