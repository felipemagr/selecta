"""Invariants checked on every eval run, whatever the scenario."""

import re

from selecta.application.speech import sentences
from selecta.domain.library import norm
from tests.evals.scenarios import Outcome
from tests.evals.world import fold

# Words that mark a language, for telling which one a short spoken reply is in.
WORDS = {
    "en": "the and is it you to of this that what for with your i how something would like it's "
    "i'd on do",
    "es": "el los las y es un una para con te lo por que algo tu este esta quieres ya pongo del "
    "muy si mas",
    "fr": "le les et est une pour avec tu vous ce cette je ca des du quel quelle oui pas mets "
    "c'est j'ai au",
}
MARKERS = {lang: set(words.split()) for lang, words in WORDS.items()}
# Ways of saying the record is playing now. A heuristic: the report says so.
PLAYING_CLAIM = re.compile(
    r"\b(now playing|is playing|it'?s playing|playing now|here it comes|starting it now|"
    r"ya suena|esta sonando|ahi va|ya lo tienes sonando|te lo pongo ya|"
    r"c'?est lance|ca joue|je le lance|c'?est parti)\b"
)
MAX_SENTENCES = 3
MARKDOWN = re.compile(r"(\*\*|^#|^\s*[-*] |`)", re.M)


def language_of(text: str) -> str | None:
    """Return the language a reply is in, or None if it's too short or mixed to tell."""
    words = re.findall(r"[a-z']+", fold(text))
    scores = {lang: sum(w in marks for w in words) for lang, marks in MARKERS.items()}
    best = max(scores, key=scores.__getitem__)
    rest = max(v for lang, v in scores.items() if lang != best)
    return best if scores[best] >= 2 and scores[best] > rest else None


def invariants(o: Outcome) -> list[str]:
    """Return what this run broke of the rules every turn must keep."""
    failures = []
    reply = o.turn.reply
    for run in o.turn.tool_runs:
        release_id = str(run.call.arguments.get("release_id", ""))
        if release_id and not (release_id.isdigit() and int(release_id) in o.world.releases):
            failures.append("used a record id that doesn't exist")
    # The player itself says whether anything started, whatever the tool's wording.
    started = o.player.works and bool(o.player.played)
    if PLAYING_CLAIM.search(fold(reply)) and not started:
        failures.append("said it's playing without starting it")
    pick = o.session.pick
    if pick and pick != o.pick_before:
        names = [norm(pick.title), *(norm(a) for a in pick.artists)]
        if not any(name in norm(reply) for name in names):
            failures.append("picked a record without naming it")
    if (spoken := language_of(reply)) and spoken != o.language:
        failures.append(f"answered in {spoken}, not {o.language}")
    if len(sentences(reply)) > MAX_SENTENCES:
        failures.append(f"more than {MAX_SENTENCES} sentences for speech")
    if MARKDOWN.search(reply):
        failures.append("markdown in a spoken reply")
    if not reply.strip():
        failures.append("said nothing")
    return failures
