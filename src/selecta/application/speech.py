"""Sentence splitting for speech, from whole replies or streamed text."""

import re
from collections.abc import Callable

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÝÑ¿¡\"“'])")
_MARKUP = re.compile(r"[*_#`]")


def sentences(text: str) -> list[str]:
    """Return the text as sentences to speak one by one, without markup."""
    return [s for s in _SENTENCE_END.split(_MARKUP.sub("", text).strip()) if s]


class SentenceStream:
    """A sentence buffer for streamed text: each sentence is spoken once the next one starts."""

    def __init__(self, speak: Callable[[str], None]) -> None:
        self._speak = speak
        self._buffer = ""
        self.spoken: list[str] = []

    def feed(self, text: str) -> None:
        """Add streamed text, speaking every sentence it completes."""
        self._buffer += text
        *done, self._buffer = _SENTENCE_END.split(self._buffer)
        for sentence in done:
            self._say(sentence)

    def close(self) -> None:
        """Speak what is left of the reply."""
        rest, self._buffer = self._buffer, ""
        self._say(rest)

    def _say(self, text: str) -> None:
        for sentence in sentences(text):
            self.spoken.append(sentence)
            self._speak(sentence)
