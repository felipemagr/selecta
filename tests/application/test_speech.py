import pytest

from selecta.application.speech import SentenceStream, sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Waves Vol. 2. It's calm.", ["Waves Vol. 2.", "It's calm."]),
        ("**Blue Harbour** it is! ¿Te gusta?", ["Blue Harbour it is!", "¿Te gusta?"]),
        ("Here\u2014Blue Harbour, 1970\u20131975.", ["Here, Blue Harbour, 1970\u20131975."]),
        ("Calm \u2013 and short.", ["Calm, and short."]),
    ],
)
def test_sentences_are_spoken_without_markup_or_dashes(text: str, expected: list[str]) -> None:
    assert sentences(text) == expected


def test_each_sentence_is_spoken_once_the_next_one_starts() -> None:
    said: list[str] = []
    stream = SentenceStream(said.append)

    for piece in ["Calm it", " is. How", " long do", " you have?"]:
        stream.feed(piece)
    spoken_while_streaming = list(said)
    stream.close()

    assert spoken_while_streaming == ["Calm it is."]
    assert said == ["Calm it is.", "How long do you have?"]
