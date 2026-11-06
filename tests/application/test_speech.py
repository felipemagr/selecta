import pytest

from selecta.application.speech import SentenceStream, sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Atmospherica Vol. 2. It's calm.", ["Atmospherica Vol. 2.", "It's calm."]),
        ("**Pink Moon** it is! ¿Te gusta?", ["Pink Moon it is!", "¿Te gusta?"]),
    ],
)
def test_sentences_split_for_speech_without_markup(text: str, expected: list[str]) -> None:
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
