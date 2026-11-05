import pytest
from pydantic import ValidationError

from selecta.domain.brief import Answer, Brief


def test_next_slot_skips_answered_slots_and_ends_when_complete() -> None:
    brief = Brief(energy=Answer(value="low", evidence="chill"))
    assert brief.next_slot() == "mood"

    brief.mood = Answer(value="UNKNOWN", evidence="no idea")
    brief.minutes = Answer(value=60, evidence="an hour")
    brief.occasion = Answer(value="dinner", evidence="cooking for friends")

    assert brief.next_slot() is None


@pytest.mark.parametrize(("minutes", "valid"), [(4, False), (5, True), (240, True), (1000, False)])
def test_minutes_must_be_a_realistic_listening_time(minutes: int, valid: bool) -> None:
    try:
        Brief(minutes=Answer(value=minutes, evidence="x"))
    except ValidationError:
        assert not valid
    else:
        assert valid
