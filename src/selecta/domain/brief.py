"""The interview brief."""

from typing import Literal

from pydantic import BaseModel, field_validator

from selecta.domain.enrichment import Energy, Mood

Occasion = Literal["solo", "dinner", "party", "focus", "UNKNOWN"]
Slot = Literal["mood", "energy", "minutes", "occasion"]

MIN_MINUTES = 5
MAX_MINUTES = 240
SLOT_ORDER: tuple[Slot, ...] = ("mood", "energy", "minutes", "occasion")
OPTIONAL_SLOTS = ("whole_album",)


class Answer[T](BaseModel):
    """A value and the listener's words it came from."""

    value: T
    evidence: str


class Brief(BaseModel):
    """The interview slots. None until answered, UNKNOWN if the listener can't say."""

    mood: Answer[Mood] | None = None
    energy: Answer[Energy] | None = None
    minutes: Answer[int | Literal["UNKNOWN"]] | None = None
    occasion: Answer[Occasion] | None = None
    whole_album: Answer[bool] | None = None

    @field_validator("minutes")
    @classmethod
    def _realistic_minutes[A: Answer](cls, answer: A | None) -> A | None:
        if answer is None or not isinstance(answer.value, int):
            return answer
        if not MIN_MINUTES <= answer.value <= MAX_MINUTES:
            raise ValueError(f"minutes must be between {MIN_MINUTES} and {MAX_MINUTES}")
        return answer

    def values(self) -> dict[str, object]:
        """Return each slot's value, without the evidence."""
        return {
            slot: answer.value if (answer := getattr(self, slot)) else None
            for slot in (*SLOT_ORDER, *OPTIONAL_SLOTS)
        }

    def listening_minutes(self) -> int | None:
        """Return the listening time in minutes, or None if not known."""
        value = self.minutes.value if self.minutes else None
        return value if isinstance(value, int) else None

    def wants_whole_album(self) -> bool:
        """Return whether the listener said they want the whole album, not one side."""
        return bool(self.whole_album and self.whole_album.value)

    def next_slot(self) -> Slot | None:
        """Return the next slot to ask, or None when complete."""
        return next((slot for slot in SLOT_ORDER if getattr(self, slot) is None), None)
