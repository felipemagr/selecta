"""How a record feels, judged once per record by an LLM."""

from typing import Literal

from pydantic import BaseModel

Mood = Literal["calm", "warm", "melancholic", "upbeat", "intense", "dark", "UNKNOWN"]
Energy = Literal["low", "medium", "high", "UNKNOWN"]


class Enrichment(BaseModel):
    """Mood and energy of a record."""

    mood: Mood
    energy: Energy
