from selecta.application.conversation import Rejection, Session
from selecta.application.picking import Candidates, describe, find_candidates
from selecta.domain.brief import Answer, Brief
from selecta.domain.collection import ReleaseSummary


class Shelf:
    def __init__(self, *releases: ReleaseSummary) -> None:
        self.releases = releases

    def owned_summaries(
        self, mood: str | None = None, energy: str | None = None
    ) -> list[ReleaseSummary]:
        return [r for r in self.releases if mood in (None, r.mood) and energy in (None, r.energy)]


def _record(id: int, mood: str = "calm", minutes: int = 40) -> ReleaseSummary:
    return ReleaseSummary(
        id=id,
        title=f"Album {id}",
        artists=["Someone"],
        original_year=1970,
        label=None,
        format=None,
        genres=[],
        styles=[],
        mood=mood,
        energy="low",
        duration_seconds=minutes * 60,
    )


def _ids(*releases: ReleaseSummary, rejected: list[int] = []) -> list[int]:  # noqa: B006
    session = Session(
        id="s",
        started_at="2026-11-05T21:00:00Z",
        brief=Brief(
            mood=Answer(value="calm", evidence="calm"),
            energy=Answer(value="low", evidence="chill"),
            minutes=Answer(value=45, evidence="45 minutes"),
        ),
        rejections=[Rejection(release_id=i, reason="no") for i in rejected],
    )
    return [r.id for r in find_candidates(Shelf(*releases), session).releases]  # type: ignore[arg-type]


def test_candidates_match_mood_and_energy() -> None:
    assert sorted(_ids(_record(1), _record(2), _record(3), _record(4, mood="dark"))) == [1, 2, 3]


def test_rejected_records_are_not_candidates() -> None:
    assert 1 not in _ids(_record(1), _record(2), _record(3), _record(4), rejected=[1])


def test_too_few_matches_relax_the_filters_keeping_strict_matches_first() -> None:
    assert _ids(_record(1), _record(2, mood="dark"), _record(3, mood="dark"))[0] == 1


def test_fitting_record_is_kept_among_many_that_dont() -> None:
    records = [_record(i, minutes=90) for i in range(1, 30)]

    assert _ids(*records, _record(99, minutes=30))[0] == 99


def test_describe_says_when_nothing_is_left() -> None:
    empty = Candidates(releases=[], filters={}, relaxed=[])

    assert describe(empty, minutes=45).startswith("No record is left to offer")
