from datetime import UTC, datetime

from selecta.application.conversation import Recalled, Rejection, Session
from selecta.application.picking import Candidates, describe, find_candidates
from selecta.domain.brief import Answer, Brief
from selecta.domain.collection import ReleaseSummary

NOW = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)


class Shelf:
    def __init__(self, *releases: ReleaseSummary) -> None:
        self.releases = releases

    def owned_summaries(
        self, mood: str | None = None, energy: str | None = None, playable: bool = False
    ) -> list[ReleaseSummary]:
        return [
            r
            for r in self.releases
            if mood in (None, r.mood)
            and energy in (None, r.energy)
            and (r.playable_count or not playable)
        ]


def _record(id: int, mood: str = "calm", minutes: int = 40, playable: int = 2) -> ReleaseSummary:
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
        track_count=2,
        playable_count=playable,
    )


def _ids(
    *releases: ReleaseSummary,
    rejected: list[int] = [],  # noqa: B006
    remembered: list[int] = [],  # noqa: B006
    whole_album: bool = False,
    playable: bool = True,
) -> list[int]:
    session = Session(
        id="s",
        started_at=NOW,
        memory=[
            Recalled(release_id=i, title="x", artists=[], outcome="picked", on=NOW)
            for i in remembered
        ],
        brief=Brief(
            mood=Answer(value="calm", evidence="calm"),
            energy=Answer(value="low", evidence="chill"),
            minutes=Answer(value=45, evidence="45 minutes"),
            whole_album=Answer(value=True, evidence="the whole thing") if whole_album else None,
        ),
        rejections=[Rejection(release_id=i, reason="no") for i in rejected],
    )
    shelf = Shelf(*releases)
    return [r.id for r in find_candidates(shelf, session, playable).releases]  # type: ignore[arg-type]


def test_candidates_match_mood_and_energy() -> None:
    assert sorted(_ids(_record(1), _record(2), _record(3), _record(4, mood="dark"))) == [1, 2, 3]


def test_rejected_records_are_not_candidates() -> None:
    assert 1 not in _ids(_record(1), _record(2), _record(3), _record(4), rejected=[1])


def test_too_few_matches_relax_the_filters_keeping_strict_matches_first() -> None:
    assert _ids(_record(1), _record(2, mood="dark"), _record(3, mood="dark"))[0] == 1


def test_remembered_match_beats_a_relaxed_one() -> None:
    ids = _ids(_record(1), _record(2, mood="dark"), _record(3, mood="dark"), remembered=[1])

    assert ids[0] == 1


def test_fresh_records_come_before_remembered_ones() -> None:
    records = [_record(i) for i in range(1, 30)]

    assert set(_ids(*records, remembered=list(range(1, 29)))[:1]) == {29}


def test_describe_says_when_nothing_is_left() -> None:
    empty = Candidates(releases=[], filters={}, relaxed=[])

    assert describe(empty, Session(id="s", started_at=NOW)).startswith("No record is left")


def test_whole_album_leaves_out_records_known_to_run_past_the_time() -> None:
    unknown = _record(3).model_copy(update={"duration_seconds": None})
    records = [_record(1, minutes=40), _record(2, minutes=50), unknown]

    assert sorted(_ids(*records, whole_album=True)) == [1, 3]
    assert sorted(_ids(*records)) == [1, 2, 3]


def test_only_records_on_the_music_server_are_candidates() -> None:
    assert sorted(_ids(_record(1), _record(2, playable=0), _record(3, playable=1))) == [1, 3]


def test_without_a_music_server_every_owned_record_is_a_candidate() -> None:
    assert sorted(_ids(_record(1), _record(2, playable=0), playable=False)) == [1, 2]


def test_without_a_music_server_candidates_say_nothing_about_playing() -> None:
    session = Session(id="s", started_at=NOW)
    records = [_record(1, playable=0)]
    candidates = Candidates(releases=records, filters={}, relaxed=[], playable=False)

    assert "can be played" not in describe(candidates, session)


def test_partly_downloaded_record_says_how_much_can_play() -> None:
    session = Session(id="s", started_at=NOW)
    candidates = Candidates(releases=[_record(1, playable=1)], filters={}, relaxed=[])

    assert "Only 1 of its 2 tracks can be played." in describe(candidates, session)
