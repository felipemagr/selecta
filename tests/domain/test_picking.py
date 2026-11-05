from selecta.domain.collection import ReleaseSummary
from selecta.domain.picking import rank


def _record(id: int, minutes: int | None) -> ReleaseSummary:
    return ReleaseSummary(
        id=id,
        title="x",
        artists=[],
        original_year=None,
        label=None,
        format=None,
        genres=[],
        styles=[],
        duration_seconds=minutes * 60 if minutes else None,
    )


def test_rank_puts_fitting_records_first_and_keeps_the_rest() -> None:
    records = [_record(1, 90), _record(2, None), _record(3, 40)]

    assert [r.id for r in rank(records, minutes=60)] == [3, 2, 1]
