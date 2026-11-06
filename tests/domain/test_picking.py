import pytest

from selecta.domain.collection import ReleaseSummary
from selecta.domain.picking import MUCH_LONGER_FACTOR, rank, runs_past


def _record(id: int) -> ReleaseSummary:
    return ReleaseSummary(
        id=id,
        title="x",
        artists=[],
        original_year=None,
        label=None,
        format=None,
        genres=[],
        styles=[],
    )


def test_rank_puts_fresh_records_first_and_keeps_the_order_otherwise() -> None:
    records = [_record(1), _record(2), _record(3), _record(4)]

    assert [r.id for r in rank(records, remembered={1, 3})] == [2, 4, 1, 3]


@pytest.mark.parametrize(
    ("duration_minutes", "minutes", "expected"),
    [(60, 45, False), (70, 45, True), (None, 45, False), (300, None, False)],
)
def test_only_a_much_longer_record_is_worth_asking_about(
    duration_minutes: int | None, minutes: int | None, expected: bool
) -> None:
    duration = duration_minutes * 60 if duration_minutes else None

    assert runs_past(duration, minutes, MUCH_LONGER_FACTOR) is expected
