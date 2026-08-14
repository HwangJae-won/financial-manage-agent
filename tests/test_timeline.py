"""은퇴 타임라인 · 행동 경로 · 능동 알림 테스트.

지켜야 하는 것 셋.

1. **기한이 조건문이 아니라 날짜로 나온다.** "첫 고지서 납부기한 +2개월"은 맞는
   말이지만 사용자가 알고 싶은 것은 "그래서 몇 월까지요?"다.
2. **각 항목에 어디서 무엇을 하는지가 붙는다.** 계산은 다 해주는데 신청 방법이
   없으면 시니어에게는 아무것도 안 해준 것과 같다.
3. **놓친 기한을 그 항목의 문의처로 안내한다.** 퇴직금 결정을 놓쳤는데 건보공단으로
   보내면 헛걸음이다.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from core.samples import DEMO_PROFILE
from core.timeline import build_timeline, changes_since

TODAY = _dt.date(2026, 8, 14)


@pytest.fixture(scope="module")
def timeline():
    return build_timeline(DEMO_PROFILE, today=TODAY)


# --------------------------------------------------------------------------- #
# 날짜로 말한다
# --------------------------------------------------------------------------- #


def test_events_are_in_chronological_order(timeline):
    keys = [(e.year, e.month) for e in timeline.events]
    assert keys == sorted(keys)


def test_the_health_insurance_deadline_has_a_real_month(timeline):
    """2026년 12월 퇴직 → 고지서가 다음 달에 나온다고 보면 2027년 3월."""
    event = next(e for e in timeline.events if "임의계속가입 신청" in e.title)

    assert (event.year, event.month) == (2027, 3)
    assert event.kind == "deadline"


def test_the_age_respects_the_birth_month(timeline):
    """12월생에게 3월을 '만 61세'로 적으면 다른 화면과 어긋난다."""
    event = next(e for e in timeline.events if "임의계속가입 신청" in e.title)
    assert event.age == 60


def test_the_cliff_appears_on_the_timeline(timeline):
    """따로 계산되던 건보 절벽이 순서 안에 들어와야 한다."""
    event = next(e for e in timeline.events if "피부양자 탈락" in e.title)
    assert event.year == 2031


def test_deadlines_are_distinguished_from_events(timeline):
    """놓치면 끝나는 것과 그냥 일어나는 것은 다르게 다뤄야 한다."""
    titles = {e.title for e in timeline.deadlines}
    assert any("임의계속가입 신청" in t for t in titles)
    assert not any("피부양자 탈락" in t for t in titles)  # 이건 자동으로 일어난다


# --------------------------------------------------------------------------- #
# 행동 경로
# --------------------------------------------------------------------------- #


def test_every_deadline_says_where_to_go(timeline):
    """기한만 알려주고 신청처를 안 알려주면 절반만 한 것이다."""
    for event in timeline.deadlines:
        assert event.where, event.title
        assert event.what, event.title


def test_the_contact_numbers_are_present(timeline):
    text = " ".join(e.where for e in timeline.events)
    assert "1577-1000" in text  # 건강보험공단
    assert "1355" in text  # 국민연금공단


def test_the_deadline_estimate_is_disclosed(timeline):
    """고지서를 봐야 정확한 날짜를 안다는 사실을 숨기지 않는다."""
    assert any("고지서를 받으시면 확인" in n for n in timeline.notes)


# --------------------------------------------------------------------------- #
# 능동 알림 (기능 ⑨)
# --------------------------------------------------------------------------- #


def test_a_missed_deadline_is_surfaced_on_return():
    """지난 상담 이후 지나간 기한을 알려준다."""
    report = changes_since(
        DEMO_PROFILE, _dt.date(2026, 5, 1), today=_dt.date(2027, 4, 1)
    )
    assert "기한이 지났습니다" in report.headline
    assert "퇴직금" in report.headline


def test_the_missed_deadline_points_at_the_right_place():
    """퇴직금 결정을 놓쳤는데 건보공단으로 보내면 헛걸음이다."""
    report = changes_since(
        DEMO_PROFILE, _dt.date(2026, 5, 1), today=_dt.date(2027, 4, 1)
    )
    assert "인사팀" in report.headline
    assert "1577-1000" not in report.headline


def test_an_imminent_deadline_is_surfaced(timeline):
    """아직 안 지났지만 곧 오는 것도 말해야 한다."""
    assert timeline.imminent
    assert "놓치면 되돌릴 수 없습니다" in timeline.headline


def test_nothing_imminent_still_says_what_is_next():
    """임박한 것이 없어도 다음 일정은 알려준다."""
    report = build_timeline(DEMO_PROFILE, today=_dt.date(2028, 1, 1))
    assert report.headline
    assert not report.imminent
    assert "다음 일정은" in report.headline
