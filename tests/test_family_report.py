"""가족 공유 리포트 테스트 (기능 5).

이 리포트가 지켜야 하는 것은 정확도가 아니라 **순서와 정직성**이다. 숫자는 이미
각 모듈에서 검증했다. 여기서 확인할 것은 세 가지다.

1. **기한이 있는 것이 맨 위에 온다.** 임의계속가입 신청 기한을 놓치면 그걸로
   끝이고, 자산 배분은 다음 달에 해도 된다. 가족이 도울 수 있는 것은 앞쪽이다.
2. **하지 않아도 되는 것을 말한다.** 가족이 가장 많이 하는 개입이 "뭐라도 해야
   하는 것 아니냐"이기 때문이다.
3. **모르는 것과 계산하지 않은 것이 함께 나간다.** 가족은 본인보다 이 리포트를
   더 꼼꼼히 읽는다.
"""

from __future__ import annotations

import pytest

from agents.profiling import build_profile
from core.family_report import build_family_report, to_text
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture(scope="module")
def demo():
    return build_family_report(DEMO_PROFILE)


@pytest.fixture(scope="module")
def diversified():
    return build_family_report(DIVERSIFIED_PROFILE)


# --------------------------------------------------------------------------- #
# 순서 — 기한이 있는 것이 먼저
# --------------------------------------------------------------------------- #


def test_deadline_items_come_first(demo):
    """기한이 있는 항목만 '지금' 칸에 들어간다."""
    assert demo.now
    assert all(item.deadline for item in demo.now)
    assert all(not item.deadline for item in demo.later)


def test_the_severance_decision_is_urgent(demo):
    """퇴직금 수령 방식은 받고 나면 되돌릴 수 없다."""
    item = next(i for i in demo.now if "퇴직금" in i.title)

    assert "되돌릴 수 없습니다" in item.deadline
    assert "308만원" in item.detail  # 근속 32년 실효세액


def test_the_pension_window_is_stated_as_a_deadline(demo):
    """임의계속가입은 연금을 받기 시작하면 자격이 사라진다."""
    item = next(i for i in demo.now if "국민연금" in i.title)

    assert "자격이 사라집니다" in item.deadline
    assert "4,049만원" in item.value  # 건보료를 뺀 순이득


def test_an_item_never_appears_in_two_columns(demo, diversified):
    """같은 항목이 두 칸에 나오면 리포트를 믿기 어려워진다."""
    for report in (demo, diversified):
        titles = [i.title for i in (*report.now, *report.later, *report.not_needed)]
        assert len(titles) == len(set(titles))


# --------------------------------------------------------------------------- #
# 하지 않아도 되는 것 — 이 서비스의 축
# --------------------------------------------------------------------------- #


def test_a_safe_plan_is_told_to_do_nothing(diversified):
    """여유 있는 사람에게 '뭐라도 하세요'라고 하지 않는다."""
    assert diversified.not_needed
    assert any("필요 없습니다" in item.value for item in diversified.not_needed)
    assert "유지됩니다" in diversified.headline


def test_a_tight_plan_gets_concrete_numbers(demo):
    """반대로 부족하면 얼마를 어떻게 바꿔야 하는지 숫자로 말한다."""
    assert "바닥납니다" in demo.headline
    assert any("월 187만원" in item.value for item in demo.later)


# --------------------------------------------------------------------------- #
# 정직성
# --------------------------------------------------------------------------- #


def test_limits_always_go_out(demo, diversified):
    """계산하지 않은 것을 숨기지 않는다. 가족은 더 꼼꼼히 읽는다."""
    for report in (demo, diversified):
        assert report.limits
        assert any("재산분이 빠져" in note for note in report.limits)
        assert any("권유하지 않습니다" in note for note in report.limits)


def test_assumed_fields_are_named_in_plain_korean():
    """필드명을 그대로 내보내면 가족이 읽을 수 없다."""
    profile = build_profile(
        {
            "birth_year": 1966,
            "retirement_year": 2026,
            "monthly_expense": 3_000_000,
            "severance_pay": 200_000_000,
            "years_employed": 32,
        }
    )
    report = build_family_report(profile)

    assert report.assumptions
    note = report.assumptions[0]
    assert "퇴직 전 월 급여" in note
    assert "last_monthly_salary" not in note


def test_a_complete_profile_has_nothing_to_disclose(demo):
    assert demo.assumptions == []


def test_the_subject_does_not_guess_whether_they_retired(demo):
    """프로파일에 오늘 날짜가 없다. '예정'을 붙였다 틀리면 리포트 전체를 의심한다."""
    assert "예정" not in demo.subject
    assert "1966년생" in demo.subject


# --------------------------------------------------------------------------- #
# 평문 — 실제 공유 경로는 카톡이다
# --------------------------------------------------------------------------- #


def test_the_text_version_carries_the_deadlines(demo):
    """화면을 캡처해 보내면 숫자만 남고 단서가 사라진다."""
    text = demo.as_text

    assert "[기한]" in text
    assert "■ 기한이 있는 것" in text
    assert "■ 이 계산의 한계" in text


def test_the_text_version_matches_the_report(demo):
    assert demo.as_text == to_text(demo)


def test_the_text_skips_empty_sections(diversified):
    """빈 칸의 제목만 남으면 리포트가 엉성해 보인다."""
    text = to_text(diversified)

    for title, items in (
        ("■ 기한이 있는 것", diversified.now),
        ("■ 천천히 보셔도 되는 것", diversified.later),
        ("■ 하지 않으셔도 되는 것", diversified.not_needed),
    ):
        assert (title in text) == bool(items), title


def test_the_report_does_not_mutate_the_profile():
    build_family_report(DEMO_PROFILE)

    assert DEMO_PROFILE.life_events == []
    assert DEMO_PROFILE.national_pension_monthly == 1_500_000


# --------------------------------------------------------------------------- #
# 모르는 값이 있어도 리포트는 나온다
# --------------------------------------------------------------------------- #


def test_the_report_survives_missing_inputs():
    """근속연수도 가입월수도 모르는 사람에게도 한 장은 나와야 한다."""
    bare = DEMO_PROFILE.model_copy(
        update={
            "years_employed": 0,
            "last_monthly_salary": 0,
            "national_pension_months": 0,
        }
    )
    report = build_family_report(bare)

    assert report.headline
    assert report.as_text
    # 계산할 수 없는 항목은 '아직 계산하지 못했습니다'로 나가고 지어내지 않는다.
    assert any("계산하지 못했습니다" in item.value for item in report.later)
