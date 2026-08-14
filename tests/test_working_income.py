"""재취업 소득의 함정 테스트.

여기서 지켜야 하는 것은 두 가지다.

1. **깎이기 시작하는 지점.** A값을 200만원 넘게 초과해야 깎인다. 2025.12.16
   개정으로 생긴 구간이라, 개정 전 자료를 보고 만들면 없는 감액이 생긴다.
   "일하면 연금 깎여요"가 대부분의 사람에게 해당하지 않는다는 것이 이 기능의 답이다.
2. **우리 자신의 권고를 검증한다.** 처방 엔진이 "월 162만원 더 버세요"라고 하는데,
   건강보험료를 제도대로 계산하면 그것으로 목표에 못 미친다. 남의 말을 정직하게
   환산하기로 해놓고 우리 권고만 낙관적으로 두면 앞뒤가 맞지 않는다.
"""

from __future__ import annotations

import pytest

from core.assumptions import load_assumptions
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.working_income import analyze_working_income


@pytest.fixture(scope="module")
def spec():
    return load_assumptions().national_pension.earned_income


# --------------------------------------------------------------------------- #
# 감액 산식 — 국민연금법 제63조의2
# --------------------------------------------------------------------------- #


def test_below_the_threshold_nothing_is_cut(spec):
    """A값 + 200만원까지는 한 푼도 깎이지 않는다."""
    pension = 1_500_000
    assert spec.reduction(spec.a_value, pension) == 0
    assert spec.reduction(spec.a_value + 1_999_999, pension) == 0


def test_the_first_band_matches_the_statute(spec):
    """초과소득월액 250만원 → 15만원 + (250만-200만) × 15% = 22만 5,000원."""
    income = spec.a_value + 2_500_000
    assert spec.reduction(income, 3_000_000) == 225_000


def test_the_second_band_matches_the_statute(spec):
    """초과 350만원 → 30만원 + (350만-300만) × 20% = 40만원."""
    income = spec.a_value + 3_500_000
    assert spec.reduction(income, 3_000_000) == 400_000


def test_the_third_band_matches_the_statute(spec):
    """초과 500만원 → 50만원 + (500만-400만) × 25% = 75만원."""
    income = spec.a_value + 5_000_000
    assert spec.reduction(income, 3_000_000) == 750_000


def test_the_cut_never_exceeds_half_the_pension(spec):
    """아무리 많이 벌어도 연금의 절반을 넘게 깎이지 않는다."""
    pension = 1_000_000
    huge = spec.a_value + 50_000_000

    assert spec.reduction(huge, pension) == pension // 2


def test_the_free_ceiling_is_the_number_to_say_first(spec):
    """화면에서 가장 먼저 말할 숫자 — 여기까지는 안 깎인다."""
    assert spec.free_income_ceiling() == spec.a_value + spec.exempt_threshold


def test_the_reduction_window_is_60_to_65(spec):
    assert not spec.applies_at(59)
    assert spec.applies_at(60)
    assert spec.applies_at(64)
    assert not spec.applies_at(65)  # 65세가 되면 소득과 무관하게 전액


def test_no_pension_means_no_reduction(spec):
    """받는 연금이 없으면 깎을 것도 없다."""
    assert spec.reduction(spec.a_value + 10_000_000, 0) == 0


# --------------------------------------------------------------------------- #
# 화면이 말하는 것
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo():
    return analyze_working_income(DEMO_PROFILE)


def test_the_headline_leads_with_what_is_not_cut(demo):
    """'일하면 연금 깎인다'는 말이 대부분에게 해당하지 않는다는 것이 답이다."""
    assert "깎이지 않습니다" in demo.headline
    assert demo.free_ceiling_label in demo.headline


def test_ordinary_incomes_are_untouched(demo):
    """월 500만원까지는 감액이 0 이어야 한다."""
    for outcome in demo.outcomes:
        if outcome.monthly_income <= demo.free_ceiling:
            assert outcome.pension_cut == 0, outcome.monthly_income


def test_health_premium_appears_before_the_pension_cut(demo):
    """연금보다 건보료가 먼저 생긴다. 그 순서가 화면에 드러나야 한다."""
    low = demo.outcomes[0]
    assert low.pension_cut == 0
    assert low.health_premium > 0


def test_a_person_below_sixty_is_told_it_does_not_apply():
    """59세 퇴직자에게는 감액 규정 자체가 적용되지 않는다."""
    report = analyze_working_income(DIVERSIFIED_PROFILE)

    assert not report.reduction_applies
    assert "해당하지 않습니다" in report.headline


def test_income_tax_is_disclosed_as_not_calculated(demo):
    """근사식을 지어내지 않았다는 사실과 오차의 방향을 함께 낸다."""
    assert any("소득세" in note and "계산하지 않았" in note for note in demo.notes)
    assert any("여기 나온 것보다 적습니다" in note for note in demo.notes)


# --------------------------------------------------------------------------- #
# 우리 자신의 권고를 검증한다
# --------------------------------------------------------------------------- #


def test_the_prescription_is_re_measured_with_real_premiums(demo):
    """처방 엔진은 건보료를 근사로 잡는다. 제도대로 계산하면 더 필요하다."""
    assert demo.prescribed_income > 0
    assert demo.actual_needed_income > demo.prescribed_income
    assert demo.gap > 0


def test_the_gap_is_said_out_loud(demo):
    """차이를 계산해 놓고 화면에서 말하지 않으면 아무 의미가 없다."""
    assert demo.actual_needed_income_label in demo.verdict
    assert demo.prescribed_income_label in demo.verdict


def test_the_analysis_does_not_mutate_the_profile():
    analyze_working_income(DEMO_PROFILE)

    assert DEMO_PROFILE.other_monthly_income == 0
    assert DEMO_PROFILE.national_pension_monthly == 1_500_000
