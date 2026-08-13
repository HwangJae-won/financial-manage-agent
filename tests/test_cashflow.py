"""캐시플로우 엔진 테스트.

이 엔진의 숫자가 서비스 전체의 신뢰도를 결정하므로, 손계산과 대조 가능한
known-answer 테스트와 불변식 테스트를 함께 둔다.
"""

from __future__ import annotations

import pytest

from core.assumptions import Assumptions
from core.cashflow import simulate
from core.models import Allocation, RiskTolerance, UserProfile
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


def _january_profile(**overrides) -> UserProfile:
    """1월 퇴직 = 첫 해도 12개월 → 연 단위 손계산이 깔끔해지는 테스트용 프로파일."""
    defaults = dict(
        birth_year=1966,
        birth_month=1,
        retirement_year=2026,
        retirement_month=1,
        severance_pay=0,
        cash_savings=240_000_000,
        monthly_expense=2_000_000,
        national_pension_monthly=0,
        national_pension_start_age=64,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


# --------------------------------------------------------------------------- #
# Known-answer: 수익률 0, 물가 0, 세금 0
# --------------------------------------------------------------------------- #


def test_zero_return_zero_inflation_matches_hand_calculation(flat_assumptions: Assumptions):
    """2.4억 / 연 2,400만 = 정확히 10년."""
    profile = _january_profile()
    result = simulate(profile, assumptions=flat_assumptions)

    assert profile.financial_assets == 240_000_000
    assert profile.annual_expense == 24_000_000
    assert result.expense_coverage_years == pytest.approx(10.0, abs=0.01)


def test_zero_return_fractional_coverage(flat_assumptions: Assumptions):
    """2.2억 / 연 3,600만 = 6.11년 — 기획서의 '약 6.2년' 예시에 대응."""
    profile = _january_profile(cash_savings=220_000_000, monthly_expense=3_000_000)
    result = simulate(profile, assumptions=flat_assumptions)

    assert result.expense_coverage_years == pytest.approx(220 / 36, abs=0.01)


def test_yearly_rows_are_exact_under_flat_assumptions(flat_assumptions: Assumptions):
    """수익·세금이 0이면 매년 잔액은 정확히 생활비만큼 줄어든다."""
    profile = _january_profile(national_pension_monthly=0)
    result = simulate(profile, assumptions=flat_assumptions)

    first = result.rows[0]
    assert first.investment_return == 0
    assert first.tax == 0
    assert first.expense == 24_000_000
    assert first.start_balance - first.end_balance == 24_000_000


# --------------------------------------------------------------------------- #
# 불변식
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", [DEMO_PROFILE, DIVERSIFIED_PROFILE])
def test_balance_never_goes_negative(profile: UserProfile):
    result = simulate(profile)
    assert all(row.end_balance >= 0 for row in result.rows)
    assert all(row.start_balance >= 0 for row in result.rows)


@pytest.mark.parametrize("profile", [DEMO_PROFILE, DIVERSIFIED_PROFILE])
def test_rows_are_chained(profile: UserProfile):
    """한 해의 기말 잔액은 다음 해의 기초 잔액과 같아야 한다."""
    rows = simulate(profile).rows
    for prev, curr in zip(rows, rows[1:]):
        assert curr.start_balance == prev.end_balance


def test_simulation_starts_at_retirement_and_ends_at_horizon():
    result = simulate(DEMO_PROFILE)
    assert result.rows[0].year == DEMO_PROFILE.retirement_year
    assert result.rows[-1].age == result.horizon_age


def test_retirement_year_is_prorated_by_month():
    """12월 퇴직이면 첫 해는 1개월만 계산된다."""
    result = simulate(DEMO_PROFILE)
    assert result.rows[0].active_months == 1
    assert result.rows[0].expense == DEMO_PROFILE.monthly_expense
    assert result.rows[1].active_months == 12


def test_pension_delays_depletion():
    """국민연금이 있으면 없을 때보다 자산이 늦게 고갈되어야 한다."""
    with_pension = simulate(DEMO_PROFILE)
    without_pension = simulate(DEMO_PROFILE.model_copy(update={"national_pension_monthly": 0}))

    assert without_pension.years_until_depletion is not None
    if with_pension.years_until_depletion is None:
        return  # 연금 덕분에 아예 고갈되지 않는 경우도 정상
    assert with_pension.years_until_depletion > without_pension.years_until_depletion


def test_pension_income_starts_at_the_right_year():
    result = simulate(DEMO_PROFILE)
    by_year = {row.year: row for row in result.rows}

    assert by_year[DEMO_PROFILE.pension_start_year - 1].pension_income == 0
    assert by_year[DEMO_PROFILE.pension_start_year].pension_income > 0
    # 1966년 12월생 → 개시 연도에는 12월 한 달치만 수령
    assert by_year[DEMO_PROFILE.pension_start_year].pension_income == pytest.approx(
        by_year[DEMO_PROFILE.pension_start_year + 1].pension_income / 12,
        rel=0.05,
    )


def test_income_gap_months_matches_profile():
    result = simulate(DEMO_PROFILE)
    assert result.income_gap_months == 48
    assert result.income_gap_months == DEMO_PROFILE.income_gap_months


def test_expense_coverage_ignores_all_income():
    """생활비 충당 연수는 연금·기타소득을 무시한 값이어야 한다.

    연금액을 바꿔도 이 지표는 변하지 않는다.
    """
    a = simulate(DEMO_PROFILE)
    b = simulate(DEMO_PROFILE.model_copy(update={"national_pension_monthly": 5_000_000}))
    assert a.expense_coverage_years == b.expense_coverage_years


def test_higher_expense_depletes_faster():
    cheap = simulate(DEMO_PROFILE.model_copy(update={"monthly_expense": 2_000_000}))
    pricey = simulate(DEMO_PROFILE.model_copy(update={"monthly_expense": 6_000_000}))
    assert cheap.expense_coverage_years > pricey.expense_coverage_years


# --------------------------------------------------------------------------- #
# 경계 조건
# --------------------------------------------------------------------------- #


def test_no_income_gap_when_pension_already_started():
    """퇴직 시점에 이미 연금 수령 연령이면 소득공백기는 0개월."""
    profile = _january_profile(
        birth_year=1960, retirement_year=2026, national_pension_start_age=63
    )
    assert profile.income_gap_months == 0
    result = simulate(profile)
    assert result.income_gap_months == 0


def test_income_exceeding_expense_never_depletes():
    profile = DEMO_PROFILE.model_copy(
        update={"other_monthly_income": 10_000_000, "monthly_expense": 3_000_000}
    )
    result = simulate(profile)
    assert result.survives_horizon
    assert result.depletion_year is None
    assert result.final_balance > result.starting_balance


def test_immediate_depletion_when_assets_are_tiny(flat_assumptions: Assumptions):
    profile = _january_profile(cash_savings=1_000_000, monthly_expense=2_000_000)
    result = simulate(profile, assumptions=flat_assumptions)
    assert result.depletion_year == profile.retirement_year
    assert result.expense_coverage_years == pytest.approx(1 / 24, abs=0.01)


# --------------------------------------------------------------------------- #
# 배분(allocation)
# --------------------------------------------------------------------------- #


def test_growth_allocation_outlasts_cash_allocation():
    """기대수익률이 높은 배분이 더 오래 버텨야 한다 (결정론적 시뮬레이션 기준)."""
    all_cash = simulate(DEMO_PROFILE, allocation=Allocation(cash=1.0, bond=0.0, equity=0.0))
    growth = simulate(DEMO_PROFILE, allocation=Allocation(cash=0.3, bond=0.2, equity=0.5))
    assert growth.expense_coverage_years > all_cash.expense_coverage_years


def test_allocation_must_sum_to_one():
    with pytest.raises(ValueError):
        Allocation(cash=0.5, bond=0.2, equity=0.1)


def test_current_allocation_reflects_holdings():
    alloc = DIVERSIFIED_PROFILE.current_allocation()
    assert alloc.cash + alloc.bond + alloc.equity == pytest.approx(1.0)
    assert alloc.equity > 0  # 주식·ETF와 ISA 일부를 보유

    conservative = DIVERSIFIED_PROFILE.model_copy(
        update={"risk_tolerance": RiskTolerance.CONSERVATIVE}
    )
    assert conservative.current_allocation().equity < alloc.equity
