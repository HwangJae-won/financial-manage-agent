from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.assumptions import Assumptions, load_assumptions
from core.formatting import fmt_krw, fmt_months, fmt_pct, fmt_years
from core.models import Allocation, UserProfile
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


# --------------------------------------------------------------------------- #
# UserProfile 파생값
# --------------------------------------------------------------------------- #


def test_demo_profile_matches_the_spec_example():
    """기획서 예시: 총자산 7.2억 (금융 2.2억 / 부동산 5억), 소득공백기 4년."""
    p = DEMO_PROFILE
    assert p.financial_assets == 220_000_000
    assert p.real_estate == 500_000_000
    assert p.total_assets == 720_000_000
    assert p.income_gap_months == 48
    assert p.retirement_age == 60
    assert p.pension_start_year == 2030


def test_net_worth_subtracts_debt():
    assert DIVERSIFIED_PROFILE.net_worth == DIVERSIFIED_PROFILE.total_assets - 50_000_000


def test_liquid_assets_exclude_equity_and_pension():
    p = DIVERSIFIED_PROFILE
    assert p.liquid_assets == p.severance_pay + p.cash_savings + p.isa
    assert p.financial_assets == p.liquid_assets + p.equity + p.pension_dc


def test_income_gap_is_zero_when_pension_precedes_retirement():
    p = DEMO_PROFILE.model_copy(update={"national_pension_start_age": 55})
    assert p.income_gap_months == 0


def test_monthly_expense_must_be_positive():
    with pytest.raises(ValidationError):
        UserProfile.model_validate(
            DEMO_PROFILE.model_dump() | {"monthly_expense": 0}
        )


def test_negative_assets_are_rejected():
    with pytest.raises(ValidationError):
        UserProfile(
            birth_year=1966,
            retirement_year=2026,
            monthly_expense=3_000_000,
            cash_savings=-1,
        )


def test_assumed_fields_default_to_empty():
    assert DEMO_PROFILE.assumed_fields == []


# --------------------------------------------------------------------------- #
# Assumptions
# --------------------------------------------------------------------------- #


def test_assumptions_load_and_have_required_asset_classes(base_assumptions: Assumptions):
    assert set(base_assumptions.asset_classes) >= {"cash", "bond", "equity"}
    assert 0 < base_assumptions.macro.inflation_rate < 0.1
    assert base_assumptions.macro.horizon_age > 80


def test_assumptions_are_cached():
    assert load_assumptions() is load_assumptions()


@pytest.mark.parametrize(
    "birth_year,expected_age",
    [(1952, 60), (1953, 61), (1957, 62), (1961, 63), (1966, 64), (1969, 65), (1975, 65)],
)
def test_national_pension_start_age_schedule(
    base_assumptions: Assumptions, birth_year: int, expected_age: int
):
    assert base_assumptions.national_pension_start_age(birth_year) == expected_age


def test_expected_return_is_weighted_average(base_assumptions: Assumptions):
    alloc = {"cash": 0.5, "bond": 0.5, "equity": 0.0}
    expected = 0.5 * base_assumptions.asset_classes["cash"].expected_return + 0.5 * (
        base_assumptions.asset_classes["bond"].expected_return
    )
    assert base_assumptions.expected_return(alloc) == pytest.approx(expected)


def test_volatility_increases_with_equity(base_assumptions: Assumptions):
    low = base_assumptions.volatility({"cash": 1.0, "bond": 0.0, "equity": 0.0})
    high = base_assumptions.volatility({"cash": 0.0, "bond": 0.0, "equity": 1.0})
    assert high > low


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #


def test_allocation_rejects_out_of_range_weights():
    with pytest.raises(ValidationError):
        Allocation(cash=1.5, bond=-0.5, equity=0.0)


def test_allocation_as_dict_round_trips():
    alloc = Allocation(cash=0.4, bond=0.3, equity=0.3)
    assert alloc.as_dict() == {"cash": 0.4, "bond": 0.3, "equity": 0.3}


# --------------------------------------------------------------------------- #
# 포맷팅
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "amount,expected",
    [
        (0, "0원"),
        (3_000_000, "300만원"),
        (720_000_000, "7억 2,000만원"),
        (200_000_000, "2억원"),
        (-50_000_000, "-5,000만원"),
        (12_345, "1만 2,345원"),
        (5_000, "5,000원"),
    ],
)
def test_fmt_krw(amount: int, expected: str):
    assert fmt_krw(amount) == expected


@pytest.mark.parametrize(
    "months,expected",
    [(0, "없음"), (3, "3개월"), (12, "1년"), (48, "4년"), (27, "2년 3개월")],
)
def test_fmt_months(months: int, expected: str):
    assert fmt_months(months) == expected


def test_fmt_years_and_pct():
    assert fmt_years(6.15) == "6.2년"
    assert fmt_years(None) == "고갈되지 않음"
    assert fmt_pct(0.184) == "18.4%"
