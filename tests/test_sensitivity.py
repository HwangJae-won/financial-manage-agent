"""민감도 분석 테스트.

두 가지를 지킨다:

  - **방향이 맞아야 한다.** 생활비가 늘면 고갈이 빨라지고, 자산이 늘면 늦어진다.
    부호가 뒤집힌 민감도 분석은 없는 것보다 나쁘다.
  - **원본 가정값을 오염시키면 안 된다.** `load_assumptions()` 는 lru_cache 로
    전역 공유되므로, 흔들어 보는 과정에서 원본을 건드리면 이후 모든 계산이
    조용히 틀어진다.
"""

from __future__ import annotations

import pytest

from core.assumptions import load_assumptions
from core.cashflow import simulate
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.sensitivity import analyze_sensitivity


@pytest.fixture(scope="module")
def demo_report():
    return analyze_sensitivity(DEMO_PROFILE)


def _case(report, key: str):
    return next(c for c in report.cases if c.key == key)


def _age(depletion_age, horizon: int) -> int:
    return horizon if depletion_age is None else depletion_age


# --------------------------------------------------------------------------- #
# 오염 방지 — 가장 조용하게 망가지는 지점
# --------------------------------------------------------------------------- #


def test_the_shared_assumptions_are_not_mutated():
    """전역 캐시를 건드리면 이후 모든 계산이 틀어진다."""
    before = load_assumptions().model_dump()
    baseline = simulate(DEMO_PROFILE).depletion_year

    analyze_sensitivity(DEMO_PROFILE)

    assert load_assumptions().model_dump() == before
    assert simulate(DEMO_PROFILE).depletion_year == baseline


def test_baseline_matches_the_plain_simulation(demo_report):
    assert demo_report.baseline_depletion_age == simulate(DEMO_PROFILE).depletion_age
    assert demo_report.horizon_age == load_assumptions().macro.horizon_age


# --------------------------------------------------------------------------- #
# 방향 — 부호가 뒤집히면 안 된다
# --------------------------------------------------------------------------- #


def test_higher_expense_depletes_sooner(demo_report):
    case = _case(demo_report, "expense")
    horizon = demo_report.horizon_age
    assert _age(case.high_depletion_age, horizon) < _age(case.low_depletion_age, horizon)


def test_higher_inflation_depletes_sooner(demo_report):
    case = _case(demo_report, "inflation")
    horizon = demo_report.horizon_age
    assert _age(case.high_depletion_age, horizon) <= _age(case.low_depletion_age, horizon)


def test_more_assets_last_longer(demo_report):
    case = _case(demo_report, "assets")
    horizon = demo_report.horizon_age
    assert _age(case.high_depletion_age, horizon) > _age(case.low_depletion_age, horizon)


def test_higher_returns_last_longer(demo_report):
    case = _case(demo_report, "returns")
    horizon = demo_report.horizon_age
    assert _age(case.high_depletion_age, horizon) >= _age(case.low_depletion_age, horizon)


def test_bigger_pension_lasts_longer(demo_report):
    case = _case(demo_report, "pension")
    horizon = demo_report.horizon_age
    assert _age(case.high_depletion_age, horizon) >= _age(case.low_depletion_age, horizon)


# --------------------------------------------------------------------------- #
# 결론 — 무엇이 결과를 지배하는가
# --------------------------------------------------------------------------- #


def test_cases_are_sorted_by_impact(demo_report):
    swings = [c.swing_years for c in demo_report.cases]
    assert swings == sorted(swings, reverse=True)


def test_summary_names_the_dominant_assumption(demo_report):
    top = demo_report.most_sensitive
    assert top is not None
    assert top.label in demo_report.summary
    assert str(top.swing_years) in demo_report.summary


def test_expense_dominates_for_both_personas():
    """생활비가 가장 큰 변수라는 것은 두 인물 모두에서 성립한다.

    사용자가 스스로 답한 값이라 오차가 크면서 결과 지배력도 가장 크다 —
    "이 값을 정확히 아는 것이 가장 중요합니다"라고 말할 근거가 된다.
    """
    for profile in (DEMO_PROFILE, DIVERSIFIED_PROFILE):
        assert analyze_sensitivity(profile).most_sensitive.key == "expense"


def test_a_safe_plan_can_still_be_fragile():
    """DIVERSIFIED 는 고갈되지 않지만 생활비가 10% 늘면 무너진다.

    안정도 98점만 보고는 알 수 없는 사실이고, 이 분석이 존재하는 이유다.
    """
    report = analyze_sensitivity(DIVERSIFIED_PROFILE)
    assert report.baseline_depletion_age is None

    expense = _case(report, "expense")
    assert expense.high_depletion_age is not None
    assert expense.high_depletion_age < report.horizon_age


# --------------------------------------------------------------------------- #
# 표시와 한계
# --------------------------------------------------------------------------- #


def test_every_case_is_explained(demo_report):
    for case in demo_report.cases:
        assert case.detail.strip()
        assert case.baseline_label.strip()
        assert case.low_label and case.high_label


def test_the_univariate_limit_is_disclosed(demo_report):
    """한 번에 하나만 흔든다는 사실을 숨기면 안 된다."""
    assert "동시에" in demo_report.caveat


def test_profile_without_a_pension_skips_that_case():
    no_pension = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 0})
    keys = {c.key for c in analyze_sensitivity(no_pension).cases}
    assert "pension" not in keys
    assert "expense" in keys
