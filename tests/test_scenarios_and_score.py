from __future__ import annotations

import pytest

from core.risk_score import GREEN, RED, WEIGHTS, compute_risk_score
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.scenarios import (
    build_scenarios,
    cash_floor_ratio,
    comparison_table,
    volatility_label,
)

# 시나리오 생성은 몬테카를로를 3번 돌리므로 모듈 단위로 캐시한다.
FAST_PATHS = 600


@pytest.fixture(scope="module")
def demo_scenarios():
    return build_scenarios(DEMO_PROFILE, n_paths=FAST_PATHS)


@pytest.fixture(scope="module")
def diversified_scenarios():
    return build_scenarios(DIVERSIFIED_PROFILE, n_paths=FAST_PATHS)


# --------------------------------------------------------------------------- #
# 시나리오
# --------------------------------------------------------------------------- #


def test_three_scenarios_are_generated(demo_scenarios):
    assert [s.key for s in demo_scenarios] == ["stable", "balanced", "growth"]
    assert [s.label for s in demo_scenarios] == ["안정형", "균형형", "성장형"]


def test_allocations_are_valid(demo_scenarios):
    for scenario in demo_scenarios:
        weights = scenario.allocation.as_dict()
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.values())


def test_amounts_sum_to_financial_assets(diversified_scenarios):
    for scenario in diversified_scenarios:
        total = sum(scenario.amounts.values())
        assert total == pytest.approx(DIVERSIFIED_PROFILE.financial_assets, abs=3)


def test_equity_share_never_decreases_across_scenarios(diversified_scenarios):
    """성장형이 안정형보다 주식 비중이 낮으면 안 된다."""
    equities = [s.allocation.equity for s in diversified_scenarios]
    assert equities == sorted(equities)


def test_volatility_increases_with_equity(diversified_scenarios):
    vols = [s.volatility for s in diversified_scenarios]
    assert vols == sorted(vols)


def test_cash_floor_protects_the_income_gap():
    """소득공백기에 필요한 돈은 어떤 성향에서도 현금으로 남아야 한다."""
    floor = cash_floor_ratio(DEMO_PROFILE)
    for scenario in build_scenarios(DEMO_PROFILE, n_paths=100):
        assert scenario.allocation.cash >= floor - 1e-9


def test_tight_budget_collapses_the_scenarios():
    """자산이 빠듯하면 세 시나리오가 수렴한다 — '위험을 질 여력이 없다'는 진단."""
    tight = DEMO_PROFILE.model_copy(
        update={"severance_pay": 100_000_000, "cash_savings": 0, "monthly_expense": 4_000_000}
    )
    assert cash_floor_ratio(tight) == 1.0
    scenarios = build_scenarios(tight, n_paths=100)
    assert all(s.allocation.cash == 1.0 for s in scenarios)
    assert all(s.allocation.equity == 0.0 for s in scenarios)


def test_growth_scenario_has_wider_outcome_spread(diversified_scenarios):
    stable, _, growth = diversified_scenarios
    stable_spread = (
        stable.monte_carlo.terminal_balance_p90 - stable.monte_carlo.terminal_balance_p10
    )
    growth_spread = (
        growth.monte_carlo.terminal_balance_p90 - growth.monte_carlo.terminal_balance_p10
    )
    assert growth_spread > stable_spread


def test_comparison_table_shape(demo_scenarios):
    rows = comparison_table(demo_scenarios)
    assert [r["지표"] for r in rows] == [
        "소득공백기 안정성",
        "85세 이전 자산 고갈 가능성",
        "변동성",
    ]
    for row in rows:
        for scenario in demo_scenarios:
            assert scenario.label in row


@pytest.mark.parametrize(
    "vol,expected", [(0.01, "낮음"), (0.07, "중간"), (0.15, "높음")]
)
def test_volatility_label(vol: float, expected: str):
    assert volatility_label(vol) == expected


def test_scenario_metrics_are_probabilities(demo_scenarios):
    for scenario in demo_scenarios:
        assert 0.0 <= scenario.prob_survive_income_gap <= 1.0
        assert 0.0 <= scenario.prob_deplete_before_85 <= 1.0
        assert scenario.downside_terminal_balance <= scenario.median_terminal_balance


# --------------------------------------------------------------------------- #
# 리스크 스코어
# --------------------------------------------------------------------------- #


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_score_components_match_spec():
    score = compute_risk_score(DEMO_PROFILE)
    assert [c.key for c in score.components] == [
        "income_gap",
        "liquidity",
        "concentration",
        "market_risk",
        "debt",
        "pension_adequacy",
    ]
    for component in score.components:
        assert 0 <= component.score <= 100
        assert component.weight == WEIGHTS[component.key]
        assert component.detail


def test_total_is_weighted_average():
    score = compute_risk_score(DIVERSIFIED_PROFILE)
    expected = sum(c.score * c.weight for c in score.components)
    assert score.total == round(expected)
    assert 0 <= score.total <= 100


def test_demo_profile_is_strong_short_term_weak_long_term():
    """데모의 핵심 메시지가 점수로도 드러나야 한다."""
    score = compute_risk_score(DEMO_PROFILE)
    assert score.by_key("income_gap").status == GREEN  # 단기는 양호
    assert score.by_key("pension_adequacy").score < 80  # 장기소득은 부족
    assert score.weakest.key in {"pension_adequacy", "concentration"}


def test_no_income_gap_scores_full_marks():
    profile = DEMO_PROFILE.model_copy(
        update={"birth_year": 1960, "national_pension_start_age": 63}
    )
    assert profile.income_gap_months == 0
    assert compute_risk_score(profile).by_key("income_gap").score == 100


def test_heavy_debt_scores_zero():
    indebted = DEMO_PROFILE.model_copy(update={"debt": DEMO_PROFILE.total_assets})
    component = compute_risk_score(indebted).by_key("debt")
    assert component.score == 0
    assert component.status == RED


def test_no_debt_scores_full_marks():
    assert compute_risk_score(DEMO_PROFILE).by_key("debt").score == 100


def test_concentration_penalises_real_estate_heavy_profiles():
    balanced = compute_risk_score(DIVERSIFIED_PROFILE).by_key("concentration")
    lopsided = compute_risk_score(
        DEMO_PROFILE.model_copy(update={"real_estate": 3_000_000_000})
    ).by_key("concentration")
    assert lopsided.score < balanced.score


def test_market_risk_depends_on_risk_tolerance():
    from core.models import RiskTolerance

    aggressive_holdings = DIVERSIFIED_PROFILE.model_copy(
        update={"equity": 400_000_000, "cash_savings": 0}
    )
    conservative = aggressive_holdings.model_copy(
        update={"risk_tolerance": RiskTolerance.CONSERVATIVE}
    )
    aggressive = aggressive_holdings.model_copy(
        update={"risk_tolerance": RiskTolerance.AGGRESSIVE}
    )
    assert (
        compute_risk_score(conservative).by_key("market_risk").score
        < compute_risk_score(aggressive).by_key("market_risk").score
    )


def test_weakest_component_is_the_lowest_scoring_one():
    score = compute_risk_score(DEMO_PROFILE)
    assert score.weakest.score == min(c.score for c in score.components)


def test_status_icons_are_assigned():
    score = compute_risk_score(DEMO_PROFILE)
    assert all(c.icon in {"🟢", "🟠", "🔴"} for c in score.components)


# --------------------------------------------------------------------------- #
# 지속가능성 상한 — "양호 91점 + 만 69세 고갈" 모순 방지
# --------------------------------------------------------------------------- #


def test_doomed_plan_cannot_score_as_healthy():
    """자산이 곧 고갈되는 계획이 '양호'로 표시되면 안 된다."""
    score = compute_risk_score(DEMO_PROFILE)
    assert score.raw_total >= 80  # 단기 지표만 보면 높은 점수가 나오지만
    assert score.cap == 40  # 지속가능성 상한이 걸리고
    assert score.total == 40
    assert score.status == RED
    assert "고갈" in score.cap_reason


def test_sustainable_plan_is_not_capped():
    score = compute_risk_score(DIVERSIFIED_PROFILE)
    assert score.cap is None
    assert score.total == score.raw_total
    assert score.status == GREEN


def test_cap_tiers_by_depletion_age():
    """고갈이 늦을수록 상한이 완화된다."""
    early = compute_risk_score(DEMO_PROFILE)  # 만 69세 고갈
    later = compute_risk_score(
        DEMO_PROFILE.model_copy(update={"cash_savings": 220_000_000})
    )
    assert early.cap == 40
    assert later.cap in (60, None)
    assert later.total > early.total


def test_long_term_component_reflects_the_simulation():
    """장기소득 점수는 연금 비율뿐 아니라 실제 고갈 시점을 반영해야 한다."""
    doomed = compute_risk_score(DEMO_PROFILE).by_key("pension_adequacy")
    safe = compute_risk_score(
        DEMO_PROFILE.model_copy(update={"cash_savings": 1_000_000_000})
    ).by_key("pension_adequacy")
    # 연금액은 같지만 자산이 버티는 기간이 다르므로 점수가 달라야 한다
    assert safe.score > doomed.score
