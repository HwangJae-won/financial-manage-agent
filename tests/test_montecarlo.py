"""몬테카를로 테스트.

가장 중요한 것은 첫 번째 테스트다: 변동성을 0으로 두면 몬테카를로가 결정론적
엔진과 정확히 같은 답을 내야 한다. 두 엔진이 조용히 어긋나는 것을 막는 장치다.
"""

from __future__ import annotations

import pytest

from core.assumptions import Assumptions
from core.cashflow import simulate
from core.models import Allocation
from core.montecarlo import run_monte_carlo
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture
def zero_vol_assumptions(base_assumptions: Assumptions) -> Assumptions:
    """수익률은 그대로 두고 변동성만 0으로 만든 가정."""
    a = base_assumptions.model_copy(deep=True)
    for cls in a.asset_classes.values():
        cls.volatility = 0.0
    return a


# --------------------------------------------------------------------------- #
# 두 엔진의 일관성
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", [DEMO_PROFILE, DIVERSIFIED_PROFILE])
def test_zero_volatility_reproduces_deterministic_engine(profile, zero_vol_assumptions):
    """변동성 0이면 모든 경로가 결정론적 시뮬레이션과 같아야 한다."""
    deterministic = simulate(profile, assumptions=zero_vol_assumptions)
    mc = run_monte_carlo(profile, assumptions=zero_vol_assumptions, n_paths=50)

    if deterministic.depletion_age is None:
        assert mc.depletion_age_median is None
        assert mc.survival_by_age[mc.horizon_age] == 1.0
        assert mc.terminal_balance_median == pytest.approx(
            deterministic.final_balance, rel=1e-6
        )
    else:
        assert mc.depletion_age_median == deterministic.depletion_age
        assert mc.depletion_age_p10 == deterministic.depletion_age


def test_zero_volatility_gives_degenerate_percentiles(zero_vol_assumptions):
    """변동성이 없으면 하위 10%와 상위 90%가 같아야 한다 (불확실성 없음)."""
    mc = run_monte_carlo(DIVERSIFIED_PROFILE, assumptions=zero_vol_assumptions, n_paths=200)
    assert mc.terminal_balance_p10 == pytest.approx(mc.terminal_balance_p90, rel=1e-6)


# --------------------------------------------------------------------------- #
# 확률의 기본 성질
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo_mc():
    return run_monte_carlo(DEMO_PROFILE, n_paths=2_000)


def test_survival_probability_is_monotonically_decreasing(demo_mc):
    """나이가 들수록 자산이 남아있을 확률은 줄기만 해야 한다."""
    by_age = [demo_mc.survival_by_age[age] for age in sorted(demo_mc.survival_by_age)]
    for earlier, later in zip(by_age, by_age[1:]):
        assert later <= earlier + 1e-9


def test_probabilities_are_within_range(demo_mc):
    assert all(0.0 <= p <= 1.0 for p in demo_mc.survival_by_age.values())
    assert 0.0 <= demo_mc.prob_survive_income_gap <= 1.0
    assert 0.0 <= demo_mc.prob_depleted_before_age(80) <= 1.0


def test_terminal_percentiles_are_ordered(demo_mc):
    assert demo_mc.terminal_balance_p10 <= demo_mc.terminal_balance_median
    assert demo_mc.terminal_balance_median <= demo_mc.terminal_balance_p90


def test_same_seed_is_reproducible():
    a = run_monte_carlo(DEMO_PROFILE, n_paths=300, seed=7)
    b = run_monte_carlo(DEMO_PROFILE, n_paths=300, seed=7)
    assert a.survival_by_age == b.survival_by_age
    assert a.terminal_balance_median == b.terminal_balance_median


def test_different_seed_changes_results():
    # 두 프로파일 모두 절반 이상의 경로가 고갈되어 중앙값이 0이므로,
    # 시드 차이는 상위 분위와 생존 확률에서 봐야 한다.
    a = run_monte_carlo(DIVERSIFIED_PROFILE, n_paths=300, seed=1)
    b = run_monte_carlo(DIVERSIFIED_PROFILE, n_paths=300, seed=2)
    assert a.terminal_balance_p90 != b.terminal_balance_p90
    assert a.survival_by_age != b.survival_by_age


def test_demo_profile_depletes_on_every_path(demo_mc):
    """기획서 예시 인물은 현재 계획대로면 사실상 100% 고갈된다 — 데모의 핵심 경고."""
    assert demo_mc.survival_by_age[demo_mc.horizon_age] == 0.0
    assert demo_mc.depletion_age_median is not None
    # 다만 소득공백기 자체는 넘긴다 (단기는 안전, 장기가 문제)
    assert demo_mc.prob_survive_income_gap == 1.0


def test_survival_after_years_helper(demo_mc):
    assert demo_mc.survival_after_years(0) == pytest.approx(
        demo_mc.survival_at_age(DEMO_PROFILE.retirement_age), abs=1e-9
    )
    assert demo_mc.survival_after_years(10) <= demo_mc.survival_after_years(1)


# --------------------------------------------------------------------------- #
# 경제적으로 말이 되는가
# --------------------------------------------------------------------------- #


def test_more_assets_means_higher_survival():
    poor = run_monte_carlo(
        DEMO_PROFILE.model_copy(update={"cash_savings": 0}), n_paths=1_000
    )
    rich = run_monte_carlo(
        DEMO_PROFILE.model_copy(update={"cash_savings": 300_000_000}), n_paths=1_000
    )
    assert rich.survival_after_years(15) > poor.survival_after_years(15)


def test_higher_expense_means_lower_survival():
    frugal = run_monte_carlo(
        DEMO_PROFILE.model_copy(update={"monthly_expense": 2_000_000}), n_paths=1_000
    )
    lavish = run_monte_carlo(
        DEMO_PROFILE.model_copy(update={"monthly_expense": 6_000_000}), n_paths=1_000
    )
    assert frugal.survival_after_years(10) > lavish.survival_after_years(10)


def test_equity_widens_the_outcome_spread():
    """주식 비중이 높으면 결과의 분산이 커져야 한다 — 변동성의 정의."""
    safe = run_monte_carlo(
        DIVERSIFIED_PROFILE,
        allocation=Allocation(cash=1.0, bond=0.0, equity=0.0),
        n_paths=1_500,
    )
    risky = run_monte_carlo(
        DIVERSIFIED_PROFILE,
        allocation=Allocation(cash=0.2, bond=0.0, equity=0.8),
        n_paths=1_500,
    )
    safe_spread = safe.terminal_balance_p90 - safe.terminal_balance_p10
    risky_spread = risky.terminal_balance_p90 - risky.terminal_balance_p10
    assert risky_spread > safe_spread


def test_ample_income_never_depletes():
    mc = run_monte_carlo(
        DEMO_PROFILE.model_copy(update={"other_monthly_income": 10_000_000}),
        n_paths=500,
    )
    assert mc.survival_by_age[mc.horizon_age] == 1.0
    assert mc.depletion_age_median is None
