"""몬테카를로 시뮬레이션 (기능 ③ 확률 파트).

결정론적 시뮬레이션은 "기대수익률이 그대로 실현되면" 이라는 하나의 미래만 보여준다.
실제로 사용자가 알고 싶은 것은 "그래서 돈이 떨어질 가능성이 얼마나 되느냐"이고,
그건 수익률 분포를 넣고 수천 번 돌려봐야 나온다.

    "10년 후 자산이 남아있을 확률 92% / 80세 이전 자산 고갈 가능성 18%"

구현 노트:
  - 연도별 확정 현금흐름은 core.schedule 을 그대로 쓴다. 결정론적 엔진과
    같은 스케줄을 공유하므로 두 엔진이 어긋날 수 없다.
  - 성능을 위해 경로를 numpy 로 벡터화했다. 규약이 결정론적 엔진과 일치하는지는
    "변동성 0이면 결정론적 결과와 같아야 한다"는 테스트로 강제한다.

명시적 한계 (발표에서 질문받을 항목):
  - 자산군 간 상관계수를 0으로, 연도 간 수익률을 독립으로 가정한다.
    실제 시장의 상관과 평균회귀를 반영하지 않으므로 꼬리 위험을 과소평가할 수 있다.
  - 수익률은 정규분포를 가정한다. 실제 수익률 분포의 두꺼운 꼬리는 반영되지 않는다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import annual_tax
from core.models import Allocation, UserProfile
from core.schedule import build_schedule

DEFAULT_PATHS = 3_000
DEFAULT_SEED = 20260813


class MonteCarloResult(BaseModel):
    """몬테카를로 결과. 여기 있는 확률만 LLM 설명 계층이 인용할 수 있다."""

    n_paths: int
    seed: int
    allocation: Allocation
    retirement_age: int
    horizon_age: int

    survival_by_age: dict[int, float] = Field(
        description="나이 → 그 나이까지 금융자산이 남아있을 확률"
    )
    depletion_age_p10: Optional[int] = Field(
        default=None, description="하위 10% 경로의 고갈 나이 (나쁜 시나리오)"
    )
    depletion_age_median: Optional[int] = None
    terminal_balance_p10: int = Field(description="종료 시점 잔액 하위 10%")
    terminal_balance_median: int
    terminal_balance_p90: int

    prob_survive_income_gap: float = Field(
        description="국민연금 개시 시점까지 자산이 버틸 확률"
    )

    def survival_at_age(self, age: int) -> float:
        """특정 나이에 자산이 남아있을 확률. 범위를 벗어나면 가장 가까운 값."""
        if not self.survival_by_age:
            return 0.0
        ages = sorted(self.survival_by_age)
        if age <= ages[0]:
            return self.survival_by_age[ages[0]]
        if age >= ages[-1]:
            return self.survival_by_age[ages[-1]]
        return self.survival_by_age[min(ages, key=lambda a: abs(a - age))]

    def survival_after_years(self, years: int) -> float:
        """퇴직 후 N년 시점에 자산이 남아있을 확률."""
        return self.survival_at_age(self.retirement_age + years)

    def prob_depleted_before_age(self, age: int) -> float:
        """특정 나이 이전에 자산이 고갈될 확률."""
        return 1.0 - self.survival_at_age(age)


def run_monte_carlo(
    profile: UserProfile,
    *,
    allocation: Optional[Allocation] = None,
    assumptions: Optional[Assumptions] = None,
    n_paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    """수익률 분포를 넣고 n_paths 개의 미래를 시뮬레이션한다.

    같은 seed 면 항상 같은 결과가 나온다 (데모 재현성).
    """
    assumptions = assumptions or load_assumptions()
    allocation = allocation or profile.current_allocation()
    plans = build_schedule(profile, assumptions)

    weights = allocation.as_dict()
    classes = assumptions.asset_classes
    taxable_keys = ("cash", "bond")

    rng = np.random.default_rng(seed)
    balance = np.full(n_paths, float(profile.financial_assets))
    alive = np.ones(n_paths, dtype=bool)
    depletion_age = np.full(n_paths, -1, dtype=np.int64)

    survival_by_age: dict[int, float] = {}

    tax_rate = assumptions.tax.financial_income_rate
    hi = assumptions.health_insurance

    for plan in plans:
        frac = plan.frac

        # 자산군별 실현 수익률을 뽑아 과세/비과세 성격으로 나눈다.
        taxable_rate = np.zeros(n_paths)
        for key in taxable_keys:
            spec = classes[key]
            draws = rng.normal(spec.expected_return, spec.volatility, n_paths)
            taxable_rate += weights[key] * draws

        equity_spec = classes["equity"]
        untaxed_rate = weights["equity"] * rng.normal(
            equity_spec.expected_return, equity_spec.volatility, n_paths
        )

        taxable_return = balance * taxable_rate * frac
        investment_return = taxable_return + balance * untaxed_rate * frac

        # annual_tax 와 동일한 규약을 벡터화한 것 (손실 연도는 0, 건보료는 공제 후)
        positive = np.maximum(taxable_return, 0.0)
        tax = positive * tax_rate + np.maximum(
            0.0, positive - hi.exemption_threshold
        ) * hi.effective_rate_on_financial_income

        new_balance = (
            balance
            + investment_return
            + plan.pension_income
            + plan.other_income
            - plan.expense
            - plan.event_expense
            - tax
        )

        newly_depleted = alive & (new_balance < 0)
        depletion_age[newly_depleted] = plan.age
        alive &= ~newly_depleted

        # 고갈된 경로는 이후로도 0으로 고정한다 (연금만으로 생활하는 상태).
        balance = np.where(alive, np.maximum(new_balance, 0.0), 0.0)

        survival_by_age[plan.age] = float(alive.mean())

    depleted = depletion_age[depletion_age >= 0]
    horizon_age = assumptions.macro.horizon_age

    pension_age = profile.pension_start_year - profile.birth_year
    prob_survive_gap = survival_by_age.get(
        pension_age, next(iter(survival_by_age.values()), 0.0)
    )

    return MonteCarloResult(
        n_paths=n_paths,
        seed=seed,
        allocation=allocation,
        retirement_age=profile.retirement_age,
        horizon_age=horizon_age,
        survival_by_age=survival_by_age,
        depletion_age_p10=int(np.percentile(depleted, 10)) if depleted.size else None,
        depletion_age_median=int(np.percentile(depleted, 50)) if depleted.size else None,
        terminal_balance_p10=int(np.percentile(balance, 10)),
        terminal_balance_median=int(np.percentile(balance, 50)),
        terminal_balance_p90=int(np.percentile(balance, 90)),
        prob_survive_income_gap=prob_survive_gap,
    )
