"""연도별 현금흐름 스케줄.

퇴직월·연금개시월 안분, 물가연동 같은 자잘하지만 틀리기 쉬운 규칙을 한 곳에 모은다.
결정론적 엔진(cashflow.py)과 몬테카를로(montecarlo.py)가 **같은 스케줄**을 쓰게 해서
두 엔진이 조용히 어긋나는 것을 막는 것이 이 모듈의 존재 이유다.

경로에 따라 달라지지 않는(= 투자수익과 무관한) 값만 여기서 계산한다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from core.assumptions import Assumptions, load_assumptions
from core.models import UserProfile


class YearPlan(BaseModel):
    """한 연도의 확정적 현금흐름."""

    year: int
    age: int
    active_months: int
    year_index: int = 0
    inflation_factor: float = 1.0
    expense: float = 0.0
    pension_income: float = 0.0
    other_income: float = 0.0

    @property
    def frac(self) -> float:
        """해당 연도에서 시뮬레이션이 적용되는 비율(0~1)."""
        return self.active_months / 12.0

    @property
    def total_income(self) -> float:
        return self.pension_income + self.other_income


def _active_months(year: int, profile: UserProfile) -> int:
    """해당 연도에 시뮬레이션이 적용되는 개월 수 (퇴직 연도는 퇴직월부터)."""
    start_month = profile.retirement_month if year == profile.retirement_year else 1
    return 13 - start_month


def _pension_months(year: int, profile: UserProfile) -> int:
    """해당 연도에 국민연금을 수령하는 개월 수."""
    if profile.national_pension_monthly <= 0:
        return 0

    pension_year = profile.pension_start_year
    if year < pension_year:
        return 0

    pension_start_month = profile.birth_month if year == pension_year else 1
    period_start_month = profile.retirement_month if year == profile.retirement_year else 1
    effective_start = max(pension_start_month, period_start_month)
    return max(0, 13 - effective_start)


def build_schedule(
    profile: UserProfile,
    assumptions: Optional[Assumptions] = None,
    *,
    include_income: bool = True,
) -> list[YearPlan]:
    """퇴직 시점부터 horizon_age 까지의 연도별 확정 현금흐름을 만든다.

    Args:
        include_income: False면 국민연금·기타소득을 0으로 둔다. 순수 인출 기준
            '생활비 충당 가능 연수'를 구할 때 쓴다.
    """
    assumptions = assumptions or load_assumptions()
    inflation = assumptions.macro.inflation_rate
    pension_indexed = assumptions.pension.indexed_to_inflation

    # 법정 개시연령보다 미루면 가산, 앞당기면 감액된 금액을 평생 받는다.
    # 프로파일이 법정 연령 그대로면 1.0 이라 기존 결과는 바뀌지 않는다.
    pension_factor = assumptions.pension_amount_factor(
        profile.birth_year, profile.national_pension_start_age
    )

    start_year = profile.retirement_year
    end_year = profile.birth_year + assumptions.macro.horizon_age

    plans: list[YearPlan] = []
    for year in range(start_year, end_year + 1):
        year_index = year - start_year
        months = _active_months(year, profile)
        inflation_factor = (1.0 + inflation) ** year_index

        if include_income:
            indexation = inflation_factor if pension_indexed else 1.0
            pension_income = (
                profile.national_pension_monthly
                * pension_factor
                * indexation
                * _pension_months(year, profile)
            )
            # 기타소득(임대·근로)은 물가에 연동하지 않는다 — 보수적 가정.
            other_income = float(profile.other_monthly_income * months)
        else:
            pension_income = 0.0
            other_income = 0.0

        plans.append(
            YearPlan(
                year=year,
                age=year - profile.birth_year,
                active_months=months,
                year_index=year_index,
                inflation_factor=inflation_factor,
                expense=profile.monthly_expense * inflation_factor * months,
                pension_income=pension_income,
                other_income=other_income,
            )
        )

    return plans
