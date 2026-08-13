"""소득공백기 캐시플로우 시뮬레이션 엔진 (기능 ③).

이 서비스의 심장이다. 기획서에 등장하는 모든 숫자 — "4년 소득공백기",
"6.2년간 생활 가능", "연금 개시 시점 잔여자산" — 가 전부 여기서 나온다.

    Asset(t+1) = Asset(t) + Return(t) + Income(t) - Expense(t) - Tax(t)
    Income(t)  = OtherIncome + (t >= pension_start ? PensionIncome : 0)

명시적 계산 규약 (발표에서 질문받을 항목):
  1. 시뮬레이션은 **퇴직 시점부터** 시작한다. 퇴직 전 축적은 반영하지 않는다(보수적).
  2. 투자수익은 **연초 잔액 기준**으로 계산하고, 인출은 연말에 발생하는 것으로 본다.
  3. 생활비는 매년 물가상승률만큼 증가한다. 국민연금은 법정 물가연동을 반영한다.
     기타소득(임대·근로)은 물가에 연동하지 않는다(보수적).
  4. 세금은 이자·배당 성격의 수익(현금성·채권)에만 부과한다. 국내주식 양도차익은
     소액주주 비과세 현실을 반영해 과세하지 않는다.
  5. 퇴직 연도와 연금 개시 연도는 개월 단위로 안분한다.
"""

from __future__ import annotations

from typing import Optional

from core.assumptions import Assumptions, load_assumptions
from core.models import Allocation, SimulationResult, UserProfile, YearRow


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


def simulate(
    profile: UserProfile,
    *,
    allocation: Optional[Allocation] = None,
    assumptions: Optional[Assumptions] = None,
    include_income: bool = True,
    _compute_coverage: bool = True,
) -> SimulationResult:
    """퇴직 시점부터 horizon_age 까지 연 단위로 자산 잔액을 시뮬레이션한다.

    Args:
        profile: 사용자 프로파일.
        allocation: 자산군 배분. 생략하면 프로파일에서 추정한 현재 배분을 쓴다.
        assumptions: 가정값. 생략하면 data/assumptions.yaml.
        include_income: False면 국민연금·기타소득을 모두 0으로 두고 순수 인출만
            시뮬레이션한다. `expense_coverage_years` 계산에 쓰인다.
        _compute_coverage: 내부용 재귀 방지 플래그.
    """
    assumptions = assumptions or load_assumptions()
    allocation = allocation or profile.current_allocation()

    weights = allocation.as_dict()
    classes = assumptions.asset_classes
    inflation = assumptions.macro.inflation_rate

    # 이자·배당 성격(과세 대상)과 주식 양도차익 성격(비과세)을 분리한다.
    taxable_return_rate = sum(
        weights[key] * classes[key].expected_return for key in ("cash", "bond")
    )
    equity_return_rate = weights["equity"] * classes["equity"].expected_return

    start_year = profile.retirement_year
    end_year = profile.birth_year + assumptions.macro.horizon_age

    balance = float(profile.financial_assets)
    starting_balance = int(round(balance))

    rows: list[YearRow] = []
    elapsed_years = 0.0
    years_until_depletion: Optional[float] = None
    depletion_year: Optional[int] = None
    balance_at_pension_start: Optional[int] = None

    for year in range(start_year, end_year + 1):
        age = year - profile.birth_year
        months = _active_months(year, profile)
        frac = months / 12.0
        year_index = year - start_year
        inflation_factor = (1.0 + inflation) ** year_index

        if balance_at_pension_start is None and year >= profile.pension_start_year:
            balance_at_pension_start = int(round(balance))

        if depletion_year is not None:
            # 이미 고갈된 이후 — 차트 길이를 맞추기 위해 0으로 채운다.
            rows.append(
                YearRow(
                    year=year,
                    age=age,
                    active_months=months,
                    start_balance=0,
                    investment_return=0,
                    pension_income=0,
                    other_income=0,
                    expense=0,
                    tax=0,
                    end_balance=0,
                )
            )
            continue

        start_balance = balance

        # --- 수익 ---
        taxable_return = start_balance * taxable_return_rate * frac
        equity_return = start_balance * equity_return_rate * frac
        investment_return = taxable_return + equity_return

        # --- 소득 ---
        if include_income:
            pension_factor = inflation_factor if assumptions.pension.indexed_to_inflation else 1.0
            pension_income = (
                profile.national_pension_monthly
                * pension_factor
                * _pension_months(year, profile)
            )
            other_income = profile.other_monthly_income * months
        else:
            pension_income = 0.0
            other_income = 0.0

        # --- 지출 ---
        expense = profile.monthly_expense * inflation_factor * months

        # --- 세금 (금융소득세 + 건강보험료 소득분 근사) ---
        income_tax = taxable_return * assumptions.tax.financial_income_rate
        hi = assumptions.health_insurance
        hi_base = max(0.0, taxable_return - hi.exemption_threshold)
        health_premium = hi_base * hi.effective_rate_on_financial_income
        tax = income_tax + health_premium

        end_balance = (
            start_balance + investment_return + pension_income + other_income - expense - tax
        )

        if end_balance < 0:
            drain = start_balance - end_balance
            portion = start_balance / drain if drain > 0 else 0.0
            years_until_depletion = elapsed_years + portion * frac
            depletion_year = year
            end_balance = 0.0

        rows.append(
            YearRow(
                year=year,
                age=age,
                active_months=months,
                start_balance=int(round(start_balance)),
                investment_return=int(round(investment_return)),
                pension_income=int(round(pension_income)),
                other_income=int(round(other_income)),
                expense=int(round(expense)),
                tax=int(round(tax)),
                end_balance=int(round(end_balance)),
            )
        )

        balance = end_balance
        elapsed_years += frac

    if balance_at_pension_start is None:
        # 연금 개시 전에 시뮬레이션이 끝나는 비정상 케이스
        balance_at_pension_start = int(round(balance))

    # 연금·기타소득을 전혀 고려하지 않은 순수 인출 기준 생존 연수
    if _compute_coverage:
        bare = simulate(
            profile,
            allocation=allocation,
            assumptions=assumptions,
            include_income=False,
            _compute_coverage=False,
        )
        coverage = (
            bare.years_until_depletion
            if bare.years_until_depletion is not None
            else bare.horizon_age - profile.retirement_age
        )
    else:
        coverage = years_until_depletion if years_until_depletion is not None else elapsed_years

    return SimulationResult(
        rows=rows,
        allocation=allocation,
        income_gap_months=profile.income_gap_months,
        pension_start_year=profile.pension_start_year,
        balance_at_pension_start=balance_at_pension_start,
        depletion_year=depletion_year,
        depletion_age=None if depletion_year is None else depletion_year - profile.birth_year,
        years_until_depletion=years_until_depletion,
        expense_coverage_years=round(coverage, 2),
        starting_balance=starting_balance,
        final_balance=rows[-1].end_balance if rows else starting_balance,
        horizon_age=assumptions.macro.horizon_age,
    )
