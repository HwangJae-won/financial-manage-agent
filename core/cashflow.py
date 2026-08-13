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
  5. 퇴직 연도와 연금 개시 연도는 개월 단위로 안분한다 (core/schedule.py).
"""

from __future__ import annotations

from typing import Optional, Protocol

from core.assumptions import Assumptions, load_assumptions
from core.models import Allocation, SimulationResult, UserProfile, YearRow
from core.schedule import build_schedule


class TaxModel(Protocol):
    """연간 세금 계산 규약.

    기본값은 아래 `annual_tax` 다. 이 자리를 열어 둔 이유는 정책 영향 분석(W7)이
    "세제가 이렇게 바뀌면 계획이 어떻게 달라지는가"를 **같은 엔진으로** 다시 돌려야
    하기 때문이다. 세제만 갈아끼우고 나머지 규약(수익 시점, 물가연동, 안분)은
    그대로 공유해야 두 결과의 차이가 오직 세제 차이가 된다.
    """

    def __call__(self, taxable_return: float, assumptions: Assumptions) -> float: ...


def split_return_rates(
    allocation: Allocation, assumptions: Assumptions
) -> tuple[float, float]:
    """배분을 (과세 대상 수익률, 비과세 수익률)로 나눈다.

    현금성·채권 수익은 이자·배당 성격이라 과세 대상이고,
    주식 수익은 국내 소액주주 양도차익 비과세를 반영해 과세하지 않는다.
    몬테카를로도 같은 규약을 써야 하므로 여기에 둔다.
    """
    weights = allocation.as_dict()
    classes = assumptions.asset_classes
    taxable = sum(weights[key] * classes[key].expected_return for key in ("cash", "bond"))
    untaxed = weights["equity"] * classes["equity"].expected_return
    return taxable, untaxed


def annual_tax(taxable_return: float, assumptions: Assumptions) -> float:
    """금융소득세 + 건강보험료 소득분 근사. 손실이 난 해에는 0."""
    if taxable_return <= 0:
        return 0.0
    income_tax = taxable_return * assumptions.tax.financial_income_rate
    hi = assumptions.health_insurance
    hi_base = max(0.0, taxable_return - hi.exemption_threshold)
    return income_tax + hi_base * hi.effective_rate_on_financial_income


def simulate(
    profile: UserProfile,
    *,
    allocation: Optional[Allocation] = None,
    assumptions: Optional[Assumptions] = None,
    include_income: bool = True,
    tax_model: Optional[TaxModel] = None,
    _compute_coverage: bool = True,
) -> SimulationResult:
    """퇴직 시점부터 horizon_age 까지 연 단위로 자산 잔액을 시뮬레이션한다.

    Args:
        profile: 사용자 프로파일.
        allocation: 자산군 배분. 생략하면 프로파일에서 추정한 현재 배분을 쓴다.
        assumptions: 가정값. 생략하면 data/assumptions.yaml.
        include_income: False면 국민연금·기타소득을 모두 0으로 두고 순수 인출만
            시뮬레이션한다. `expense_coverage_years` 계산에 쓰인다.
        tax_model: 세금 계산기. 생략하면 `annual_tax`(기본 규약)를 쓴다.
            정책 영향 분석에서 세제만 바꿔 재시뮬레이션할 때 주입한다.
        _compute_coverage: 내부용 재귀 방지 플래그.
    """
    assumptions = assumptions or load_assumptions()
    allocation = allocation or profile.current_allocation()
    compute_tax: TaxModel = tax_model or annual_tax

    taxable_rate, untaxed_rate = split_return_rates(allocation, assumptions)
    plans = build_schedule(profile, assumptions, include_income=include_income)

    balance = float(profile.financial_assets)
    starting_balance = int(round(balance))

    rows: list[YearRow] = []
    elapsed_years = 0.0
    years_until_depletion: Optional[float] = None
    depletion_year: Optional[int] = None
    balance_at_pension_start: Optional[int] = None

    for plan in plans:
        if balance_at_pension_start is None and plan.year >= profile.pension_start_year:
            balance_at_pension_start = int(round(balance))

        if depletion_year is not None:
            # 이미 고갈된 이후 — 차트 길이를 맞추기 위해 0으로 채운다.
            rows.append(
                YearRow(
                    year=plan.year,
                    age=plan.age,
                    active_months=plan.active_months,
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
        frac = plan.frac

        taxable_return = start_balance * taxable_rate * frac
        investment_return = taxable_return + start_balance * untaxed_rate * frac
        tax = compute_tax(taxable_return, assumptions)

        end_balance = (
            start_balance
            + investment_return
            + plan.pension_income
            + plan.other_income
            - plan.expense
            - tax
        )

        if end_balance < 0:
            drain = start_balance - end_balance
            portion = start_balance / drain if drain > 0 else 0.0
            years_until_depletion = elapsed_years + portion * frac
            depletion_year = plan.year
            end_balance = 0.0

        rows.append(
            YearRow(
                year=plan.year,
                age=plan.age,
                active_months=plan.active_months,
                start_balance=int(round(start_balance)),
                investment_return=int(round(investment_return)),
                pension_income=int(round(plan.pension_income)),
                other_income=int(round(plan.other_income)),
                expense=int(round(plan.expense)),
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
            tax_model=tax_model,
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
