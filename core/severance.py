"""퇴직금을 일시금으로 받을까, 연금으로 받을까 (기능 ⑤).

**타겟이 지금 당장 내려야 하는 결정이다.** 그런데 지금까지의 엔진은 퇴직금 2억을
그냥 현금 2억으로 봤다. 세금이 얼마인지도, 연금으로 받으면 무엇이 달라지는지도
화면에 없었다.

두 가지가 다르다:

  1. **세액** — IRP 등으로 연금 수령하면 퇴직소득세가 30% 감면된다(11년차부터 40%).
  2. **납부 시점** — 일시금은 퇴직할 때 한 번에 낸다. 연금은 받을 때마다 나눠 낸다.
     늦게 내는 만큼 그 돈이 더 오래 운용된다.

2번은 세율표만 봐서는 안 보이는데, 실제로는 이쪽 효과가 더 클 때도 있다. 그래서
세액 차이만 비교하지 않고 **두 경우를 각각 끝까지 시뮬레이션한다.**

구현 방식: 세금을 `LifeEvent`(일회성 지출)로 표현한다. 일시금은 퇴직 연도에 한 번,
연금은 수령 기간에 걸쳐 나눠서. 이렇게 하면 캐시플로우 엔진을 고치지 않고도
납부 시점 차이가 그대로 반영되고, 몬테카를로·처방·민감도에도 자동으로 얹힌다.

한계 (발표에서 질문받을 항목):
  - IRP 계좌 안의 운용수익에 대한 과세이연 효과는 반영하지 않는다. 실제로는 연금
    쪽이 여기서도 유리하므로, 이 비교는 **연금에 불리한 쪽으로 보수적**이다.
  - 연금 수령액이 연 1,500만원을 넘을 때의 종합과세 전환은 다루지 않는다.
  - 세금은 참고용이다. 실제 세액은 퇴직 사유·중간정산 이력에 따라 달라진다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, RetirementIncomeTaxAssumption, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_pct
from core.models import LifeEvent, UserProfile

LUMP_SUM = "일시금"
PENSION = "연금(IRP)"


# --------------------------------------------------------------------------- #
# 퇴직소득세
# --------------------------------------------------------------------------- #


def service_deduction(years: int, spec: RetirementIncomeTaxAssumption) -> int:
    """근속연수공제. 장기근속일수록 커져서 실효세율을 크게 낮춘다."""
    if years <= 0:
        return 0
    band = max(
        (b for b in spec.service_deduction if years > b.over_years),
        key=lambda b: b.over_years,
    )
    return band.base + band.per_year * (years - band.over_years)


def converted_deduction(converted: float, spec: RetirementIncomeTaxAssumption) -> float:
    """환산급여공제."""
    if converted <= 0:
        return 0.0
    band = max(
        (b for b in spec.converted_deduction if converted > b.over),
        key=lambda b: b.over,
    )
    return band.base + (converted - band.over) * band.rate


def progressive_tax(base: float, spec: RetirementIncomeTaxAssumption) -> float:
    """기본세율(누진)을 적용한다."""
    if base <= 0:
        return 0.0
    tax = 0.0
    for index, bracket in enumerate(spec.brackets):
        upper = (
            spec.brackets[index + 1].over
            if index + 1 < len(spec.brackets)
            else float("inf")
        )
        if base <= bracket.over:
            break
        tax += (min(base, upper) - bracket.over) * bracket.rate
    return tax


def retirement_income_tax(
    severance: int, years_employed: int, assumptions: Optional[Assumptions] = None
) -> int:
    """퇴직소득세 (지방소득세 포함).

        환산급여 = (퇴직급여 - 근속연수공제) ÷ 근속연수 × 12
        과세표준 = 환산급여 - 환산급여공제
        산출세액 = 과세표준 × 기본세율 ÷ 12 × 근속연수   (연분연승법)

    근속연수가 0이면 계산할 수 없으므로 0을 돌려준다 — 입력이 불완전해도
    화면이 멈추지 않아야 한다.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.retirement_income_tax

    if severance <= 0 or years_employed <= 0:
        return 0

    taxable = severance - service_deduction(years_employed, spec)
    if taxable <= 0:
        return 0  # 공제가 퇴직금보다 크면 세금이 없다 (장기근속 소액 퇴직금)

    converted = taxable / years_employed * 12
    base = converted - converted_deduction(converted, spec)
    if base <= 0:
        return 0

    income_tax = progressive_tax(base, spec) / 12 * years_employed
    return int(round(income_tax * (1 + spec.local_tax_rate)))


def pension_income_tax(
    severance: int,
    years_employed: int,
    pension_years: int,
    assumptions: Optional[Assumptions] = None,
) -> int:
    """연금으로 나눠 받을 때의 총 세액.

    수령 10년차까지는 30%, 11년차부터는 40% 감면된다. 오래 나눠 받을수록
    감면 폭이 커지는 구조다.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.retirement_income_tax

    full = retirement_income_tax(severance, years_employed, assumptions)
    if full <= 0 or pension_years <= 0:
        return 0

    early_years = min(pension_years, spec.pension_discount_after_years)
    late_years = max(0, pension_years - spec.pension_discount_after_years)

    per_year = full / pension_years
    total = per_year * early_years * (1 - spec.pension_discount)
    total += per_year * late_years * (1 - spec.pension_discount_after)
    return int(round(total))


# --------------------------------------------------------------------------- #
# 비교
# --------------------------------------------------------------------------- #


class SeveranceOption(BaseModel):
    """한 가지 수령 방식의 결과."""

    method: str
    total_tax: int
    total_tax_label: str
    effective_rate_label: str
    when_paid: str = Field(description="세금을 언제 내는가 — 이 차이가 결과를 가른다")

    depletion_age: Optional[int] = None
    depletion_label: str = ""
    final_balance: int = 0
    final_balance_label: str = ""


class SeveranceComparison(BaseModel):
    """일시금 vs 연금 비교 결과."""

    severance: int
    severance_label: str
    years_employed: int
    pension_years: int

    lump_sum: SeveranceOption
    pension: SeveranceOption

    computable: bool = Field(
        default=True, description="세금을 계산할 수 있는 상태인가 (근속연수 등)"
    )
    tax_saved: int = Field(description="연금으로 받을 때 줄어드는 세액")
    tax_saved_label: str = ""
    final_balance_gap: int = Field(description="시뮬레이션 종료 시점 잔액 차이")
    depletion_gap_years: int = 0

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


def _with_tax_events(
    profile: UserProfile, events: list[LifeEvent]
) -> UserProfile:
    return profile.model_copy(update={"life_events": [*profile.life_events, *events]})


def _option(
    method: str,
    profile: UserProfile,
    assumptions: Assumptions,
    total_tax: int,
    when_paid: str,
) -> SeveranceOption:
    sim = simulate(profile, assumptions=assumptions)
    severance = profile.severance_pay
    return SeveranceOption(
        method=method,
        total_tax=total_tax,
        total_tax_label=fmt_krw(total_tax),
        effective_rate_label=(
            fmt_pct(total_tax / severance, precision=2) if severance > 0 else "0%"
        ),
        when_paid=when_paid,
        depletion_age=sim.depletion_age,
        depletion_label=(
            f"만 {sim.horizon_age}세까지 유지"
            if sim.depletion_age is None
            else f"만 {sim.depletion_age}세 고갈"
        ),
        final_balance=sim.final_balance,
        final_balance_label=fmt_krw(sim.final_balance),
    )


def compare_severance_options(
    profile: UserProfile,
    *,
    pension_years: Optional[int] = None,
    assumptions: Optional[Assumptions] = None,
) -> SeveranceComparison:
    """퇴직금을 일시금으로 받을 때와 연금으로 받을 때를 각각 끝까지 시뮬레이션한다."""
    assumptions = assumptions or load_assumptions()
    spec = assumptions.retirement_income_tax
    pension_years = pension_years or spec.default_pension_years

    severance = profile.severance_pay
    years = profile.years_employed

    lump_tax = retirement_income_tax(severance, years, assumptions)
    split_tax = pension_income_tax(severance, years, pension_years, assumptions)

    # 세금을 일회성 지출로 표현한다. 일시금은 퇴직 연도에 한 번, 연금은 나눠서.
    # 이렇게 하면 엔진을 고치지 않고도 '언제 내는가'의 차이가 그대로 반영된다.
    lump_events = (
        [LifeEvent(year=profile.retirement_year, amount=lump_tax, label="퇴직소득세")]
        if lump_tax > 0
        else []
    )
    per_year = split_tax // pension_years if pension_years else 0
    pension_events = (
        [
            LifeEvent(
                year=profile.retirement_year + offset,
                amount=per_year,
                label="연금소득세",
            )
            for offset in range(pension_years)
        ]
        if per_year > 0
        else []
    )

    lump = _option(
        LUMP_SUM,
        _with_tax_events(profile, lump_events),
        assumptions,
        lump_tax,
        f"퇴직하시는 {profile.retirement_year}년에 한 번에 냅니다",
    )
    pension = _option(
        PENSION,
        _with_tax_events(profile, pension_events),
        assumptions,
        split_tax,
        f"{pension_years}년에 걸쳐 나눠 냅니다",
    )

    saved = lump_tax - split_tax
    gap = pension.final_balance - lump.final_balance
    depletion_gap = _age_gap(lump.depletion_age, pension.depletion_age, assumptions)

    computable = severance > 0 and years > 0

    if severance <= 0:
        headline = "퇴직금이 없으셔서 비교할 내용이 없습니다."
        verdict = "이 항목은 고객님께 해당하지 않습니다."
    elif years <= 0:
        # 근속연수를 모르면 세금을 계산할 수 없다. 기본값으로 채우면 실효세율이
        # 몇 배씩 어긋나므로(5년 근속과 40년 근속은 18배 차이) 추측하지 않는다.
        headline = (
            "몇 년 근무하셨는지 알려주시면 퇴직소득세까지 계산해 드립니다."
        )
        verdict = (
            "근속연수에 따라 세금이 크게 달라집니다. 같은 퇴직금이라도 5년 근무와 "
            "40년 근무는 세금이 10배 넘게 차이 납니다. 추측해서 알려드리지 않겠습니다."
        )
    elif saved <= 0 and gap <= 0:
        headline = "두 방식의 차이가 거의 없습니다."
        verdict = "어느 쪽을 고르셔도 결과가 크게 달라지지 않습니다."
    else:
        parts = [f"연금으로 받으시면 세금이 {fmt_krw(saved)} 줄어듭니다."]
        if depletion_gap > 0:
            parts.append(f"금융자산이 바닥나는 시점도 {depletion_gap}년 늦춰집니다.")
        elif gap > 0:
            parts.append(
                f"은퇴 마지막 시점에 남는 돈이 {fmt_krw(gap)} 많아집니다."
            )
        else:
            # 두 경우 모두 고갈되면 잔액 차이가 0으로 나온다. 세액 차이만 말한다.
            parts.append(
                "다만 이 정도로는 자산이 바닥나는 시점이 달라지지 않습니다."
            )
        headline = " ".join(parts)
        verdict = (
            "연금 수령이 유리합니다. 다만 연금은 중간에 목돈으로 꺼내 쓰기 어렵습니다. "
            "큰 지출이 예정되어 있다면 그 부분을 먼저 확인하세요."
        )

    notes = [
        f"근속 {years}년 기준 퇴직소득세 실효세율은 {lump.effective_rate_label} 입니다. "
        "근속연수공제가 커서 장기근속자는 생각보다 세금이 적습니다.",
        "IRP 계좌 안에서 발생하는 운용수익의 과세이연 효과는 반영하지 않았습니다. "
        "실제로는 연금 쪽이 여기서도 유리하므로 이 비교는 보수적입니다.",
        "세금은 참고용입니다. 실제 세액은 퇴직 사유와 중간정산 이력에 따라 달라지므로 "
        "회사나 세무 전문가에게 확인하세요.",
    ]

    return SeveranceComparison(
        severance=severance,
        severance_label=fmt_krw(severance),
        years_employed=years,
        pension_years=pension_years,
        lump_sum=lump,
        pension=pension,
        computable=computable,
        tax_saved=saved,
        tax_saved_label=fmt_krw(saved),
        final_balance_gap=gap,
        depletion_gap_years=depletion_gap,
        headline=headline,
        verdict=verdict,
        notes=notes,
    )


def _age_gap(
    lump_age: Optional[int], pension_age: Optional[int], assumptions: Assumptions
) -> int:
    horizon = assumptions.macro.horizon_age
    return (pension_age or horizon) - (lump_age or horizon)
