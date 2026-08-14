"""퇴직하고 일하면 손에 얼마가 남는가.

이 모듈은 **우리 자신의 권고를 검증하려고** 만들었다. 처방 엔진이 "월 162만원의
수입을 더 만드시면 만 95세까지 유지됩니다"라고 말하는데, 그 162만원이 통째로
손에 들어온다는 전제가 깔려 있었다. 실제로는 그렇지 않다.

퇴직 후 소득이 생기면 세 가지가 동시에 움직인다.

  1. **노령연금이 깎인다** — 60~65세 구간에 한해. 소득월액이 A값을 200만원 넘게
     초과하면 구간별로 감액되고, 최대 연금의 절반까지 깎인다 (국민연금법 제63조의2).
  2. **건강보험 피부양자에서 탈락한다** — 합산소득이 늘어 기준을 넘으면 그 순간
     지역가입자가 되어 보험료가 생긴다.
  3. 소득세·지방소득세를 낸다 — **이 모듈은 계산하지 않는다.** 근로소득·사업소득·
     기타소득이 각각 다른 체계이고, 근사식을 지어내지 않는다는 기존 결정과 같다.
     따라서 여기 나오는 "손에 남는 금액"은 **세전이고, 실제는 이보다 적다.**

그런데 계산해 보면 이 서비스의 축이 그대로 나온다. **깎이기 시작하는 지점이
생각보다 높다.** A값 319만원 + 200만원 = 월 519만원까지는 한 푼도 안 깎인다.
"일하면 연금 깎여요"라는 말이 대부분의 사람에게는 해당하지 않는다는 뜻이다.
"그건 당신에게 중요하지 않습니다"를 또 하나 말할 수 있는 자리다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_pct
from core.health_insurance import (
    analyze_health_insurance,
    financial_income_tax_only,
    premium_events,
)
from core.models import UserProfile


class IncomeOutcome(BaseModel):
    """월 소득 하나에 대한 결과."""

    monthly_income: int
    monthly_income_label: str = ""

    pension_cut: int = Field(default=0, description="깎이는 노령연금 (월)")
    pension_cut_label: str = "0원"
    pension_after: int = Field(default=0, description="깎인 뒤 받는 연금 (월)")
    pension_after_label: str = "0원"

    health_premium: int = Field(default=0, description="새로 생기는 건강보험료 (월)")
    health_premium_label: str = "0원"

    net: int = Field(default=0, description="손에 남는 것 (월, 세전)")
    net_label: str = "0원"
    keep_rate_label: str = Field(default="", description="번 돈 중 남는 비율")

    depletion_age: Optional[int] = None
    depletion_label: str = ""


class WorkingIncomeReport(BaseModel):
    """재취업 소득을 얼마나 벌면 실제로 얼마가 남는가."""

    computable: bool = True

    a_value: int = 0
    a_value_label: str = ""
    free_ceiling: int = Field(
        default=0, description="여기까지는 연금이 한 푼도 안 깎인다"
    )
    free_ceiling_label: str = ""
    reduction_applies: bool = Field(
        default=False, description="이 분의 나이에 감액 규정이 적용되는가"
    )
    reduction_window: str = ""

    outcomes: list[IncomeOutcome] = Field(default_factory=list)

    # 처방 엔진의 권고와, 건보료까지 넣었을 때 실제로 필요한 금액.
    # 우리 자신의 출력을 같은 잣대로 환산하는 자리다.
    prescribed_income: int = 0
    prescribed_income_label: str = ""
    actual_needed_income: int = 0
    actual_needed_income_label: str = ""
    gap: int = Field(default=0, description="처방이 과소평가한 금액 (월)")
    gap_label: str = ""

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


# 화면에서 비교해 보여줄 월 소득 후보. 이 연령대의 재취업 소득 분포를 감안한 값이다.
DEFAULT_LEVELS: tuple[int, ...] = (
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_000_000,
)


def _outcome(
    profile: UserProfile,
    assumptions: Assumptions,
    monthly_income: int,
    *,
    baseline_premium: int,
) -> IncomeOutcome:
    """소득 하나를 넣고 연금 감액·건보료·고갈 시점을 한 번에 본다."""
    spec = assumptions.national_pension.earned_income
    pension = profile.national_pension_monthly

    applies = spec.applies_at(profile.retirement_age)
    cut = spec.reduction(monthly_income, pension) if applies else 0

    # 소득이 생긴 상태를 그대로 시뮬레이션한다. 건보료는 제도 구조대로 다시 계산해
    # 지출로 얹는다 — 엔진의 실효부담률 근사를 두 번 세지 않기 위해서다.
    variant = profile.model_copy(
        update={
            "other_monthly_income": profile.other_monthly_income + monthly_income,
            "national_pension_monthly": max(0, pension - cut),
        }
    )
    health = analyze_health_insurance(variant, assumptions=assumptions)
    with_premiums = variant.model_copy(
        update={
            "life_events": [
                *variant.life_events,
                *premium_events(health.years, variant, assumptions),
            ]
        }
    )
    sim = simulate(
        with_premiums, assumptions=assumptions, tax_model=financial_income_tax_only
    )

    premium = max(0, health.local.monthly - baseline_premium)
    net = monthly_income - cut - premium

    return IncomeOutcome(
        monthly_income=monthly_income,
        monthly_income_label=fmt_krw(monthly_income),
        pension_cut=cut,
        pension_cut_label=fmt_krw(cut),
        pension_after=max(0, pension - cut),
        pension_after_label=fmt_krw(max(0, pension - cut)),
        health_premium=premium,
        health_premium_label=fmt_krw(premium),
        net=net,
        net_label=fmt_krw(net),
        keep_rate_label=fmt_pct(net / monthly_income) if monthly_income else "",
        depletion_age=sim.depletion_age,
        depletion_label=(
            f"만 {sim.horizon_age}세까지 유지"
            if sim.depletion_age is None
            else f"만 {sim.depletion_age}세 고갈"
        ),
    )


def analyze_working_income(
    profile: UserProfile,
    *,
    levels: Optional[tuple[int, ...]] = None,
    assumptions: Optional[Assumptions] = None,
) -> WorkingIncomeReport:
    """월 소득 구간별로 실제 순증을 계산한다.

    Args:
        levels: 비교할 월 소득들. 생략하면 기본 다섯 구간.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.national_pension.earned_income

    if profile.national_pension_monthly <= 0:
        return _no_pension(profile, assumptions, levels or DEFAULT_LEVELS)

    # 지금(추가 소득 없음) 상태의 건보료. 여기서 늘어난 만큼만 '새로 생긴' 것이다.
    baseline = analyze_health_insurance(profile, assumptions=assumptions)
    baseline_premium = baseline.local.monthly if not baseline.qualifies_at_retirement else 0

    outcomes = [
        _outcome(profile, assumptions, level, baseline_premium=baseline_premium)
        for level in (levels or DEFAULT_LEVELS)
    ]

    applies = spec.applies_at(profile.retirement_age)
    report = WorkingIncomeReport(
        computable=True,
        a_value=spec.a_value,
        a_value_label=fmt_krw(spec.a_value),
        free_ceiling=spec.free_income_ceiling(),
        free_ceiling_label=fmt_krw(spec.free_income_ceiling()),
        reduction_applies=applies,
        reduction_window=f"만 {spec.start_age}세부터 {spec.end_age}세까지",
        outcomes=outcomes,
    )
    _compare_with_prescription(report, profile, assumptions, baseline_premium)
    report.headline, report.verdict = _verdict(report, profile, spec)
    report.notes = _notes(report, spec)
    return report


def _compare_with_prescription(
    report: WorkingIncomeReport,
    profile: UserProfile,
    assumptions: Assumptions,
    baseline_premium: int,
) -> None:
    """처방 엔진의 '추가 수입' 권고를 건보료까지 넣어 다시 재 본다.

    **우리 자신의 출력에 같은 잣대를 대는 자리다.** 남이 하는 말을 정직하게
    환산하기로 해놓고 우리 권고만 낙관적으로 두면 앞뒤가 맞지 않는다.
    """
    from core.prescribe import prescribe

    try:
        plan = prescribe(profile, assumptions=assumptions)
    except Exception:  # pragma: no cover - 처방 실패로 이 화면을 죽이지 않는다
        return

    option = next(
        (o for o in plan.options if o.required_value and "수입" in o.label), None
    )
    if option is None or plan.already_safe:
        return

    needed = _needed_income(
        profile,
        assumptions,
        target_age=plan.target_age,
        start=option.required_value,
        baseline_premium=baseline_premium,
    )
    if needed <= 0:
        return

    report.prescribed_income = option.required_value
    report.prescribed_income_label = fmt_krw(option.required_value)
    report.actual_needed_income = needed
    report.actual_needed_income_label = fmt_krw(needed)
    report.gap = max(0, needed - option.required_value)
    report.gap_label = fmt_krw(report.gap)


def _needed_income(
    profile: UserProfile,
    assumptions: Assumptions,
    *,
    target_age: int,
    start: int,
    baseline_premium: int,
) -> int:
    """목표 나이까지 유지하려면 실제로 월 얼마를 벌어야 하는가.

    처방 엔진은 캐시플로우 엔진의 **건보료 실효부담률 근사**로 역산한다. 그런데
    이 서비스는 퇴직 후 건보료를 제도 구조대로 다시 계산하는 모듈을 따로 갖고
    있고(`core/health_insurance.py`), 그쪽이 더 정확하다. 두 값이 벌어지면
    **화면은 낙관적인 쪽을 말하게 된다.**

    비싼 계산(소득 한 점마다 건보료 재계산 + 시뮬레이션)이라 이분 탐색을 짧게
    돌린다. 못 찾으면 0 을 돌려주고 화면은 그 사실을 말하지 않는다 — 틀린 숫자를
    내놓느니 말하지 않는 편이 낫다.
    """
    if start <= 0:
        return 0

    def survives(income: int) -> bool:
        outcome = _outcome(
            profile, assumptions, income, baseline_premium=baseline_premium
        )
        return outcome.depletion_age is None or outcome.depletion_age >= target_age

    low, high = start, start * 3
    if survives(low):
        return low
    if not survives(high):
        return 0

    for _ in range(6):  # 10만원 단위까지 좁히면 충분하다
        mid = (low + high) // 2
        if survives(mid):
            high = mid
        else:
            low = mid
    return int(round(high / 100_000) * 100_000)


def _no_pension(
    profile: UserProfile, assumptions: Assumptions, levels: tuple[int, ...]
) -> WorkingIncomeReport:
    """연금을 안 받는 사람은 감액이 없다. 건보료만 본다."""
    spec = assumptions.national_pension.earned_income
    baseline = analyze_health_insurance(profile, assumptions=assumptions)
    baseline_premium = baseline.local.monthly if not baseline.qualifies_at_retirement else 0
    outcomes = [
        _outcome(profile, assumptions, level, baseline_premium=baseline_premium)
        for level in levels
    ]
    report = WorkingIncomeReport(
        computable=True,
        a_value=spec.a_value,
        a_value_label=fmt_krw(spec.a_value),
        free_ceiling=spec.free_income_ceiling(),
        free_ceiling_label=fmt_krw(spec.free_income_ceiling()),
        reduction_applies=False,
        reduction_window=f"만 {spec.start_age}세부터 {spec.end_age}세까지",
        outcomes=outcomes,
        headline=(
            "아직 국민연금을 받지 않으셔서 연금이 깎일 일은 없습니다. "
            "다만 소득이 생기면 건강보험료는 달라집니다."
        ),
        verdict="아래 표에서 소득 구간별로 건강보험료가 얼마나 붙는지 보실 수 있습니다.",
    )
    report.notes = _notes(report, spec)
    return report


def _verdict(report: WorkingIncomeReport, profile: UserProfile, spec) -> tuple[str, str]:
    """한 문장과 결론.

    대부분의 경우 "생각하시는 것만큼 깎이지 않습니다"가 답이 된다. 그것이
    사실이기 때문이다.
    """
    if not report.reduction_applies:
        return (
            f"고객님은 퇴직 시점이 만 {profile.retirement_age}세라 "
            f"연금 감액 구간({report.reduction_window})에 해당하지 않습니다. "
            "일하셔도 연금은 그대로입니다.",
            "다만 소득이 늘면 건강보험 피부양자에서 탈락할 수 있습니다. "
            "아래 표에서 소득별로 확인하세요.",
        )

    cut_free = [o for o in report.outcomes if o.pension_cut == 0]
    ceiling = report.free_ceiling_label

    if len(cut_free) == len(report.outcomes):
        return (
            f"**월 {ceiling}까지는 연금이 한 푼도 깎이지 않습니다.** "
            "'일하면 연금 깎인다'는 말은 고객님께 해당하지 않습니다.",
            "지금 하실 일은 없습니다. 소득이 이 선을 넘을 때만 다시 확인하세요. "
            "건강보험료는 그보다 먼저 생기니 아래 표를 함께 보세요.",
        )

    first_cut = next(o for o in report.outcomes if o.pension_cut > 0)
    tail = (
        "깎이는 금액은 아무리 많이 버셔도 **연금의 절반을 넘지 않습니다.** "
        "그리고 이 감액은 만 65세가 되면 끝납니다 — 평생 깎이는 것이 아닙니다."
    )
    if report.gap > 0:
        tail += (
            f" 참고로 '그래서 무엇을 하면 되나' 화면의 "
            f"{report.prescribed_income_label}은 건강보험료를 근사로 잡은 값입니다. "
            f"제도대로 계산하면 **{report.actual_needed_income_label}**이 필요합니다."
        )
    return (
        f"**월 {ceiling}까지는 연금이 깎이지 않습니다.** 그보다 많이 버시면 "
        f"깎이기 시작하는데, 월 {first_cut.monthly_income_label}을 버실 때 "
        f"연금이 {first_cut.pension_cut_label} 줄어 손에 남는 것은 "
        f"{first_cut.net_label}입니다.",
        tail,
    )


def _notes(report: WorkingIncomeReport, spec) -> list[str]:
    return [
        "**소득세·지방소득세는 계산하지 않았습니다.** 근로·사업·기타소득이 각각 다른 "
        "체계라 근사식을 지어내지 않았습니다. 실제로 손에 남는 금액은 여기 나온 "
        "것보다 적습니다.",
        f"연금 감액은 **만 {spec.start_age}세부터 {spec.end_age}세까지**만 적용됩니다. "
        f"{spec.end_age}세가 되면 소득과 무관하게 전액을 받습니다.",
        f"기준이 되는 A값({report.a_value_label})은 전체 가입자의 평균소득월액으로, "
        "**매년 1월 고시됩니다.** 바뀌면 깎이지 않는 상한도 함께 움직입니다.",
        "깎이는 금액은 노령연금액의 2분의 1을 넘지 않습니다 (국민연금법 제63조의2).",
        "건강보험료는 소득분만 계산했습니다. 재산분이 빠져 있어 실제 부담은 더 큽니다.",
    ]
