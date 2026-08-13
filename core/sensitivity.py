"""민감도 분석 — "그 가정이 틀리면 어떻게 되나".

이 서비스가 내놓는 모든 숫자는 가정 위에 서 있다. 물가 2.3%, 주식 기대수익률
7.0%, 사용자가 기억으로 답한 생활비. 심사·발표에서 "그 가정이 틀리면요?"는
반드시 나오는 질문이고, 답을 준비하는 것보다 **화면에 미리 띄워두는 편**이 낫다.

이 모듈은 이 서비스의 축을 우리 자신에게 적용한 것이다. 남의 말을 "내 자산 기준
얼마"로 환산해 왔으니, 우리 출력도 같은 방식으로 환산한다:

    "물가가 3.0%로 오르면 고갈이 만 68세로 1년 앞당겨집니다."

가정을 하나씩 위아래로 흔들어 고갈 나이가 얼마나 움직이는지 재고, 흔들림이 큰
순서로 정렬한다(토네이도 차트의 자료 형태). 무엇이 결과를 지배하는지 보이면
사용자도 어디를 관리해야 하는지 알게 된다.

한계: 한 번에 하나씩만 흔든다(일변량). 가정들이 동시에 나빠지는 경우는 반영하지
않으므로, 실제 최악의 경우는 여기 나온 것보다 나쁠 수 있다. 여러 가정이 동시에
움직이는 상황은 몬테카를로(core/montecarlo.py)가 다룬다.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_pct
from core.models import UserProfile


class SensitivityCase(BaseModel):
    """가정 하나를 위아래로 흔든 결과."""

    key: str
    label: str
    baseline_label: str
    low_label: str
    high_label: str

    low_depletion_age: Optional[int] = None
    high_depletion_age: Optional[int] = None
    swing_years: int = Field(description="위아래 결과의 차이(년). 클수록 결과를 지배한다")

    detail: str = Field(description="화면에 그대로 쓰는 한 문장")


class SensitivityReport(BaseModel):
    """민감도 분석 결과. 흔들림이 큰 순서로 정렬되어 있다."""

    baseline_depletion_age: Optional[int] = None
    horizon_age: int
    cases: list[SensitivityCase] = Field(default_factory=list)
    summary: str = ""
    caveat: str = (
        "한 번에 한 가지 가정만 바꿔 본 결과입니다. 여러 가정이 동시에 나빠지면 "
        "실제로는 이보다 나쁠 수 있습니다."
    )

    @property
    def most_sensitive(self) -> Optional[SensitivityCase]:
        return self.cases[0] if self.cases else None


def _effective_age(depletion_age: Optional[int], horizon_age: int) -> int:
    """고갈되지 않는 경우를 비교 가능한 숫자로 바꾼다.

    None 을 그대로 두면 흔들림 폭을 잴 수 없다. 시뮬레이션 종료 나이로 본다 —
    그 이후는 어차피 계산하지 않으므로 흔들림을 과소평가하는 쪽이고, 보수적이다.
    """
    return horizon_age if depletion_age is None else depletion_age


def _age_label(depletion_age: Optional[int], horizon_age: int) -> str:
    if depletion_age is None:
        return f"만 {horizon_age}세까지 유지"
    return f"만 {depletion_age}세 고갈"


def _shift_assumptions(
    assumptions: Assumptions, mutate: Callable[[Assumptions], None]
) -> Assumptions:
    """가정값을 복사해 흔든다. 원본은 lru_cache 로 공유되므로 절대 건드리지 않는다."""
    shifted = assumptions.model_copy(deep=True)
    mutate(shifted)
    return shifted


def _case(
    key: str,
    label: str,
    baseline_label: str,
    low: tuple[str, Optional[int]],
    high: tuple[str, Optional[int]],
    horizon_age: int,
    detail: str,
) -> SensitivityCase:
    low_label, low_age = low
    high_label, high_age = high
    swing = abs(
        _effective_age(high_age, horizon_age) - _effective_age(low_age, horizon_age)
    )
    return SensitivityCase(
        key=key,
        label=label,
        baseline_label=baseline_label,
        low_label=low_label,
        high_label=high_label,
        low_depletion_age=low_age,
        high_depletion_age=high_age,
        swing_years=swing,
        detail=detail,
    )


def analyze_sensitivity(
    profile: UserProfile, assumptions: Optional[Assumptions] = None
) -> SensitivityReport:
    """가정을 하나씩 흔들어 고갈 시점이 얼마나 움직이는지 계산한다."""
    assumptions = assumptions or load_assumptions()
    spec = assumptions.sensitivity
    horizon = assumptions.macro.horizon_age

    baseline = simulate(profile, assumptions=assumptions).depletion_age

    def run(*, prof: Optional[UserProfile] = None, assm: Optional[Assumptions] = None):
        return simulate(prof or profile, assumptions=assm or assumptions).depletion_age

    cases: list[SensitivityCase] = []

    # --- 물가상승률 ---------------------------------------------------------
    inflation = assumptions.macro.inflation_rate
    delta = spec.inflation_delta

    def set_inflation(value: float):
        return lambda a: setattr(a.macro, "inflation_rate", value)

    low_age = run(assm=_shift_assumptions(assumptions, set_inflation(inflation - delta)))
    high_age = run(assm=_shift_assumptions(assumptions, set_inflation(inflation + delta)))
    cases.append(
        _case(
            "inflation",
            "물가상승률",
            fmt_pct(inflation),
            (fmt_pct(inflation - delta), low_age),
            (fmt_pct(inflation + delta), high_age),
            horizon,
            f"물가가 {fmt_pct(inflation + delta)}로 오르면 "
            f"{_age_label(high_age, horizon)}, "
            f"{fmt_pct(inflation - delta)}로 내리면 {_age_label(low_age, horizon)}입니다.",
        )
    )

    # --- 투자 기대수익률 ----------------------------------------------------
    return_delta = spec.return_delta

    def shift_returns(amount: float):
        def mutate(a: Assumptions) -> None:
            for spec_ in a.asset_classes.values():
                spec_.expected_return = max(0.0, spec_.expected_return + amount)

        return mutate

    low_age = run(assm=_shift_assumptions(assumptions, shift_returns(-return_delta)))
    high_age = run(assm=_shift_assumptions(assumptions, shift_returns(+return_delta)))
    cases.append(
        _case(
            "returns",
            "투자 기대수익률",
            "가정값 그대로",
            (f"-{fmt_pct(return_delta)}p", low_age),
            (f"+{fmt_pct(return_delta)}p", high_age),
            horizon,
            f"수익률이 가정보다 {fmt_pct(return_delta)}p 낮으면 "
            f"{_age_label(low_age, horizon)}, 높으면 {_age_label(high_age, horizon)}입니다.",
        )
    )

    # --- 월 생활비 ----------------------------------------------------------
    expense = profile.monthly_expense
    ratio = spec.expense_ratio
    low_value = int(expense * (1 - ratio))
    high_value = int(expense * (1 + ratio))

    low_age = run(prof=profile.model_copy(update={"monthly_expense": low_value}))
    high_age = run(prof=profile.model_copy(update={"monthly_expense": high_value}))
    cases.append(
        _case(
            "expense",
            "월 생활비",
            fmt_krw(expense),
            (fmt_krw(low_value), low_age),
            (fmt_krw(high_value), high_age),
            horizon,
            f"생활비가 {fmt_pct(ratio)} 더 든다면 {_age_label(high_age, horizon)}, "
            f"{fmt_pct(ratio)} 아낀다면 {_age_label(low_age, horizon)}입니다.",
        )
    )

    # --- 국민연금 수령액 ----------------------------------------------------
    if profile.national_pension_monthly > 0:
        pension = profile.national_pension_monthly
        ratio = spec.pension_ratio
        low_value = int(pension * (1 - ratio))
        high_value = int(pension * (1 + ratio))

        low_age = run(
            prof=profile.model_copy(update={"national_pension_monthly": low_value})
        )
        high_age = run(
            prof=profile.model_copy(update={"national_pension_monthly": high_value})
        )
        cases.append(
            _case(
                "pension",
                "국민연금 예상 수령액",
                fmt_krw(pension),
                (fmt_krw(low_value), low_age),
                (fmt_krw(high_value), high_age),
                horizon,
                f"실제 수령액이 기억하신 것보다 {fmt_pct(ratio)} 적다면 "
                f"{_age_label(low_age, horizon)}입니다. 공단에서 확인해 보실 값입니다.",
            )
        )

    # --- 금융자산 평가액 ----------------------------------------------------
    ratio = spec.assets_ratio
    scale_low = 1 - ratio
    scale_high = 1 + ratio

    def scaled(scale: float) -> UserProfile:
        return profile.model_copy(
            update={
                "severance_pay": int(profile.severance_pay * scale),
                "cash_savings": int(profile.cash_savings * scale),
                "equity": int(profile.equity * scale),
                "isa": int(profile.isa * scale),
                "pension_dc": int(profile.pension_dc * scale),
            }
        )

    low_age = run(prof=scaled(scale_low))
    high_age = run(prof=scaled(scale_high))
    cases.append(
        _case(
            "assets",
            "금융자산 평가액",
            fmt_krw(profile.financial_assets),
            (fmt_krw(int(profile.financial_assets * scale_low)), low_age),
            (fmt_krw(int(profile.financial_assets * scale_high)), high_age),
            horizon,
            f"금융자산이 {fmt_pct(ratio)} 적다면 {_age_label(low_age, horizon)}, "
            f"{fmt_pct(ratio)} 많다면 {_age_label(high_age, horizon)}입니다.",
        )
    )

    # 흔들림이 큰 순서로. 무엇이 결과를 지배하는지가 이 분석의 결론이다.
    cases.sort(key=lambda c: c.swing_years, reverse=True)

    top = cases[0] if cases else None
    if top is None or top.swing_years == 0:
        summary = (
            "가정을 흔들어도 결과가 크게 달라지지 않습니다. "
            "지금 계획은 가정에 민감하지 않은 편입니다."
        )
    else:
        summary = (
            f"결과를 가장 크게 좌우하는 것은 **{top.label}**입니다. "
            f"이 가정 하나만으로 고갈 시점이 {top.swing_years}년 움직입니다."
        )

    return SensitivityReport(
        baseline_depletion_age=baseline,
        horizon_age=horizon,
        cases=cases,
        summary=summary,
    )
