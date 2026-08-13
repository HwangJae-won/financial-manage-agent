"""국민연금을 더 낼까 — 임의계속가입과 추후납부(추납).

연기·조기 수령이 **"언제 받을까"**의 레버라면 이쪽은 **"얼마나 쌓았나"**의 레버다.
둘은 성격이 다르다. 가입기간이 모자란 사람에게는 앞의 레버가 아무 의미가 없다.
받을 연금 자체가 없기 때문이다.

여기에도 계단이 있다. **가입 119개월과 120개월은 연속적이지 않다.** 119개월이면
노령연금이 평생 0원이고 낸 돈에 이자를 붙인 반환일시금만 받는다. 120개월을 채우면
죽을 때까지 매달 나온다. 건강보험 피부양자 1,000만원 계단과 같은 구조다.

수단은 두 가지다.

  - **추납(추후납부)** — 실직·휴직으로 납부예외였거나 전업주부로 적용제외였던
    기간의 보험료를 나중에 내고 그 기간을 가입기간으로 인정받는다. 최대 119개월.
    목돈이 한 번에 나가지만 가입기간이 즉시 늘어난다.
  - **임의계속가입** — 60세가 되어 의무가입이 끝난 뒤에도 65세까지 계속 낸다.
    다만 **노령연금을 받기 시작하면 자격이 사라진다.** 수급 개시가 64세인 사람은
    65세가 아니라 64세까지만 쓸 수 있다.

그런데 이 기능의 값어치는 "연금이 늘어납니다"를 계산하는 데 있지 않다.

**연금을 늘리면 건강보험 피부양자에서 더 빨리 탈락한다.** 공적연금소득은 피부양자
판정에서 전액이 합산되기 때문이다. 연금만 보면 이득인데 건보료까지 보면 손해인
구간이 실제로 존재한다. 한쪽만 계산해서 "연금 늘리세요"라고 말하는 것이 정확히
이 서비스가 하지 않기로 한 일이다. 그래서 수단마다 **늘어난 연금 · 낸 돈 ·
추가 건보료**를 같은 기준(퇴직 시점 화폐)으로 나란히 놓고 순효과를 낸다.

연금액을 어떻게 구하는가:
    기본연금액 산식(국민연금법 별표 1)의 가입기간 계수만 쓴다.

        계수 = 1 + 0.05 × (가입월수 - 240) ÷ 12

    A값(전체 가입자 평균소득)과 B값(본인 평균소득)은 **지어내지 않는다.** 대신
    사용자가 알려준 공단 예상연금액을 기준점으로 삼고 계수의 **비율만** 적용한다.
    그래서 가입월수를 모르면 계산하지 않고 물어본다 — 근속연수와 같은 규약이다.

계산하지 않는 것 (화면에 함께 내보낸다):
  - 추납·임의계속으로 낸 기간이 **B값(본인 평균소득)을 끌어올리는 효과**는 빼고
    계산했다. 실제로는 연금이 이보다 조금 더 는다. 이 비교는 **보수적**이다.
  - 반환일시금의 이자는 계산하지 않는다. 가입기간이 모자란 경우 "연금이 없다"는
    사실만 말한다.
  - 추납 보험료는 신청 당시 기준소득월액으로 매겨진다. 분할납부(최대 60회)를
    선택하면 부담 시점이 달라지지만 여기서는 일시납으로 본다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, NationalPensionAssumption, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_eul, fmt_euro, fmt_krw, fmt_months
from core.health_insurance import (
    analyze_health_insurance,
    financial_income_tax_only,
    premium_events,
)
from core.models import LifeEvent, UserProfile

NONE = "지금 그대로"
CATCHUP = "추납"
VOLUNTARY = "임의계속가입"
BOTH = "둘 다"


# --------------------------------------------------------------------------- #
# 제도 계산
# --------------------------------------------------------------------------- #


def monthly_pension_for(
    total_months: int,
    *,
    anchor_monthly: int,
    anchor_months: int,
    spec: NationalPensionAssumption,
) -> int:
    """가입월수가 `total_months` 일 때의 월 연금액.

    공단 예상연금액(`anchor_monthly`, 가입월수 `anchor_months` 기준)을 기준점으로 두고
    가입기간 계수의 **비율만** 적용한다. 최소가입기간에 못 미치면 0이다 — 절벽이다.
    """
    base = spec.period_factor(anchor_months)
    if base <= 0 or anchor_monthly <= 0:
        return 0
    return int(round(anchor_monthly * spec.period_factor(total_months) / base))


def contribution_cost(
    income_base: int, months: int, spec: NationalPensionAssumption
) -> int:
    """임의계속가입·추납 보험료 총액 — **전액 본인 부담**이다.

    직장에 다닐 때는 회사가 절반을 냈다. 그 절반이 없어졌다는 사실이 이 금액의
    체감을 만든다.
    """
    if months <= 0 or income_base <= 0:
        return 0
    return int(round(spec.clamp_income_base(income_base) * spec.contribution_rate * months))


def voluntary_window_months(
    profile: UserProfile, spec: NationalPensionAssumption
) -> int:
    """임의계속가입으로 더 낼 수 있는 개월 수.

    시작은 의무가입이 끝나는 60세(이미 지났으면 퇴직 시점), 끝은 65세와 **노령연금
    수급 개시 연령 중 빠른 쪽**이다. 연금을 받기 시작하면 자격이 사라지기 때문이다.
    조기수령을 선택해 60세부터 받는 사람은 이 창이 아예 열리지 않는다.
    """
    start = max(spec.contribution_age_limit, profile.retirement_age)
    end = min(spec.voluntary_max_age, profile.national_pension_start_age)
    return max(0, (end - start) * 12)


def catchup_window_months(
    profile: UserProfile, spec: NationalPensionAssumption
) -> int:
    """추납할 수 있는 개월 수. 상한 119개월에서 자른다."""
    return max(0, min(profile.pension_catchup_months, spec.max_catchup_months))


def _receiving_months(profile: UserProfile, assumptions: Assumptions) -> int:
    """연금을 받는 총 개월 수 (수급 개시 ~ 시뮬레이션 종료)."""
    start_year = profile.pension_start_year
    end_year = profile.birth_year + assumptions.macro.horizon_age
    if end_year < start_year:
        return 0
    return (end_year - start_year) * 12 + (13 - profile.birth_month)


# --------------------------------------------------------------------------- #
# 결과 모델
# --------------------------------------------------------------------------- #


class PensionOption(BaseModel):
    """수단 하나의 결과. 화면에서 나란히 비교한다."""

    key: str
    label: str
    available: bool = True
    unavailable_reason: str = ""

    months_added: int = 0
    total_months: int = 0
    total_months_label: str = ""

    cost: int = 0
    cost_label: str = "0원"
    cost_basis: str = ""

    monthly_pension: int = 0
    monthly_pension_label: str = "0원"
    monthly_gain: int = 0
    monthly_gain_label: str = "0원"
    qualifies: bool = True

    lifetime_gain: int = Field(
        default=0, description="평생 더 받는 연금 (퇴직 시점 화폐 기준)"
    )
    lifetime_gain_label: str = "0원"
    extra_premium: int = Field(default=0, description="연금이 늘어 새로 생기는 건강보험료")
    extra_premium_label: str = "0원"
    net_gain: int = Field(default=0, description="평생 연금 증가 - 낸 돈 - 추가 건보료")
    net_gain_label: str = "0원"

    breakeven_years: Optional[float] = None
    breakeven_age: Optional[int] = None
    breakeven_label: str = ""

    cliff_year: Optional[int] = None
    cliff_shift_years: int = Field(
        default=0, description="건강보험 피부양자 탈락이 몇 년 앞당겨지는가"
    )

    depletion_age: Optional[int] = None
    depletion_label: str = ""


class NationalPensionReport(BaseModel):
    """국민연금 임의계속가입·추납 분석 결과."""

    computable: bool = True

    contributed_months: int = 0
    contributed_label: str = ""
    qualifies_now: bool = True
    months_to_qualify: int = 0
    # 가입기간이 모자란 사람에게 이 두 값이 전부다. "얼마를 내면 평생 연금이
    # 생기는가" — 수단을 고르기 전에 먼저 알아야 하는 숫자다.
    cost_to_qualify: int = Field(
        default=0, description="최소가입기간만 채우는 데 드는 보험료"
    )
    cost_to_qualify_label: str = "0원"
    monthly_at_minimum: int = Field(
        default=0, description="최소가입기간을 딱 채웠을 때의 월 연금액"
    )
    monthly_at_minimum_label: str = "0원"

    income_base: int = 0
    income_base_label: str = ""
    income_base_capped: bool = False

    current: PensionOption
    catchup: PensionOption
    voluntary: PensionOption
    both: PensionOption

    best_key: str = ""
    best_label: str = ""

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def options(self) -> list[PensionOption]:
        return [self.current, self.catchup, self.voluntary, self.both]


# --------------------------------------------------------------------------- #
# 분석
# --------------------------------------------------------------------------- #


def _cost_events(
    profile: UserProfile,
    spec: NationalPensionAssumption,
    *,
    catchup_months: int,
    voluntary_months: int,
    income_base: int,
) -> list[LifeEvent]:
    """보험료를 일회성 지출로 바꾼다 — 퇴직소득세·건보료와 같은 방식이다.

    추납은 신청한 해에 한 번(일시납), 임의계속가입은 내는 기간에 걸쳐 매년 나눈다.
    `LifeEvent` 금액은 퇴직 시점 기준으로 해석되므로 그 기준 그대로 넣는다.
    """
    events: list[LifeEvent] = []

    catchup_cost = contribution_cost(income_base, catchup_months, spec)
    if catchup_cost > 0:
        events.append(
            LifeEvent(
                year=profile.retirement_year, amount=catchup_cost, label="국민연금 추납"
            )
        )

    if voluntary_months > 0:
        start_age = max(spec.contribution_age_limit, profile.retirement_age)
        yearly = contribution_cost(income_base, 12, spec)
        full_years, rest = divmod(voluntary_months, 12)
        for offset in range(full_years):
            events.append(
                LifeEvent(
                    year=profile.birth_year + start_age + offset,
                    amount=yearly,
                    label="국민연금 임의계속가입",
                )
            )
        rest_cost = contribution_cost(income_base, rest, spec)
        if rest_cost > 0:
            events.append(
                LifeEvent(
                    year=profile.birth_year + start_age + full_years,
                    amount=rest_cost,
                    label="국민연금 임의계속가입",
                )
            )

    return events


def _simulate_option(
    profile: UserProfile, assumptions: Assumptions
) -> tuple[Optional[int], int, Optional[int], int]:
    """이 프로파일을 건보료까지 얹어 끝까지 돌린다.

    돌려주는 값: (고갈 나이, 평생 건보료, 피부양자 탈락 연도, 시뮬레이션 종료 나이)

    건보료는 기본 엔진의 실효부담률 **근사**가 아니라 제도 구조 그대로 계산한 값을
    지출로 얹는다(`core/health_insurance.py` 와 같은 방식). 근사를 그대로 두면 같은
    부담을 두 번 세게 되므로 `tax_model` 을 바꿔 끼운다.
    """
    report = analyze_health_insurance(profile, assumptions=assumptions)
    events = premium_events(report.years, profile, assumptions)
    with_premiums = profile.model_copy(
        update={"life_events": [*profile.life_events, *events]}
    )
    sim = simulate(
        with_premiums, assumptions=assumptions, tax_model=financial_income_tax_only
    )
    return sim.depletion_age, report.total_premiums, report.cliff_year, sim.horizon_age


def _disabled(option: PensionOption, reason: str) -> PensionOption:
    """쓸 수 없는 수단에서 숫자를 지운다. 남는 것은 이름과 **왜 못 쓰는지**뿐이다."""
    return PensionOption(
        key=option.key, label=option.label, available=False, unavailable_reason=reason
    )


def _build_option(
    key: str,
    label: str,
    profile: UserProfile,
    assumptions: Assumptions,
    *,
    catchup_months: int,
    voluntary_months: int,
    income_base: int,
    contributed_months: int,
    anchor_months: int,
    baseline: Optional[tuple[int, int, Optional[int]]] = None,
) -> tuple[PensionOption, tuple[int, int, Optional[int]]]:
    """수단 하나를 끝까지 계산한다.

    `baseline` 은 (월 연금액, 평생 건보료, 탈락 연도) — '지금 그대로'의 값이다.
    비교 대상이므로 첫 호출(현재)에서는 None 이고, 그 결과가 이후 호출의 기준이 된다.
    """
    spec = assumptions.national_pension
    months_added = catchup_months + voluntary_months
    total_months = contributed_months + months_added

    monthly = monthly_pension_for(
        total_months,
        anchor_monthly=profile.national_pension_monthly,
        anchor_months=anchor_months,
        spec=spec,
    )
    cost = contribution_cost(income_base, months_added, spec)

    variant = profile.model_copy(
        update={
            "national_pension_monthly": monthly,
            "life_events": [
                *profile.life_events,
                *_cost_events(
                    profile,
                    spec,
                    catchup_months=catchup_months,
                    voluntary_months=voluntary_months,
                    income_base=income_base,
                ),
            ],
        }
    )
    depletion, premiums, cliff_year, horizon = _simulate_option(variant, assumptions)

    base_monthly, base_premiums, base_cliff = baseline or (monthly, premiums, cliff_year)
    gain = monthly - base_monthly
    lifetime = gain * _receiving_months(profile, assumptions)
    # 음수일 수 있다. 목돈을 내면 금융자산이 줄어 금융소득도 줄고, 그만큼 건보료가
    # 오히려 낮아지는 구간이 있다. 같은 결정에서 나온 효과이므로 깎지 않고 그대로 센다.
    extra_premium = premiums - base_premiums
    net = lifetime - cost - extra_premium

    breakeven_years: Optional[float] = None
    breakeven_age: Optional[int] = None
    breakeven_label = ""
    if gain > 0 and cost > 0:
        breakeven_years = cost / (gain * 12)
        breakeven_age = int(profile.national_pension_start_age + breakeven_years)
        breakeven_label = f"만 {breakeven_age}세 (연금 받기 시작하고 {breakeven_years:.1f}년)"

    cliff_shift = 0
    if cliff_year is not None and base_cliff is not None:
        cliff_shift = max(0, base_cliff - cliff_year)
    elif cliff_year is not None and base_cliff is None:
        # 원래는 오지 않던 절벽이 생겼다. 몇 년 앞당겨졌는지가 아니라 '생겼다'가 사실이다.
        cliff_shift = 0

    option = PensionOption(
        key=key,
        label=label,
        months_added=months_added,
        total_months=total_months,
        total_months_label=fmt_months(total_months),
        cost=cost,
        cost_label=fmt_krw(cost),
        cost_basis=(
            f"기준소득월액 {fmt_krw(spec.clamp_income_base(income_base))} × "
            f"{spec.contribution_rate * 100:.0f}% × {months_added}개월 (전액 본인 부담)"
            if months_added > 0
            else "더 내지 않습니다"
        ),
        monthly_pension=monthly,
        monthly_pension_label=fmt_krw(monthly),
        monthly_gain=gain,
        monthly_gain_label=fmt_krw(gain),
        qualifies=total_months >= spec.min_months,
        lifetime_gain=lifetime,
        lifetime_gain_label=fmt_krw(lifetime),
        extra_premium=extra_premium,
        extra_premium_label=fmt_krw(extra_premium),
        net_gain=net,
        net_gain_label=fmt_krw(net),
        breakeven_years=breakeven_years,
        breakeven_age=breakeven_age,
        breakeven_label=breakeven_label,
        cliff_year=cliff_year,
        cliff_shift_years=cliff_shift,
        depletion_age=depletion,
        depletion_label=(
            f"만 {horizon}세까지 유지" if depletion is None else f"만 {depletion}세 고갈"
        ),
    )
    return option, (monthly, premiums, cliff_year)


def analyze_national_pension(
    profile: UserProfile,
    *,
    contributed_months: Optional[int] = None,
    catchup_months: Optional[int] = None,
    monthly_income_base: Optional[int] = None,
    assumptions: Optional[Assumptions] = None,
) -> NationalPensionReport:
    """국민연금을 더 냈을 때 실제로 얼마가 남는지 계산한다.

    Args:
        contributed_months: 국민연금 가입월수. 생략하면 프로파일 값을 쓰고, 그것도
            0이면 **계산하지 않고 물어본다.** 가입월수를 모르면 연금 증가분을
            낼 수 없다 — 근속연수와 같은 규약이다.
        catchup_months: 추납할 수 있는 개월 수. 생략하면 프로파일 값을 쓴다.
        monthly_income_base: 기준소득월액. 생략하면 퇴직 전 월 급여를 쓴다.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.national_pension

    contributed = (
        contributed_months
        if contributed_months is not None
        else profile.national_pension_months
    )
    contributed = max(0, contributed)

    raw_base = (
        monthly_income_base
        if monthly_income_base is not None
        else profile.last_monthly_salary
    )
    raw_base = max(0, raw_base)
    income_base = spec.clamp_income_base(raw_base) if raw_base > 0 else 0

    catchup_available = (
        catchup_months if catchup_months is not None else profile.pension_catchup_months
    )
    catchup_window = max(0, min(catchup_available, spec.max_catchup_months))
    voluntary_window = voluntary_window_months(profile, spec)

    # 산식은 최소가입기간 위에서만 의미가 있다. 그 아래에서는 연금이 아예 없으므로
    # 사용자가 알려준 예상액을 '자격을 채웠을 때' 값으로 보고 기준점을 잡는다.
    # 지어낸 값이 아니라 **해석**이므로 화면에 적어 내보낸다.
    anchor_months = max(contributed, spec.min_months)

    # 셋 중 하나라도 모르면 계산하지 않는다. 특히 기준소득월액을 0으로 두면 보험료가
    # 0원으로 나와 **공짜로 연금이 늘어나는 것처럼** 보인다.
    missing = ""
    if contributed <= 0:
        missing = "국민연금 가입월수"
    elif profile.national_pension_monthly <= 0:
        missing = "국민연금 예상 월 수령액"
    elif income_base <= 0:
        missing = "퇴직 전 월 급여(기준소득월액)"
    if missing:
        return _not_computable(contributed, income_base, spec, missing)

    current, base = _build_option(
        "none",
        NONE,
        profile,
        assumptions,
        catchup_months=0,
        voluntary_months=0,
        income_base=income_base,
        contributed_months=contributed,
        anchor_months=anchor_months,
    )

    def build(key: str, label: str, catch: int, vol: int) -> PensionOption:
        option, _ = _build_option(
            key,
            label,
            profile,
            assumptions,
            catchup_months=catch,
            voluntary_months=vol,
            income_base=income_base,
            contributed_months=contributed,
            anchor_months=anchor_months,
            baseline=base,
        )
        return option

    catchup = build("catchup", CATCHUP, catchup_window, 0)
    voluntary = build("voluntary", VOLUNTARY, 0, voluntary_window)
    both = build("both", BOTH, catchup_window, voluntary_window)

    # 쓸 수 없는 수단은 **숫자를 지우고 이유만 남긴다.** 숫자를 남겨 두면 화면도
    # 에이전트도 그것을 '고를 수 있는 선택지'로 인용한다.
    if catchup_window <= 0:
        catchup = _disabled(
            catchup, "납부예외·적용제외였던 기간이 없거나 아직 알려주지 않으셨습니다"
        )
    if voluntary_window <= 0:
        voluntary = _disabled(
            voluntary,
            f"만 {profile.national_pension_start_age}세부터 연금을 받으시면 그 시점에 "
            "자격이 사라져 더 낼 기간이 없습니다",
        )
    if not (catchup.available and voluntary.available):
        both = _disabled(both, "두 수단을 함께 쓸 수 있을 때만 계산합니다")

    usable = [o for o in (catchup, voluntary, both) if o.available and o.months_added > 0]
    best = max(usable, key=lambda o: o.net_gain, default=None)

    need = max(0, spec.min_months - contributed)
    at_minimum = monthly_pension_for(
        spec.min_months,
        anchor_monthly=profile.national_pension_monthly,
        anchor_months=anchor_months,
        spec=spec,
    )

    report = NationalPensionReport(
        computable=True,
        contributed_months=contributed,
        contributed_label=fmt_months(contributed),
        qualifies_now=contributed >= spec.min_months,
        months_to_qualify=need,
        cost_to_qualify=contribution_cost(income_base, need, spec),
        cost_to_qualify_label=fmt_krw(contribution_cost(income_base, need, spec)),
        monthly_at_minimum=at_minimum,
        monthly_at_minimum_label=fmt_krw(at_minimum),
        income_base=income_base,
        income_base_label=fmt_krw(income_base),
        income_base_capped=raw_base > spec.income_base_max,
        current=current,
        catchup=catchup,
        voluntary=voluntary,
        both=both,
        best_key=best.key if best and best.net_gain > 0 else "none",
        best_label=best.label if best and best.net_gain > 0 else NONE,
    )
    report.headline, report.verdict = _verdict(report, best, spec)
    report.notes = _notes(report, spec, raw_base)
    return report


def _not_computable(
    contributed: int,
    income_base: int,
    spec: NationalPensionAssumption,
    missing: str,
) -> NationalPensionReport:
    """모르는 값이 있으면 계산하지 않는다.

    기본값으로 채우면 사용자가 그 숫자를 믿는다. 가입월수는 특히 위험하다 —
    119개월과 120개월은 '연금 0원'과 '평생 연금'을 가른다.
    """
    blank = PensionOption(key="none", label=NONE, available=False)
    report = NationalPensionReport(
        computable=False,
        contributed_months=contributed,
        contributed_label=fmt_months(contributed),
        qualifies_now=contributed >= spec.min_months,
        months_to_qualify=max(0, spec.min_months - contributed),
        income_base=income_base,
        income_base_label=fmt_krw(income_base),
        current=blank.model_copy(update={"key": "none", "label": NONE}),
        catchup=blank.model_copy(update={"key": "catchup", "label": CATCHUP}),
        voluntary=blank.model_copy(update={"key": "voluntary", "label": VOLUNTARY}),
        both=blank.model_copy(update={"key": "both", "label": BOTH}),
        headline=f"{fmt_eul(missing)} 알려주시면 계산해 드립니다.",
        verdict=(
            "국민연금공단 앱이나 1355 로 '가입내역 조회'를 하시면 가입월수와 예상 "
            "수령액이 함께 나옵니다. 가입 119개월과 120개월은 연금이 평생 0원이냐 "
            "아니냐를 가르기 때문에 추측해서 알려드리지 않겠습니다."
        ),
    )
    report.notes = [
        f"노령연금을 받으려면 가입기간이 최소 {spec.min_months // 12}년"
        f"({spec.min_months}개월) 이어야 합니다. 모자라면 연금 대신 반환일시금만 "
        "받습니다.",
        "추납은 가입자 자격이 있을 때만 신청할 수 있습니다. 퇴직해서 자격이 없으면 "
        "임의가입·임의계속가입으로 자격을 먼저 만들어야 합니다.",
    ]
    return report


def _verdict(
    report: NationalPensionReport,
    best: Optional[PensionOption],
    spec: NationalPensionAssumption,
) -> tuple[str, str]:
    """한 문장과 결론.

    이 서비스의 축대로, 이득이 없으면 없다고 말한다. 그리고 이득이 있을 때도
    **건보료를 빼고 남는 것**으로 말한다.
    """
    if not report.qualifies_now:
        need = report.months_to_qualify
        filler = report.both if report.both.available else (
            report.catchup if report.catchup.available else report.voluntary
        )
        reachable = filler.available and filler.total_months >= spec.min_months
        if reachable:
            # 자격을 채우는 값(딱 120개월)과 수단을 끝까지 쓴 값을 **분리해서** 말한다.
            # 한 문장에 섞으면 "10개월만 더 내면 227만원"처럼 읽힌다.
            return (
                f"지금 가입기간이 {report.contributed_label}이라 **노령연금이 평생 "
                f"0원**입니다. {need}개월치 {report.cost_to_qualify_label}만 내시면 "
                f"매달 {report.monthly_at_minimum_label}이 죽을 때까지 나옵니다.",
                f"이 항목만큼은 고민하실 필요가 없습니다. 더 채우실 수도 있습니다 — "
                f"{fmt_euro(filler.label)} {filler.months_added}개월"
                f"({filler.cost_label})까지 채우시면 매달 "
                f"{filler.monthly_pension_label}이 되고, "
                f"{filler.breakeven_label or '회수 시점은 수령액에 따라 달라집니다'}에 "
                "본전입니다.",
            )
        return (
            f"지금 가입기간이 {report.contributed_label}이라 노령연금이 평생 0원입니다. "
            f"{need}개월이 모자랍니다.",
            "임의계속가입이나 추납으로 채울 수 있는지 공단(1355)에 확인해 보세요. "
            "가입기간을 채우지 못하면 낸 보험료에 이자를 붙인 반환일시금만 받습니다.",
        )

    if best is None or best.net_gain <= 0:
        # 가장 흔한 경우. "그건 당신에게 중요하지 않습니다"를 숫자로 말한다.
        candidate = max(
            (o for o in (report.catchup, report.voluntary, report.both) if o.available),
            key=lambda o: o.months_added,
            default=None,
        )
        if candidate is None or candidate.months_added <= 0:
            return (
                "더 내실 수 있는 기간이 없습니다. 이 항목은 고객님께 해당하지 않습니다.",
                "지금 하실 일은 없습니다.",
            )
        # 왜 손해인지가 수단마다 다르다. 건보료가 잡아먹는 경우와, 애초에 늘어나는
        # 연금이 낸 돈에 못 미치는 경우는 사용자가 할 일도 다르다.
        if candidate.extra_premium > 0:
            cause = f"건강보험료가 평생 {candidate.extra_premium_label} 새로 생겨"
            advice = (
                "연금만 보면 이득처럼 보이는 구간입니다. 건강보험 피부양자 소득기준까지 "
                "함께 보면 남는 것이 없습니다. 지금 하실 일은 없습니다."
            )
        else:
            cause = f"평생 더 받는 연금이 {candidate.lifetime_gain_label}뿐이라"
            advice = (
                "내신 돈을 만 95세까지도 다 회수하지 못합니다. 더 오래 사실 것 같다면 "
                "달라지지만, 지금 계산으로는 하실 이유가 없습니다."
            )
        return (
            f"{fmt_euro(candidate.label)} {candidate.cost_label}을 더 내시면 연금이 매달 "
            f"{candidate.monthly_gain_label} 늘지만, {cause} 결국 "
            f"{fmt_krw(abs(candidate.net_gain))} **손해**입니다.",
            advice,
        )

    parts = [
        f"{fmt_euro(best.label)} {best.cost_label}을 더 내시면 연금이 매달 "
        f"{best.monthly_gain_label} 늘어, 평생 {best.lifetime_gain_label}을 더 받습니다."
    ]
    if best.extra_premium > 0:
        parts.append(
            f"다만 늘어난 연금 때문에 건강보험료가 {best.extra_premium_label} 생겨 "
            f"실제로 남는 것은 {best.net_gain_label}입니다."
        )
    if best.cliff_shift_years > 0:
        parts.append(
            f"건강보험 피부양자 탈락도 {best.cliff_shift_years}년 앞당겨집니다."
        )

    return " ".join(parts), (
        f"{best.breakeven_label}에 본전입니다. 그 전에 돌아가시면 낸 돈이 손해고, "
        "오래 사실수록 이득이 커집니다. 국민연금은 물가에 연동되므로 오래 사는 위험에 "
        "대한 보험으로 보시는 편이 맞습니다."
    )


def _notes(
    report: NationalPensionReport, spec: NationalPensionAssumption, raw_base: int
) -> list[str]:
    notes = [
        f"노령연금 수급 최소 가입기간은 {spec.min_months}개월입니다. "
        f"{spec.min_months - 1}개월이면 연금이 평생 0원이고 반환일시금만 받습니다. "
        "1개월 차이로 결과가 갈립니다.",
        "추납·임의계속으로 낸 기간이 본인 평균소득(B값)을 끌어올리는 효과는 빼고 "
        "계산했습니다. 실제 연금은 여기 나온 금액보다 조금 더 늘어납니다.",
        "공적연금소득은 건강보험 피부양자 판정에서 **전액** 반영됩니다. 연금을 늘리면 "
        "그만큼 탈락이 빨라진다는 뜻이라, 두 계산을 함께 냈습니다.",
        "추납은 가입자 자격이 있을 때만 신청할 수 있습니다. 퇴직해서 자격이 없으면 "
        "임의가입·임의계속가입으로 자격을 먼저 만들어야 합니다.",
        f"추납 보험료는 최대 {spec.max_installments}회로 나눠 낼 수 있습니다. "
        "여기서는 일시납으로 계산했습니다.",
    ]
    if report.voluntary.available:
        notes.append(
            f"임의계속가입은 만 {spec.voluntary_max_age}세까지지만 **노령연금을 받기 "
            "시작하면 자격이 사라집니다.** 수급 개시 연령이 더 이르면 그때까지만 "
            "낼 수 있습니다."
        )
    if report.income_base_capped:
        notes.append(
            f"기준소득월액에는 상한({fmt_krw(spec.income_base_max)})이 있어 급여 "
            f"{fmt_krw(raw_base)} 전부에 보험료가 매겨지지는 않습니다. "
            "상한은 매년 7월에 조정됩니다."
        )
    if raw_base <= 0:
        notes.append(
            "퇴직 전 월 급여를 알려주시면 보험료를 정확히 계산해 드립니다. "
            "기준소득월액에 따라 금액이 달라져 추측하지 않았습니다."
        )
    return notes
