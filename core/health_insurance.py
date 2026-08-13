"""퇴직하면 건강보험료가 0원에서 연 수백만원이 된다 (피부양자 절벽).

직장에 다니는 동안 건강보험료는 급여에서 빠져나가고 회사가 절반을 낸다. 퇴직하면
그 자격이 사라진다. 직장 다니는 배우자·자녀의 **피부양자**로 들어가면 보험료가
0원이지만, 요건을 하나라도 넘기면 **지역가입자**가 되어 연 수백만원이 새로 생긴다.

이 서비스의 축에 정확히 맞는 항목이다. 이유는 두 가지다.

1. **절벽이다.** 금융소득이 연 1,000만원 **이하면 합산소득에서 아예 빠지고,
   1원이라도 넘으면 전액이 들어간다.** 이자를 1만원 더 받았다고 연 수백만원이
   생길 수 있다. "예금 금리 높은 상품으로 갈아타세요"라는 말이 이 구간의
   고객에게 실제로 얼마인지는 이렇게 계산해야만 나온다.

2. **가만히 있어도 다가온다.** 국민연금은 물가에 연동되어 매년 오르는데 피부양자
   소득기준 2,000만원은 그대로다. 지금 통과해도 몇 년 뒤 넘는다. 그래서 한 시점이
   아니라 **연도별로 판정하고, 넘는 해를 찾는다.**

완충장치가 하나 있다. **임의계속가입** — 퇴직 후 최대 36개월간 직장가입자 시절
보험료(본인부담분)를 그대로 낸다. 지역가입자 보험료보다 싸면 그쪽을 쓴다.

계산하지 않는 것 (화면에 함께 내보낸다):
  - **재산분 보험료를 넣지 않았다.** 재산 점수표를 확보하지 않았기 때문이다.
    근사식을 지어내지 않는다(주택연금과 같은 결정). 따라서 여기 나오는 지역가입자
    보험료는 **하한**이고, 실제 부담은 이보다 크다. 방향을 명시해서 내보낸다.
  - 피부양자 **부양요건**(직장가입자인 배우자·자녀가 있는지)은 모른다. 모르면
    조건부로 말한다.
  - 기준금액(2,000만원 등)이 앞으로 바뀌면 절벽 시점이 달라진다. 지금 기준이
    유지된다고 보고 계산한다.

구현 방식은 퇴직소득세와 같다. 보험료를 `LifeEvent`(일회성 지출)로 표현해
캐시플로우 엔진을 고치지 않고 반영한다. 다만 기본 엔진은 금융소득에 건보료
실효부담률 **근사**를 이미 얹고 있으므로, 이 모듈에서 시뮬레이션할 때는 그
근사를 빼고 돌린다 — 같은 부담을 두 번 세지 않기 위해서다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, HealthInsuranceAssumption, load_assumptions
from core.cashflow import simulate, split_return_rates
from core.formatting import fmt_krw, fmt_pct
from core.models import LifeEvent, UserProfile
from core.schedule import build_schedule

DEPENDENT = "피부양자"
LOCAL = "지역가입자"
VOLUNTARY = "임의계속가입"


# --------------------------------------------------------------------------- #
# 세금 계산기 — 건보료 근사를 뺀 것
# --------------------------------------------------------------------------- #


def financial_income_tax_only(
    taxable_return: float, assumptions: Assumptions
) -> float:
    """금융소득세만 매긴다. 건강보험료 실효부담률 근사는 빼고 계산한다.

    기본 엔진(`annual_tax`)은 금융소득에 건보료 근사를 얹는다. 이 모듈은 건보료를
    제도 구조대로 다시 계산해 지출로 넣으므로, 근사를 그대로 두면 **같은 부담을
    두 번** 세게 된다. `simulate(tax_model=...)` 자리가 열려 있는 이유다.
    """
    if taxable_return <= 0:
        return 0.0
    return taxable_return * assumptions.tax.financial_income_rate


# --------------------------------------------------------------------------- #
# 요건 판정
# --------------------------------------------------------------------------- #


def counted_financial_income(financial_income: int, spec: HealthInsuranceAssumption) -> int:
    """피부양자 판정·보험료 부과에 들어가는 금융소득.

    **한도 이하면 0, 넘으면 전액이다.** 이 계단이 이 모듈의 존재 이유이므로
    비례식으로 뭉개지 않는다.
    """
    limit = spec.dependent.financial_income_limit
    return financial_income if financial_income > limit else 0


def dependent_income(
    pension_income: int,
    financial_income: int,
    other_income: int,
    spec: HealthInsuranceAssumption,
) -> int:
    """피부양자 소득요건에서 세는 합산소득.

    공적연금소득은 **전액** 반영된다(보험료를 부과할 때 50%만 보는 것과 다르다).
    사적연금(IRP·연금저축)은 합산소득에 들어가지 않으므로 여기서도 세지 않는다.
    """
    return pension_income + other_income + counted_financial_income(financial_income, spec)


def property_tax_base(
    profile: UserProfile,
    spec: HealthInsuranceAssumption,
    override: Optional[int] = None,
) -> tuple[int, bool]:
    """재산세 과세표준. (금액, 추정값인가) 를 돌려준다.

    시가 → 공시가격 → 과세표준으로 두 단계 근사가 겹쳐 있다. 그래서 경계 근처에서는
    믿을 수 없고, 화면에서 그렇게 말한다. 고지서의 실제 과세표준을 받으면 그것을 쓴다.
    """
    if override is not None:
        return max(0, override), False
    return int(round(profile.real_estate * spec.property_tax_base_ratio)), True


def property_failure(
    base: int, counted_income: int, spec: HealthInsuranceAssumption
) -> Optional[str]:
    """재산요건으로 탈락하면 그 이유를, 통과하면 None 을 돌려준다."""
    rule = spec.dependent
    if base > rule.property_limit:
        return (
            f"재산세 과세표준이 {fmt_krw(base)}으로 한도 {fmt_krw(rule.property_limit)}을 "
            "넘습니다"
        )
    if base > rule.property_soft_limit and counted_income > rule.soft_limit_income_limit:
        return (
            f"재산세 과세표준이 {fmt_krw(rule.property_soft_limit)}을 넘어 소득 기준이 "
            f"{fmt_krw(rule.soft_limit_income_limit)}으로 강화되는데, 합산소득이 "
            f"{fmt_krw(counted_income)}입니다"
        )
    return None


# --------------------------------------------------------------------------- #
# 보험료
# --------------------------------------------------------------------------- #


def local_monthly_premium(
    pension_income: int,
    financial_income: int,
    other_income: int,
    spec: HealthInsuranceAssumption,
) -> int:
    """지역가입자 월 보험료 — **소득분만.**

    재산분은 넣지 않았다(모듈 docstring 참조). 따라서 이 값은 하한이다.
    """
    local = spec.local
    base = (
        pension_income * local.pension_income_share
        + other_income
        + counted_financial_income(financial_income, spec)
    )
    health = max(base * local.health_rate / 12.0, float(local.minimum_monthly))
    return int(round(health * (1 + local.long_term_care_rate)))


def voluntary_monthly_premium(
    monthly_salary: int, spec: HealthInsuranceAssumption
) -> int:
    """임의계속가입 월 보험료.

    직장가입자 보험료의 **본인부담분**(절반)만 낸다. 나머지 절반은 회사가 내던
    몫인데, 임의계속가입에서는 그 부분이 면제되는 것이 아니라 공단이 부담을
    조정한다 — 결과적으로 본인이 내는 금액은 재직 때와 같다.
    """
    local = spec.local
    health = monthly_salary * local.health_rate * spec.voluntary.employee_share
    return int(round(health * (1 + local.long_term_care_rate)))


# --------------------------------------------------------------------------- #
# 결과 모델
# --------------------------------------------------------------------------- #


class YearStatus(BaseModel):
    """한 연도의 자격과 보험료."""

    year: int
    age: int
    active_months: int = Field(
        default=12, description="그 해에 적용되는 개월 수 — 퇴직 연도는 12개월이 아니다"
    )
    pension_income: int
    financial_income: int
    other_income: int
    counted_income: int = Field(description="피부양자 판정에 들어가는 합산소득")
    qualifies: bool = Field(description="피부양자 자격을 유지하는가")
    reason: str = ""
    method: str
    monthly_premium: int
    annual_premium: int


class PremiumPath(BaseModel):
    """수단 하나의 보험료. 화면에서 나란히 비교한다."""

    method: str
    available: bool = True
    monthly: int = 0
    monthly_label: str = "0원"
    annual: int = 0
    annual_label: str = "0원"
    basis: str = Field(default="", description="이 숫자가 어디서 나왔는가")


class HealthInsuranceReport(BaseModel):
    """퇴직 후 건강보험료 절벽 분석 결과."""

    # --- 절벽 ---
    qualifies_at_retirement: bool
    cliff_year: Optional[int] = None
    cliff_age: Optional[int] = None
    cliff_reason: str = ""

    # --- 절벽까지 남은 거리 ---
    counted_income: int = Field(description="퇴직 직후 기준 합산소득")
    counted_income_label: str = ""
    income_headroom: int = Field(description="소득 한도까지 남은 금액")
    income_headroom_label: str = ""
    financial_income: int = 0
    financial_income_label: str = ""
    financial_headroom: int = Field(
        default=0, description="금융소득 계단까지 남은 금액 — 넘으면 전액이 합산된다"
    )
    financial_headroom_label: str = ""
    headroom_rate_label: str = Field(
        default="", description="금융자산 수익률로 환산한 여유 (예: 1.5%p)"
    )

    # --- 수단 비교 ---
    dependent: PremiumPath
    local: PremiumPath
    voluntary: PremiumPath

    # --- 재산 ---
    property_tax_base: int = 0
    property_tax_base_label: str = ""
    property_assumed: bool = True
    property_near_limit: bool = False

    # --- 계획에 미치는 영향 ---
    total_premiums: int = 0
    total_premiums_label: str = ""
    depletion_age_before: Optional[int] = None
    depletion_age_after: Optional[int] = None
    depletion_advanced_years: int = 0

    years: list[YearStatus] = Field(default_factory=list)

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 분석
# --------------------------------------------------------------------------- #


def _yearly_status(
    profile: UserProfile,
    assumptions: Assumptions,
    *,
    monthly_salary: int,
    property_base: int,
    has_employed_family: bool,
) -> list[YearStatus]:
    """연도별로 자격을 판정하고 보험료를 붙인다.

    금융소득은 기준 시뮬레이션의 연초 잔액에서 나온 **이자·배당 성격의 수익**이다.
    주식 양도차익은 건강보험료 부과 대상이 아니므로 캐시플로우 엔진이 이미 나눠 둔
    과세대상 수익률을 그대로 쓴다 — 두 곳에서 따로 정의하면 조용히 어긋난다.
    """
    spec = assumptions.health_insurance
    allocation = profile.current_allocation()
    taxable_rate, _ = split_return_rates(allocation, assumptions)

    base_sim = simulate(
        profile, assumptions=assumptions, tax_model=financial_income_tax_only
    )
    balances = {row.year: row.start_balance for row in base_sim.rows}
    plans = build_schedule(profile, assumptions)

    voluntary_available = (
        monthly_salary > 0
        and profile.years_employed >= spec.voluntary.min_service_years
    )
    voluntary_monthly = (
        voluntary_monthly_premium(monthly_salary, spec) if voluntary_available else 0
    )

    elapsed_months = 0
    rows: list[YearStatus] = []

    for plan in plans:
        months = plan.active_months
        financial = int(round(balances.get(plan.year, 0) * taxable_rate * plan.frac))
        pension = int(round(plan.pension_income))
        other = int(round(plan.other_income))

        counted = dependent_income(pension, financial, other, spec)
        rule = spec.dependent

        reason = ""
        qualifies = has_employed_family
        if not qualifies:
            reason = "피부양자로 올려 줄 직장가입자 가족이 없습니다"
        else:
            property_reason = property_failure(property_base, counted, spec)
            if property_reason is not None:
                qualifies, reason = False, property_reason
            elif counted > rule.income_limit:
                qualifies = False
                reason = (
                    f"합산소득이 {fmt_krw(counted)}으로 한도 {fmt_krw(rule.income_limit)}을 "
                    "넘습니다"
                )

        if qualifies:
            method, monthly = DEPENDENT, 0
            annual = 0
        else:
            local_monthly = local_monthly_premium(pension, financial, other, spec)
            # 임의계속가입은 **퇴직 직후**에만 신청할 수 있다. 지나간 개월 수로
            # 창을 닫는 이유다 — 나중에 자격을 잃었다고 다시 열리지 않는다.
            window_left = max(0, spec.voluntary.max_months - elapsed_months)
            use_voluntary = (
                voluntary_available
                and window_left > 0
                and voluntary_monthly < local_monthly
            )
            if use_voluntary:
                v_months = min(window_left, months)
                l_months = months - v_months
                method = VOLUNTARY if l_months == 0 else f"{VOLUNTARY}+{LOCAL}"
                annual = voluntary_monthly * v_months + local_monthly * l_months
                monthly = int(round(annual / months)) if months else 0
            else:
                method, monthly = LOCAL, local_monthly
                annual = local_monthly * months

        rows.append(
            YearStatus(
                year=plan.year,
                age=plan.age,
                active_months=months,
                pension_income=pension,
                financial_income=financial,
                other_income=other,
                counted_income=counted,
                qualifies=qualifies,
                reason=reason,
                method=method,
                monthly_premium=monthly,
                annual_premium=annual,
            )
        )
        elapsed_months += months

    return rows


def _first_full_year(rows: list[YearStatus]) -> Optional[YearStatus]:
    """기준으로 삼을 **온전한 1년치** 연도.

    퇴직 연도는 퇴직월부터라 몇 달뿐이다. 그 해의 소득으로 "한도까지 얼마 남았다"를
    말하면 실제보다 훨씬 여유 있어 보인다 — 12월 퇴직이면 12분의 1로 보인다.
    """
    if not rows:
        return None
    return next((row for row in rows if row.active_months == 12), rows[0])


def premium_events(
    rows: list[YearStatus], profile: UserProfile, assumptions: Assumptions
) -> list[LifeEvent]:
    """보험료를 일회성 지출로 바꾼다.

    `LifeEvent` 금액은 **퇴직 시점 기준**으로 해석되어 엔진이 물가만큼 키운다.
    보험료는 그 해의 명목 소득에서 이미 계산했으므로, 넣기 전에 물가로 되돌린다.
    이 한 줄을 빼먹으면 물가가 두 번 곱해진다.

    공개 함수인 이유: 국민연금 모듈(`core/national_pension.py`)이 연금을 늘렸을 때의
    건보료를 같은 규칙으로 얹어야 한다. 되돌림 규칙을 두 곳에서 따로 구현하면
    조용히 어긋난다.
    """
    factors = {
        plan.year: plan.inflation_factor
        for plan in build_schedule(profile, assumptions)
    }
    events: list[LifeEvent] = []
    for row in rows:
        if row.annual_premium <= 0:
            continue
        amount = int(round(row.annual_premium / factors.get(row.year, 1.0)))
        if amount > 0:
            events.append(
                LifeEvent(year=row.year, amount=amount, label="건강보험료")
            )
    return events


def analyze_health_insurance(
    profile: UserProfile,
    *,
    monthly_salary: Optional[int] = None,
    property_tax_base_override: Optional[int] = None,
    has_employed_family: Optional[bool] = None,
    assumptions: Optional[Assumptions] = None,
) -> HealthInsuranceReport:
    """퇴직 후 건강보험료가 언제 얼마나 생기는지 계산한다.

    Args:
        monthly_salary: 퇴직 전 월 급여(보수월액). 생략하면 프로파일 값을 쓰고,
            그것도 0이면 임의계속가입 보험료를 **계산하지 않고 물어본다.**
        property_tax_base_override: 재산세 고지서의 과세표준. 있으면 추정을 쓰지 않는다.
        has_employed_family: 직장 다니는 배우자·자녀가 있는지. 모르면(None) 있다고
            보고 계산하되, 결과에 조건을 붙여 내보낸다.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.health_insurance
    rule = spec.dependent

    salary = monthly_salary if monthly_salary is not None else profile.last_monthly_salary
    salary = max(0, salary)
    family_known = has_employed_family is not None
    family = True if has_employed_family is None else has_employed_family

    base, assumed = property_tax_base(profile, spec, property_tax_base_override)

    rows = _yearly_status(
        profile,
        assumptions,
        monthly_salary=salary,
        property_base=base,
        has_employed_family=family,
    )

    first = rows[0] if rows else None

    # "한도까지 얼마 남았나"는 **온전한 1년치**로 말해야 한다. 12월 퇴직이면 퇴직
    # 연도의 소득이 12분의 1이라, 그 해로 말하면 없는 여유가 있는 것처럼 보인다.
    reference = _first_full_year([row for row in rows if row.qualifies]) or first
    counted = reference.counted_income if reference else 0
    financial = reference.financial_income if reference else 0

    # 절벽 = 피부양자 자격을 처음 잃는 해.
    failing = next((row for row in rows if not row.qualifies), None)

    # 두 계단까지의 거리. 금융소득 계단이 더 위험하다 — 넘는 순간 전액이 합산된다.
    income_headroom = max(0, rule.income_limit - counted)
    financial_headroom = (
        max(0, rule.financial_income_limit - financial)
        if financial <= rule.financial_income_limit
        else 0
    )
    assets = profile.financial_assets
    headroom_rate_label = (
        fmt_pct(financial_headroom / assets, precision=2) + "p"
        if assets > 0 and financial_headroom > 0
        else ""
    )

    # 재산 경계 근처인가 — 추정값이라 이 구간에서는 믿을 수 없다고 말해야 한다.
    near_limit = assumed and any(
        0.8 * limit <= base <= 1.2 * limit
        for limit in (rule.property_limit, rule.property_soft_limit)
    )

    paths = _paths(rows, salary, spec, family)
    before, after, advanced, total = _plan_impact(profile, assumptions, rows)

    report = HealthInsuranceReport(
        qualifies_at_retirement=bool(first and first.qualifies),
        cliff_year=failing.year if failing else None,
        cliff_age=failing.age if failing else None,
        cliff_reason=failing.reason if failing else "",
        counted_income=counted,
        counted_income_label=fmt_krw(counted),
        income_headroom=income_headroom,
        income_headroom_label=fmt_krw(income_headroom),
        financial_income=financial,
        financial_income_label=fmt_krw(financial),
        financial_headroom=financial_headroom,
        financial_headroom_label=fmt_krw(financial_headroom),
        headroom_rate_label=headroom_rate_label,
        dependent=paths[DEPENDENT],
        local=paths[LOCAL],
        voluntary=paths[VOLUNTARY],
        property_tax_base=base,
        property_tax_base_label=fmt_krw(base),
        property_assumed=assumed,
        property_near_limit=near_limit,
        total_premiums=total,
        total_premiums_label=fmt_krw(total),
        depletion_age_before=before,
        depletion_age_after=after,
        depletion_advanced_years=advanced,
        years=rows,
    )
    report.headline, report.verdict = _verdict(report, rows, family, family_known, salary)
    report.notes = _notes(report, spec, assumed, family_known, salary)
    return report


def _paths(
    rows: list[YearStatus],
    salary: int,
    spec: HealthInsuranceAssumption,
    family: bool,
) -> dict[str, PremiumPath]:
    """세 가지 수단을 나란히 놓는다 — 화면이 비교표를 그리기 좋게."""
    # 지역가입자 보험료는 **자격을 잃은 첫 해** 기준으로 말한다. 지금 0원인 사람에게
    # "그때 가면 얼마"를 알려주는 것이 이 기능의 목적이기 때문이다.
    failing = _first_full_year([row for row in rows if not row.qualifies])
    local_monthly = 0
    basis = "피부양자 자격이 유지되어 계산할 필요가 없습니다"
    if failing is not None:
        pension, financial, other = (
            failing.pension_income,
            failing.financial_income,
            failing.other_income,
        )
        local_monthly = local_monthly_premium(pension, financial, other, spec)
        basis = (
            f"{failing.year}년 기준 소득분만 계산했습니다 "
            f"(연금 {fmt_krw(pension)}의 "
            f"{fmt_pct(spec.local.pension_income_share, precision=0)} + "
            f"금융소득 {fmt_krw(counted_financial_income(financial, spec))} + "
            f"기타 {fmt_krw(other)}). **재산분은 빠져 있습니다.**"
        )

    voluntary_available = salary > 0
    voluntary_monthly = (
        voluntary_monthly_premium(salary, spec) if voluntary_available else 0
    )

    return {
        DEPENDENT: PremiumPath(
            method=DEPENDENT,
            available=family,
            basis=(
                "직장 다니는 배우자·자녀의 피부양자로 들어가면 보험료가 없습니다"
                if family
                else "피부양자로 올려 줄 직장가입자 가족이 없으면 쓸 수 없습니다"
            ),
        ),
        LOCAL: PremiumPath(
            method=LOCAL,
            monthly=local_monthly,
            monthly_label=fmt_krw(local_monthly),
            annual=local_monthly * 12,
            annual_label=fmt_krw(local_monthly * 12),
            basis=basis,
        ),
        VOLUNTARY: PremiumPath(
            method=VOLUNTARY,
            available=voluntary_available,
            monthly=voluntary_monthly,
            monthly_label=fmt_krw(voluntary_monthly),
            annual=voluntary_monthly * 12,
            annual_label=fmt_krw(voluntary_monthly * 12),
            basis=(
                f"퇴직 전 월 급여 {fmt_krw(salary)}의 "
                f"{fmt_pct(spec.local.health_rate, precision=2)} 중 본인부담 "
                f"{fmt_pct(spec.voluntary.employee_share, precision=0)}, "
                f"최대 {spec.voluntary.max_months}개월"
                if voluntary_available
                else "퇴직 전 월 급여를 알려주시면 계산해 드립니다"
            ),
        ),
    }


def _plan_impact(
    profile: UserProfile, assumptions: Assumptions, rows: list[YearStatus]
) -> tuple[Optional[int], Optional[int], int, int]:
    """보험료를 넣기 전과 후의 고갈 시점. 두 시뮬레이션 모두 건보료 근사를 뺀 상태다."""
    events = premium_events(rows, profile, assumptions)
    before = simulate(
        profile, assumptions=assumptions, tax_model=financial_income_tax_only
    )
    if not events:
        return before.depletion_age, before.depletion_age, 0, 0

    with_premiums = profile.model_copy(
        update={"life_events": [*profile.life_events, *events]}
    )
    after = simulate(
        with_premiums, assumptions=assumptions, tax_model=financial_income_tax_only
    )

    horizon = assumptions.macro.horizon_age
    advanced = (before.depletion_age or horizon) - (after.depletion_age or horizon)
    return (
        before.depletion_age,
        after.depletion_age,
        max(0, advanced),
        sum(event.amount for event in events),
    )


def _verdict(
    report: HealthInsuranceReport,
    rows: list[YearStatus],
    family: bool,
    family_known: bool,
    salary: int,
) -> tuple[str, str]:
    """한 문장과 결론. 대부분의 경우 '아직 아닙니다'라고 말하게 된다."""
    condition = "" if family_known else "직장 다니는 배우자나 자녀가 계시다면, "

    if not family:
        return (
            f"퇴직하시는 즉시 지역가입자가 되어 매달 {report.local.monthly_label}, "
            f"연 {report.local.annual_label}이 생깁니다.",
            "임의계속가입을 신청하면 최대 36개월간 재직 때 내던 금액으로 유지할 수 있습니다. "
            "신청 기한이 첫 지역보험료 고지서의 납부기한에서 2개월까지라 놓치기 쉽습니다.",
        )

    if report.cliff_year is None:
        headline = (
            f"{condition}건강보험료는 계속 0원입니다. 이 항목은 고객님께 큰 문제가 "
            "아닙니다."
        )
        if report.financial_headroom > 0:
            headline += (
                f" 다만 이자·배당이 연 {report.financial_headroom_label}만 늘면"
                f"(퇴직 시점 금융자산 기준 수익률 {report.headroom_rate_label}) "
                f"그 순간 전액이 합산소득에 들어가 연 {report.local.annual_label}이 "
                "생깁니다."
            )
        return headline, (
            "지금 하실 일은 없습니다. 예금 금리를 갈아타거나 배당을 늘리실 때만 "
            "이 선을 다시 확인하세요."
        )

    if not report.qualifies_at_retirement:
        return (
            f"{condition}퇴직하시는 {report.cliff_year}년부터 건강보험료가 매달 "
            f"{report.local.monthly_label}, 연 {report.local.annual_label} 생깁니다. "
            f"{report.cliff_reason}.",
            (
                f"임의계속가입을 쓰시면 최대 36개월간 매달 {report.voluntary.monthly_label}로 "
                "낮출 수 있습니다. 신청 기한이 첫 고지서 납부기한에서 2개월까지입니다."
                if report.voluntary.available
                and report.voluntary.monthly < report.local.monthly
                else "퇴직 전 월 급여를 알려주시면 임의계속가입으로 얼마나 낮출 수 있는지 "
                "계산해 드립니다."
            ),
        )

    # 가장 흔하고 가장 안 보이는 경우 — 지금은 0원인데 몇 년 뒤 절벽이 온다.
    return (
        f"{condition}지금은 0원이지만 {report.cliff_year}년(만 {report.cliff_age}세)부터 "
        f"매달 {report.local.monthly_label}, 연 {report.local.annual_label}이 새로 "
        f"생깁니다. {report.cliff_reason}.",
        "국민연금이 시작되면 합산소득이 한 번에 올라갑니다. 연금 수령 시기를 정하실 때 "
        "이 시점을 같이 보세요. 지금 하실 일은 없습니다.",
    )


def _notes(
    report: HealthInsuranceReport,
    spec: HealthInsuranceAssumption,
    assumed: bool,
    family_known: bool,
    salary: int,
) -> list[str]:
    notes = [
        "지역가입자 보험료는 **소득분만** 계산했습니다. 재산분 점수표를 확보하지 못해 "
        "근사식을 지어내지 않았습니다. 실제 부담은 여기 나온 금액보다 큽니다.",
        f"피부양자 소득기준 {fmt_krw(spec.dependent.income_limit)}은 지금 기준이 그대로 "
        "유지된다고 보고 계산했습니다. 기준이 바뀌면 시점이 달라집니다.",
        "국민연금은 물가에 연동되어 매년 오르는데 기준금액은 그대로입니다. 지금 통과해도 "
        "언젠가 넘게 되는 구조입니다.",
    ]
    if assumed:
        notes.append(
            f"재산세 과세표준을 부동산 시가의 {fmt_pct(spec.property_tax_base_ratio)}로 "
            f"추정했습니다({report.property_tax_base_label}). 고지서의 실제 과세표준을 "
            "넣으면 정확해집니다."
        )
    if report.property_near_limit:
        notes.append(
            "추정한 과세표준이 재산 기준선 근처입니다. 이 구간에서는 추정값을 믿을 수 "
            "없으니 재산세 고지서를 확인해 주세요."
        )
    if not family_known:
        notes.append(
            "피부양자는 직장에 다니는 배우자·자녀가 있어야 올릴 수 있습니다. "
            "안 계시면 퇴직 즉시 지역가입자가 됩니다."
        )
    if salary <= 0:
        notes.append(
            "퇴직 전 월 급여를 알려주시면 임의계속가입 보험료까지 계산해 드립니다. "
            "급여에 따라 금액이 달라져 추측하지 않았습니다."
        )
    return notes
