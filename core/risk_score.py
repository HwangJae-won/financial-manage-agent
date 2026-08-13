"""은퇴 재무 안정도 점수 (기능 ⑤).

    🟢 유동성        양호
    🟠 소득공백기    주의
    🔴 장기소득      부족
    종합 72 / 100

가중치는 기획서를 그대로 따른다. 각 항목은 "무엇이 몇이면 몇 점인지"를
_score_* 함수 하나에 가두어, 발표에서 근거를 설명할 수 있게 했다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.asset_map import build_asset_map
from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.models import RiskTolerance, UserProfile

GREEN, AMBER, RED = "양호", "주의", "부족"

# 기획서의 Retirement Risk Score 가중치
WEIGHTS: dict[str, float] = {
    "income_gap": 0.30,
    "liquidity": 0.25,
    "concentration": 0.15,
    "market_risk": 0.10,
    "debt": 0.10,
    "pension_adequacy": 0.10,
}

# 투자성향별로 감당할 수 있다고 보는 주식 비중 상한
_EQUITY_LIMIT = {
    RiskTolerance.CONSERVATIVE: 0.20,
    RiskTolerance.MODERATE: 0.40,
    RiskTolerance.AGGRESSIVE: 0.60,
}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _status(score: float) -> str:
    if score >= 80:
        return GREEN
    if score >= 60:
        return AMBER
    return RED


class ScoreComponent(BaseModel):
    """점수 한 항목."""

    key: str
    label: str
    weight: float
    score: float = Field(ge=0, le=100)
    detail: str = Field(description="이 점수가 나온 이유 — 화면과 LLM 설명에 그대로 쓴다")

    @property
    def status(self) -> str:
        return _status(self.score)

    @property
    def icon(self) -> str:
        return {GREEN: "🟢", AMBER: "🟠", RED: "🔴"}[self.status]

    @property
    def weighted(self) -> float:
        return self.score * self.weight


class RiskScore(BaseModel):
    """은퇴 재무 안정도 종합."""

    components: list[ScoreComponent]
    cap: Optional[int] = Field(
        default=None, description="계획 지속가능성에 따른 종합 점수 상한"
    )
    cap_reason: Optional[str] = None

    @property
    def raw_total(self) -> int:
        """가중합 원점수 (상한 적용 전)."""
        return int(round(sum(c.weighted for c in self.components)))

    @property
    def total(self) -> int:
        if self.cap is None:
            return self.raw_total
        return min(self.raw_total, self.cap)

    @property
    def status(self) -> str:
        return _status(self.total)

    @property
    def weakest(self) -> ScoreComponent:
        """가장 낮은 항목 — "지금 가장 중요한 것은 ○○입니다"의 근거."""
        return min(self.components, key=lambda c: c.score)

    def by_key(self, key: str) -> ScoreComponent:
        for component in self.components:
            if component.key == key:
                return component
        raise KeyError(key)


def compute_risk_score(
    profile: UserProfile, assumptions: Optional[Assumptions] = None
) -> RiskScore:
    """프로파일에서 은퇴 재무 안정도를 계산한다."""
    assumptions = assumptions or load_assumptions()
    amap = build_asset_map(profile, assumptions)
    sim = simulate(profile, assumptions=assumptions)
    allocation = profile.current_allocation()

    components = [
        _score_income_gap(profile, amap),
        _score_liquidity(profile, amap),
        _score_concentration(profile),
        _score_market_risk(profile, allocation.equity),
        _score_debt(profile),
        _score_pension_adequacy(profile, sim),
    ]
    cap, cap_reason = _sustainability_cap(sim)
    return RiskScore(components=components, cap=cap, cap_reason=cap_reason)


def _sustainability_cap(sim) -> tuple[Optional[int], Optional[str]]:
    """계획이 지속 불가능하면 종합 점수에 상한을 건다.

    기획서의 가중치는 단기 항목(소득공백기 30% + 유동성 25%)에 55%가 몰려 있어,
    "지금 당장은 현금이 넉넉하지만 20년 뒤 파산이 확실한" 사람에게도 90점대가 나온다.
    화면에 '양호 91점'과 '만 69세 자산 고갈'이 나란히 뜨면 사용자는 무엇을 믿어야
    할지 알 수 없다.

    가중치 자체는 기획서를 그대로 두고, 대신 시뮬레이션 결과가 파산을 가리키면
    종합 점수가 그 사실을 넘어설 수 없게 상한을 둔다.
    """
    if sim.depletion_age is None:
        return None, None
    if sim.depletion_age < 75:
        return 40, f"만 {sim.depletion_age}세에 금융자산이 고갈되어 계획을 유지할 수 없습니다."
    if sim.depletion_age < 85:
        return 60, f"만 {sim.depletion_age}세에 금융자산이 고갈됩니다. 장기 대비가 필요합니다."
    return None, None


# --------------------------------------------------------------------------- #
# 항목별 채점
# --------------------------------------------------------------------------- #


def _score_income_gap(profile: UserProfile, amap) -> ScoreComponent:
    """소득공백기(30%) — 연금 개시까지 필요한 생활비를 유동자산으로 덮는가."""
    if amap.income_gap_need <= 0:
        score, detail = 100.0, "소득공백기가 없습니다. 퇴직과 동시에 연금을 받습니다."
    else:
        ratio = amap.gap_coverage_ratio
        score = _clamp(ratio * 100)
        if ratio >= 1.0:
            detail = (
                f"소득공백기 {amap.income_gap_months}개월에 필요한 생활비를 "
                "유동자산으로 모두 충당할 수 있습니다."
            )
        else:
            detail = (
                f"소득공백기 {amap.income_gap_months}개월 생활비 중 "
                f"{ratio * 100:.0f}%만 유동자산으로 충당됩니다."
            )
    return ScoreComponent(
        key="income_gap",
        label="소득공백기",
        weight=WEIGHTS["income_gap"],
        score=score,
        detail=detail,
    )


def _score_liquidity(profile: UserProfile, amap) -> ScoreComponent:
    """유동성(25%) — 즉시 쓸 수 있는 돈이 생활비 몇 개월치인가. 24개월이면 만점."""
    months = amap.liquid_assets / profile.monthly_expense if profile.monthly_expense else 0.0
    score = _clamp(months / 24 * 100)
    return ScoreComponent(
        key="liquidity",
        label="유동성",
        weight=WEIGHTS["liquidity"],
        score=score,
        detail=f"바로 쓸 수 있는 자산이 생활비 약 {months:.0f}개월치입니다 (24개월이면 만점).",
    )


def _score_concentration(profile: UserProfile) -> ScoreComponent:
    """자산집중도(15%) — 한 자산에 얼마나 쏠려 있는가. 대개 부동산 쏠림이 잡힌다."""
    total = profile.total_assets
    if total <= 0:
        return ScoreComponent(
            key="concentration",
            label="자산집중도",
            weight=WEIGHTS["concentration"],
            score=0.0,
            detail="자산이 없습니다.",
        )

    holdings = {
        "부동산": profile.real_estate,
        "현금성": profile.severance_pay + profile.cash_savings,
        "투자자산": profile.equity + profile.isa,
        "연금": profile.pension_dc,
    }
    name, amount = max(holdings.items(), key=lambda kv: kv[1])
    share = amount / total
    # 비중 50%면 100점, 100%면 0점
    score = _clamp((1 - share) / 0.5 * 100)
    return ScoreComponent(
        key="concentration",
        label="자산집중도",
        weight=WEIGHTS["concentration"],
        score=score,
        detail=f"{name}이(가) 총자산의 {share * 100:.0f}%를 차지합니다 (50% 이하면 만점).",
    )


def _score_market_risk(profile: UserProfile, equity_weight: float) -> ScoreComponent:
    """시장위험(10%) — 투자성향 대비 주식 비중이 과한가."""
    limit = _EQUITY_LIMIT[profile.risk_tolerance]
    if equity_weight <= limit:
        score = 100.0
        detail = (
            f"주식 비중 {equity_weight * 100:.0f}%로, {profile.risk_tolerance.value} 성향에 "
            f"맞는 범위(≤{limit * 100:.0f}%)입니다."
        )
    else:
        over = (equity_weight - limit) / max(1e-9, 1 - limit)
        score = _clamp(100 * (1 - over))
        detail = (
            f"주식 비중 {equity_weight * 100:.0f}%가 {profile.risk_tolerance.value} 성향의 "
            f"권장 상한 {limit * 100:.0f}%를 넘습니다."
        )
    return ScoreComponent(
        key="market_risk",
        label="시장위험",
        weight=WEIGHTS["market_risk"],
        score=score,
        detail=detail,
    )


def _score_debt(profile: UserProfile) -> ScoreComponent:
    """부채(10%) — 총자산 대비 부채 비율. 50% 이상이면 0점."""
    total = profile.total_assets
    ratio = profile.debt / total if total > 0 else (1.0 if profile.debt else 0.0)
    score = _clamp((1 - ratio / 0.5) * 100)
    detail = (
        "부채가 없습니다."
        if profile.debt == 0
        else f"부채가 총자산의 {ratio * 100:.0f}%입니다 (50% 이상이면 0점)."
    )
    return ScoreComponent(
        key="debt", label="부채", weight=WEIGHTS["debt"], score=score, detail=detail
    )


def _score_pension_adequacy(profile: UserProfile, sim) -> ScoreComponent:
    """장기소득(10%) — 연금이 생활비를 얼마나 감당하는가 + 자산이 언제까지 버티는가.

    연금/생활비 비율만 보면 "연금은 부족하지만 자산이 충분해서 괜찮은" 사람과
    "연금도 부족하고 자산도 곧 바닥나는" 사람을 구분하지 못한다. 그래서 비율 점수와
    시뮬레이션 결과 점수 중 **낮은 쪽**을 택한다.
    """
    covered = (
        (profile.national_pension_monthly + profile.other_monthly_income)
        / profile.monthly_expense
        if profile.monthly_expense
        else 0.0
    )
    ratio_score = _clamp(covered / 0.7 * 100)
    detail = (
        f"연금과 기타소득이 월 생활비의 {covered * 100:.0f}%를 감당합니다 "
        "(70% 이상이면 만점)."
    )

    if sim.depletion_age is None:
        longevity_score = 100.0
        detail += f" 금융자산은 만 {sim.horizon_age}세까지 유지됩니다."
    else:
        span = max(1, sim.horizon_age - profile.retirement_age)
        longevity_score = _clamp((sim.depletion_age - profile.retirement_age) / span * 100)
        detail += f" 현재 계획대로면 만 {sim.depletion_age}세에 금융자산이 고갈됩니다."

    return ScoreComponent(
        key="pension_adequacy",
        label="장기소득",
        weight=WEIGHTS["pension_adequacy"],
        score=min(ratio_score, longevity_score),
        detail=detail,
    )
