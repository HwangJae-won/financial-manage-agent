"""집을 줄이면 실제로 얼마가 남는가 — 주택 다운사이징.

이 서비스의 처방 엔진(`core/prescribe.py`)이 쥐고 있는 레버는 셋이다. 생활비,
연금 개시 시기, 기타 소득. **부동산이 없다.** 그런데 데모 인물의 부동산은 총자산의
69%다. 자산의 3분의 2를 빼놓고 "생활비를 줄이세요"라고 말해 온 셈이다.

    세상이 하는 말 — "집 있으시니 괜찮죠", "정 어려우면 집 팔면 되죠"
    실제로는      — 5억 집을 3억으로 줄여도 손에 2억이 들어오지 않는다.
                    새 집 취득세, 매도·매수 중개보수, 등기비, 이사비가 먼저 나간다.

그래서 이 모듈이 하는 일은 **차액에서 거래비용을 빼는 것**이다. 그리고 그 결과를
현금흐름에 넣어 고갈 시점이 실제로 얼마나 밀리는지 본다.

여기에 한 가지가 더 붙는다. **재산이 줄면 건강보험 피부양자 재산요건이 완화된다.**
과세표준이 5.4억을 넘으면 소득과 무관하게 탈락하고, 3.6억을 넘으면 소득 기준이
2,000만원에서 1,000만원으로 강화된다(`core/health_insurance.py`). 집을 줄이는 것은
현금을 만드는 동시에 이 계단에서 내려오는 일이기도 하다. 한쪽만 계산하면
다운사이징의 효과를 과소평가한다.

**조심한 것 두 가지**

  1. 거래비용은 **하한**이다. 지방교육세·농어촌특별세는 취득세에 부가되지만 요율
     체계가 별도라 근사식을 지어내지 않았다. 양도소득세도 넣지 않았다 —
     1세대 1주택 비과세 여부가 보유·거주기간에 달려 있어 물어보지 않고는 알 수 없다.
     화면에서 이 사실을 말한다.
  2. **집값 자체는 이 서비스가 모른다.** 사용자가 알려준 평가액을 쓰고, 새 집 가격은
     사용자가 고른다. 시세를 추정하지 않는다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, HousingAssumption, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw
from core.health_insurance import analyze_health_insurance, property_tax_base
from core.models import UserProfile

# 기본으로 보여줄 축소 폭. 현재 집값 대비 비율이며, 사용자가 직접 금액을 넣으면
# 이 사다리 대신 그 금액을 쓴다.
DEFAULT_RATIOS: tuple[float, ...] = (0.75, 0.55, 0.35)

# 후보 금액을 이 단위로 반올림한다. "3억 7,432만원짜리 집" 같은 것은 없다.
ROUND_UNIT = 50_000_000

# 양도소득세를 넣지 않은 이유를 화면에서 그대로 쓴다.
CAPITAL_GAINS_NOTE = (
    "**양도소득세는 넣지 않았습니다.** 1세대 1주택이고 2년 이상 보유·거주하셨다면 "
    "12억원까지 비과세라 대부분 해당이 없지만, 다주택이거나 보유기간이 짧으면 "
    "여기 계산보다 훨씬 많이 나갑니다. 이 부분은 세무서나 세무사에게 확인하셔야 합니다."
)


class TransactionCost(BaseModel):
    """거래비용 내역. 화면에서 항목별로 펼친다 — 합계만 보여주면 안 믿는다."""

    acquisition_tax: int = 0
    acquisition_tax_label: str = "0원"
    acquisition_tax_rate_label: str = ""

    brokerage_sell: int = 0
    brokerage_sell_label: str = "0원"
    brokerage_buy: int = 0
    brokerage_buy_label: str = "0원"

    registration: int = 0
    registration_label: str = "0원"
    moving: int = 0
    moving_label: str = "0원"

    total: int = 0
    total_label: str = "0원"


class DownsizingOption(BaseModel):
    """집을 이만큼 줄였을 때."""

    new_home_price: int
    new_home_label: str = ""
    label: str = ""

    gross_difference: int = Field(default=0, description="집값 차액 (비용 차감 전)")
    gross_difference_label: str = ""
    costs: TransactionCost = Field(default_factory=TransactionCost)

    net_proceeds: int = Field(default=0, description="비용을 빼고 손에 남는 돈")
    net_proceeds_label: str = ""
    cost_share_label: str = Field(
        default="", description="차액 대비 비용 비중 — '2억 중 얼마가 사라지나'"
    )

    depletion_age: Optional[int] = None
    depletion_label: str = ""
    years_gained: int = Field(default=0, description="고갈이 몇 년 밀리는가")

    # 건강보험 재산요건 — 집을 줄이는 두 번째 효과
    property_tax_base: int = 0
    property_tax_base_label: str = ""
    relieves_property_test: bool = Field(
        default=False, description="재산요건 계단에서 내려오는가"
    )
    health_effect: str = ""


class DownsizingReport(BaseModel):
    """주택 다운사이징 분석."""

    computable: bool = True
    reason: str = Field(default="", description="계산할 수 없으면 왜인지")

    current_home: int = 0
    current_home_label: str = ""
    home_share_label: str = Field(default="", description="총자산에서 부동산이 차지하는 비중")

    baseline_depletion_age: Optional[int] = None
    baseline_property_tax_base: int = 0
    baseline_property_tax_base_label: str = ""

    options: list[DownsizingOption] = Field(default_factory=list)

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
def _round_price(amount: int) -> int:
    return max(ROUND_UNIT, int(round(amount / ROUND_UNIT)) * ROUND_UNIT)


def transaction_cost(
    sell_price: int, buy_price: int, spec: HousingAssumption
) -> TransactionCost:
    """집을 팔고 사는 데 드는 비용.

    매도 쪽은 중개보수만, 매수 쪽은 취득세·중개보수·등기비가 붙는다. 이사비는 한 번.
    """
    tax_rate = spec.acquisition_tax.rate_for(buy_price)
    tax = int(round(buy_price * tax_rate))
    sell_fee = spec.brokerage_fee(sell_price)
    buy_fee = spec.brokerage_fee(buy_price)
    registration = int(round(buy_price * spec.registration_rate))
    moving = spec.moving_cost

    total = tax + sell_fee + buy_fee + registration + moving
    return TransactionCost(
        acquisition_tax=tax,
        acquisition_tax_label=fmt_krw(tax),
        acquisition_tax_rate_label=f"{tax_rate * 100:g}%",
        brokerage_sell=sell_fee,
        brokerage_sell_label=fmt_krw(sell_fee),
        brokerage_buy=buy_fee,
        brokerage_buy_label=fmt_krw(buy_fee),
        registration=registration,
        registration_label=fmt_krw(registration),
        moving=moving,
        moving_label=fmt_krw(moving),
        total=total,
        total_label=fmt_krw(total),
    )


def _option(
    profile: UserProfile,
    assumptions: Assumptions,
    new_price: int,
    *,
    baseline_age: Optional[int],
    baseline_qualifies: bool,
    baseline_cliff: Optional[int],
) -> DownsizingOption:
    spec = assumptions.housing
    sell_price = profile.real_estate

    costs = transaction_cost(sell_price, new_price, spec)
    gross = sell_price - new_price
    net = gross - costs.total

    # 남은 돈은 현금성 자산으로 들어간다. 주식에 넣는다고 가정하면 수익률이
    # 붙어 다운사이징 효과가 부풀려진다 — 가장 보수적인 쪽을 쓴다.
    variant = profile.model_copy(
        update={
            "real_estate": new_price,
            "cash_savings": max(0, profile.cash_savings + net),
        }
    )
    sim = simulate(variant, assumptions=assumptions)
    horizon = assumptions.macro.horizon_age
    gained = max(0, (sim.depletion_age or horizon) - (baseline_age or horizon))

    base, _ = property_tax_base(variant, assumptions.health_insurance)
    health = analyze_health_insurance(variant, assumptions=assumptions)
    relieved = (health.qualifies_at_retirement and not baseline_qualifies) or (
        health.cliff_year is not None
        and baseline_cliff is not None
        and health.cliff_year > baseline_cliff
    )

    option = DownsizingOption(
        new_home_price=new_price,
        new_home_label=fmt_krw(new_price),
        label=f"{fmt_krw(new_price)} 집으로",
        gross_difference=gross,
        gross_difference_label=fmt_krw(gross),
        costs=costs,
        net_proceeds=net,
        net_proceeds_label=fmt_krw(net),
        cost_share_label=(
            f"차액 {fmt_krw(gross)} 중 {fmt_krw(costs.total)}이 거래비용으로 나갑니다"
            if gross > 0
            else ""
        ),
        depletion_age=sim.depletion_age,
        depletion_label=(
            f"만 {horizon}세까지 유지"
            if sim.depletion_age is None
            else f"만 {sim.depletion_age}세 고갈"
        ),
        years_gained=gained,
        property_tax_base=base,
        property_tax_base_label=fmt_krw(base),
        relieves_property_test=relieved,
    )
    option.health_effect = _health_effect(option, health, baseline_cliff)
    return option


def _health_effect(option, health, baseline_cliff: Optional[int]) -> str:
    if option.relieves_property_test:
        if health.qualifies_at_retirement and baseline_cliff is None:
            return (
                "재산이 줄어 건강보험 **피부양자 재산요건**을 통과하게 됩니다. "
                "보험료가 0원이 되는 것이라 현금 이상의 효과입니다."
            )
        if health.cliff_year and baseline_cliff:
            return (
                f"피부양자 탈락 시점이 {baseline_cliff}년에서 {health.cliff_year}년으로 "
                "미뤄집니다."
            )
        return "건강보험 피부양자 재산요건이 완화됩니다."
    return "건강보험 피부양자 판정은 달라지지 않습니다 — 재산이 아니라 소득이 걸림돌입니다."


# --------------------------------------------------------------------------- #
def analyze_downsizing(
    profile: UserProfile,
    *,
    new_home_price: Optional[int] = None,
    ratios: Optional[tuple[float, ...]] = None,
    assumptions: Optional[Assumptions] = None,
) -> DownsizingReport:
    """집을 줄였을 때 손에 남는 돈과 계획 변화.

    Args:
        new_home_price: 옮겨 갈 집의 가격. 주시면 그 금액 하나만 계산한다.
            생략하면 현재 집값의 75% / 55% / 35% 사다리를 보여준다.
    """
    assumptions = assumptions or load_assumptions()
    spec = assumptions.health_insurance

    if profile.real_estate <= 0:
        base, _ = property_tax_base(profile, spec)
        return DownsizingReport(
            computable=False,
            reason="부동산 평가액을 알려주셔야 계산할 수 있습니다.",
            baseline_property_tax_base=base,
            headline="보유하신 집의 시세를 알려주시면 줄였을 때 손에 남는 돈을 계산해 드립니다.",
            verdict=(
                "네이버 부동산이나 국토교통부 실거래가 공개시스템에서 같은 단지 "
                "최근 거래가를 보시면 됩니다."
            ),
            notes=[CAPITAL_GAINS_NOTE],
        )

    baseline = simulate(profile, assumptions=assumptions)
    baseline_health = analyze_health_insurance(profile, assumptions=assumptions)
    base_property, _ = property_tax_base(profile, spec)

    if new_home_price is not None:
        prices = [max(0, int(new_home_price))]
    else:
        prices = []
        for ratio in ratios or DEFAULT_RATIOS:
            price = _round_price(profile.real_estate * ratio)
            if price < profile.real_estate and price not in prices:
                prices.append(price)

    options = [
        _option(
            profile,
            assumptions,
            price,
            baseline_age=baseline.depletion_age,
            baseline_qualifies=baseline_health.qualifies_at_retirement,
            baseline_cliff=baseline_health.cliff_year,
        )
        for price in prices
    ]

    total = profile.total_assets
    report = DownsizingReport(
        current_home=profile.real_estate,
        current_home_label=fmt_krw(profile.real_estate),
        home_share_label=(
            f"총자산 {fmt_krw(total)} 중 {profile.real_estate / total * 100:.0f}%"
            if total > 0
            else ""
        ),
        baseline_depletion_age=baseline.depletion_age,
        baseline_property_tax_base=base_property,
        baseline_property_tax_base_label=fmt_krw(base_property),
        options=options,
    )
    report.headline, report.verdict = _verdict(report, profile)
    report.notes = _notes(report)
    return report


def _verdict(report: DownsizingReport, profile: UserProfile) -> tuple[str, str]:
    if not report.options:
        return (
            "지금보다 작은 집을 고르셔야 계산할 수 있습니다.",
            "옮겨 가실 집의 가격을 알려주시면 손에 남는 돈을 계산해 드립니다.",
        )

    best = max(report.options, key=lambda o: o.years_gained)
    first = report.options[0]

    headline = (
        f"{report.current_home_label} 집을 {first.new_home_label}으로 줄이시면 "
        f"차액 {first.gross_difference_label} 중 거래비용 {first.costs.total_label}을 "
        f"빼고 **{first.net_proceeds_label}**이 남습니다."
    )

    if report.baseline_depletion_age is None:
        verdict = (
            "지금 계획으로도 자산이 고갈되지 않으므로 집을 줄이실 이유는 "
            "현금흐름이 아니라 다른 데 있습니다. "
            + (
                best.health_effect
                if best.relieves_property_test
                else "건강보험 피부양자 판정도 달라지지 않습니다."
            )
        )
    elif best.years_gained > 0:
        verdict = (
            f"가장 크게 줄이시면 고갈이 만 {report.baseline_depletion_age}세에서 "
            f"만 {best.depletion_age}세로 **{best.years_gained}년** 밀립니다. "
            + (best.health_effect if best.relieves_property_test else "")
        ).strip()
    else:
        verdict = (
            "집을 줄여도 고갈 시점이 눈에 띄게 밀리지는 않습니다. "
            "거래비용이 차액의 상당 부분을 가져가기 때문입니다. "
            "생활비나 연금 개시 시기 쪽이 더 큰 레버입니다."
        )

    if report.home_share_label:
        verdict += f" 참고로 부동산이 {report.home_share_label}입니다."
    return headline, verdict


def _notes(report: DownsizingReport) -> list[str]:
    notes = [
        "취득세는 **지방세법 제11조 제1항 제8호**의 주택 유상거래 표준세율입니다 "
        "(6억 이하 1%, 6~9억 계산식, 9억 초과 3%). 다주택·조정대상지역 중과세율은 "
        "반영하지 않았습니다.",
        "여기 거래비용은 **하한**입니다. 취득세에 붙는 지방교육세·농어촌특별세는 "
        "요율 체계가 달라 넣지 않았습니다.",
        CAPITAL_GAINS_NOTE,
        "중개보수는 법정 **상한요율**로 계산했습니다. 협의로 낮추실 수 있고, "
        "지자체 조례에 따라 다를 수 있습니다.",
        "등기·법무비와 이사비는 가정값입니다. 지역과 규모에 따라 크게 달라집니다.",
    ]
    if report.options:
        notes.append(
            "집을 판 돈은 **예금에 넣는다고 보고** 계산했습니다. 투자하시면 수익률이 "
            "붙지만, 그만큼 위험도 함께 옵니다."
        )
    return notes
