"""의료비가 계획을 얼마나 흔드나 — 그리고 본인부담상한제.

이 연령대의 가장 큰 재무 공포는 의료비다. 그리고 그 공포는 정확히 이 서비스가
다루는 종류다.

    세상이 하는 말 — "암 걸리면 수억 듭니다. 보험 지금 드세요"
    실제로는      — 건강보험 **본인부담상한제**가 있다. 1년 동안 낸 본인부담금이
                    소득수준별로 정한 상한액을 넘으면 **초과분을 공단이 돌려준다**
                    (국민건강보험법 제44조 제2항).

그래서 "수억"은 대부분의 경우 사실이 아니다. 다만 여기서 조심할 것이 세 가지 있다.

  1. **상한제는 급여 항목에만 적용된다.** 비급여(상급병실료, 선택진료, 신약 등)와
     간병비는 상한에 포함되지 않는다. 이것이 실제 부담의 큰 몫이다.
  2. **상한액은 소득분위별로 다르다.** 시행령 별표의 표에 있는데 **첨부파일(hwp/pdf)
     형태라 기계로 읽을 수 없다.** 그래서 이 모듈은 그 표를 들고 있지 않다.
  3. **돌려받는 것은 나중이다.** 사후환급은 다음 해에 이루어지므로, 그해에는
     일단 내야 한다. 현금흐름 관점에서는 시점이 중요하다.

그래서 이 모듈이 하는 일은 이렇다.

  - 상한액을 **알려주시면** 상한제를 반영해 실제 부담을 계산한다
  - 모르면 **추측하지 않고 확인 경로를 안내한다** (근속연수·가입월수와 같은 규약)
  - 어느 쪽이든 **의료비가 계획에 미치는 영향은 계산한다** — 그건 상한액을 몰라도
    할 수 있고, 사용자가 실제로 알고 싶은 것이다
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw
from core.models import LifeEvent, UserProfile


class MedicalScenario(BaseModel):
    """의료비 한 건이 계획에 미치는 영향."""

    label: str
    total_cost: int = Field(description="병원비 총액 (급여 + 비급여)")
    total_cost_label: str = ""

    covered_share: int = Field(default=0, description="상한제가 적용되는 급여 본인부담")
    out_of_pocket: int = Field(default=0, description="상한제를 반영한 실제 부담")
    out_of_pocket_label: str = ""
    refund: int = Field(default=0, description="공단이 돌려주는 금액")
    refund_label: str = ""
    ceiling_applied: bool = False

    depletion_age: Optional[int] = None
    depletion_label: str = ""
    years_lost: int = Field(default=0, description="고갈이 몇 년 앞당겨지는가")


class MedicalCostReport(BaseModel):
    """의료비 충격 분석."""

    ceiling_known: bool = Field(
        default=False, description="본인부담상한액을 알고 있는가"
    )
    ceiling: int = 0
    ceiling_label: str = ""

    baseline_depletion_age: Optional[int] = None
    scenarios: list[MedicalScenario] = Field(default_factory=list)

    headline: str = ""
    verdict: str = ""
    notes: list[str] = Field(default_factory=list)


# 화면에서 비교할 의료비 규모. "암 걸리면 수억"이라는 말과 실제 사이의 간격을
# 보여주려면 큰 금액까지 올라가 봐야 한다.
DEFAULT_COSTS: tuple[tuple[str, int], ...] = (
    ("입원·수술 한 번", 5_000_000),
    ("중증질환 치료", 30_000_000),
    ("장기 투병", 100_000_000),
)

# 병원비 중 건강보험 급여 항목의 본인부담이 차지하는 비중.
# **이 값은 질환에 따라 크게 다르다.** 상한제가 얼마나 막아주는지의 감을 주기 위한
# 것이지 개인의 실제 비율이 아니며, 화면에 그렇게 적는다.
COVERED_SHARE = 0.6


def _scenario(
    profile: UserProfile,
    assumptions: Assumptions,
    label: str,
    total: int,
    *,
    ceiling: Optional[int],
    baseline_age: Optional[int],
) -> MedicalScenario:
    """의료비 한 건을 넣고 실제 부담과 고갈 시점을 본다."""
    covered = int(total * COVERED_SHARE)
    if ceiling is not None and covered > ceiling:
        refund = covered - ceiling
        out_of_pocket = total - refund
        applied = True
    else:
        refund = 0
        out_of_pocket = total
        applied = False

    # 의료비는 언제 생길지 모른다. 퇴직 직후로 잡는 것이 가장 보수적이다 —
    # 자산이 가장 많이 남아 있어야 할 시점에 빠져나가기 때문이다.
    variant = profile.model_copy(
        update={
            "life_events": [
                *profile.life_events,
                LifeEvent(
                    year=profile.retirement_year, amount=out_of_pocket, label="의료비"
                ),
            ]
        }
    )
    sim = simulate(variant, assumptions=assumptions)
    horizon = assumptions.macro.horizon_age
    lost = max(0, (baseline_age or horizon) - (sim.depletion_age or horizon))

    return MedicalScenario(
        label=label,
        total_cost=total,
        total_cost_label=fmt_krw(total),
        covered_share=covered,
        out_of_pocket=out_of_pocket,
        out_of_pocket_label=fmt_krw(out_of_pocket),
        refund=refund,
        refund_label=fmt_krw(refund),
        ceiling_applied=applied,
        depletion_age=sim.depletion_age,
        depletion_label=(
            f"만 {sim.horizon_age}세까지 유지"
            if sim.depletion_age is None
            else f"만 {sim.depletion_age}세 고갈"
        ),
        years_lost=lost,
    )


def analyze_medical_cost(
    profile: UserProfile,
    *,
    annual_ceiling: Optional[int] = None,
    costs: Optional[tuple[tuple[str, int], ...]] = None,
    assumptions: Optional[Assumptions] = None,
) -> MedicalCostReport:
    """의료비가 계획을 얼마나 흔드는지, 상한제가 얼마나 막아주는지.

    Args:
        annual_ceiling: 본인부담상한액(연). **모르면 생략한다** — 소득분위별로
            다르고 시행령 별표가 첨부파일이라 이 서비스가 들고 있지 않다.
            생략하면 상한제를 반영하지 않은 값으로 계산하고 그 사실을 말한다.
    """
    assumptions = assumptions or load_assumptions()
    baseline = simulate(profile, assumptions=assumptions)

    scenarios = [
        _scenario(
            profile,
            assumptions,
            label,
            total,
            ceiling=annual_ceiling,
            baseline_age=baseline.depletion_age,
        )
        for label, total in (costs or DEFAULT_COSTS)
    ]

    report = MedicalCostReport(
        ceiling_known=annual_ceiling is not None,
        ceiling=annual_ceiling or 0,
        ceiling_label=fmt_krw(annual_ceiling) if annual_ceiling else "",
        baseline_depletion_age=baseline.depletion_age,
        scenarios=scenarios,
    )
    report.headline, report.verdict = _verdict(report)
    report.notes = _notes(report)
    return report


def _verdict(report: MedicalCostReport) -> tuple[str, str]:
    worst = report.scenarios[-1] if report.scenarios else None

    if not report.ceiling_known:
        return (
            "건강보험에는 **본인부담상한제**가 있습니다. 1년 동안 내신 병원비 "
            "본인부담금이 정해진 금액을 넘으면 **초과분을 공단이 돌려줍니다.** "
            "'암 걸리면 수억'이라는 말이 대부분의 경우 사실이 아닌 이유입니다.",
            "고객님의 상한액은 소득분위에 따라 달라서 여기서 계산하지 않았습니다. "
            "건강보험공단(1577-1000)이나 The건강보험 앱에서 확인하신 뒤 넣으시면 "
            "실제 부담을 계산해 드립니다. 아래 표는 **상한제를 반영하지 않은** "
            "금액이라 실제보다 큽니다.",
        )

    if worst is not None and worst.ceiling_applied:
        return (
            f"병원비가 {worst.total_cost_label}이 나와도, 본인부담상한제 덕분에 "
            f"실제로 부담하시는 것은 **{worst.out_of_pocket_label}**입니다 "
            f"({worst.refund_label}은 공단이 돌려줍니다).",
            "다만 **비급여와 간병비는 상한에 포함되지 않습니다.** 상급병실료, "
            "선택진료비, 일부 신약이 여기 해당하고 실제 부담의 큰 몫입니다. "
            "그리고 돌려받는 것은 보통 다음 해라, 그해에는 일단 내셔야 합니다.",
        )

    return (
        "예상하신 의료비는 본인부담상한액에 못 미쳐 상한제가 적용되지 않습니다.",
        "지금 규모라면 계획에 큰 영향이 없습니다. 아래 표에서 고갈 시점 변화를 "
        "확인하세요.",
    )


def _notes(report: MedicalCostReport) -> list[str]:
    notes = [
        "**비급여와 간병비는 본인부담상한제에 포함되지 않습니다.** 상급병실료·"
        "선택진료비·일부 신약이 여기 해당하며, 실제 부담의 큰 몫입니다.",
        "상한을 넘은 금액은 보통 **다음 해에 돌려받습니다.** 그해에는 일단 내셔야 "
        "하므로, 현금이 묶이는 시점이 있습니다.",
        f"병원비 중 급여 항목 본인부담을 {int(COVERED_SHARE * 100)}%로 잡았습니다. "
        "질환에 따라 크게 다르므로 **감을 주기 위한 값이지 고객님의 실제 비율이 "
        "아닙니다.**",
        "의료비가 퇴직 직후에 발생한다고 보고 계산했습니다. 자산이 가장 많이 남아 "
        "있어야 할 시점이라 가장 보수적인 가정입니다.",
    ]
    if not report.ceiling_known:
        notes.append(
            "본인부담상한액은 소득분위별로 정해집니다(국민건강보험법 시행령 별표). "
            "표가 첨부파일 형태라 이 서비스가 들고 있지 않습니다 — 추측하지 "
            "않으려고 비워 두었습니다."
        )
    return notes
