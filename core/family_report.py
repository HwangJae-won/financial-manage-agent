"""가족에게 보여줄 한 장 (기능 5).

시니어의 금융 결정은 실제로 혼자 내려지지 않는다. 자녀가 "그거 하지 마세요"
하고, 배우자가 "옆집은 했다던데" 한다. 그런데 지금까지 이 서비스가 만든 것은
전부 **본인 화면**이었다. 가족은 그 화면을 보지 못하고, 본인은 화면에 있던
숫자를 정확히 옮기지 못한다.

이 모듈은 나머지 기능의 결과를 담는 그릇이다. 그래서 마지막에 만들었다 —
담을 것이 다 나온 뒤라야 무엇을 담을지 정할 수 있다.

가족이 읽는다는 전제가 내용을 바꾼다:

  - **기한이 있는 것을 맨 위로.** 임의계속가입은 신청 기한을 놓치면 그걸로
    끝이다. 자산 배분은 다음 달에 해도 된다. 가족이 도울 수 있는 것은 주로
    앞쪽이다.
  - **하지 않아도 되는 것을 명시한다.** 이 서비스의 축이 그대로 여기 온다.
    가족이 가장 많이 하는 개입이 "뭐라도 해야 하는 것 아니냐"이기 때문이다.
  - **모르는 것을 숨기지 않는다.** 가정한 값과 계산하지 않은 것을 함께 낸다.
    가족은 본인보다 이 리포트를 더 꼼꼼히 읽는다.

평문(`as_text`)을 함께 만든다. 카톡으로 보내는 것이 실제 공유 경로이기 때문이다.
화면을 캡처해 보내면 숫자만 남고 단서가 사라진다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_eul, fmt_krw, fmt_years
from core.health_insurance import analyze_health_insurance
from core.models import UserProfile
from core.national_pension import analyze_national_pension
from core.prescribe import prescribe
from core.risk_score import compute_risk_score
from core.severance import compare_severance_options

# 사용자가 답하지 않아 기본값으로 채운 항목을 사람 말로 옮긴다.
# 필드명("national_pension_months")을 그대로 내보내면 가족이 읽을 수 없다.
FIELD_LABELS: dict[str, str] = {
    "birth_month": "생월",
    "retirement_month": "퇴직 월",
    "severance_pay": "퇴직금",
    "years_employed": "근속연수",
    "last_monthly_salary": "퇴직 전 월 급여",
    "cash_savings": "예금·적금",
    "national_pension_monthly": "국민연금 예상 수령액",
    "national_pension_months": "국민연금 가입기간",
    "equity": "주식·펀드",
    "isa": "ISA",
    "pension_dc": "퇴직연금·IRP",
    "real_estate": "부동산",
    "other_monthly_income": "기타 월소득",
    "debt": "부채",
    "risk_tolerance": "투자성향",
}


class ReportItem(BaseModel):
    """리포트 한 줄. 제목·숫자·왜 그런지."""

    title: str
    value: str = ""
    detail: str = ""
    deadline: str = Field(default="", description="기한이 있으면 그 기한")


class FamilyReport(BaseModel):
    """가족에게 보여줄 한 장."""

    subject: str = Field(description="누구의 계획인지 — 이름 대신 나이와 퇴직 시점")
    headline: str
    summary: str

    # 기한이 있는 것부터. 가족이 실제로 도울 수 있는 것이 여기 모인다.
    now: list[ReportItem] = Field(default_factory=list)
    later: list[ReportItem] = Field(default_factory=list)
    # 이 서비스의 축 — 하지 않아도 되는 것
    not_needed: list[ReportItem] = Field(default_factory=list)
    watch_outs: list[str] = Field(default_factory=list)

    assumptions: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)

    as_text: str = Field(default="", description="카톡으로 보낼 수 있는 평문")


# --------------------------------------------------------------------------- #
# 구성
# --------------------------------------------------------------------------- #


def _subject(profile: UserProfile) -> str:
    """누구의 계획인지. 이름 대신 나이와 퇴직 시점으로 적는다.

    퇴직을 했는지 안 했는지는 쓰지 않는다. 프로파일에 오늘 날짜가 없어서
    알 수 없고, '예정'을 붙였다 틀리면 가족이 리포트 전체를 의심하게 된다.
    """
    return (
        f"{profile.birth_year}년생 (퇴직 시점 만 {profile.retirement_age}세) · "
        f"{profile.retirement_year}년 {profile.retirement_month}월 퇴직"
    )


def _diagnosis(profile: UserProfile, assumptions: Assumptions) -> tuple[str, str]:
    """한 문장 진단과 가족에게 하는 말."""
    sim = simulate(profile, assumptions=assumptions)
    score = compute_risk_score(profile, assumptions)

    if sim.depletion_age is None:
        headline = (
            f"만 {sim.horizon_age}세까지 금융자산이 유지됩니다. "
            "지금 계획을 크게 바꾸실 필요가 없습니다."
        )
    else:
        headline = (
            f"지금 계획대로면 만 {sim.depletion_age}세에 금융자산이 바닥납니다 "
            f"({sim.depletion_year}년)."
        )

    summary = (
        f"금융자산 {fmt_krw(profile.financial_assets)}, 월 생활비 "
        f"{fmt_krw(profile.monthly_expense)} 기준입니다. "
        f"연금·기타소득 없이 이 자산만으로는 {fmt_years(sim.expense_coverage_years)} "
        f"버팁니다. 은퇴 재무 안정도는 {score.total}점({score.status})입니다."
    )
    return headline, summary


def _severance_item(
    profile: UserProfile, assumptions: Assumptions
) -> Optional[ReportItem]:
    """퇴직금 수령 방식 — 퇴직 시점에 한 번뿐인 결정이라 기한이 있다."""
    result = compare_severance_options(profile, assumptions=assumptions)
    if not result.computable:
        return None
    if result.tax_saved <= 0 and result.final_balance_gap <= 0:
        return ReportItem(
            title="퇴직금 수령 방식",
            value="어느 쪽이든 비슷합니다",
            detail=result.verdict,
        )
    return ReportItem(
        title="퇴직금을 연금(IRP)으로 받기",
        value=f"세금 {result.tax_saved_label} 절약",
        detail=(
            f"일시금이면 {result.lump_sum.total_tax_label}, 연금이면 "
            f"{result.pension.total_tax_label}입니다 "
            f"(근속 {result.years_employed}년 기준)."
        ),
        deadline="퇴직금을 받기 전에 정해야 합니다. 받고 나면 되돌릴 수 없습니다.",
    )


def _health_item(profile: UserProfile, assumptions: Assumptions) -> ReportItem:
    """건강보험 — 절벽이 언제 오는지. 신청 기한이 있는 완충장치가 딸려 있다."""
    report = analyze_health_insurance(profile, assumptions=assumptions)

    if report.cliff_year is None:
        return ReportItem(
            title="건강보험료",
            value="지금도 앞으로도 0원",
            detail=(
                "피부양자 자격이 유지됩니다. "
                f"다만 이자·배당이 연 {report.financial_headroom_label}만 늘면 "
                f"그 순간 연 {report.local.annual_label}이 생깁니다."
            ),
        )

    item = ReportItem(
        title="건강보험료가 새로 생기는 시점",
        value=f"{report.cliff_year}년부터 연 {report.local.annual_label}",
        detail=f"{report.cliff_reason}. 지금은 {report.dependent.monthly_label}입니다.",
    )
    if report.voluntary.available and report.voluntary.monthly < report.local.monthly:
        item.deadline = (
            f"임의계속가입을 신청하면 최대 36개월간 월 "
            f"{report.voluntary.monthly_label}로 낮출 수 있습니다. "
            "신청 기한은 첫 지역보험료 고지서 납부기한에서 2개월까지입니다."
        )
    return item


def _pension_item(
    profile: UserProfile, assumptions: Assumptions
) -> Optional[ReportItem]:
    """국민연금 추가납부 — 이득이면 이득, 손해면 손해라고 말한다."""
    report = analyze_national_pension(profile, assumptions=assumptions)
    if not report.computable:
        return ReportItem(
            title="국민연금을 더 낼지",
            value="아직 계산하지 못했습니다",
            detail=report.verdict,
        )

    if not report.qualifies_now:
        return ReportItem(
            title="국민연금 수급자격이 아직 없습니다",
            value=f"{report.months_to_qualify}개월 부족 · {report.cost_to_qualify_label}",
            detail=(
                f"지금 상태로는 노령연금이 평생 0원입니다. 채우시면 매달 "
                f"{report.monthly_at_minimum_label}이 죽을 때까지 나옵니다."
            ),
            deadline="임의계속가입은 만 65세까지, 연금을 받기 시작하면 자격이 사라집니다.",
        )

    best = next(
        (o for o in report.options if o.key == report.best_key and o.available), None
    )
    if best is None:
        return ReportItem(
            title="국민연금을 더 내는 것",
            value="하실 필요 없습니다",
            detail=report.verdict,
        )
    return ReportItem(
        title=f"국민연금 {best.label}",
        value=f"{best.cost_label} 내고 평생 {best.net_gain_label} 이득",
        detail=(
            f"연금이 매달 {best.monthly_gain_label} 늘어납니다. "
            f"늘어난 연금 때문에 생기는 건강보험료 "
            f"{fmt_eul(best.extra_premium_label)} 뺀 금액입니다. "
            f"{best.breakeven_label}에 본전입니다."
        ),
        deadline=(
            f"만 {profile.national_pension_start_age}세에 연금을 받기 시작하면 "
            "자격이 사라집니다."
        ),
    )


def _prescription_items(
    profile: UserProfile, assumptions: Assumptions
) -> tuple[list[ReportItem], list[ReportItem]]:
    """무엇을 얼마나 바꿔야 하는지. 되는 것과 안 되는 것을 나눈다."""
    result = prescribe(profile, assumptions=assumptions)
    if result.already_safe:
        return [], [
            ReportItem(
                title="생활비를 줄이는 것",
                value="지금은 필요 없습니다",
                detail=result.summary,
            )
        ]

    doable = [
        ReportItem(
            title=option.label,
            value=f"{option.current_label} → {option.required_label}",
            detail=(
                f"이 하나만 하셔도 만 {result.target_age}세까지 유지됩니다."
                if option.depletion_age is None
                else f"만 {option.depletion_age}세까지 늘어납니다."
            ),
        )
        for option in result.feasible_options[:3]
    ]
    return doable, []


def _watch_outs(profile: UserProfile) -> list[str]:
    """가족이 지켜봐야 할 것. 이 연령대에 실제로 벌어지는 일이다."""
    notes = [
        "'제도가 바뀌니 지금 갈아타세요'는 이 연령대를 노리는 가장 흔한 문구입니다. "
        "발표와 시행은 다르고, 대부분의 제도 변화는 개인에게 연 몇 만원 수준입니다.",
        "원금 보장과 높은 확정 수익을 함께 말하는 상품은 없습니다.",
    ]
    if profile.financial_assets > 0:
        notes.append(
            f"금융자산 {fmt_krw(profile.financial_assets)} 중 큰 금액이 한 번에 "
            "움직이려 할 때는 가족과 먼저 이야기하도록 정해 두시면 좋습니다."
        )
    return notes


def _assumptions(profile: UserProfile) -> list[str]:
    """가정한 값. 가족이 바로잡을 수 있는 것들이다."""
    if not profile.assumed_fields:
        return []
    labels = [FIELD_LABELS.get(field, field) for field in profile.assumed_fields]
    return [
        f"말씀하지 않으셔서 가정한 값: {', '.join(labels)}. "
        "실제 값을 아시면 알려주세요 — 결과가 달라질 수 있습니다."
    ]


LIMITS = [
    "지역가입자 건강보험료는 소득분만 계산했습니다. 재산분이 빠져 있어 실제 부담은 "
    "이보다 큽니다.",
    "투자 수익률은 가정값이며 미래 수익을 보장하지 않습니다.",
    "이 리포트는 특정 금융상품의 가입을 권유하지 않습니다. 참고용 안내입니다.",
]


# --------------------------------------------------------------------------- #
# 조립
# --------------------------------------------------------------------------- #


def build_family_report(
    profile: UserProfile, *, assumptions: Optional[Assumptions] = None
) -> FamilyReport:
    """가족에게 보여줄 한 장을 만든다.

    나머지 기능의 결과를 모아 **기한이 있는 것 / 나중에 해도 되는 것 /
    하지 않아도 되는 것** 세 칸으로 나눈다. 가족이 실제로 도울 수 있는 것은
    주로 첫 칸이고, 가장 많이 하는 개입은 셋째 칸을 향한 것이다.
    """
    assumptions = assumptions or load_assumptions()

    headline, summary = _diagnosis(profile, assumptions)

    candidates = [
        _severance_item(profile, assumptions),
        _health_item(profile, assumptions),
        _pension_item(profile, assumptions),
    ]
    items = [item for item in candidates if item is not None]

    doable, unnecessary = _prescription_items(profile, assumptions)

    # 기한이 있으면 앞 칸으로. 놓치면 되돌릴 수 없는 것이 먼저다.
    now = [item for item in items if item.deadline]
    later = [item for item in items if not item.deadline] + doable
    not_needed = unnecessary + [
        item for item in items if "필요 없습니다" in item.value
    ]
    # 위 두 줄에서 겹칠 수 있다 — 같은 항목이 두 칸에 나오면 리포트를 믿기 어려워진다.
    not_needed_titles = {item.title for item in not_needed}
    later = [item for item in later if item.title not in not_needed_titles]

    report = FamilyReport(
        subject=_subject(profile),
        headline=headline,
        summary=summary,
        now=now,
        later=later,
        not_needed=not_needed,
        watch_outs=_watch_outs(profile),
        assumptions=_assumptions(profile),
        limits=LIMITS,
    )
    report.as_text = to_text(report)
    return report


def to_text(report: FamilyReport) -> str:
    """카톡으로 보낼 수 있는 평문.

    화면을 캡처해 보내면 숫자만 남고 단서가 사라진다. 기한과 한계가 함께 가야
    가족이 제대로 판단한다.
    """
    lines = [
        "[은퇴 자산 정리]",
        report.subject,
        "",
        report.headline,
        report.summary,
    ]

    def block(title: str, items: list[ReportItem]) -> None:
        if not items:
            return
        lines.extend(["", title])
        for item in items:
            lines.append(f"- {item.title}: {item.value}")
            if item.detail:
                lines.append(f"  {item.detail}")
            if item.deadline:
                lines.append(f"  [기한] {item.deadline}")

    block("■ 기한이 있는 것 — 놓치면 되돌릴 수 없습니다", report.now)
    block("■ 천천히 보셔도 되는 것", report.later)
    block("■ 하지 않으셔도 되는 것", report.not_needed)

    if report.watch_outs:
        lines.extend(["", "■ 가족이 함께 봐 주실 것"])
        lines.extend(f"- {note}" for note in report.watch_outs)
    if report.assumptions:
        lines.extend(["", "■ 가정한 값"])
        lines.extend(f"- {note}" for note in report.assumptions)

    lines.extend(["", "■ 이 계산의 한계"])
    lines.extend(f"- {note}" for note in report.limits)
    return "\n".join(lines)
