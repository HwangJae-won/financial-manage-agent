"""언제 무엇을 해야 하는가 — 은퇴 타임라인과 능동 알림.

지금까지 각 결정은 **따로따로** 계산됐다. 퇴직금 수령 방식, 건강보험 임의계속가입,
국민연금 개시, 피부양자 절벽. 그런데 타겟은 이것들을 **순서대로** 마주한다.

가족 리포트에 "기한이 있는 것"이 있지만 그건 **조건문**이지 날짜가 아니다.
"임의계속가입은 첫 고지서 납부기한에서 2개월까지"는 맞는 말인데, 사용자가 알고
싶은 것은 "그래서 몇 월까지요?"다. 생년월일과 퇴직 시점이 있으니 실제 연월로
뽑을 수 있다. **시니어에게 "언제"는 "얼마"만큼 중요하다.**

여기에 두 가지를 더 붙였다.

  - **행동 경로.** 계산은 다 해주는데 "임의계속가입 신청"을 어디서 어떻게 하는지가
    없었다. 시니어에게는 이것이 진짜 벽이다. 각 항목에 신청처·준비물을 적는다.
  - **능동 알림(기능 ⑨).** 저장된 프로파일과 오늘 날짜를 비교해 "그동안 무엇이
    다가왔는지"를 말한다. 보류 사유가 "유저 DB 선행"이었는데 그것이 생겼다.
    푸시·이메일은 인프라가 필요하지만, **다시 오셨을 때 알려주는 것**은 지금 된다.

이 모듈은 계산을 새로 하지 않는다. 이미 있는 모듈들이 낸 결론에 **날짜와 행동을
붙이는 일**만 한다.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.health_insurance import analyze_health_insurance
from core.models import UserProfile
from core.national_pension import analyze_national_pension

# 문의처. 화면에 그대로 나가므로 한 곳에서만 관리한다.
NHIS = "국민건강보험공단 1577-1000"
NPS = "국민연금공단 1355"


class TimelineEvent(BaseModel):
    """타임라인 한 칸."""

    year: int
    month: int = 1
    when: str = Field(default="", description="화면에 쓰는 시점 문구")
    age: int = 0

    title: str
    detail: str = ""
    kind: str = Field(
        default="info", description="deadline / decision / event — 화면이 어떻게 그릴지"
    )

    # 행동 경로 — 계산이 아니라 안내다. 시니어에게는 이것이 진짜 벽이다.
    where: str = Field(default="", description="어디서 하나")
    what: str = Field(default="", description="무엇을 준비하나")

    # 능동 알림용
    days_away: Optional[int] = None
    passed: bool = False


class TimelineReport(BaseModel):
    """은퇴 타임라인과, 오늘 기준으로 다가온 것."""

    today: str = ""
    events: list[TimelineEvent] = Field(default_factory=list)

    upcoming: list[TimelineEvent] = Field(
        default_factory=list, description="아직 오지 않은 것 (가까운 순)"
    )
    imminent: list[TimelineEvent] = Field(
        default_factory=list, description="기한이 임박한 것 — 놓치면 되돌릴 수 없다"
    )

    headline: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def deadlines(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.kind == "deadline"]


# 기한이 이 안으로 들어오면 '임박'으로 본다. 임의계속가입처럼 놓치면 끝나는
# 것들이라 넉넉히 잡는다.
IMMINENT_DAYS = 180


def _months_between(year: int, month: int, today: _dt.date) -> int:
    return (year - today.year) * 12 + (month - today.month)


def _when(year: int, month: int) -> str:
    return f"{year}년 {month}월"


def build_timeline(
    profile: UserProfile,
    *,
    today: Optional[_dt.date] = None,
    assumptions: Optional[Assumptions] = None,
) -> TimelineReport:
    """생년월일과 퇴직 시점에서 실제 연월을 뽑아 순서대로 세운다."""
    assumptions = assumptions or load_assumptions()
    today = today or _dt.date.today()
    events: list[TimelineEvent] = []

    def add(year, month, title, **kw):
        # 생월을 지나야 한 살 더 먹는다. 12월생에게 3월을 '만 61세'로 적으면
        # 화면의 다른 나이 표기와 어긋난다.
        age = year - profile.birth_year - (1 if month < profile.birth_month else 0)
        events.append(
            TimelineEvent(
                year=year,
                month=month,
                when=_when(year, month),
                age=age,
                title=title,
                **kw,
            )
        )

    # --- 퇴직 --------------------------------------------------------------- #
    add(
        profile.retirement_year,
        profile.retirement_month,
        "퇴직",
        kind="event",
        detail="여기서부터 건강보험 자격과 소득이 동시에 바뀝니다.",
    )

    if profile.severance_pay > 0:
        add(
            profile.retirement_year,
            profile.retirement_month,
            "퇴직금 수령 방식 결정",
            kind="deadline",
            detail=(
                "일시금으로 받을지 IRP 로 연금 수령할지 정합니다. "
                "**받고 나면 되돌릴 수 없습니다.**"
            ),
            where="퇴직 전 회사 인사팀, IRP 는 은행·증권사",
            what="IRP 계좌를 미리 만들어 두셔야 연금 수령을 고를 수 있습니다",
        )

    # --- 건강보험 임의계속가입 (기한이 있는 것) ------------------------------ #
    #
    # 법은 "첫 지역보험료 고지서 납부기한에서 2개월 이내"로 정한다. 고지서는 보통
    # 퇴직 다음 달에 나오므로 **퇴직 +3개월** 정도가 실질적인 마감이다.
    # 정확한 날짜는 고지서를 봐야 알 수 있고, 화면에서 그렇게 말한다.
    deadline_month = profile.retirement_month + 3
    deadline_year = profile.retirement_year + (deadline_month - 1) // 12
    deadline_month = (deadline_month - 1) % 12 + 1
    add(
        deadline_year,
        deadline_month,
        "건강보험 임의계속가입 신청 기한",
        kind="deadline",
        detail=(
            "직장 다닐 때 내던 보험료로 최대 36개월 유지할 수 있습니다. "
            "**이 기한을 놓치면 다시 신청할 수 없습니다.**"
        ),
        where=f"{NHIS} · 지사 방문 · 홈페이지 · The건강보험 앱",
        what="신분증. 첫 지역보험료 고지서를 받으시면 그 납부기한을 확인하세요",
    )

    # --- 국민연금 ----------------------------------------------------------- #
    if profile.national_pension_monthly > 0:
        add(
            profile.pension_start_year,
            profile.birth_month,
            "국민연금 수령 시작",
            kind="decision",
            detail=(
                f"만 {profile.national_pension_start_age}세부터 매달 "
                f"{profile.national_pension_monthly:,}원. 이 시점에 임의계속가입 "
                "자격이 사라집니다."
            ),
            where=f"{NPS} · 지사 방문 · 내곁에국민연금 앱",
            what="신분증, 본인 명의 통장. 개시 시기는 미루거나 앞당길 수 있습니다",
        )

    # --- 건강보험 피부양자 절벽 --------------------------------------------- #
    health = analyze_health_insurance(profile, assumptions=assumptions)
    if health.cliff_year:
        add(
            health.cliff_year,
            1,
            "건강보험 피부양자 탈락",
            kind="event",
            detail=(
                f"연 {health.local.annual_label}이 새로 생깁니다. {health.cliff_reason}."
            ),
            where=f"{NHIS} — 자격 변동은 자동으로 처리됩니다",
            what="따로 하실 일은 없습니다. 다만 그해부터 지출이 늘어납니다",
        )

    # --- 임의계속가입(국민연금) 창이 닫히는 시점 ----------------------------- #
    pension = analyze_national_pension(profile, assumptions=assumptions)
    if pension.computable and pension.voluntary.available:
        add(
            profile.pension_start_year,
            profile.birth_month,
            "국민연금 임의계속가입 종료",
            kind="deadline",
            detail=(
                "연금을 받기 시작하면 더 낼 수 없습니다. "
                f"그전까지 {pension.voluntary.months_added}개월을 더 채우실 수 있습니다."
            ),
            where=f"{NPS} · 지사 방문 · 내곁에국민연금 앱",
            what="신분증. 60세 도달 시점에 신청하실 수 있습니다",
        )

    # --- 재직자 감액이 끝나는 시점 ------------------------------------------- #
    earned = assumptions.national_pension.earned_income
    if profile.national_pension_monthly > 0:
        add(
            profile.birth_year + earned.end_age,
            profile.birth_month,
            f"소득이 있어도 연금이 깎이지 않음 (만 {earned.end_age}세)",
            kind="event",
            detail="이때부터는 얼마를 버셔도 노령연금 전액을 받습니다.",
        )

    events.sort(key=lambda e: (e.year, e.month))
    for event in events:
        months = _months_between(event.year, event.month, today)
        event.passed = months < 0
        event.days_away = months * 30 if months >= 0 else None

    report = TimelineReport(
        today=today.isoformat(),
        events=events,
        upcoming=[e for e in events if not e.passed],
        imminent=[
            e
            for e in events
            if not e.passed
            and e.kind == "deadline"
            and (e.days_away or 0) <= IMMINENT_DAYS
        ],
    )
    report.headline = _headline(report)
    report.notes = [
        "임의계속가입 신청 기한은 **첫 지역보험료 고지서의 납부기한에서 2개월까지**"
        "입니다. 여기 적힌 달은 고지서가 퇴직 다음 달에 나온다고 보고 계산한 것이라, "
        "실제 날짜는 고지서를 받으시면 확인하셔야 합니다.",
        "신청처와 준비물은 참고용입니다. 방문 전에 전화로 한 번 확인하시면 헛걸음을 "
        "줄이실 수 있습니다.",
    ]
    return report


def _headline(report: TimelineReport) -> str:
    if report.imminent:
        first = report.imminent[0]
        return (
            f"**{first.when}까지 {first.title}**을 하셔야 합니다. "
            "놓치면 되돌릴 수 없습니다."
        )
    if report.upcoming:
        first = report.upcoming[0]
        return f"다음 일정은 {first.when} **{first.title}**입니다."
    return "지금 기준으로 다가오는 일정이 없습니다."


def changes_since(
    profile: UserProfile,
    last_seen: _dt.date,
    *,
    today: Optional[_dt.date] = None,
    assumptions: Optional[Assumptions] = None,
) -> TimelineReport:
    """지난 상담 이후 무엇이 다가왔는가 (기능 ⑨의 정직한 축소판).

    푸시·이메일은 인프라가 필요하다. 하지만 **다시 오셨을 때 알려주는 것**은 저장된
    프로파일과 오늘 날짜만 있으면 된다. 이 서비스는 이미 "2031년부터 연 80만원",
    "신청 기한은 고지서 +2개월" 같은 것을 계산해 놓고 **알려줄 방법이 없었다.**
    """
    today = today or _dt.date.today()
    report = build_timeline(profile, today=today, assumptions=assumptions)

    # 지난번에는 아직 안 왔는데 지금은 지나간 것
    newly_passed = [
        e
        for e in report.events
        if e.passed and _months_between(e.year, e.month, last_seen) >= 0
    ]
    gap_days = (today - last_seen).days

    if newly_passed:
        missed = [e for e in newly_passed if e.kind == "deadline"]
        if missed:
            # 문의처는 그 항목의 것을 쓴다. 퇴직금 결정을 놓쳤는데 건보공단으로
            # 안내하면 헛걸음을 시킨다.
            where = missed[0].where or "해당 기관"
            report.headline = (
                f"지난 상담({last_seen}) 이후 **{missed[0].title}** 기한이 "
                f"지났습니다. 아직 안 하셨다면 {where} 에 확인해 보세요."
            )
        else:
            report.headline = (
                f"지난 상담 이후 **{newly_passed[0].title}**이(가) 지났습니다. "
                "계획을 다시 보실 때가 되었습니다."
            )
    elif report.imminent:
        first = report.imminent[0]
        report.headline = (
            f"{gap_days}일 만에 오셨습니다. **{first.when}까지 {first.title}**을 "
            "하셔야 합니다."
        )
    return report
