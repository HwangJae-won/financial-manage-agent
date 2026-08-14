"""계산 엔진을 에이전트의 도구로 노출한다.

W1 에서 "core/ 는 LLM 도 UI 도 모른다"고 정한 것의 두 번째 배당금이다. `core/` 가
전부 순수 함수라서, 도구화는 인자 모델을 붙이고 출력을 줄이는 일이 전부다.

**LLM 은 인자만 고른다.** 숫자는 여전히 `core/` 에서만 나온다. 에이전트가 자율적으로
움직이는 것처럼 보여도 환각 통제 구조는 그대로다 — 오히려 강해진다. 지금까지는
"프롬프트에 넣은 사실 목록"이 인용 가능 범위였는데, 이제는 "실제로 실행된 계산의
출력"이 그 범위가 된다.

설계 원칙 셋:

1. **프로파일은 도구가 들고 있는다.** 모델이 사용자 자산 전체를 인자로 넘기게 하면
   거기서 숫자를 지어낼 여지가 생긴다. 모델은 "무엇을 바꿔볼지"만 말한다.

2. **오류는 예외가 아니라 결과로 돌려준다.** 잘못된 인자로 대화가 끊기면 안 된다.
   무엇이 잘못됐는지 모델에게 알려주면 다시 시도할 수 있다.

3. **출력은 작게.** 40년치 시뮬레이션 행을 통째로 돌려주면 토큰만 먹고 모델이
   길을 잃는다. 결론에 해당하는 값만, 표시 문자열과 함께 돌려준다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.llm import ToolCall, ToolResult, ToolSpec
from agents.fraud import analyze_message
from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_years
from core.models import LifeEvent, UserProfile
from core.policy import DEFAULT_POLICY_KEY, analyze_policy_impact, load_policy_catalog
from core.prescribe import prescribe
from core.risk_score import compute_risk_score
from core.health_insurance import analyze_health_insurance
from core.national_pension import analyze_national_pension
from core.downsizing import analyze_downsizing
from core.medical_cost import analyze_medical_cost
from core.timeline import build_timeline
from core.working_income import analyze_working_income
from core.severance import compare_severance_options
from core.sensitivity import analyze_sensitivity

# 금액 인자의 상한. 모델이 자릿수를 잘못 세는 일이 있어서 막아 둔다.
MAX_WON = 10_000_000_000


# --------------------------------------------------------------------------- #
# 인자 모델 — JSON 스키마와 검증을 한 벌로 얻는다
# --------------------------------------------------------------------------- #


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulateArgs(_Args):
    """무엇을 바꿔서 다시 계산해 볼지. 전부 생략하면 현재 계획 그대로 계산한다."""

    monthly_expense: Optional[int] = Field(
        default=None, gt=0, le=MAX_WON, description="바꿔볼 월 생활비(원)"
    )
    national_pension_start_age: Optional[int] = Field(
        default=None, ge=55, le=75, description="바꿔볼 국민연금 수령 개시 연령"
    )
    other_monthly_income: Optional[int] = Field(
        default=None, ge=0, le=MAX_WON, description="바꿔볼 기타 월소득(임대·근로 등, 원)"
    )
    one_off_expense_amount: Optional[int] = Field(
        default=None,
        gt=0,
        le=MAX_WON,
        description="한 번에 나가는 큰 지출 금액(자녀 결혼자금·의료비 등, 원)",
    )
    one_off_expense_year: Optional[int] = Field(
        default=None, ge=2000, le=2100, description="그 지출이 발생하는 연도"
    )
    one_off_expense_label: str = Field(
        default="큰 지출", max_length=40, description="지출 이름 (예: 자녀 결혼자금)"
    )

    @model_validator(mode="after")
    def _amount_and_year_go_together(self) -> "SimulateArgs":
        """금액만 주고 연도를 빼면 언제 나가는 돈인지 알 수 없다.

        조용히 기본값을 넣으면 모델이 의도하지 않은 연도로 계산된 답을 그대로
        말하게 된다. 오류로 돌려주면 모델이 다시 물어보거나 채워 넣는다.
        """
        if (self.one_off_expense_amount is None) != (self.one_off_expense_year is None):
            raise ValueError(
                "one_off_expense_amount 와 one_off_expense_year 는 함께 지정해야 합니다"
            )
        return self


class PrescribeArgs(_Args):
    target_age: Optional[int] = Field(
        default=None, ge=60, le=110, description="목표 나이. 생략하면 95세"
    )


class SensitivityArgs(_Args):
    pass


class PolicyArgs(_Args):
    policy_key: str = Field(
        default=DEFAULT_POLICY_KEY, description="정책 키. 목록은 도구 설명 참조"
    )


class SeveranceArgs(_Args):
    pension_years: Optional[int] = Field(
        default=None, ge=1, le=40, description="연금으로 나눠 받을 기간(년). 생략하면 10년"
    )


class HealthInsuranceArgs(_Args):
    monthly_salary: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description="퇴직 전 월 급여(원). 임의계속가입 보험료 계산에 쓴다. 모르면 생략",
    )
    property_tax_base: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description="재산세 고지서의 과세표준(원). 생략하면 부동산 시가에서 추정한다",
    )
    has_employed_family: Optional[bool] = Field(
        default=None,
        description="직장에 다니는 배우자·자녀가 있는지. 모르면 생략",
    )


class NationalPensionArgs(_Args):
    contributed_months: Optional[int] = Field(
        default=None,
        ge=0,
        le=600,
        description="국민연금 가입월수. 모르면 생략한다 — 추측해서 넣지 말 것",
    )
    catchup_months: Optional[int] = Field(
        default=None,
        ge=0,
        le=600,
        description="추납할 수 있는 개월 수(납부예외·적용제외 기간). 모르면 생략",
    )
    monthly_income_base: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description="기준소득월액(원). 생략하면 퇴직 전 월 급여를 쓴다",
    )


class WorkingIncomeArgs(_Args):
    monthly_income: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description="벌어볼 월 소득(원). 생략하면 여러 구간을 한 번에 비교한다",
    )


class MedicalCostArgs(_Args):
    annual_ceiling: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description="본인부담상한액(연, 원). 모르면 생략한다 — 추측해서 넣지 말 것",
    )


class DownsizingArgs(_Args):
    new_home_price: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_WON,
        description=(
            "옮겨 갈 집의 가격(원). 사용자가 정해 두었으면 넣고, 아니면 생략한다 — "
            "생략하면 여러 축소 폭을 한 번에 비교한다. **시세를 추측해서 넣지 말 것**"
        ),
    )


class TimelineArgs(_Args):
    pass


class MessageArgs(_Args):
    text: str = Field(
        min_length=1, max_length=10_000, description="사용자가 받은 문자·카톡 원문"
    )


# --------------------------------------------------------------------------- #
# 도구 구현 — 출력은 결론만
# --------------------------------------------------------------------------- #


def _plan_summary(
    profile: UserProfile, assumptions: Assumptions, label: str
) -> dict[str, Any]:
    sim = simulate(profile, assumptions=assumptions)
    score = compute_risk_score(profile, assumptions)
    return {
        "기준": label,
        "금융자산_고갈_나이": sim.depletion_age,
        "고갈_여부": "고갈되지 않음" if sim.depletion_age is None else f"만 {sim.depletion_age}세 고갈",
        "시뮬레이션_종료_나이": sim.horizon_age,
        "연금개시시점_잔액": fmt_krw(sim.balance_at_pension_start),
        "생활비_충당_가능_기간": fmt_years(sim.expense_coverage_years),
        "은퇴재무안정도": f"{score.total}점 ({score.status})",
    }


def _run_simulate(
    profile: UserProfile, assumptions: Assumptions, args: SimulateArgs
) -> dict[str, Any]:
    plain = ("monthly_expense", "national_pension_start_age", "other_monthly_income")
    overrides: dict[str, Any] = {
        key: getattr(args, key) for key in plain if getattr(args, key) is not None
    }
    if args.one_off_expense_amount is not None:
        overrides["life_events"] = [
            *profile.life_events,
            LifeEvent(
                year=args.one_off_expense_year,
                amount=args.one_off_expense_amount,
                label=args.one_off_expense_label,
            ),
        ]

    baseline = _plan_summary(profile, assumptions, "현재 계획")
    if not overrides:
        return {"현재_계획": baseline, "바꾼_것": "없음"}

    changed = profile.model_copy(update=overrides)
    after = _plan_summary(changed, assumptions, "바꾼 계획")

    labels = []
    if args.monthly_expense is not None:
        labels.append(
            f"월 생활비 {fmt_krw(profile.monthly_expense)} → {fmt_krw(args.monthly_expense)}"
        )
    if args.national_pension_start_age is not None:
        labels.append(
            f"국민연금 개시 만 {profile.national_pension_start_age}세 → "
            f"만 {args.national_pension_start_age}세"
        )
    if args.other_monthly_income is not None:
        labels.append(
            f"기타 월소득 {fmt_krw(profile.other_monthly_income)} → "
            f"{fmt_krw(args.other_monthly_income)}"
        )
    if args.one_off_expense_amount is not None:
        labels.append(
            f"{args.one_off_expense_label} {fmt_krw(args.one_off_expense_amount)} "
            f"({args.one_off_expense_year}년) 반영"
        )

    before_age = baseline["금융자산_고갈_나이"] or baseline["시뮬레이션_종료_나이"]
    after_age = after["금융자산_고갈_나이"] or after["시뮬레이션_종료_나이"]

    return {
        "바꾼_것": " / ".join(labels),
        "현재_계획": baseline,
        "바꾼_계획": after,
        "고갈_시점_변화": f"{after_age - before_age:+d}년",
    }


def _run_prescribe(
    profile: UserProfile, assumptions: Assumptions, args: PrescribeArgs
) -> dict[str, Any]:
    plan = prescribe(profile, target_age=args.target_age, assumptions=assumptions)
    return {
        "목표_나이": plan.target_age,
        "현재_고갈_나이": plan.baseline_depletion_age,
        "요약": plan.summary,
        "선택지": [
            {
                "방법": option.label,
                "목표_달성_가능": option.feasible,
                "한_문장": option.headline,
                "해야_할_일": option.change_label,
                "필요한_값": option.required_value,
                "늘어나는_기간": f"{option.gained_years}년",
                "주의": option.caution or None,
            }
            for option in plan.options
        ],
    }


def _run_sensitivity(
    profile: UserProfile, assumptions: Assumptions, args: SensitivityArgs
) -> dict[str, Any]:
    report = analyze_sensitivity(profile, assumptions)
    return {
        "요약": report.summary,
        "가정별_흔들림": [
            {
                "가정": case.label,
                "흔들림": f"{case.swing_years}년",
                "설명": case.detail,
            }
            for case in report.cases
        ],
        "한계": report.caveat,
    }


def _run_policy(
    profile: UserProfile, assumptions: Assumptions, args: PolicyArgs
) -> dict[str, Any]:
    impact = analyze_policy_impact(
        profile, policy_key=args.policy_key, assumptions=assumptions
    )
    return {
        "제도": impact.policy.title,
        "확정여부": impact.policy.badge,
        "출처": impact.policy.trust_note,
        "미확정_고지": impact.policy.notice or None,
        "고객님께_해당되는가": impact.applicable,
        "이유": impact.reason,
        "한_문장": impact.headline,
        "연간_절감액": fmt_krw(impact.annual_tax_saving) if impact.applicable else None,
        "절감액_상한": (
            fmt_krw(impact.max_annual_saving) if impact.max_annual_saving else None
        ),
        "결론": impact.verdict,
    }


def _run_severance(
    profile: UserProfile, assumptions: Assumptions, args: SeveranceArgs
) -> dict[str, Any]:
    result = compare_severance_options(
        profile, pension_years=args.pension_years, assumptions=assumptions
    )
    return {
        "퇴직금": result.severance_label,
        "근속연수": result.years_employed,
        "일시금": {
            "세금": result.lump_sum.total_tax_label,
            "실효세율": result.lump_sum.effective_rate_label,
            "납부시점": result.lump_sum.when_paid,
            "고갈": result.lump_sum.depletion_label,
        },
        "연금": {
            "세금": result.pension.total_tax_label,
            "실효세율": result.pension.effective_rate_label,
            "납부시점": result.pension.when_paid,
            "고갈": result.pension.depletion_label,
            "수령기간": f"{result.pension_years}년",
        },
        "줄어드는_세금": result.tax_saved_label,
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _run_health_insurance(
    profile: UserProfile, assumptions: Assumptions, args: HealthInsuranceArgs
) -> dict[str, Any]:
    result = analyze_health_insurance(
        profile,
        monthly_salary=args.monthly_salary,
        property_tax_base_override=args.property_tax_base,
        has_employed_family=args.has_employed_family,
        assumptions=assumptions,
    )
    return {
        "지금_피부양자인가": result.qualifies_at_retirement,
        "탈락_시점": (
            f"{result.cliff_year}년 (만 {result.cliff_age}세)"
            if result.cliff_year is not None
            else "시뮬레이션 기간 안에는 없음"
        ),
        "탈락_이유": result.cliff_reason,
        "합산소득": result.counted_income_label,
        "한도까지_여유": result.income_headroom_label,
        "금융소득": result.financial_income_label,
        "금융소득_계단까지_여유": result.financial_headroom_label,
        "지역가입자": {
            "월": result.local.monthly_label,
            "연": result.local.annual_label,
            "근거": result.local.basis,
        },
        "임의계속가입": {
            "가능": result.voluntary.available,
            "월": result.voluntary.monthly_label,
            "근거": result.voluntary.basis,
        },
        "평생_부담": result.total_premiums_label,
        "고갈_변화": f"{result.depletion_age_before} → {result.depletion_age_after}",
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _option_payload(option) -> dict[str, Any]:
    """수단 하나를 모델이 인용할 수 있는 형태로. 못 쓰는 수단은 이유만 남긴다."""
    if not option.available:
        return {"가능": False, "이유": option.unavailable_reason}
    return {
        "가능": True,
        "더_내는_기간": f"{option.months_added}개월",
        "내는_돈": option.cost_label,
        "월_연금_증가": option.monthly_gain_label,
        "평생_더_받는_금액": option.lifetime_gain_label,
        "추가_건강보험료": option.extra_premium_label,
        "빼고_남는_것": option.net_gain_label,
        "본전_시점": option.breakeven_label,
        "피부양자_탈락_변화": (
            f"{option.cliff_shift_years}년 앞당겨짐"
            if option.cliff_shift_years > 0
            else "달라지지 않음"
        ),
    }


def _run_national_pension(
    profile: UserProfile, assumptions: Assumptions, args: NationalPensionArgs
) -> dict[str, Any]:
    result = analyze_national_pension(
        profile,
        contributed_months=args.contributed_months,
        catchup_months=args.catchup_months,
        monthly_income_base=args.monthly_income_base,
        assumptions=assumptions,
    )
    if not result.computable:
        return {
            "계산_가능": False,
            "한_문장": result.headline,
            "결론": result.verdict,
            "주의": result.notes,
        }
    return {
        "계산_가능": True,
        "가입월수": result.contributed_months,
        "수급자격": "있음" if result.qualifies_now else f"{result.months_to_qualify}개월 부족",
        "자격까지_드는_돈": result.cost_to_qualify_label,
        "지금_월_연금": result.current.monthly_pension_label,
        "추납": _option_payload(result.catchup),
        "임의계속가입": _option_payload(result.voluntary),
        "둘_다": _option_payload(result.both),
        "가장_나은_선택": result.best_label,
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _run_working_income(
    profile: UserProfile, assumptions: Assumptions, args: WorkingIncomeArgs
) -> dict[str, Any]:
    levels = (args.monthly_income,) if args.monthly_income else None
    result = analyze_working_income(profile, levels=levels, assumptions=assumptions)
    return {
        "연금이_안_깎이는_상한": result.free_ceiling_label,
        "감액_적용": result.reduction_applies,
        "감액_구간": result.reduction_window,
        "소득별_결과": [
            {
                "월_소득": o.monthly_income_label,
                "연금_감액": o.pension_cut_label,
                "건강보험료": o.health_premium_label,
                "손에_남는_것": o.net_label,
                "고갈": o.depletion_label,
            }
            for o in result.outcomes
        ],
        "처방_대비_실제_필요액": (
            f"{result.prescribed_income_label} → {result.actual_needed_income_label}"
            if result.gap
            else "차이 없음"
        ),
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _run_medical_cost(
    profile: UserProfile, assumptions: Assumptions, args: MedicalCostArgs
) -> dict[str, Any]:
    result = analyze_medical_cost(
        profile, annual_ceiling=args.annual_ceiling, assumptions=assumptions
    )
    return {
        "상한액_알고_있나": result.ceiling_known,
        "상한액": result.ceiling_label or "모름",
        "의료비별_영향": [
            {
                "상황": s.label,
                "병원비": s.total_cost_label,
                "실제_부담": s.out_of_pocket_label,
                "공단_환급": s.refund_label,
                "고갈": s.depletion_label,
                "앞당겨지는_햇수": s.years_lost,
            }
            for s in result.scenarios
        ],
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _run_downsizing(
    profile: UserProfile, assumptions: Assumptions, args: DownsizingArgs
) -> dict[str, Any]:
    result = analyze_downsizing(
        profile, new_home_price=args.new_home_price, assumptions=assumptions
    )
    if not result.computable:
        return {
            "계산_가능": False,
            "이유": result.reason,
            "한_문장": result.headline,
            "결론": result.verdict,
            "주의": result.notes,
        }
    return {
        "계산_가능": True,
        "지금_집": result.current_home_label,
        "총자산_중_비중": result.home_share_label,
        "지금_고갈": (
            "고갈되지 않음"
            if result.baseline_depletion_age is None
            else f"만 {result.baseline_depletion_age}세"
        ),
        "선택지": [
            {
                "옮길_집": o.new_home_label,
                "집값_차액": o.gross_difference_label,
                "거래비용": o.costs.total_label,
                "비용_내역": {
                    "취득세": f"{o.costs.acquisition_tax_label} ({o.costs.acquisition_tax_rate_label})",
                    "중개보수_매도": o.costs.brokerage_sell_label,
                    "중개보수_매수": o.costs.brokerage_buy_label,
                    "등기·법무": o.costs.registration_label,
                    "이사비": o.costs.moving_label,
                },
                "손에_남는_돈": o.net_proceeds_label,
                "고갈": o.depletion_label,
                "밀리는_햇수": o.years_gained,
                "건강보험_영향": o.health_effect,
            }
            for o in result.options
        ],
        "한_문장": result.headline,
        "결론": result.verdict,
        "주의": result.notes,
    }


def _run_timeline(
    profile: UserProfile, assumptions: Assumptions, args: TimelineArgs
) -> dict[str, Any]:
    result = build_timeline(profile, assumptions=assumptions)
    return {
        "일정": [
            {
                "시점": e.when,
                "나이": f"만 {e.age}세",
                "할_일": e.title,
                "기한인가": e.kind == "deadline",
                "어디서": e.where,
                "준비물": e.what,
                "지났나": e.passed,
            }
            for e in result.events
        ],
        "임박한_기한": [e.title for e in result.imminent],
        "한_문장": result.headline,
        "주의": result.notes,
    }


def _run_message(
    profile: UserProfile, assumptions: Assumptions, args: MessageArgs
) -> dict[str, Any]:
    result = analyze_message(args.text, profile=profile)
    return {
        "위험도": result.risk_level.value,
        "요약": result.summary,
        "걸린_신호": [
            {"항목": s.label, "심각도": s.severity.value, "걸린_표현": s.evidence}
            for s in result.signals
        ],
        "언급된_제도": [
            {"제도": p.title, "확정여부": p.badge, "주의": p.caution}
            for p in result.policy_checks
        ],
        "고객님_자산과의_관계": result.profile_notes,
    }


def _sum_simulate(args: SimulateArgs, out: dict[str, Any]) -> str:
    if out["바꾼_것"] == "없음":
        return f"현재 계획 확인 → {out['현재_계획']['고갈_여부']}"
    return f"{out['바꾼_것']} → {out['바꾼_계획']['고갈_여부']} ({out['고갈_시점_변화']})"


def _sum_prescribe(args: PrescribeArgs, out: dict[str, Any]) -> str:
    가능 = sum(1 for o in out["선택지"] if o["목표_달성_가능"])
    return f"만 {out['목표_나이']}세 목표로 역산 → 가능한 방법 {가능}개"


def _sum_sensitivity(args: SensitivityArgs, out: dict[str, Any]) -> str:
    top = out["가정별_흔들림"][0]
    return f"가정별 민감도 → {top['가정']}이(가) 가장 큼 ({top['흔들림']})"


def _sum_policy(args: PolicyArgs, out: dict[str, Any]) -> str:
    return f"{out['제도']} {out['확정여부']} → {out['한_문장']}"


def _sum_severance(args: SeveranceArgs, out: dict[str, Any]) -> str:
    return (
        f"퇴직금 {out['퇴직금']} 수령 방식 비교 → 일시금 세금 {out['일시금']['세금']} / "
        f"연금 {out['연금']['세금']}"
    )


def _sum_health_insurance(args: HealthInsuranceArgs, out: dict[str, Any]) -> str:
    return (
        f"퇴직 후 건강보험료 → 피부양자 탈락 {out['탈락_시점']} · "
        f"지역가입자 월 {out['지역가입자']['월']}"
    )


def _sum_national_pension(args: NationalPensionArgs, out: dict[str, Any]) -> str:
    if not out["계산_가능"]:
        return "국민연금 추가납부 검토 → 가입월수를 몰라 계산하지 않음"
    return (
        f"국민연금 추가납부 검토 → 수급자격 {out['수급자격']} · "
        f"가장 나은 선택 {out['가장_나은_선택']}"
    )


def _sum_working_income(args: WorkingIncomeArgs, out: dict[str, Any]) -> str:
    return (
        f"재취업 소득 검토 → 월 {out['연금이_안_깎이는_상한']}까지 연금 감액 없음 · "
        f"구간 {len(out['소득별_결과'])}개 비교"
    )


def _sum_medical_cost(args: MedicalCostArgs, out: dict[str, Any]) -> str:
    return (
        f"의료비 충격 검토 → 상한액 {out['상한액']} · "
        f"{len(out['의료비별_영향'])}개 규모 비교"
    )


def _sum_downsizing(args: DownsizingArgs, out: dict[str, Any]) -> str:
    if not out.get("계산_가능"):
        return "주택 다운사이징 검토 → 부동산 평가액 미확인"
    return (
        f"주택 다운사이징 검토 → 지금 집 {out['지금_집']} · "
        f"선택지 {len(out['선택지'])}개 비교"
    )


def _sum_timeline(args: TimelineArgs, out: dict[str, Any]) -> str:
    return (
        f"은퇴 일정 확인 → {len(out['일정'])}개 · "
        f"임박한 기한 {len(out['임박한_기한'])}건"
    )


def _sum_message(args: MessageArgs, out: dict[str, Any]) -> str:
    return f"받은 메시지 확인 → 위험도 {out['위험도']} · 신호 {len(out['걸린_신호'])}개"


# --------------------------------------------------------------------------- #
# 등록
# --------------------------------------------------------------------------- #


@dataclass
class ToolRun:
    """도구를 한 번 실행한 기록. 화면의 trace 가 이것을 그대로 그린다.

    에이전트가 자율적으로 움직여도 **무엇을 어떤 인자로 불러 무슨 값이 나왔는지**가
    남는다. 사기 탐지가 '걸린 표현'을 그대로 보여주는 것과 같은 원리다.
    """

    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """도구 하나 — 모델에게 알려줄 정의와 실제 실행 함수."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[UserProfile, Assumptions, Any], dict[str, Any]]
    summarize: Callable[[Any, dict[str, Any]], str] = lambda args, out: ""

    @property
    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(
            name=self.name, description=self.description, input_schema=schema
        )


# 순서가 의미를 갖는다. 첫 번째 도구는 인자 없이 불러도 안전해야 한다 —
# mock 의 기본 동작이 첫 도구를 빈 인자로 부르기 때문이다(현재 계획 조회가 된다).
TOOLS: tuple[Tool, ...] = (
    Tool(
        name="simulate_plan",
        description=(
            "은퇴 계획을 다시 계산한다. 생활비·국민연금 개시 연령·기타 월소득을 바꿔보거나, "
            "자녀 결혼자금처럼 한 번에 나가는 큰 지출을 넣어보고 금융자산이 언제 바닥나는지 "
            "확인한다. **인자를 모두 생략하면 현재 계획 그대로 계산해 기준값을 준다.** "
            "'생활비를 50만원 줄이면?', '내후년에 결혼자금 5천만원이 나가면?' 같은 질문에 쓴다."
        ),
        args_model=SimulateArgs,
        handler=_run_simulate,
        summarize=_sum_simulate,
    ),
    Tool(
        name="prescribe",
        description=(
            "목표 나이까지 자산을 유지하려면 무엇을 얼마나 바꿔야 하는지 역산한다. "
            "'어떻게 해야 하나요', '얼마나 줄여야 하나요' 같은 질문에 쓴다. "
            "생활비·연금 개시 시기·추가 수입 세 가지 방법을 각각 계산해 준다."
        ),
        args_model=PrescribeArgs,
        handler=_run_prescribe,
        summarize=_sum_prescribe,
    ),
    Tool(
        name="sensitivity",
        description=(
            "계산에 쓴 가정이 틀렸을 때 결과가 얼마나 달라지는지 본다. "
            "'이 결과를 믿어도 되나요', '물가가 오르면요' 같은 질문에 쓴다."
        ),
        args_model=SensitivityArgs,
        handler=_run_sensitivity,
        summarize=_sum_sensitivity,
    ),
    Tool(
        name="policy_impact",
        description=(
            "발표되거나 시행된 제도가 이 사용자에게 실제로 얼마인지 계산한다. "
            "출처와 확정 단계(발표/입법예고/국회통과/공포/시행)를 함께 돌려준다. "
            "'ISA 세제가 바뀐다던데요' 같은 질문에 쓴다. policy_key 기본값은 isa_reform."
        ),
        args_model=PolicyArgs,
        handler=_run_policy,
        summarize=_sum_policy,
    ),
    Tool(
        name="severance_options",
        description=(
            "퇴직금을 일시금으로 받을 때와 IRP 등으로 연금 수령할 때를 비교한다. "
            "퇴직소득세는 근속연수공제가 커서 장기근속자는 실효세율이 낮고, 연금으로 "
            "받으면 30%(11년차부터 40%) 감면된다. '퇴직금 일시금으로 받을까요' 같은 "
            "질문에 쓴다."
        ),
        args_model=SeveranceArgs,
        handler=_run_severance,
        summarize=_sum_severance,
    ),
    Tool(
        name="health_insurance_cliff",
        description=(
            "퇴직 후 건강보험료가 언제 얼마나 생기는지 계산한다. 직장 다니는 가족의 "
            "피부양자로 들어가면 0원이지만, 합산소득이 연 2,000만원을 넘으면 지역가입자가 "
            "되어 연 수백만원이 생긴다. 특히 금융소득은 연 1,000만원을 1원이라도 넘으면 "
            "전액이 합산되는 절벽이다. '퇴직하면 건강보험료 얼마 나오나요', '예금 금리 "
            "높은 데로 옮길까요' 같은 질문에 쓴다."
        ),
        args_model=HealthInsuranceArgs,
        handler=_run_health_insurance,
        summarize=_sum_health_insurance,
    ),
    Tool(
        name="national_pension_options",
        description=(
            "국민연금을 더 낼지 검토한다. 60세 이후에도 65세까지 계속 내는 임의계속가입과, "
            "납부예외였던 기간의 보험료를 나중에 내는 추납(최대 119개월)을 각각 계산한다. "
            "가입기간이 120개월에 못 미치면 노령연금이 평생 0원이라는 점, 연금을 늘리면 "
            "건강보험 피부양자에서 더 빨리 탈락한다는 점을 함께 돌려준다. "
            "'국민연금 더 낼까요', '추납하는 게 이득인가요' 같은 질문에 쓴다."
        ),
        args_model=NationalPensionArgs,
        handler=_run_national_pension,
        summarize=_sum_national_pension,
    ),
    Tool(
        name="working_income",
        description=(
            "퇴직 후 일해서 소득이 생기면 손에 얼마가 남는지 계산한다. 60~65세에는 "
            "소득이 많으면 노령연금이 깎이고(국민연금법 제63조의2), 건강보험 피부양자에서 "
            "탈락해 보험료가 생긴다. 다만 **A값+200만원까지는 한 푼도 깎이지 않는다.** "
            "'퇴직하고 일하면 연금 깎이나요', '월 200만원 벌면 얼마 남나요' 같은 질문에 쓴다."
        ),
        args_model=WorkingIncomeArgs,
        handler=_run_working_income,
        summarize=_sum_working_income,
    ),
    Tool(
        name="medical_cost",
        description=(
            "의료비가 은퇴 계획을 얼마나 흔드는지 계산한다. 건강보험 **본인부담상한제**가 "
            "있어 1년 본인부담금이 상한을 넘으면 초과분을 공단이 돌려주므로, "
            "'암 걸리면 수억'은 대부분 사실이 아니다. 다만 비급여·간병비는 상한에 "
            "포함되지 않는다. '병원비 많이 나오면 어쩌죠', '암보험 들어야 하나요' 같은 "
            "질문에 쓴다. 상한액은 소득분위별로 다르므로 **모르면 생략한다.**"
        ),
        args_model=MedicalCostArgs,
        handler=_run_medical_cost,
        summarize=_sum_medical_cost,
    ),
    Tool(
        name="housing_downsizing",
        description=(
            "집을 줄였을 때 **실제로 손에 남는 돈**과 계획 변화를 계산한다. 차액이 그대로 "
            "들어오지 않는다 — 새 집 취득세(지방세법 제11조), 매도·매수 중개보수, 등기비, "
            "이사비가 먼저 나간다. 재산이 줄면 건강보험 피부양자 재산요건도 함께 완화된다. "
            "'집 줄일까요', '집 팔면 어떻게 되나요', '작은 데로 이사하면' 같은 질문에 쓴다. "
            "양도소득세는 반영하지 않는다."
        ),
        args_model=DownsizingArgs,
        handler=_run_downsizing,
        summarize=_sum_downsizing,
    ),
    Tool(
        name="retirement_timeline",
        description=(
            "언제 무엇을 해야 하는지 순서대로 알려준다. 퇴직금 수령 방식 결정, "
            "건강보험 임의계속가입 신청 기한, 국민연금 개시, 피부양자 탈락 시점을 "
            "실제 연월로 뽑고 **어디서 어떻게 신청하는지**까지 준다. "
            "'언제까지 해야 하나요', '뭐부터 해야 하죠', '어디서 신청하나요' 같은 질문에 쓴다."
        ),
        args_model=TimelineArgs,
        handler=_run_timeline,
        summarize=_sum_timeline,
    ),
    Tool(
        name="check_message",
        description=(
            "사용자가 받은 문자·카톡이 금융사기나 불완전판매인지 확인한다. "
            "받은 내용을 그대로 text 에 넣는다. 어떤 표현이 왜 걸렸는지와 "
            "사용자의 실제 보유 자산과의 관계를 함께 돌려준다."
        ),
        args_model=MessageArgs,
        handler=_run_message,
        summarize=_sum_message,
    ),
)


class Toolbox:
    """프로파일에 묶인 도구 모음.

    모델은 "무엇을 바꿔볼지"만 말하고, 사용자 자산은 여기서 들고 있는다. 프로파일을
    인자로 받게 만들면 모델이 거기서 숫자를 지어낼 여지가 생긴다.

    실행 결과는 `outputs` 에 쌓인다. 다음 단계(A7)에서 "모델이 쓴 숫자가 실제로
    계산된 값인가"를 검사할 때 이 기록이 인용 가능 범위가 된다.
    """

    def __init__(
        self,
        profile: UserProfile,
        *,
        assumptions: Optional[Assumptions] = None,
        tools: tuple[Tool, ...] = TOOLS,
    ):
        self.profile = profile
        self.assumptions = assumptions or load_assumptions()
        self.tools = {tool.name: tool for tool in tools}
        self.runs: list[ToolRun] = []

    @property
    def outputs(self) -> list[dict[str, Any]]:
        """성공한 호출의 출력만. 이것이 곧 '인용 가능한 숫자'의 범위가 된다."""
        return [run.output for run in self.runs if run.ok]

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """도구를 실행한다. **어떤 경우에도 예외를 밖으로 내지 않는다.**

        잘못된 인자로 대화가 끊기면 안 된다. 무엇이 잘못됐는지 모델에게 돌려주면
        다시 시도할 수 있고, 그 과정 자체가 화면의 trace 에 남는다.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            return self._error(
                call, f"'{call.name}' 은(는) 없는 도구입니다.", 사용가능=list(self.tools)
            )

        try:
            args = tool.args_model.model_validate(call.arguments or {})
        except ValidationError as exc:
            return self._error(
                call,
                "인자가 올바르지 않습니다. 아래 내용을 고쳐 다시 시도하세요.",
                문제=[
                    f"{'.'.join(str(p) for p in e['loc']) or '(전체)'}: {e['msg']}"
                    for e in exc.errors()
                ],
            )

        try:
            output = tool.handler(self.profile, self.assumptions, args)
        except Exception as exc:  # 엔진이 터져도 대화는 끊지 않는다
            return self._error(call, f"계산 중 문제가 발생했습니다: {exc}")

        try:
            summary = tool.summarize(args, output)
        except Exception:  # 요약 실패가 계산 결과를 버리게 두지 않는다
            summary = f"{tool.name} 실행"

        self.runs.append(
            ToolRun(
                name=call.name,
                arguments=dict(call.arguments or {}),
                ok=True,
                summary=summary,
                output=output,
            )
        )
        return ToolResult(call_id=call.id, content=_dump(output))

    def _error(self, call: ToolCall, message: str, **extra: Any) -> ToolResult:
        payload = {"오류": message, **extra}
        self.runs.append(
            ToolRun(
                name=call.name,
                arguments=dict(call.arguments or {}),
                ok=False,
                summary=message,
            )
        )
        return ToolResult(call_id=call.id, content=_dump(payload), is_error=True)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def policy_keys() -> list[str]:
    """등록된 정책 키 목록. 시스템 프롬프트에 넣어 모델이 고를 수 있게 한다."""
    return [fact.key for fact in load_policy_catalog()]
