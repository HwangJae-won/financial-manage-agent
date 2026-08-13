"""계산 결과 설명 에이전트 (기능 ⑤·⑧).

core/ 가 만든 숫자를 시니어가 읽을 수 있는 말로 옮긴다. LLM 은 **문장만 쓰고
숫자는 만들지 않는다.** 이를 강제하기 위해 두 겹으로 막는다:

  1. 사전: 프롬프트에 계산 결과를 사실 목록으로 넣고, 그 안의 숫자만 쓰라고 지시
  2. 사후: 생성된 문장의 모든 숫자가 사실 목록에 있는지 기계적으로 검사

2번이 이 모듈의 핵심이다. 프롬프트 지시는 지켜지지 않을 수 있지만 검사는 확실하다.
기획서 ⑧ "AI 금융정보 신뢰 검증"을 가장 값싸고 확실하게 구현한 형태다.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.llm import NUMERIC_GUARDRAIL, LLMClient, get_client
from core.asset_map import AssetMap
from core.formatting import fmt_krw, fmt_months, fmt_pct, fmt_years
from core.models import SimulationResult, UserProfile
from core.montecarlo import MonteCarloResult
from core.risk_score import RiskScore

BRIEFING_SYSTEM = f"""당신은 은퇴 자산관리를 돕는 상담사입니다.
55~64세 시니어에게 계산 결과를 설명합니다.

말투와 형식:
- 존댓말을 쓰고, 어려운 금융용어는 풀어서 씁니다.
- 짧은 문장으로 씁니다. 한 문장에 한 가지만 담습니다.
- 겁주지 않되 사실을 흐리지도 않습니다. 문제가 있으면 분명히 말합니다.
- 특정 금융상품이나 회사를 언급하지 않습니다.

절대 규칙:
- **아래 '계산 결과'에 있는 숫자만 쓰세요.** 없는 숫자는 문장에서 빼세요.
- 직접 계산하지 마세요. 더하기, 나누기, 비율 계산 모두 금지입니다.
- 계산 결과에 없는 수치가 필요하면 그 부분을 언급하지 마세요.

{NUMERIC_GUARDRAIL}"""

BRIEFING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "지금 상황을 한 문장으로 요약. 40자 이내",
        },
        "situation": {
            "type": "string",
            "description": "현재 자산과 소득공백기 상황 설명. 3~4문장",
        },
        "priority": {
            "type": "string",
            "description": "지금 가장 먼저 챙겨야 할 것과 그 이유. 2~3문장",
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "구체적인 다음 행동 2~3가지. 각 한 문장",
        },
    },
    "required": ["headline", "situation", "priority", "next_steps"],
    "additionalProperties": False,
}


class Briefing(BaseModel):
    """설명 결과."""

    headline: str
    situation: str
    priority: str
    next_steps: list[str]

    unverified_numbers: list[str] = Field(
        default_factory=list,
        description="계산 결과에 없는데 문장에 등장한 숫자. 비어 있어야 정상이다.",
    )

    @property
    def is_verified(self) -> bool:
        return not self.unverified_numbers

    def full_text(self) -> str:
        steps = "\n".join(f"- {s}" for s in self.next_steps)
        return f"{self.headline}\n\n{self.situation}\n\n{self.priority}\n\n{steps}"


# --------------------------------------------------------------------------- #
# 사실 목록 만들기
# --------------------------------------------------------------------------- #


def build_facts(
    profile: UserProfile,
    sim: SimulationResult,
    asset_map: AssetMap,
    score: Optional[RiskScore] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
) -> dict[str, str]:
    """LLM 에게 넘길 계산 결과. **이 안의 숫자만 문장에 쓸 수 있다.**"""
    facts: dict[str, str] = {
        "총자산": fmt_krw(profile.total_assets),
        "금융자산(부동산 제외)": fmt_krw(profile.financial_assets),
        "바로 쓸 수 있는 돈": fmt_krw(profile.liquid_assets),
        "부동산": fmt_krw(profile.real_estate),
        "월 생활비": fmt_krw(profile.monthly_expense),
        "소득공백기": fmt_months(sim.income_gap_months),
        "국민연금 개시 연도": f"{sim.pension_start_year}년",
        "국민연금 월 수령 예상액": fmt_krw(profile.national_pension_monthly),
        "연금·기타소득 없이 생활비를 감당할 수 있는 기간": fmt_years(
            sim.expense_coverage_years
        ),
        "국민연금 개시 시점의 남은 금융자산": fmt_krw(sim.balance_at_pension_start),
    }

    if sim.depletion_age is not None:
        facts["금융자산이 바닥나는 시점"] = f"{sim.depletion_year}년, 만 {sim.depletion_age}세"
    else:
        facts["금융자산이 바닥나는 시점"] = f"만 {sim.horizon_age}세까지 바닥나지 않음"

    facts["소득공백기에 필요한 생활비 총액"] = fmt_krw(asset_map.income_gap_need)
    if asset_map.gap_shortfall > 0:
        facts["소득공백기 생활비 부족액"] = fmt_krw(asset_map.gap_shortfall)
    else:
        facts["소득공백기 생활비 부족액"] = "없음 (필요한 만큼 확보되어 있음)"

    for bucket in asset_map.buckets:
        if bucket.amount > 0:
            facts[f"{bucket.label} 금액"] = fmt_krw(bucket.amount)

    if score is not None:
        facts["은퇴 재무 안정도 점수"] = f"{score.total}점 (100점 만점, {score.status})"
        facts["가장 취약한 항목"] = f"{score.weakest.label} ({score.weakest.score:.0f}점)"
        if score.cap is not None:
            facts["점수 상한이 걸린 이유"] = score.cap_reason or ""

    if monte_carlo is not None:
        facts["10년 뒤 자산이 남아있을 가능성"] = fmt_pct(
            monte_carlo.survival_after_years(10), precision=0
        )
        facts["20년 뒤 자산이 남아있을 가능성"] = fmt_pct(
            monte_carlo.survival_after_years(20), precision=0
        )
        facts["80세 이전에 자산이 바닥날 가능성"] = fmt_pct(
            monte_carlo.prob_depleted_before_age(80), precision=0
        )

    if profile.assumed_fields:
        facts["사용자가 답하지 않아 가정한 항목 수"] = f"{len(profile.assumed_fields)}개"

    return facts


def facts_block(facts: dict[str, str]) -> str:
    return "\n".join(f"- {label}: {value}" for label, value in facts.items())


# --------------------------------------------------------------------------- #
# 숫자 검증 — 이 모듈의 핵심
# --------------------------------------------------------------------------- #

# 금융적으로 의미 있는 수치만 검사한다. 단위가 붙었거나 두 자리 이상인 숫자.
_MEANINGFUL_NUMBER = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(억|만원|만|원|세|년|개월|%|점|퍼센트)?"
)

# 순서를 세는 표현("세 가지", "1단계")은 사실이 아니므로 검사 대상이 아니다.
_SAFE_SMALL = {str(n) for n in range(0, 10)}


def _numbers_in(text: str) -> set[str]:
    """검사 대상 숫자를 뽑는다. 쉼표는 제거해 표기 차이를 흡수한다."""
    found: set[str] = set()
    for raw, unit in _MEANINGFUL_NUMBER.findall(text):
        digits = raw.replace(",", "")
        if not digits or digits in (".",):
            continue
        if not unit and digits in _SAFE_SMALL:
            continue  # 단위 없는 한 자리 수는 순서 표현으로 본다
        found.add(digits)
    return found


def verify_numbers(text: str, facts: dict[str, str]) -> list[str]:
    """문장에 등장한 숫자 중 계산 결과에 없는 것을 돌려준다.

    비어 있으면 정상 — LLM 이 숫자를 지어내지 않았다는 뜻이다.
    """
    allowed = _numbers_in(" ".join(facts.values()))
    used = _numbers_in(text)
    return sorted(used - allowed, key=lambda s: (len(s), s))


# --------------------------------------------------------------------------- #
# 에이전트
# --------------------------------------------------------------------------- #


def explain(
    profile: UserProfile,
    sim: SimulationResult,
    asset_map: AssetMap,
    *,
    score: Optional[RiskScore] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
    client: Optional[LLMClient] = None,
) -> Briefing:
    """계산 결과를 브리핑으로 옮긴다. 생성 후 숫자를 검증해 결과에 표시한다."""
    client = client or get_client()
    facts = build_facts(profile, sim, asset_map, score, monte_carlo)

    prompt = (
        "아래는 이 분의 은퇴 자산을 계산한 결과입니다.\n\n"
        f"[계산 결과]\n{facts_block(facts)}\n\n"
        "이 결과를 바탕으로 브리핑을 작성하세요. "
        "위 목록에 있는 숫자만 사용하고, 없는 숫자는 문장에서 빼세요."
    )

    data = client.structured(
        prompt, BRIEFING_SCHEMA, system=BRIEFING_SYSTEM, max_tokens=1500
    )
    briefing = Briefing.model_validate(data)
    briefing.unverified_numbers = verify_numbers(briefing.full_text(), facts)
    return briefing
