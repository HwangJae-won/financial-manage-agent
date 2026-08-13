"""키 없이 동작하는 슬롯 추출기.

MockClient 에 꽂으면 API 키 없이도 12턴짜리 프로파일링 대화가 끝까지 돌아간다.
용도는 세 가지다:
  - CI 에서 LLM 호출 없이 그래프 전체를 검증
  - 발표 당일 네트워크가 끊겼을 때의 안전망
  - LLM 추출 결과의 교차검증 기준

동작 원리는 단순하다. 프롬프트에 담긴 대화록에서 **마지막 상담사 질문**을 보고
어떤 슬롯을 묻고 있었는지 알아낸 뒤, **마지막 사용자 답변**에서 금액·연도를 뽑는다.
사용자가 먼저 말한 항목(“퇴직금 2억이고 예금 5천”)은 키워드로 잡는다.

LLM 을 대체하려는 게 아니다. 문맥 이해가 필요한 표현은 놓치고, 그건 정상이다.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.parsing import parse_korean_amount, parse_month, parse_year
from agents.slots import QUESTIONS
from core.models import RiskTolerance

# 답변에 특정 낱말이 있으면 그 슬롯으로 본다 (사용자가 먼저 말한 경우).
KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("severance_pay", ("퇴직금", "명예퇴직금", "명퇴금")),
    ("cash_savings", ("예금", "적금", "예적금", "저축")),
    ("national_pension_monthly", ("국민연금", "노령연금")),
    ("isa", ("ISA", "isa", "아이에스에이")),
    ("equity", ("주식", "펀드", "ETF", "etf")),
    ("pension_dc", ("IRP", "irp", "퇴직연금", "개인연금", "DC")),
    ("real_estate", ("부동산", "아파트", "집", "주택", "빌라")),
    ("other_monthly_income", ("월세", "임대", "아르바이트", "알바")),
    ("debt", ("대출", "빚", "융자", "담보")),
    ("monthly_expense", ("생활비", "쓰고", "지출")),
)

# "없어요"는 0 이다. "모르겠어요"는 미상이므로 여기 넣으면 안 된다 —
# 0(보유 안 함)과 null(아직 모름)이 섞이면 잘못된 시뮬레이션이 나간다.
NEGATIVE = ("없어", "없습니다", "없음", "안 해", "안해", "하나도")

MONTHLY_SLOTS = {"monthly_expense", "national_pension_monthly", "other_monthly_income"}

# 한 문장에 여러 항목을 말한 경우를 나눈다.
# "퇴직금은 2억, 예금은 5천" 에서 퇴직금이 5천까지 삼키지 않게 하기 위함.
_SEGMENT = re.compile(r"[,，]|그리고|이고요|이고|이며|있고")


def _last_line(transcript: str, speaker: str) -> str:
    lines = [
        line[len(speaker) + 1 :].strip()
        for line in transcript.splitlines()
        if line.startswith(f"{speaker}:")
    ]
    return lines[-1] if lines else ""


def _question_for(assistant_text: str):
    """상담사의 마지막 발화가 어떤 질문인지 찾는다."""
    for question in QUESTIONS:
        # 질문 문장 앞부분이 일치하면 그 질문으로 본다 (뒤에 안내문이 붙을 수 있음).
        head = question.text[:18]
        if head and head in assistant_text:
            return question
    return None


def _risk_from(text: str) -> Optional[str]:
    if any(word in text for word in ("원금", "안정", "지키", "보수", "잃고 싶지")):
        return RiskTolerance.CONSERVATIVE.value
    if any(word in text for word in ("공격", "수익", "적극", "감수", "위험 감수")):
        return RiskTolerance.AGGRESSIVE.value
    if any(word in text for word in ("중간", "적당", "반반", "보통")):
        return RiskTolerance.MODERATE.value
    return None


def extract_slots(transcript: str, today_year: int) -> dict[str, Any]:
    """대화록에서 슬롯을 뽑는다. 못 알아낸 항목은 넣지 않는다(= null)."""
    answer = _last_line(transcript, "사용자")
    if not answer:
        return {}

    asked = _question_for(_last_line(transcript, "상담사"))
    result: dict[str, Any] = {}
    denied = any(word in answer for word in NEGATIVE)

    # 1) 질문이 특정된 경우 — 그 질문이 채우는 슬롯을 먼저 시도한다.
    if asked is not None:
        if "birth_year" in asked.fills:
            year = parse_year(answer, current_year=today_year)
            if year is not None:
                result["birth_year"] = year
            month = parse_month(answer)
            if month is not None:
                result["birth_month"] = month
        elif "retirement_year" in asked.fills:
            year = parse_year(answer, current_year=today_year)
            if year is not None:
                result["retirement_year"] = year
            month = parse_month(answer)
            if month is not None:
                result["retirement_month"] = month
        elif "risk_tolerance" in asked.fills:
            risk = _risk_from(answer)
            if risk is not None:
                result["risk_tolerance"] = risk
        else:
            amount = parse_korean_amount(answer)
            target = asked.fills[0]
            if amount is not None:
                result[target] = _scale(target, amount)
            elif denied:
                for field in asked.fills:
                    result[field] = 0

    # 2) 사용자가 먼저 말한 항목 — 문장을 항목별로 쪼갠 뒤 각 조각에서 금액을 잡는다.
    for segment in _SEGMENT.split(answer):
        segment = segment.strip()
        if not segment:
            continue
        for slot, words in KEYWORDS:
            if slot in result:
                continue
            hit = next((w for w in words if w in segment), None)
            if hit is None:
                continue
            tail = segment[segment.find(hit) + len(hit) :]
            amount = parse_korean_amount(tail)
            if amount is not None:
                result[slot] = _scale(slot, amount)
            elif any(word in segment for word in NEGATIVE):
                result[slot] = 0
            break

    return result


def _scale(slot: str, amount: int) -> int:
    """월 단위 항목에 억 단위 값이 들어오면 파서가 과하게 읽은 것으로 본다.

    "생활비 300만원"은 3,000,000 이 맞지만, 구어체 보정이 겹쳐 3천억이 나올 수는 없다.
    안전장치일 뿐이며, 정상 입력에서는 아무 일도 하지 않는다.
    """
    if slot in MONTHLY_SLOTS and amount >= 100_000_000:
        return amount // 10_000
    return amount


def korean_slot_extractor(today_year: int):
    """MockClient(structured_handler=...) 에 꽂을 핸들러를 만든다."""

    def handler(prompt: str, schema: dict[str, Any], system: Optional[str]) -> dict[str, Any]:
        transcript = prompt
        match = re.search(r"대화 내용:\n(.*?)\n\n", prompt, re.S)
        if match:
            transcript = match.group(1)
        extracted = extract_slots(transcript, today_year)
        # 스키마의 모든 필드를 채우되, 모르는 것은 null
        return {field: extracted.get(field) for field in schema.get("properties", {})}

    return handler
