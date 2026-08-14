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
    ("last_monthly_salary", ("월급", "급여", "보수월액", "월 소득")),
)

# "없어요"는 0 이다. "모르겠어요"는 미상이므로 여기 넣으면 안 된다 —
# 0(보유 안 함)과 null(아직 모름)이 섞이면 잘못된 시뮬레이션이 나간다.
NEGATIVE = ("없어", "없습니다", "없음", "안 해", "안해", "하나도")

MONTHLY_SLOTS = {
    "monthly_expense",
    "national_pension_monthly",
    "other_monthly_income",
    "last_monthly_salary",
}

# 한 문장에 여러 항목을 말한 경우를 나눈다.
# "퇴직금은 2억, 예금은 5천" 에서 퇴직금이 5천까지 삼키지 않게 하기 위함.
_SEGMENT = re.compile(r"[,，]|그리고|이고요|이고|이며|있고")

# "32년 다녔습니다" 처럼 근속연수를 말하는 표현. 퇴직소득세의 핵심 변수라
# 금액과 같은 문장에 섞여 와도 놓치면 안 된다.
_SERVICE_YEARS = re.compile(
    r"(\d{1,2})\s*년\s*(?:정도\s*)?(?:간\s*|동안\s*)?(?:다녔|다니|근무|재직|일했|몸담)"
)

# 국민연금 가입기간. "32년 넣었습니다" / "가입 기간은 30년" / "384개월 냈어요".
# 119개월과 120개월이 '연금 0원'과 '평생 연금'을 가르므로 개월까지 받는다.
# 근속연수와 표현이 겹쳐서(둘 다 "N년") 납부를 뜻하는 동사나 '가입' 이라는 말이
# 붙은 경우에만 잡는다. 겹치면 근속연수로 오인해 엉뚱한 세금이 계산된다.
_PENSION_PERIOD = re.compile(
    r"(?:(\d{1,3})\s*개월|(\d{1,2})\s*년)\s*(?:정도\s*|쯤\s*)?(?:간\s*|동안\s*)?"
    r"(?:넣었|넣고|부었|붓고|냈|내고|납부|가입)"
)
_PENSION_PERIOD_PREFIX = re.compile(
    r"가입\s*(?:기간|한\s*기간)?[은는이가]?\s*(?:(\d{1,3})\s*개월|(\d{1,2})\s*년)"
)


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


def _pension_months(answer: str) -> Optional[int]:
    """국민연금 가입기간을 개월 수로. 못 알아내면 None."""
    for pattern in (_PENSION_PERIOD, _PENSION_PERIOD_PREFIX):
        match = pattern.search(answer)
        if match is None:
            continue
        months, years = match.group(1), match.group(2)
        if months is not None:
            return int(months)
        return int(years) * 12
    return None


def extract_slots(transcript: str, today_year: int) -> dict[str, Any]:
    """대화록에서 슬롯을 뽑는다. 못 알아낸 항목은 넣지 않는다(= null)."""
    answer = _last_line(transcript, "사용자")
    if not answer:
        return {}

    asked = _question_for(_last_line(transcript, "상담사"))
    result: dict[str, Any] = {}
    denied = any(word in answer for word in NEGATIVE)

    # 근속연수는 어느 질문에 대한 답에서든 나올 수 있다.
    service = _SERVICE_YEARS.search(answer)
    if service is not None:
        result["years_employed"] = int(service.group(1))

    # 국민연금 가입기간도 마찬가지다.
    pension_months = _pension_months(answer)
    if pension_months is not None:
        result["national_pension_months"] = pension_months

    # 금액을 읽기 전에 기간 표현을 떼어낸다. "2억 정도요, 32년 다녔습니다" 에서
    # 금액 파서가 32 를 금액의 일부로 삼켜 2억 32원이 되는 것을 막는다.
    money_text = _SERVICE_YEARS.sub(" ", answer)
    money_text = _PENSION_PERIOD.sub(" ", money_text)
    money_text = _PENSION_PERIOD_PREFIX.sub(" ", money_text)

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
            amount = parse_korean_amount(money_text)
            target = asked.fills[0]
            if amount is not None:
                result[target] = _scale(target, amount)
            elif denied:
                for field in asked.fills:
                    result[field] = 0

    # 2) 사용자가 먼저 말한 항목 — 문장을 항목별로 쪼갠 뒤 각 조각에서 금액을 잡는다.
    for segment in _SEGMENT.split(money_text):
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


# --------------------------------------------------------------------------- #
# 브리핑 — 키 없이도 결과 화면이 채워지도록
# --------------------------------------------------------------------------- #


def _fact(prompt: str, label: str) -> str:
    """프롬프트의 사실 목록에서 값을 꺼낸다."""
    match = re.search(rf"^- {re.escape(label)}: (.+)$", prompt, re.M)
    return match.group(1).strip() if match else ""


def briefing_from_facts(prompt: str, schema: dict[str, Any], system: Optional[str]) -> dict[str, Any]:
    """계산 결과를 그대로 문장에 끼워 넣는 결정론적 브리핑.

    LLM 흉내를 내려는 게 아니라, 사실 목록의 값만 쓰기 때문에 숫자 검증을
    항상 통과한다. 발표 당일 LLM 이 막혔을 때 그대로 화면에 띄울 수 있다.
    """
    gap = _fact(prompt, "소득공백기")
    coverage = _fact(prompt, "연금·기타소득 없이 생활비를 감당할 수 있는 기간")
    depletion = _fact(prompt, "금융자산이 바닥나는 시점")
    shortfall = _fact(prompt, "소득공백기 생활비 부족액")
    total = _fact(prompt, "총자산")
    liquid = _fact(prompt, "바로 쓸 수 있는 돈")
    pension_year = _fact(prompt, "국민연금 개시 연도")
    weakest = _fact(prompt, "가장 취약한 항목")

    has_shortfall = shortfall and not shortfall.startswith("없음")

    if has_shortfall:
        headline = "소득공백기 생활비 확보가 가장 급합니다."
        priority = (
            f"지금 가장 중요한 것은 생활비를 담을 유동자산 확보입니다. "
            f"소득공백기 생활비가 {shortfall} 부족합니다. "
            "투자보다 먼저 이 부분을 채우시는 것이 좋습니다."
        )
    elif "바닥나지 않음" not in depletion:
        headline = "단기는 안정적이지만 장기 대비가 필요합니다."
        priority = (
            f"소득공백기는 넘기실 수 있습니다. 다만 현재 계획대로면 {depletion}에 "
            "금융자산이 바닥납니다. 장기적으로 쓸 수 있는 소득을 늘리거나 "
            "생활비를 조정하는 검토가 필요합니다."
        )
    else:
        headline = "현재 계획은 안정적으로 유지될 수 있습니다."
        priority = (
            "지금 계획대로면 자산이 오래 유지됩니다. "
            "무리한 변경보다는 현재 구성을 지키시는 것이 좋습니다."
        )

    situation = (
        f"현재 총자산은 {total}이고, 이 가운데 바로 쓸 수 있는 돈은 {liquid}입니다. "
        f"국민연금은 {pension_year}부터 받으시게 되어, 그때까지 {gap}의 소득공백기가 있습니다. "
        f"연금이나 다른 소득 없이 지금 자산만으로 생활하신다면 {coverage} 정도 가능합니다."
    )

    steps = [
        f"소득공백기 {gap} 동안 쓸 생활비를 언제든 찾을 수 있는 곳에 따로 두세요.",
        "국민연금 예상 수령액을 공단에서 다시 확인해 보세요.",
    ]
    if weakest:
        steps.append(f"가장 약한 부분은 {weakest}입니다. 이 항목부터 살펴보세요.")

    return {
        "headline": headline,
        "situation": situation,
        "priority": priority,
        "next_steps": steps,
    }


def demo_handler(today_year: int):
    """프로파일링과 브리핑을 한 클라이언트에서 처리하는 핸들러.

    프롬프트 모양을 보고 어느 쪽인지 판단한다. Streamlit 데모가 키 없이
    처음부터 끝까지 돌아가게 하는 것이 목적이다.
    """
    extractor = korean_slot_extractor(today_year)

    def handler(prompt: str, schema: dict[str, Any], system: Optional[str]) -> dict[str, Any]:
        if "[계산 결과]" in prompt:
            return briefing_from_facts(prompt, schema, system)
        return extractor(prompt, schema, system)

    return handler


# --------------------------------------------------------------------------- #
# 도구 선택 — 키 없이 도는 데모에서 "에이전트가 도구를 고른다"가 보이게
# --------------------------------------------------------------------------- #

# 질문에 담긴 말과 도구의 연결.
#
# mock 이 질문을 이해하는 척하려는 것이 아니다. 기본 동작은 **언제나 첫 번째
# 도구**를 부르는 것이라, 키 없이 시연하면 "국민연금을 더 낼까요?"에도 실행
# 기록에 simulate_plan 이 찍힌다. 화면에 trace 를 띄우기 시작한 이상(A4) 그
# 장면은 에이전트가 질문을 못 알아듣는 것처럼 보인다.
TOOL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retirement_timeline", ("언제까지", "언제 해", "뭐부터", "순서", "일정", "어디서 신청", "신청 방법", "기한")),
    ("housing_downsizing", ("집을 줄", "집 줄", "집 팔", "집을 팔", "이사", "다운사이징", "평수", "작은 집", "주택 처분")),
    ("medical_cost", ("의료비", "병원비", "암", "아프", "치료비", "간병", "입원", "수술")),
    ("working_income", ("일하면", "재취업", "일자리", "벌면", "취업", "알바", "소득이 생기")),
    ("national_pension_options", ("국민연금", "추납", "임의계속", "가입기간")),
    ("health_insurance_cliff", ("건강보험", "건보료", "피부양자", "지역가입자")),
    ("severance_options", ("퇴직금", "일시금", "IRP", "연금으로 받")),
    ("policy_impact", ("제도", "세법", "ISA", "개편", "바뀐다")),
    ("sensitivity", ("믿어도", "가정", "틀리면", "정확한가")),
    ("prescribe", ("어떻게 해야", "얼마나 줄", "유지하려면", "까지 가려면", "방법")),
    ("simulate_plan", ("줄이면", "늘리면", "바꾸면", "어떻게 되")),
)


def pick_tool(question: str, names: list[str]) -> Optional[str]:
    """질문에 맞는 도구 이름. 못 고르면 None."""
    for name, words in TOOL_KEYWORDS:
        if name in names and any(word in question for word in words):
            return name
    return None


def demo_tool_handler():
    """MockClient(tool_handler=...) 에 꽂을 핸들러.

    질문에 맞는 도구를 한 번 부르고, 결과를 받으면 그것으로 답한다.
    도구를 고르지 못하면 기본 동작(첫 번째 도구)에 맡긴다.
    """
    from agents.llm import ToolCall, ToolTurn, _default_mock_turn

    def handler(turns, tools):
        started = max(
            (i for i, turn in enumerate(turns) if turn.role == "user" and turn.text),
            default=-1,
        )
        segment = turns[started + 1 :]
        if not tools or any(turn.tool_results for turn in segment):
            return _default_mock_turn(turns, tools)

        question = turns[started].text if started >= 0 else ""
        chosen = pick_tool(question, [tool.name for tool in tools])
        if chosen is None:
            return _default_mock_turn(turns, tools)

        return ToolTurn(
            tool_calls=[ToolCall(id="mock-call-1", name=chosen, arguments={})],
            provider="mock",
            model="mock",
            stop_reason="tool_use",
        )

    return handler
