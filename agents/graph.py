"""Supervisor — 사용자 입력을 어느 에이전트로 보낼지 정한다.

목적지가 둘 이상이 된 시점(사기 탐지 추가)에 만들었다. 하나뿐일 때 만들었다면
순수 오버헤드였을 것이다.

**규칙이 먼저, LLM 은 애매할 때만.** 사기 탐지 요청은 구조적으로 알아볼 수 있다 —
남이 보낸 메시지를 통째로 붙여넣기 때문에 길고, 광고성 표현이 들어 있다.
이런 건 LLM 을 부를 이유가 없다. 빠르고, 공짜고, 왜 그렇게 분류했는지 설명된다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from agents.fraud import detect_signals
from agents.llm import LLMClient

# 붙여넣은 메시지로 보는 길이 기준. 대화형 답변("2억 정도요")은 이보다 훨씬 짧다.
PASTED_MESSAGE_LENGTH = 80


class Intent(str, Enum):
    PROFILING = "profiling"
    FRAUD_CHECK = "fraud_check"
    RESULT = "result"
    OTHER = "other"


class Routing(BaseModel):
    """분류 결과. reason 이 있어야 왜 그리로 갔는지 설명할 수 있다."""

    intent: Intent
    reason: str
    by_rule: bool = True


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "reason": {"type": "string", "description": "그렇게 판단한 이유 한 문장"},
    },
    "required": ["intent", "reason"],
    "additionalProperties": False,
}

CLASSIFY_SYSTEM = """당신은 은퇴 자산 상담 서비스의 안내 담당입니다.
사용자의 말이 어디로 가야 하는지만 판단합니다.

- profiling: 본인의 자산·소득·퇴직 시점 등 정보를 말하고 있다
- fraud_check: 남에게 받은 투자 권유나 상품 안내가 믿을 만한지 묻고 있다
- result: 이미 나온 분석 결과를 다시 보거나 더 설명해 달라고 한다
- other: 위 어디에도 해당하지 않는다"""

# 사기 확인 요청에서 흔한 표현
_FRAUD_ASKS = (
    "이거 사기",
    "사기인가",
    "사기 아닌",
    "믿어도",
    "진짜인가",
    "진짜예요",
    "괜찮은가",
    "괜찮을까",
    "이런 문자",
    "이런 연락",
    "받았는데",
    "왔는데",
    "확인해 주",
    "확인해줘",
    "봐 주세요",
    "봐주세요",
)

_RESULT_ASKS = (
    "다시 보",
    "결과",
    "다시 설명",
    "무슨 뜻",
    "왜 그런",
    "자세히",
)


def classify_intent(
    text: str, *, client: Optional[LLMClient] = None
) -> Routing:
    """입력을 어디로 보낼지 정한다.

    규칙으로 확실히 판단되면 LLM 을 부르지 않는다. client 가 없으면 규칙만 쓴다.
    """
    stripped = text.strip()
    if not stripped:
        return Routing(intent=Intent.OTHER, reason="빈 입력입니다.")

    # 1) 붙여넣은 메시지에 광고성 신호가 있으면 확실하다.
    signals = detect_signals(stripped)
    if signals and len(stripped) >= PASTED_MESSAGE_LENGTH:
        return Routing(
            intent=Intent.FRAUD_CHECK,
            reason=f"광고성 표현({signals[0].label})이 있는 긴 메시지입니다.",
        )

    # 2) 명시적으로 확인을 요청하는 말투
    if any(ask in stripped for ask in _FRAUD_ASKS):
        return Routing(
            intent=Intent.FRAUD_CHECK, reason="받은 연락이 믿을 만한지 묻고 있습니다."
        )

    # 3) 신호가 여러 개면 길이가 짧아도 확인 대상으로 본다
    if len(signals) >= 2:
        return Routing(
            intent=Intent.FRAUD_CHECK, reason="위험 신호가 두 개 이상 확인되었습니다."
        )

    if any(ask in stripped for ask in _RESULT_ASKS):
        return Routing(intent=Intent.RESULT, reason="분석 결과에 대해 묻고 있습니다.")

    # 4) 규칙으로 애매하면 그때 LLM 에 물어본다
    if client is not None:
        try:
            data = client.structured(
                f"사용자의 말: {stripped}", INTENT_SCHEMA, system=CLASSIFY_SYSTEM, max_tokens=200
            )
            intent = Intent(data.get("intent") or Intent.PROFILING.value)
            return Routing(
                intent=intent,
                reason=data.get("reason") or "LLM 이 분류했습니다.",
                by_rule=False,
            )
        except Exception:
            pass  # 분류 실패로 대화를 끊지 않는다 — 기본 경로로 보낸다

    return Routing(
        intent=Intent.PROFILING, reason="본인의 정보를 말하는 것으로 봅니다."
    )
