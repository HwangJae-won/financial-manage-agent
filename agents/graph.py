"""Supervisor — 사용자 입력을 어느 에이전트로 보낼지 정하고, 실제로 보낸다.

목적지가 둘 이상이 된 시점(사기 탐지 추가)에 만들었다. 하나뿐일 때 만들었다면
순수 오버헤드였을 것이다.

**규칙이 먼저, LLM 은 애매할 때만.** 사기 탐지 요청은 구조적으로 알아볼 수 있다 —
남이 보낸 메시지를 통째로 붙여넣기 때문에 길고, 광고성 표현이 들어 있다.
이런 건 LLM 을 부를 이유가 없다. 빠르고, 공짜고, 왜 그렇게 분류했는지 설명된다.

분류(`classify_intent`)와 배달(`Supervisor`)을 나눠 두었다. 분류는 순수 함수라
테스트하기 쉽고, 배달은 에이전트 셋을 쥐고 있어야 해서 객체다.

**같은 말이라도 대화가 어디까지 왔느냐에 따라 갈 곳이 다르다.** "생활비를 줄이면
어때요?"는 프로파일링 중이라면 답변이지만, 결과가 나온 뒤라면 다시 계산해 달라는
요청이다. 그래서 배달 단계에서 진행 상태를 함께 본다.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.advisor import Advisor, AdvisorReply
from agents.fraud import FraudAssessment, analyze_message, detect_signals
from agents.llm import LLMClient, get_client
from agents.profiling import ProfilingAgent
from core.models import UserProfile

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


# --------------------------------------------------------------------------- #
# 배달 — 분류한 곳으로 실제로 보낸다 (A5)
# --------------------------------------------------------------------------- #


class SupervisorReply(BaseModel):
    """한 번의 입력에 대한 응답과, **어디로 갔는지**.

    라우팅 결과를 응답에 담아 보내는 이유는 화면에서 보여주기 위해서다. 사용자가
    사기 문자를 붙여넣었는데 프로파일링 질문이 돌아오면 서비스가 고장난 것처럼
    보인다. 어디로 갔고 왜 그랬는지가 보이면 그게 설명이 된다.
    """

    text: str
    intent: Intent
    reason: str
    by_rule: bool = True
    kind: str = Field(description="question / advice / fraud — 화면이 어떻게 그릴지")

    advice: Optional[AdvisorReply] = None
    fraud: Optional[FraudAssessment] = None
    done: bool = False
    ready: bool = Field(
        default=False, description="결과를 보여줘도 되는 상태인가"
    )


NOT_READY_YET = (
    "그 질문은 계산이 끝난 다음에 정확히 답해 드릴 수 있습니다.\n"
    "몇 가지만 더 여쭙고 바로 정리해 드릴게요."
)


class Supervisor:
    """대화의 단일 창구. 사용자가 무엇을 말하든 여기로 들어와 여기서 갈린다.

    웹 세션 하나 = Supervisor 하나. 프로파일링이 끝나면 그 프로파일로 상담
    에이전트를 만들어 이후 질문을 그쪽으로 넘긴다.
    """

    def __init__(
        self,
        *,
        client: Optional[LLMClient] = None,
        today: Optional[_dt.date] = None,
    ):
        self.client = client or get_client()
        self.profiler = ProfilingAgent(client=self.client, today=today)
        self._advisor: Optional[Advisor] = None
        # 프로파일링이 끝난 뒤의 주고받음. 프로파일링 대화는 profiler 가 들고 있으므로
        # 여기에는 그 이후 것만 쌓인다. 화면은 둘을 이어 붙인 하나의 대화를 본다.
        self._after: list[dict[str, str]] = []

    # -- 상태 ------------------------------------------------------------- #

    def start(self) -> str:
        return self.profiler.start()

    @property
    def messages(self) -> list[dict[str, str]]:
        """화면에 그리는 대화 전체 — 프로파일링과 그 이후를 이어 붙인 것.

        사용자에게는 하나의 대화다. 어느 에이전트가 답했는지는 중간에 갈리지만
        말풍선은 끊기지 않아야 한다.
        """
        return [*(self.profiler.state.get("messages") or []), *self._after]

    @property
    def followups(self) -> list[dict[str, str]]:
        """프로파일링이 끝난 뒤의 주고받음만.

        화면에 따라 둘을 붙여 보이기도 하고(웹의 말풍선 하나짜리 대화창),
        나눠 보이기도 한다(Streamlit 은 결과가 사이에 끼어 있다).
        """
        return list(self._after)

    @property
    def progress(self) -> tuple[int, int]:
        return self.profiler.progress

    def _record(self, question: str, answer: str) -> None:
        self._after.append({"role": "user", "content": question})
        self._after.append({"role": "assistant", "content": answer})

    @property
    def done(self) -> bool:
        return self.profiler.done

    @property
    def ready(self) -> bool:
        return self.profiler.ready

    def build_profile(self) -> UserProfile:
        """완성된 프로파일. 아직이면 ValueError — 호출자가 400 으로 옮긴다."""
        return self.profiler.build_profile()

    def profile(self) -> Optional[UserProfile]:
        """완성된 프로파일. 아직이면 None — 예외로 흐름을 끊지 않는다."""
        try:
            return self.build_profile()
        except ValueError:
            return None

    def advisor(self) -> Optional[Advisor]:
        """상담 에이전트. 프로파일이 준비된 다음에만 생긴다.

        한 번 만들면 계속 쓴다. 대화 이력을 들고 있어야 "그럼 60만원은요?" 같은
        후속 질문이 이어지기 때문이다.
        """
        if self._advisor is None:
            profile = self.profile()
            if profile is not None:
                self._advisor = Advisor(profile, client=self.client)
        return self._advisor

    # -- 배달 ------------------------------------------------------------- #

    def send(self, text: str) -> SupervisorReply:
        """입력 하나를 분류하고 해당 에이전트로 보낸다."""
        routing = self._resolve(classify_intent(text, client=self.client))

        if routing.intent is Intent.FRAUD_CHECK:
            return self._to_fraud(text, routing)
        if routing.intent is Intent.RESULT:
            return self._to_advisor(text, routing)
        return self._to_profiler(text, routing)

    def _resolve(self, routing: Routing) -> Routing:
        """진행 상태를 반영해 목적지를 정한다.

        **결과가 나온 뒤의 평범한 질문은 프로파일링이 아니라 상담이다.** 이 한 줄이
        없으면 대화를 마친 사용자가 무슨 말을 해도 아무 일도 일어나지 않는다.

        바꾼 결과를 `Routing` 에 다시 담아 돌려주는 이유는 화면 때문이다. 분류는
        'profiling' 인데 상담 답변이 나오면 표시와 실제가 어긋난다 — trace 를
        보여주는 화면(A4)에서 바로 눈에 띈다.
        """
        if routing.intent is Intent.FRAUD_CHECK:
            return routing
        if self.done and routing.intent is not Intent.RESULT:
            return Routing(
                intent=Intent.RESULT,
                reason=f"{routing.reason} 계산이 끝나 상담으로 이어갑니다.",
                by_rule=routing.by_rule,
            )
        if not self.done and routing.intent is Intent.RESULT:
            return routing  # 아직 답할 수 없다 — _to_advisor 가 진행 상황을 말한다
        return routing

    def _base(self, routing: Routing, kind: str, text: str) -> dict:
        return {
            "text": text,
            "intent": routing.intent,
            "reason": routing.reason,
            "by_rule": routing.by_rule,
            "kind": kind,
            "done": self.done,
            "ready": self.ready,
        }

    def _to_profiler(self, text: str, routing: Routing) -> SupervisorReply:
        answer = self.profiler.respond(text)
        return SupervisorReply(**self._base(routing, "question", answer))

    def _to_fraud(self, text: str, routing: Routing) -> SupervisorReply:
        assessment = analyze_message(text, profile=self.profile())
        self._record(text, assessment.summary)
        return SupervisorReply(
            **self._base(routing, "fraud", assessment.summary), fraud=assessment
        )

    def _to_advisor(self, text: str, routing: Routing) -> SupervisorReply:
        advisor = self.advisor()
        if advisor is None:
            # 결과를 묻는데 아직 계산할 수 없다. 되묻지 않고 진행 상황을 말한다.
            self._record(text, NOT_READY_YET)
            return SupervisorReply(**self._base(routing, "question", NOT_READY_YET))

        reply = advisor.ask(text)
        self._record(text, reply.text)
        return SupervisorReply(
            **self._base(routing, "advice", reply.text), advice=reply
        )
