"""도구를 쥔 상담 에이전트 (A3).

지금까지의 LLM 호출은 전부 단발이었다. 질문을 받아 한 번 답하고 끝난다. 그래서
"생활비를 50만원 줄이면 어때요?" 같은 질문에 답할 수 없었다 — 그 답은 계산을
다시 돌려야 나오기 때문이다.

이 모듈이 그 루프다:

    사용자: "생활비 50만원만 줄이면 어때요?"
      → 모델: simulate_plan(monthly_expense=2_500_000)
      → 엔진: 만 73세 고갈
      → 모델: prescribe(target_age=95)          ← 스스로 한 번 더 판단
      → 엔진: 월 187만원 필요
      → 답변: "만 73세로 4년 늘어납니다. 95세까지 가시려면 187만원이 필요합니다."

**LLM 은 어떤 계산을 할지만 정한다.** 숫자는 전부 `core/` 에서 나온다. 에이전트가
자율적으로 움직이지만 환각 통제 구조는 그대로다.

세 가지를 반드시 지킨다:

  1. **반복 상한.** 모델이 도구를 무한히 부를 수 있으면 비용도 대기 시간도 통제할
     수 없다. 상한에 닿으면 도구 없이 한 번 더 물어 답을 받아낸다 — 빈손으로
     끝내지 않는다.
  2. **실패해도 대화를 끊지 않는다.** 네트워크가 끊기든 모델이 거절하든, 지금까지
     계산한 것을 담아 안내 문구를 돌려준다. 발표 중에 화면이 죽으면 안 된다.
  3. **한 일을 전부 남긴다.** 어떤 도구를 어떤 인자로 불러 무슨 값이 나왔는지가
     `trace` 에 쌓인다. 이것을 화면에 그대로 보여주는 것이 이 기능의 핵심이다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from agents.llm import (
    NUMERIC_GUARDRAIL,
    LLMClient,
    LLMRefusal,
    ToolTurn,
    Turn,
    get_client,
)
from agents.explain import numbers_not_in
from agents.tools import Toolbox, ToolRun, policy_keys
from core.assumptions import Assumptions
from core.models import UserProfile

# 도구를 몇 번까지 부르게 둘 것인가. 실제 질문은 두세 번이면 끝난다.
MAX_ROUNDS = 4

SYSTEM = f"""당신은 은퇴 자산관리를 돕는 상담사입니다.
55~64세 시니어와 대화하며, 이미 계산이 끝난 계획에 대해 이어서 답합니다.

**숫자는 절대 직접 만들지 마세요.** 금액·연도·나이·비율이 필요하면 반드시 도구를
불러서 계산하고, 도구가 돌려준 값만 인용하세요. 더하기 빼기도 직접 하지 마세요.
필요한 값을 도구로 얻을 수 없으면 그 부분은 말하지 마세요.

사용자의 자산 정보는 도구가 이미 알고 있습니다. 자산을 인자로 넘기지 마세요.
현재 상태가 필요하면 simulate_plan 을 인자 없이 부르면 됩니다.

도구를 고르는 법:
- "○○하면 어떻게 되나요" → simulate_plan 으로 그 값을 넣어 계산
- "어떻게 해야 하나요", "얼마나 줄여야 하나요" → prescribe
- "이 결과 믿어도 되나요", "가정이 틀리면요" → sensitivity
- 제도·세법 이야기 → policy_impact (사용 가능한 키: {", ".join(policy_keys())})
- 받은 문자·카톡을 확인해 달라 → check_message

말투와 태도:
- 존댓말로, 짧은 문장으로 씁니다. 한 문장에 한 가지만 담습니다.
- 어려운 금융용어는 풀어서 씁니다.
- 겁주지 않되 사실을 흐리지 않습니다. 문제가 있으면 분명히 말합니다.
- 특정 금융상품이나 회사를 권하지 않습니다.
- 확정되지 않은 제도는 반드시 "아직 확정되지 않았다"고 함께 말합니다.

{NUMERIC_GUARDRAIL}"""

# 도구 없이 마지막으로 한 번 더 물을 때 덧붙이는 지시
WRAP_UP = (
    "\n\n지금까지 계산한 내용만으로 사용자에게 답해 주세요. "
    "더 계산할 수는 없으니, 확인된 값으로 말할 수 있는 것만 말하고 "
    "부족한 부분은 무엇을 더 확인해야 하는지 알려 주세요."
)

FALLBACK_TEXT = (
    "죄송합니다. 지금은 답변을 만들지 못했습니다. "
    "잠시 후 다시 여쭤봐 주시거나, 화면의 계산 결과를 함께 봐 주세요."
)

# 도구 출력에 없는 숫자를 말했을 때 모델에게 돌려주는 지시 (A7).
# 한 번은 스스로 고칠 기회를 준다. 그래도 남으면 답변을 내보내지 않는다.
CORRECTION = (
    "\n\n방금 답변에 계산 결과에 없는 숫자가 있습니다: {numbers}.\n"
    "이 숫자들을 빼고 다시 쓰세요. 도구가 돌려준 값만 인용하고, "
    "직접 더하거나 빼서 만든 숫자는 쓰지 마세요."
)

BLOCKED_PREFIX = (
    "확인되지 않은 숫자가 있어 답변을 그대로 전해 드리지 않았습니다. "
    "계산된 값만 아래에 정리했습니다.\n"
)


class TraceStep(BaseModel):
    """화면에 그대로 그리는 실행 기록 한 줄."""

    tool: str
    arguments: dict = Field(default_factory=dict)
    ok: bool
    summary: str
    output: dict = Field(default_factory=dict)

    @classmethod
    def from_run(cls, run: ToolRun) -> "TraceStep":
        return cls(
            tool=run.name,
            arguments=run.arguments,
            ok=run.ok,
            summary=run.summary,
            output=run.output,
        )


class AdvisorReply(BaseModel):
    """에이전트의 한 번의 답변."""

    text: str
    trace: list[TraceStep] = Field(default_factory=list)
    rounds: int = Field(default=0, description="도구를 부른 횟수")
    stop_reason: str = Field(
        default="answered",
        description="answered / max_rounds / llm_error / blocked",
    )
    unverified_numbers: list[str] = Field(
        default_factory=list,
        description="도구 출력에 없어 걸러낸 숫자. 비어 있으면 전부 계산에서 나온 값이다.",
    )

    @property
    def used_tools(self) -> bool:
        return bool(self.trace)

    @property
    def blocked(self) -> bool:
        """지어낸 숫자 때문에 원문을 내보내지 않았는가."""
        return self.stop_reason == "blocked"


class Advisor:
    """계산 결과를 두고 이어서 대화하는 에이전트.

    대화 이력을 들고 있으므로 "그럼 60만원은요?" 같은 후속 질문이 이어진다.
    프로파일링 에이전트와 마찬가지로 상태를 밖에서 들고 다닐 수 있게 만들었다.
    """

    def __init__(
        self,
        profile: UserProfile,
        *,
        client: Optional[LLMClient] = None,
        assumptions: Optional[Assumptions] = None,
        max_rounds: int = MAX_ROUNDS,
    ):
        self.client = client or get_client()
        self.toolbox = Toolbox(profile, assumptions=assumptions)
        self.max_rounds = max_rounds
        self.turns: list[Turn] = []

    # -- 진입점 ----------------------------------------------------------- #

    def ask(self, question: str) -> AdvisorReply:
        """질문 하나에 답한다. 필요하면 도구를 여러 번 부른다."""
        self.turns.append(Turn(role="user", text=question))
        before = len(self.toolbox.runs)

        for round_index in range(self.max_rounds):
            reply = self._converse(self.toolbox.specs)
            if reply is None:
                return self._fallback(before, rounds=round_index)

            self.turns.append(reply.as_turn())

            if not reply.wants_tools:
                return self._reply(reply.text, before, round_index, "answered")

            results = [self.toolbox.execute(call) for call in reply.tool_calls]
            self.turns.append(Turn(role="user", tool_results=results))

        # 상한에 닿았다. 도구를 빼고 한 번 더 물어 답을 받아낸다.
        final = self._converse([], system=SYSTEM + WRAP_UP)
        if final is None:
            return self._fallback(before, rounds=self.max_rounds)
        self.turns.append(final.as_turn())
        return self._reply(final.text, before, self.max_rounds, "max_rounds")

    # -- 내부 ------------------------------------------------------------- #

    def _converse(self, specs, *, system: str = SYSTEM) -> Optional[ToolTurn]:
        """모델을 한 번 부른다. 실패하면 None — 예외를 위로 올리지 않는다."""
        try:
            return self.client.converse(self.turns, specs, system=system)
        except LLMRefusal:
            return None
        except Exception:
            # 네트워크·인증·SDK 어떤 실패든 대화를 끊지 않는다.
            return None

    def _trace(self, before: int) -> list[TraceStep]:
        return [TraceStep.from_run(run) for run in self.toolbox.runs[before:]]

    def _quotable(self) -> str:
        """인용해도 되는 숫자의 출처 (A7).

        두 가지다. **지금까지의 도구 출력 전부** — 전부 `core/` 에서 나온 값이다.
        그리고 **사용자가 직접 말한 것** — "생활비 50만원 줄이면요?"에 답하면서
        50만원을 되뇌는 것까지 지어냈다고 볼 수는 없다.

        이번 질문의 출력만 보면 안 된다. "그럼 60만원은요?" 같은 후속 질문에서
        모델이 앞 턴의 계산 결과를 다시 인용하는 것이 정상이기 때문이다.
        """
        parts = [_flatten(output) for output in self.toolbox.outputs]
        parts += [turn.text for turn in self.turns if turn.role == "user" and turn.text]
        return " ".join(parts)

    def _reply(
        self, text: str, before: int, rounds: int, stop_reason: str
    ) -> AdvisorReply:
        trace = self._trace(before)
        if not text.strip():
            # 모델이 도구만 부르고 말을 안 하는 경우가 있다. 빈 답변을 내보내지 않는다.
            text = _text_from_trace(trace)

        unverified = numbers_not_in(text, self._quotable())
        if unverified:
            text, stop_reason = self._rewrite_or_block(text, trace, unverified)

        return AdvisorReply(
            text=text,
            trace=trace,
            rounds=rounds,
            stop_reason=stop_reason,
            unverified_numbers=unverified,
        )

    def _rewrite_or_block(
        self, text: str, trace: list[TraceStep], unverified: list[str]
    ) -> tuple[str, str]:
        """지어낸 숫자가 있으면 한 번 고쳐 쓰게 하고, 그래도 남으면 막는다.

        브리핑(`agents/explain.py`)은 표시만 하고 원문을 내보낸다. 여기서는 막는다.
        브리핑은 이미 화면에 있는 계산 결과를 옮겨 적는 것이라 사용자가 대조할 수
        있지만, 상담 답변은 **대조할 대상이 화면에 없다.** 틀린 숫자가 그대로
        사용자의 판단이 된다.
        """
        retry = self._converse(
            [], system=SYSTEM + CORRECTION.format(numbers=", ".join(unverified))
        )
        if retry is not None and retry.text.strip():
            still = numbers_not_in(retry.text, self._quotable())
            if not still:
                self.turns.append(retry.as_turn())
                return retry.text, "answered"

        # 고쳐 쓰지 못했다. 계산된 값만으로 만든 문장으로 대체한다.
        return BLOCKED_PREFIX + _text_from_trace(trace), "blocked"

    def _fallback(self, before: int, *, rounds: int) -> AdvisorReply:
        """모델 호출이 실패했을 때. 계산한 것이 있으면 그것만이라도 보여준다."""
        trace = self._trace(before)
        text = FALLBACK_TEXT
        if trace:
            text = (
                "답변을 정리하지는 못했지만, 계산한 내용은 아래와 같습니다.\n"
                + _text_from_trace(trace)
            )
        return AdvisorReply(
            text=text, trace=trace, rounds=rounds, stop_reason="llm_error"
        )


def _flatten(value) -> str:
    """중첩된 도구 출력을 숫자 검사용 한 줄로 편다.

    출력은 `{"임의계속가입": {"내는_돈": "2,665만원"}}` 처럼 겹쳐 있다. 겉만 보면
    안쪽 금액이 '인용 가능'에서 빠져 멀쩡한 답변이 막힌다.
    """
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _text_from_trace(trace: list[TraceStep]) -> str:
    """도구 실행 기록만으로 만든 답변.

    LLM 없이도 사용자가 빈 화면을 보지 않게 하는 마지막 방어선이다.
    여기 문장은 전부 계산 결과에서 나오므로 숫자 검증도 자동으로 통과한다.
    """
    lines = [f"- {step.summary}" for step in trace if step.ok]
    return "\n".join(lines) if lines else FALLBACK_TEXT
