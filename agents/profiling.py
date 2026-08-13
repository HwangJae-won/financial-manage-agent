"""AI 금융 프로파일링 에이전트 (기능 ①).

LangGraph 로 짜여 있지만 **흐름은 결정론적**이다. 그래프는
`추출 → 판단 → (질문 | 완료)` 세 노드뿐이고, 어떤 질문을 언제 할지는
agents/slots.py 의 우선순위가 정한다. LLM 이 하는 일은 사용자의 답을
구조화하는 것 하나다.

이렇게 나눈 이유:
  - 데모 중 대화가 엉뚱한 데로 새지 않는다.
  - LLM 없이(mock) 전체 흐름을 테스트할 수 있다.
  - 어떤 질문이 왜 나왔는지 설명할 수 있다.

상태를 밖에서 들고 다니는 stateless 설계다. Streamlit 의 session_state 나
웹 세션에 그대로 얹으면 되고, 체크포인터 설정 없이 테스트할 수 있다.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.llm import NUMERIC_GUARDRAIL, LLMClient, get_client
from agents.slots import (
    DEFAULTS,
    QUESTION_BY_KEY,
    Priority,
    QUESTIONS,
    all_important_answered,
    extraction_schema,
    is_ready,
    missing_mandatory,
    next_question,
    question_for_slot,
)
from core.assumptions import load_assumptions
from core.models import UserProfile

GREETING = (
    "안녕하세요. 퇴직 이후의 자산 계획을 함께 살펴보겠습니다.\n"
    "몇 가지만 여쭤보고, 지금 상황에서 무엇을 먼저 챙기셔야 할지 정리해 드릴게요.\n"
    "정확한 금액이 기억나지 않으시면 대략적으로 말씀해 주셔도 괜찮습니다."
)

EXTRACTION_SYSTEM = f"""당신은 은퇴 자산 상담의 접수 담당자입니다.
사용자의 답변에서 금융 정보를 추출해 JSON 으로 정리하는 일만 합니다.

규칙:
- 사용자가 **명시적으로 말한 것만** 채우세요. 언급하지 않은 항목은 반드시 null 로 두세요.
- 추측하지 마세요. "없어요", "하나도 없습니다" 라고 답한 항목만 0 으로 두세요.
- 금액은 원 단위 정수입니다. "2억" → 200000000, "300만원" → 3000000, "5천" → 50000000.
- 자산 대화에서 "5천", "3백" 처럼 만 단위를 생략하는 표현이 흔합니다. 문맥을 보고 판단하세요.
- 연도는 4자리입니다. "올해", "내년" 은 아래에 주어진 오늘 날짜를 기준으로 계산하세요.
- "66년생" 은 1966년생입니다.

{NUMERIC_GUARDRAIL}"""


class ProfilingState(TypedDict, total=False):
    """대화 한 턴의 상태. 밖에서 들고 다닌다."""

    messages: list[dict[str, str]]
    slots: dict[str, Any]
    asked: list[str]
    pending_question: Optional[str]
    done: bool
    ready: bool
    error: Optional[str]
    retries: int
    today: str


def new_state(today: Optional[_dt.date] = None) -> ProfilingState:
    """대화 시작 상태."""
    today = today or _dt.date.today()
    return ProfilingState(
        messages=[],
        slots={},
        asked=[],
        pending_question=None,
        done=False,
        ready=False,
        error=None,
        retries=0,
        today=today.isoformat(),
    )


# --------------------------------------------------------------------------- #
# 노드
# --------------------------------------------------------------------------- #


def _make_extract_node(client: LLMClient):
    def extract(state: ProfilingState) -> ProfilingState:
        messages = state.get("messages") or []
        user_turns = [m for m in messages if m["role"] == "user"]
        if not user_turns:
            return {}

        transcript = "\n".join(
            f"{'상담사' if m['role'] == 'assistant' else '사용자'}: {m['content']}"
            for m in messages[-6:]
        )
        prompt = (
            f"오늘 날짜: {state.get('today')}\n\n"
            f"대화 내용:\n{transcript}\n\n"
            "위 대화에서 사용자가 밝힌 금융 정보를 추출하세요. "
            "언급되지 않은 항목은 null 입니다."
        )

        extracted = client.structured(
            prompt, extraction_schema(), system=EXTRACTION_SYSTEM, max_tokens=1000
        )

        slots = dict(state.get("slots") or {})
        for key, value in (extracted or {}).items():
            if value is not None:
                slots[key] = value
        return {"slots": slots}

    return extract


# 필수 정보를 못 받았을 때 되묻는 횟수 상한.
# 없으면 "질문도 없고 완료도 안 되는" 상태로 대화가 멎는다.
MAX_RETRIES = 3


def decide(state: ProfilingState) -> ProfilingState:
    """다음 질문을 고르거나 완료로 넘긴다."""
    slots = state.get("slots") or {}
    asked = set(state.get("asked") or [])
    retries = int(state.get("retries") or 0)

    question = next_question(slots, asked)
    ready = is_ready(slots)

    if question is None:
        missing = missing_mandatory(slots)
        if not missing:
            return {"pending_question": None, "done": True, "ready": ready}

        # 물을 질문은 다 물었는데 필수 정보가 없다 — 되묻는다.
        if retries >= MAX_RETRIES:
            return {
                "pending_question": None,
                "done": False,
                "ready": False,
                "error": f"필수 정보가 없습니다: {', '.join(missing)}",
            }

        retry_question = question_for_slot(missing[0])
        if retry_question is None:  # 방어적 — 정의상 발생하지 않는다
            return {"pending_question": None, "done": False, "ready": False}

        return {
            "pending_question": retry_question.key,
            "asked": [k for k in (state.get("asked") or []) if k != retry_question.key],
            "retries": retries + 1,
            "done": False,
            "ready": False,
        }

    return {
        "pending_question": question.key,
        "done": False,
        "ready": ready and all_important_answered(slots),
    }


def route(state: ProfilingState) -> str:
    return "finalize" if state.get("done") else "ask"


def ask(state: ProfilingState) -> ProfilingState:
    """고른 질문을 대화에 붙인다."""
    key = state.get("pending_question")
    if not key:
        return {}
    question = QUESTION_BY_KEY[key]

    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "content": question.text})
    return {
        "messages": messages,
        "asked": [*(state.get("asked") or []), key],
    }


def finalize(state: ProfilingState) -> ProfilingState:
    """더 물을 게 없으면 마무리한다. 실제 프로파일 생성은 build_profile 이 한다."""
    missing = missing_mandatory(state.get("slots") or {})
    if missing:
        return {"done": False, "error": f"필수 정보가 없습니다: {', '.join(missing)}"}
    return {"done": True, "error": None}


# --------------------------------------------------------------------------- #
# 그래프
# --------------------------------------------------------------------------- #


def build_graph(client: Optional[LLMClient] = None):
    """프로파일링 그래프를 만든다. 한 번 호출 = 사용자 한 턴 처리."""
    client = client or get_client()

    graph = StateGraph(ProfilingState)
    graph.add_node("extract", _make_extract_node(client))
    graph.add_node("decide", decide)
    graph.add_node("ask", ask)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "decide")
    graph.add_conditional_edges("decide", route, {"ask": "ask", "finalize": "finalize"})
    graph.add_edge("ask", END)
    graph.add_edge("finalize", END)

    return graph.compile()


# --------------------------------------------------------------------------- #
# 프로파일 생성
# --------------------------------------------------------------------------- #


def build_profile(slots: dict[str, Any]) -> UserProfile:
    """수집한 슬롯으로 UserProfile 을 만든다.

    빠진 값은 DEFAULTS 로 채우고, 그 사실을 assumed_fields 에 남긴다.
    화면에서 '가정한 값'으로 표시해 사용자가 바로잡을 수 있게 하기 위함이다.
    """
    missing = missing_mandatory(slots)
    if missing:
        raise ValueError(f"필수 정보가 없습니다: {', '.join(missing)}")

    values = dict(slots)
    assumed: list[str] = []
    for key, default in DEFAULTS.items():
        if values.get(key) is None:
            values[key] = default
            assumed.append(key)

    # 국민연금 수급 개시 연령은 묻지 않고 출생연도에서 법정 스케줄로 결정한다.
    assumptions = load_assumptions()
    values["national_pension_start_age"] = assumptions.national_pension_start_age(
        values["birth_year"]
    )

    allowed = set(UserProfile.model_fields)
    payload = {k: v for k, v in values.items() if k in allowed}
    payload["assumed_fields"] = sorted(assumed)
    return UserProfile.model_validate(payload)


class ProfilingAgent:
    """대화형 프로파일링. Streamlit·웹 어느 쪽에서든 이 인터페이스만 쓴다."""

    def __init__(self, client: Optional[LLMClient] = None, today: Optional[_dt.date] = None):
        self.client = client or get_client()
        self.graph = build_graph(self.client)
        self.state: ProfilingState = new_state(today)

    def start(self) -> str:
        """인사와 첫 질문을 돌려준다."""
        self.state["messages"] = [{"role": "assistant", "content": GREETING}]
        self.state = {**self.state, **self.graph.invoke(self.state)}
        return self.last_assistant_message()

    def respond(self, user_text: str) -> str:
        """사용자 답변을 받아 다음 질문(또는 마무리 안내)을 돌려준다."""
        messages = list(self.state.get("messages") or [])
        messages.append({"role": "user", "content": user_text})
        self.state = {**self.state, "messages": messages}
        self.state = {**self.state, **self.graph.invoke(self.state)}

        if self.state.get("error") and not self.state.get("pending_question"):
            notice = (
                "죄송합니다. 계산에 꼭 필요한 정보를 아직 확인하지 못했습니다.\n"
                "생년, 퇴직 시점, 월 생활비만 알려주시면 바로 정리해 드릴 수 있습니다. "
                "왼쪽 입력창에 직접 적어주셔도 됩니다."
            )
            self.state["messages"] = [
                *self.state["messages"],
                {"role": "assistant", "content": notice},
            ]
            return notice

        if self.state.get("done"):
            closing = "말씀해 주셔서 감사합니다. 이제 상황을 정리해 드리겠습니다."
            self.state["messages"] = [
                *self.state["messages"],
                {"role": "assistant", "content": closing},
            ]
            return closing
        return self.last_assistant_message()

    def last_assistant_message(self) -> str:
        for message in reversed(self.state.get("messages") or []):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    @property
    def done(self) -> bool:
        return bool(self.state.get("done"))

    @property
    def ready(self) -> bool:
        """필수·중요 질문에 모두 답해 결과를 보여줄 수 있는 상태인가."""
        return bool(self.state.get("ready"))

    @property
    def progress(self) -> tuple[int, int]:
        """(답변한 질문 수, 전체 질문 수)."""
        slots = self.state.get("slots") or {}
        answered = sum(1 for q in QUESTIONS if q.is_answered(slots))
        return answered, len(QUESTIONS)

    def build_profile(self) -> UserProfile:
        return build_profile(self.state.get("slots") or {})
