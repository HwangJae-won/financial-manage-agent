"""LLM 클라이언트 계층 — anthropic / openai / mock 을 같은 인터페이스로 감싼다.

`.env` 의 FINAGENT_LLM_PROVIDER 한 줄로 전환된다. 프로바이더를 바꿔도
agents/ 의 나머지 코드는 손대지 않는다.

세 가지를 제공한다. 이 서비스에 필요한 게 그것뿐이기 때문이다:
  - complete()  : 자연어 응답 (설명 생성)
  - structured(): JSON 스키마에 맞는 구조화 응답 (슬롯 추출, 사기 판정)
  - converse()  : 도구를 쥔 대화 한 턴 (계산 엔진을 부르는 에이전트)

`converse()` 는 **한 턴만** 처리한다. 반복 루프는 여기 두지 않고 에이전트 쪽에
둔다. 이 파일은 프로바이더 차이를 흡수하는 얇은 어댑터로 남아야 하고, "몇 번까지
돌 것인가 / 실패하면 어떻게 할 것인가" 같은 판단은 프로바이더와 무관하기 때문이다.

대화 이력은 `Turn` 이라는 중립 형태로 주고받는다. 도구 호출 이력의 표현은
프로바이더마다 다른데(anthropic 은 tool_use/tool_result 블록, openai 는
tool_calls/tool 역할), 그 차이를 에이전트가 알게 되면 프로바이더 전환이 깨진다.

mock 은 임시방편이 아니라 계속 쓰는 코드다. CI 는 키 없이 그래프 로직을
검증해야 하고, 발표 당일 네트워크가 끊겨도 데모는 돌아가야 한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol

from agents.config import Provider, api_key, model_name, resolve_provider

# 금융 계산 결과를 설명할 때 모든 프롬프트에 공통으로 붙는 제약.
# core/ 가 만든 숫자를 LLM 이 재계산하거나 지어내지 못하게 막는 방어선이다.
NUMERIC_GUARDRAIL = (
    "제시된 계산 결과에 있는 숫자만 인용하세요. 직접 계산하거나, 주어지지 않은 "
    "금액·비율·연도를 추정해서 쓰지 마세요. 필요한 숫자가 없으면 그 부분은 언급하지 "
    "마세요. 특정 금융상품의 가입을 권유하지 마세요."
)


@dataclass
class LLMResponse:
    """LLM 호출 결과."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


# --------------------------------------------------------------------------- #
# 도구 사용 (tool use)
# --------------------------------------------------------------------------- #


@dataclass
class ToolSpec:
    """모델에게 알려줄 도구 하나의 정의."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """모델이 요청한 도구 호출."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """도구를 실행한 결과. 다음 턴에 모델에게 돌려준다."""

    call_id: str
    content: str = ""  # JSON 으로 직렬화한 도구 출력
    is_error: bool = False


@dataclass
class Turn:
    """대화 이력 한 칸 — 프로바이더 중립 형태.

    도구 호출 이력의 표현은 프로바이더마다 다르다. 그 차이는 각 클라이언트의
    `_encode` 가 흡수하고, 에이전트는 이 형태만 다룬다.
    """

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


def turn_to_dict(turn: Turn) -> dict[str, Any]:
    """대화 한 칸을 저장 가능한 형태로.

    `dataclasses.asdict` 를 쓰지 않는 이유는 빈 값까지 전부 실어 나르기 때문이다.
    대화가 길어지면 대부분의 칸이 도구 없는 평범한 발화라, 빈 리스트를 빼면
    저장 크기가 눈에 띄게 줄어든다.
    """
    data: dict[str, Any] = {"role": turn.role}
    if turn.text:
        data["text"] = turn.text
    if turn.tool_calls:
        data["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in turn.tool_calls
        ]
    if turn.tool_results:
        data["tool_results"] = [
            {
                "call_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in turn.tool_results
        ]
    return data


def turn_from_dict(data: dict[str, Any]) -> Turn:
    """`turn_to_dict` 의 역. 모르는 키는 무시한다.

    저장해 둔 대화를 나중 버전이 읽을 때 필드가 늘거나 줄 수 있다. 그때 예외로
    터지면 사용자의 대화 이력이 통째로 날아간다 — 읽을 수 있는 만큼만 읽는다.
    """
    return Turn(
        role="assistant" if data.get("role") == "assistant" else "user",
        text=data.get("text") or "",
        tool_calls=[
            ToolCall(
                id=call.get("id", ""),
                name=call.get("name", ""),
                arguments=call.get("arguments") or {},
            )
            for call in data.get("tool_calls") or []
        ],
        tool_results=[
            ToolResult(
                call_id=result.get("call_id", ""),
                content=result.get("content", ""),
                is_error=bool(result.get("is_error", False)),
            )
            for result in data.get("tool_results") or []
        ],
    )


@dataclass
class ToolTurn:
    """모델 응답 한 턴. 말을 하거나, 도구를 부르거나, 둘 다 한다."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def as_turn(self) -> Turn:
        """이 응답을 대화 이력에 넣을 형태로 바꾼다."""
        return Turn(role="assistant", text=self.text, tool_calls=list(self.tool_calls))


class LLMClient(Protocol):
    """agents/ 가 의존하는 유일한 LLM 인터페이스."""

    provider: str
    model: str

    def complete(
        self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 2000
    ) -> LLMResponse: ...

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> dict[str, Any]: ...

    def converse(
        self,
        turns: list[Turn],
        tools: list[ToolSpec],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> ToolTurn: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicClient:
    """Claude API 클라이언트."""

    def __init__(self, model: Optional[str] = None, key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - 설치 환경 의존
            raise RuntimeError(
                "anthropic SDK 가 설치되지 않았습니다: pip install anthropic"
            ) from exc

        self.provider = Provider.ANTHROPIC.value
        self.model = model or model_name(Provider.ANTHROPIC)
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=key or api_key(Provider.ANTHROPIC))

    def _create(self, **kwargs):
        response = self._client.messages.create(model=self.model, **kwargs)
        # 안전 분류기가 요청을 거절하면 content 가 비어 있다. 인덱싱 전에 확인한다.
        if response.stop_reason == "refusal":
            raise LLMRefusal(
                "모델이 요청을 거절했습니다.",
                category=getattr(getattr(response, "stop_details", None), "category", None),
            )
        return response

    def complete(
        self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 2000
    ) -> LLMResponse:
        response = self._create(
            max_tokens=max_tokens,
            system=system or self._sdk.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        response = self._create(
            max_tokens=max_tokens,
            system=system or self._sdk.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    # -- 도구 사용 -------------------------------------------------------- #

    @staticmethod
    def _encode(turns: list[Turn]) -> list[dict[str, Any]]:
        """중립 이력을 Anthropic 메시지로 옮긴다.

        도구 결과는 **user 역할의 tool_result 블록**으로 들어간다.
        빈 텍스트 블록은 API 가 거부하므로 넣지 않는다.
        """
        messages: list[dict[str, Any]] = []
        for turn in turns:
            content: list[dict[str, Any]] = []
            for result in turn.tool_results:
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            for call in turn.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            if content:
                messages.append({"role": turn.role, "content": content})
        return messages

    def converse(
        self,
        turns: list[Turn],
        tools: list[ToolSpec],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> ToolTurn:
        response = self._create(
            max_tokens=max_tokens,
            system=system or self._sdk.NOT_GIVEN,
            messages=self._encode(turns),
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ],
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in response.content
            if b.type == "tool_use"
        ]
        return ToolTurn(
            text=text,
            tool_calls=calls,
            provider=self.provider,
            model=self.model,
            stop_reason=response.stop_reason or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class OpenAIClient:
    """OpenAI API 클라이언트."""

    def __init__(self, model: Optional[str] = None, key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 설치 환경 의존
            raise RuntimeError(
                "openai SDK 가 설치되지 않았습니다: pip install openai"
            ) from exc

        self.provider = Provider.OPENAI.value
        self.model = model or model_name(Provider.OPENAI)
        self._client = OpenAI(api_key=key or api_key(Provider.OPENAI))

    def complete(
        self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 2000
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw=response,
        )

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            },
        )
        return json.loads(response.choices[0].message.content or "{}")

    # -- 도구 사용 -------------------------------------------------------- #

    @staticmethod
    def _encode(turns: list[Turn]) -> list[dict[str, Any]]:
        """중립 이력을 OpenAI 메시지로 옮긴다.

        도구 결과는 **별도의 tool 역할 메시지**가 된다 (Anthropic 과 다른 지점).
        """
        messages: list[dict[str, Any]] = []
        for turn in turns:
            for result in turn.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content,
                    }
                )
            if turn.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": turn.text or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                            for call in turn.tool_calls
                        ],
                    }
                )
            elif turn.text:
                messages.append({"role": turn.role, "content": turn.text})
        return messages

    def converse(
        self,
        turns: list[Turn],
        tools: list[ToolSpec],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> ToolTurn:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._encode(turns))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ],
        )

        choice = response.choices[0]
        calls = []
        for raw_call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                # 인자가 깨져 와도 대화를 끊지 않는다. 도구 계층이 스키마 오류로
                # 걸러내고, 모델에게 그 사실을 돌려주면 다시 시도할 수 있다.
                arguments = {}
            calls.append(
                ToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments)
            )

        usage = response.usage
        return ToolTurn(
            text=choice.message.content or "",
            tool_calls=calls,
            provider=self.provider,
            model=self.model,
            stop_reason=choice.finish_reason or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw=response,
        )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


@dataclass
class MockClient:
    """키 없이 동작하는 결정론적 클라이언트.

    등록된 핸들러가 프롬프트를 보고 응답을 만든다. 핸들러가 없으면 스키마의
    기본 형태를 채워 돌려준다 — 그래프가 어디서도 None 으로 터지지 않게 하기 위함.

    호출 기록(`calls`)이 남으므로 "어떤 프롬프트가 몇 번 나갔는지"를 테스트할 수 있다.
    """

    provider: str = Provider.MOCK.value
    model: str = "mock"
    text_handler: Optional[Callable[[str, Optional[str]], str]] = None
    structured_handler: Optional[
        Callable[[str, dict[str, Any], Optional[str]], dict[str, Any]]
    ] = None
    tool_handler: Optional[Callable[[list["Turn"], list["ToolSpec"]], "ToolTurn"]] = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 2000
    ) -> LLMResponse:
        self.calls.append({"kind": "complete", "prompt": prompt, "system": system})
        text = (
            self.text_handler(prompt, system)
            if self.text_handler
            else _default_mock_text(prompt)
        )
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
        )

    def structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        self.calls.append(
            {"kind": "structured", "prompt": prompt, "schema": schema, "system": system}
        )
        if self.structured_handler:
            return self.structured_handler(prompt, schema, system)
        return _empty_from_schema(schema)

    def converse(
        self,
        turns: list[Turn],
        tools: list[ToolSpec],
        *,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> ToolTurn:
        self.calls.append(
            {
                "kind": "converse",
                "turns": turns,
                "tools": [t.name for t in tools],
                "system": system,
            }
        )
        if self.tool_handler:
            return self.tool_handler(turns, tools)
        return _default_mock_turn(turns, tools)


def _default_mock_turn(turns: list[Turn], tools: list[ToolSpec]) -> ToolTurn:
    """핸들러가 없을 때의 기본 도구 사용 동작.

    도구 결과를 아직 못 받았으면 첫 번째 도구를 빈 인자로 부르고, 결과를 받았으면
    그것을 요약한 텍스트로 마무리한다. **키 없이도 루프가 두 바퀴 도는지** 확인할
    수 있게 하려는 것이다. 인자를 비워 두는 것도 의도적이다 — 도구 계층이 스키마
    오류를 어떻게 돌려주는지가 함께 확인된다.

    판단은 **이번 질문 구간만** 보고 한다. 이력 전체를 보면 두 번째 질문부터는
    "이미 도구를 썼다"고 보고 계산 없이 답해 버린다. 키 없이 도는 데모에서
    질문을 이어갈 때 두 번째부터 아무 계산도 하지 않는 화면이 된다.
    """
    started = max(
        (i for i, turn in enumerate(turns) if turn.role == "user" and turn.text),
        default=-1,
    )
    segment = turns[started + 1 :]
    answered = any(turn.tool_results for turn in segment)

    if tools and not answered:
        return ToolTurn(
            text="",
            tool_calls=[ToolCall(id="mock-call-1", name=tools[0].name, arguments={})],
            provider=Provider.MOCK.value,
            model="mock",
            stop_reason="tool_use",
        )

    outputs = [r.content for turn in segment for r in turn.tool_results]
    return ToolTurn(
        text=f"[mock 응답] {_mock_answer(outputs[-1] if outputs else '')}",
        provider=Provider.MOCK.value,
        model="mock",
        stop_reason="end_turn",
    )


# 도구 출력에서 사람이 읽을 문장을 찾을 때 먼저 보는 키.
# 계산 모듈들이 이미 '한 문장 결론'을 만들어 두었으므로 그것을 쓴다.
_MOCK_PREFERRED = ("한_문장", "결론", "요약", "고갈_여부")


def _find_readable(data: Any, keys: tuple[str, ...]) -> Optional[str]:
    """중첩된 출력에서 사람이 읽을 만한 문장 하나를 찾는다."""
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            found = _find_readable(value, keys)
            if found:
                return found
    return None


def _first_value(data: Any) -> Optional[str]:
    """읽을 만한 키가 없을 때의 최후 수단 — 아무 값이라도 하나 집어 온다.

    실제 도구는 전부 '한_문장' 같은 한국어 결론 키를 들고 있지만, 그렇지 않은
    출력이 들어와도 mock 이 **도구 결과를 받았다는 사실**은 보여야 한다.
    """
    if isinstance(data, dict):
        for value in data.values():
            found = _first_value(value)
            if found:
                return found
    elif isinstance(data, (list, tuple)):
        for value in data:
            found = _first_value(value)
            if found:
                return found
    elif isinstance(data, (str, int, float)) and not isinstance(data, bool):
        text = str(data).strip()
        return text or None
    return None


def _mock_answer(content: str) -> str:
    """도구 출력을 사람이 읽을 문장으로 옮긴다.

    원시 JSON 을 잘라 붙이던 자리다. 상담 화면이 생기면서 이 문자열이 그대로
    사용자에게 보이게 되었다 — 키 없이 도는 데모에서 대화창에 중괄호가 뜨면
    서비스가 고장난 것처럼 보인다.

    문장은 도구 출력에서 그대로 가져온다. 지어내지 않으므로 숫자 검증(A7)도
    자연히 통과한다.
    """
    if not content:
        return "확인할 계산 결과가 없습니다."
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return "계산 결과를 확인했습니다."

    readable = _find_readable(data, _MOCK_PREFERRED) or _first_value(data)
    if readable:
        return f"계산해 보았습니다. {readable}"
    return "계산 결과를 확인했습니다."


def _default_mock_text(prompt: str) -> str:
    """프롬프트에서 첫 문장을 따와 그럴듯한 한국어 응답을 만든다."""
    head = re.sub(r"\s+", " ", prompt).strip()[:60]
    return f"[mock 응답] 요청을 확인했습니다: {head}"


def _empty_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON 스키마에서 타입에 맞는 빈 값을 채운 객체를 만든다."""
    if schema.get("type") != "object":
        return {}
    result: dict[str, Any] = {}
    for name, spec in (schema.get("properties") or {}).items():
        result[name] = _empty_value(spec)
    return result


def _empty_value(spec: dict[str, Any]) -> Any:
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    kind = spec.get("type")
    if isinstance(kind, list):
        # nullable 이면 null 을 택한다. 슬롯 채우기에서 "퇴직금 0원"과
        # "아직 모름"이 구분되지 않으면 잘못된 시뮬레이션이 나간다.
        kind = "null" if "null" in kind else next(iter(kind), "null")
    return {
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": _empty_from_schema(spec) if spec.get("properties") else {},
        "null": None,
    }.get(kind)


class LLMRefusal(RuntimeError):
    """모델이 안전상의 이유로 응답을 거절한 경우."""

    def __init__(self, message: str, *, category: Optional[str] = None):
        super().__init__(message)
        self.category = category


# --------------------------------------------------------------------------- #
# 팩토리
# --------------------------------------------------------------------------- #


def get_client(provider: Optional[Provider] = None, **kwargs) -> LLMClient:
    """설정에 맞는 클라이언트를 만든다.

    provider 를 생략하면 .env / 환경변수에서 결정한다 (agents.config.resolve_provider).
    """
    provider = provider or resolve_provider()
    if provider is Provider.ANTHROPIC:
        return AnthropicClient(**kwargs)
    if provider is Provider.OPENAI:
        return OpenAIClient(**kwargs)
    return MockClient(**kwargs)
