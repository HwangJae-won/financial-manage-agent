"""LLM 클라이언트 계층 — anthropic / openai / mock 을 같은 인터페이스로 감싼다.

`.env` 의 FINAGENT_LLM_PROVIDER 한 줄로 전환된다. 프로바이더를 바꿔도
agents/ 의 나머지 코드는 손대지 않는다.

두 가지만 제공한다. 이 서비스에 필요한 게 그것뿐이기 때문이다:
  - complete()  : 자연어 응답 (설명 생성)
  - structured(): JSON 스키마에 맞는 구조화 응답 (슬롯 추출, 사기 판정)

mock 은 임시방편이 아니라 계속 쓰는 코드다. CI 는 키 없이 그래프 로직을
검증해야 하고, 발표 당일 네트워크가 끊겨도 데모는 돌아가야 한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

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
