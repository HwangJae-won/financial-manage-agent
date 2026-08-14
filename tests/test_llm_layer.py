"""LLM 계층 테스트.

실제 API 는 호출하지 않는다. 프로바이더 전환 로직, .env 파싱, mock 동작만 검증한다.
실제 호출 검증은 별도의 수동 스모크 테스트로 분리한다 (scripts/smoke_llm.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agents import config as cfg
from agents.llm import (
    NUMERIC_GUARDRAIL,
    AnthropicClient,
    MockClient,
    OpenAIClient,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolTurn,
    Turn,
    _empty_from_schema,
    get_client,
)

SLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "retirement_year": {"type": ["integer", "null"]},
        "severance_pay": {"type": ["integer", "null"]},
        "confidence": {"type": "string", "enum": ["high", "low"]},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["retirement_year"],
    "additionalProperties": False,
}


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """.env 와 환경변수를 격리한다.

    load_env(override=True) 는 os.environ 에 직접 쓰므로 monkeypatch 만으로는
    되돌려지지 않는다. 스냅샷을 떠서 테스트가 끝나면 통째로 복원한다 —
    안 그러면 여기서 설정한 프로바이더가 다른 테스트의 앱을 죽인다.
    """
    snapshot = dict(os.environ)
    for key in [
        "FINAGENT_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cfg, "ENV_PATH", tmp_path / ".env")
    cfg.describe.cache_clear()

    yield tmp_path

    os.environ.clear()
    os.environ.update(snapshot)
    cfg.describe.cache_clear()


# --------------------------------------------------------------------------- #
# .env 파싱
# --------------------------------------------------------------------------- #


def test_load_env_parses_comments_quotes_and_export(clean_env, monkeypatch):
    env = clean_env / ".env"
    env.write_text(
        "\n".join(
            [
                "# 주석은 무시",
                "",
                "ANTHROPIC_API_KEY=sk-ant-real",
                'OPENAI_MODEL="gpt-x"',
                "export FINAGENT_LLM_PROVIDER=openai",
                "잘못된줄",
            ]
        ),
        encoding="utf-8",
    )
    loaded = cfg.load_env(env, override=True)

    assert loaded["ANTHROPIC_API_KEY"] == "sk-ant-real"
    assert loaded["OPENAI_MODEL"] == "gpt-x"  # 따옴표 제거
    assert loaded["FINAGENT_LLM_PROVIDER"] == "openai"  # export 접두어 제거
    assert "잘못된줄" not in loaded


def test_real_environment_wins_over_env_file(clean_env, monkeypatch):
    """배포 환경에서 파일 없이 주입할 수 있어야 한다."""
    (clean_env / ".env").write_text("ANTHROPIC_API_KEY=from-file", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-environment")

    cfg.load_env(clean_env / ".env")
    assert os.environ["ANTHROPIC_API_KEY"] == "from-environment"


def test_missing_env_file_is_not_an_error(clean_env):
    assert cfg.load_env(clean_env / "없는파일.env") == {}


# --------------------------------------------------------------------------- #
# 프로바이더 결정
# --------------------------------------------------------------------------- #


def test_no_key_falls_back_to_mock(clean_env):
    assert cfg.resolve_provider() is cfg.Provider.MOCK


def test_placeholder_key_is_treated_as_missing(clean_env, monkeypatch):
    """.env.example 을 그대로 복사한 상태를 '키 있음'으로 오인하면 안 된다."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-여기에_키를_붙여넣으세요")
    assert cfg.api_key(cfg.Provider.ANTHROPIC) is None
    assert cfg.resolve_provider() is cfg.Provider.MOCK


def test_real_key_selects_anthropic(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    assert cfg.resolve_provider() is cfg.Provider.ANTHROPIC


def test_explicit_provider_overrides_available_keys(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    monkeypatch.setenv("FINAGENT_LLM_PROVIDER", "mock")
    assert cfg.resolve_provider() is cfg.Provider.MOCK


def test_unknown_provider_raises_with_helpful_message(clean_env, monkeypatch):
    monkeypatch.setenv("FINAGENT_LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="anthropic"):
        cfg.resolve_provider()


def test_anthropic_model_defaults_to_opus5(clean_env):
    assert cfg.model_name(cfg.Provider.ANTHROPIC) == "claude-opus-5"


def test_anthropic_model_is_overridable(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    assert cfg.model_name(cfg.Provider.ANTHROPIC) == "claude-sonnet-5"


def test_openai_model_must_be_specified(clean_env):
    """모델 ID 를 하드코딩하지 않으므로, 미지정이면 명확히 알려줘야 한다."""
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        cfg.model_name(cfg.Provider.OPENAI)


def test_describe_never_leaks_the_key(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    cfg.describe.cache_clear()
    summary = cfg.describe()
    assert "supersecret" not in summary
    assert "anthropic" in summary


# --------------------------------------------------------------------------- #
# Mock 클라이언트
# --------------------------------------------------------------------------- #


def test_factory_returns_mock_without_keys(clean_env):
    client = get_client()
    assert isinstance(client, MockClient)
    assert client.provider == "mock"


def test_mock_complete_records_calls():
    client = MockClient()
    response = client.complete("소득공백기를 설명해줘", system="너는 상담사다")

    assert response.text
    assert response.provider == "mock"
    assert client.calls[0]["kind"] == "complete"
    assert client.calls[0]["system"] == "너는 상담사다"


def test_mock_structured_fills_schema_shape():
    """핸들러가 없어도 스키마 형태를 채워야 그래프가 None 으로 터지지 않는다."""
    result = MockClient().structured("추출해줘", SLOT_SCHEMA)

    assert set(result) == set(SLOT_SCHEMA["properties"])
    assert result["retirement_year"] is None  # ["integer","null"] → null
    assert result["confidence"] == "high"  # enum 첫 값
    assert result["notes"] == []


def test_mock_structured_handler_is_used():
    client = MockClient(
        structured_handler=lambda prompt, schema, system: {"retirement_year": 2026}
    )
    assert client.structured("...", SLOT_SCHEMA) == {"retirement_year": 2026}


def test_mock_text_handler_is_used():
    client = MockClient(text_handler=lambda prompt, system: "고정 응답")
    assert client.complete("아무거나").text == "고정 응답"


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"type": "string"}, ""),
        ({"type": "integer"}, 0),
        ({"type": "number"}, 0.0),
        ({"type": "boolean"}, False),
        ({"type": "array"}, []),
        ({"type": ["integer", "null"]}, None),
        ({"type": "string", "enum": ["a", "b"]}, "a"),
    ],
)
def test_empty_value_per_type(spec, expected):
    schema = {"type": "object", "properties": {"f": spec}}
    assert _empty_from_schema(schema)["f"] == expected


# --------------------------------------------------------------------------- #
# 가드레일
# --------------------------------------------------------------------------- #


def test_numeric_guardrail_covers_the_key_risks():
    """숫자 환각과 상품 권유는 이 서비스에서 가장 위험한 두 가지다."""
    assert "직접 계산" in NUMERIC_GUARDRAIL
    assert "권유하지" in NUMERIC_GUARDRAIL


# --------------------------------------------------------------------------- #
# 도구 사용 (tool use) — 프로바이더 중립 이력을 각자의 형식으로 옮기는가
# --------------------------------------------------------------------------- #

TOOLS = [
    ToolSpec(
        name="simulate_plan",
        description="생활비를 바꿔 은퇴 계획을 다시 계산한다",
        input_schema={
            "type": "object",
            "properties": {"monthly_expense": {"type": "integer"}},
        },
    ),
    ToolSpec(name="prescribe", description="목표 역산", input_schema={"type": "object"}),
]


def _two_turn_history():
    """도구를 한 번 부르고 결과를 받은 상태의 이력."""
    return [
        Turn(role="user", text="생활비를 줄이면 어떻게 되나요?"),
        Turn(
            role="assistant",
            tool_calls=[
                ToolCall(id="call-1", name="simulate_plan", arguments={"monthly_expense": 2_500_000})
            ],
        ),
        Turn(
            role="user",
            tool_results=[ToolResult(call_id="call-1", content='{"depletion_age": 73}')],
        ),
    ]


def test_mock_runs_a_full_tool_loop_without_a_key():
    """키 없이도 루프가 두 바퀴 돌아야 한다 — CI 와 발표 당일 안전망."""
    client = MockClient()

    first = client.converse([Turn(role="user", text="안녕하세요")], TOOLS)
    assert first.wants_tools
    assert first.tool_calls[0].name == "simulate_plan"

    history = _two_turn_history()
    second = client.converse(history, TOOLS)
    assert not second.wants_tools
    assert "73" in second.text  # 도구 결과를 받아 마무리한다

    assert [c["kind"] for c in client.calls] == ["converse", "converse"]


def test_mock_tool_handler_is_used():
    def handler(turns, tools):
        return ToolTurn(text="정해진 답", provider="mock", model="mock")

    client = MockClient(tool_handler=handler)
    assert client.converse([Turn(role="user", text="무엇이든")], TOOLS).text == "정해진 답"


def test_tool_turn_round_trips_into_history():
    """응답을 이력에 다시 넣을 때 도구 호출이 보존되어야 한다."""
    call = ToolCall(id="call-9", name="prescribe", arguments={"target_age": 95})
    turn = ToolTurn(text="계산해 보겠습니다", tool_calls=[call]).as_turn()

    assert turn.role == "assistant"
    assert turn.text == "계산해 보겠습니다"
    assert turn.tool_calls == [call]


def test_anthropic_encodes_tool_use_and_tool_result_blocks():
    """anthropic 은 tool_use 가 assistant, tool_result 가 user 블록이다."""
    encoded = AnthropicClient._encode(_two_turn_history())

    assert [m["role"] for m in encoded] == ["user", "assistant", "user"]
    assert encoded[1]["content"][0]["type"] == "tool_use"
    assert encoded[1]["content"][0]["input"] == {"monthly_expense": 2_500_000}
    assert encoded[2]["content"][0]["type"] == "tool_result"
    assert encoded[2]["content"][0]["tool_use_id"] == "call-1"


def test_anthropic_never_emits_an_empty_text_block():
    """빈 텍스트 블록은 API 가 거부한다."""
    encoded = AnthropicClient._encode(_two_turn_history())
    for message in encoded:
        for block in message["content"]:
            assert block.get("type") != "text" or block["text"]


def test_openai_encodes_tool_results_as_their_own_messages():
    """openai 는 도구 결과가 별도의 tool 역할 메시지다 — anthropic 과 다른 지점."""
    encoded = OpenAIClient._encode(_two_turn_history())

    assert [m["role"] for m in encoded] == ["user", "assistant", "tool"]
    assert encoded[1]["content"] is None  # 텍스트 없이 도구만 부른 턴
    assert encoded[1]["tool_calls"][0]["function"]["name"] == "simulate_plan"
    assert encoded[2]["tool_call_id"] == "call-1"


def test_both_providers_accept_the_same_neutral_history():
    """에이전트가 프로바이더 차이를 몰라도 되어야 한다."""
    history = _two_turn_history()
    assert AnthropicClient._encode(history)
    assert OpenAIClient._encode(history)
    # 인코딩이 원본 이력을 건드리지 않는다
    assert history[1].tool_calls[0].arguments == {"monthly_expense": 2_500_000}


def test_empty_turns_are_dropped():
    """말도 도구도 없는 턴이 메시지로 나가면 API 가 거부한다."""
    history = [Turn(role="user", text="질문"), Turn(role="assistant")]
    assert len(AnthropicClient._encode(history)) == 1
    assert len(OpenAIClient._encode(history)) == 1


# --------------------------------------------------------------------------- #
# 로컬 프로바이더 — vLLM 등 OpenAI 호환 서버
# --------------------------------------------------------------------------- #

from agents.config import (  # noqa: E402
    DEFAULT_LOCAL_BASE_URL,
    LOCAL_PLACEHOLDER_KEY,
    Provider,
    api_key,
    local_base_url,
    model_name,
)


def test_local_is_a_known_provider():
    assert Provider("local") is Provider.LOCAL


def test_the_local_key_is_a_placeholder_not_a_secret():
    """로컬 서버는 키를 검사하지 않지만 SDK 가 빈 값을 거부한다."""
    assert api_key(Provider.LOCAL) == LOCAL_PLACEHOLDER_KEY


def test_the_local_base_url_has_a_default(monkeypatch):
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    assert local_base_url() == DEFAULT_LOCAL_BASE_URL

    monkeypatch.setenv("LOCAL_BASE_URL", "http://gpu-node:9000/v1")
    assert local_base_url() == "http://gpu-node:9000/v1"


def test_a_missing_local_model_says_how_to_find_it(monkeypatch):
    """서버가 올린 이름과 달라 404 가 나면 원인이 잘 안 보인다."""
    monkeypatch.delenv("LOCAL_MODEL", raising=False)

    with pytest.raises(ValueError) as exc:
        model_name(Provider.LOCAL)
    assert "LOCAL_MODEL" in str(exc.value)
    assert "/models" in str(exc.value)


def test_describe_does_not_explode_without_a_model(monkeypatch):
    """이 문구는 화면 상단에 쓰인다. 여기서 터지면 앱 전체가 죽는다."""
    import agents.config as config

    monkeypatch.setenv("FINAGENT_LLM_PROVIDER", "local")
    monkeypatch.delenv("LOCAL_MODEL", raising=False)
    config.describe.cache_clear()

    assert "LOCAL_MODEL" in config.describe()
    config.describe.cache_clear()
