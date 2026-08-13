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
    MockClient,
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
