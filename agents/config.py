"""환경설정 로더.

.env 를 읽는다. python-dotenv 를 쓰지 않는 이유는 HPC 환경에서 의존성을 하나라도
줄이기 위해서다. 필요한 문법(주석, KEY=VALUE, 따옴표)만 지원한다.

키는 코드나 커밋에 절대 들어가지 않는다. 이 모듈은 값을 읽기만 하고, 로그나
예외 메시지에 값을 노출하지 않는다.
"""

from __future__ import annotations

import functools
import os
from enum import Enum
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

# OpenAI 는 기본 모델을 정해두지 않는다. 모델 ID 라인업이 자주 바뀌어서
# 하드코딩해두면 어느 날 조용히 404 가 나고 원인을 찾기 어려워진다.
# openai 프로바이더를 쓰려면 .env 에 OPENAI_MODEL 을 명시해야 한다.
OPENAI_MODEL_DOCS = "https://platform.openai.com/docs/models"


class Provider(str, Enum):
    """LLM 프로바이더."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"
    MOCK = "mock"


# 로컬 서버(vLLM 등)의 기본 주소. OpenAI 호환 API 를 그대로 쓰므로 클라이언트는
# 새로 만들지 않고 base_url 만 갈아 끼운다.
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8001/v1"

# 로컬 서버는 키를 검사하지 않지만 OpenAI SDK 가 값을 요구한다.
LOCAL_PLACEHOLDER_KEY = "EMPTY"


def load_env(path: Path = ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """.env 를 읽어 os.environ 에 채운다. 이미 설정된 값은 기본적으로 덮지 않는다.

    (실제 환경변수가 .env 보다 우선한다 — 배포 환경에서 파일 없이 주입하기 위함)
    """
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded


def _placeholder(value: Optional[str]) -> bool:
    """.env.example 의 자리표시자가 그대로 남아있는지 판별한다."""
    if not value:
        return True
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in ("여기에", "paste", "your_key", "changeme", "xxx", "<", "...")
    )


def api_key(provider: Provider) -> Optional[str]:
    """프로바이더의 API 키. 없거나 자리표시자면 None.

    로컬 서버는 키가 필요 없지만 OpenAI SDK 가 빈 값을 거부하므로 자리표시자를
    돌려준다. 이 값은 네트워크를 나가지 않는다.
    """
    load_env()
    if provider is Provider.LOCAL:
        return os.environ.get("LOCAL_API_KEY") or LOCAL_PLACEHOLDER_KEY

    env_name = {
        Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
        Provider.OPENAI: "OPENAI_API_KEY",
        Provider.MOCK: "",
    }[provider]
    if not env_name:
        return None
    value = os.environ.get(env_name)
    return None if _placeholder(value) else value


def local_base_url() -> str:
    """로컬 OpenAI 호환 서버 주소."""
    load_env()
    return os.environ.get("LOCAL_BASE_URL") or DEFAULT_LOCAL_BASE_URL


def resolve_provider() -> Provider:
    """사용할 프로바이더를 결정한다.

    우선순위:
      1. FINAGENT_LLM_PROVIDER 명시값
      2. 사용 가능한 키가 있는 프로바이더 (anthropic 우선)
      3. mock
    """
    load_env()

    explicit = os.environ.get("FINAGENT_LLM_PROVIDER", "").strip().lower()
    if explicit:
        try:
            return Provider(explicit)
        except ValueError as exc:
            valid = ", ".join(p.value for p in Provider)
            raise ValueError(
                f"FINAGENT_LLM_PROVIDER='{explicit}' 은(는) 알 수 없는 값입니다. "
                f"사용 가능: {valid}"
            ) from exc

    for provider in (Provider.ANTHROPIC, Provider.OPENAI):
        if api_key(provider):
            return provider
    return Provider.MOCK


def model_name(provider: Provider) -> str:
    """프로바이더별 모델 ID."""
    load_env()
    if provider is Provider.ANTHROPIC:
        return os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    if provider is Provider.LOCAL:
        model = os.environ.get("LOCAL_MODEL")
        if not model:
            raise ValueError(
                "local 프로바이더를 쓰려면 .env 에 LOCAL_MODEL 을 지정해야 합니다. "
                "서버가 올린 것과 같은 이름이어야 합니다 — "
                "`curl $LOCAL_BASE_URL/models` 로 확인하세요."
            )
        return model
    if provider is Provider.OPENAI:
        model = os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError(
                "openai 프로바이더를 쓰려면 .env 에 OPENAI_MODEL 을 지정해야 합니다. "
                f"사용 가능한 모델 ID는 {OPENAI_MODEL_DOCS} 에서 확인하세요."
            )
        return model
    return "mock"


@functools.lru_cache(maxsize=1)
def describe() -> str:
    """현재 설정 요약. 키 값은 절대 포함하지 않는다."""
    provider = resolve_provider()
    if provider is Provider.MOCK:
        return "LLM: mock (API 키 없음 — 정해진 응답으로 동작합니다)"
    if provider is Provider.LOCAL:
        # 모델 이름이 없으면 예외 대신 그 사실을 문구로 낸다. 이 함수는 화면
        # 상단에 쓰이므로 여기서 터지면 앱 전체가 죽는다.
        try:
            model = model_name(provider)
        except ValueError:
            return "LLM: local (LOCAL_MODEL 이 설정되지 않았습니다)"
        return f"LLM: 로컬 모델 / {model} ({local_base_url()})"
    return f"LLM: {provider.value} / {model_name(provider)}"
