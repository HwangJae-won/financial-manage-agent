from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.assumptions import Assumptions, load_assumptions


@pytest.fixture(autouse=True, scope="session")
def isolated_env():
    """테스트는 개발자의 `.env` 를 읽지 않는다.

    이걸 막지 않으면 각자의 로컬 설정이 테스트 결과를 바꾼다. 실제로 로컬 모델을
    붙여 보려고 `.env` 에 `FINAGENT_LLM_PROVIDER=local` 을 넣자 세 개가 깨졌다 —
    "LOCAL_MODEL 이 없을 때" 를 검증하는 테스트가 개발자의 .env 에서 값을 주워
    왔고, Streamlit 스모크 테스트는 mock 대신 실제 모델로 대화를 시작했다.

    `load_env` 의 기본 인자가 정의 시점에 묶여 있어 `config.ENV_PATH` 를 바꿔서는
    막히지 않는다. 그렇다고 함수를 통째로 무력화하면 `load_env` 자체를 검증하는
    테스트가 깨진다 — **기본 경로만** 없는 파일로 돌린다.
    """
    import agents.config as config

    original = config.load_env.__defaults__
    config.load_env.__defaults__ = (Path("/nonexistent/finagent/.env"),)
    for key in (
        "FINAGENT_LLM_PROVIDER",
        "LOCAL_MODEL",
        "LOCAL_BASE_URL",
        "LOCAL_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ):
        os.environ.pop(key, None)
    config.describe.cache_clear()
    yield
    config.load_env.__defaults__ = original


@pytest.fixture(autouse=True, scope="session")
def isolated_db(tmp_path_factory):
    """테스트는 저장소를 건드리지 않는다.

    기본 경로(`var/finagent.db`)를 그대로 쓰면 테스트를 돌릴 때마다 개발자의 실제
    DB 에 세션이 쌓이고, 집계(`storage.advice_stats`)가 테스트 데이터로 오염된다.
    """
    import storage

    path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["FINAGENT_DB_PATH"] = str(path)
    storage.reset()
    yield path
    storage.reset()


@pytest.fixture
def base_assumptions() -> Assumptions:
    return load_assumptions()


@pytest.fixture
def flat_assumptions(base_assumptions: Assumptions) -> Assumptions:
    """수익률·물가·세금이 전부 0인 가정.

    이 조건에서는 시뮬레이션 결과가 손계산과 정확히 일치해야 한다.
    엔진의 known-answer 테스트에 쓰인다.
    """
    a = base_assumptions.model_copy(deep=True)
    a.macro.inflation_rate = 0.0
    for cls in a.asset_classes.values():
        cls.expected_return = 0.0
        cls.volatility = 0.0
    a.tax.financial_income_rate = 0.0
    a.health_insurance.effective_rate_on_financial_income = 0.0
    return a
