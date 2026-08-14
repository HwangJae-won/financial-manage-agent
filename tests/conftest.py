from __future__ import annotations

import os

import pytest

from core.assumptions import Assumptions, load_assumptions


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
