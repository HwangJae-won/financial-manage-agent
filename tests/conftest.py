from __future__ import annotations

import pytest

from core.assumptions import Assumptions, load_assumptions


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
