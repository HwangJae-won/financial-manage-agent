"""데모 회귀 테스트 — 발표에서 말할 숫자를 고정한다.

가정값(assumptions.yaml)이나 엔진 규약을 바꾸면 여기가 먼저 깨진다.
깨졌을 때 해야 할 일은 테스트를 고치는 게 아니라, **바뀐 숫자를 의도한 것인지
확인하고 데모 스크립트를 함께 갱신하는 것**이다.
"""

from __future__ import annotations

import pytest

from core.asset_map import build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_years
from core.samples import DEMO_PROFILE


@pytest.fixture(scope="module")
def demo():
    return simulate(DEMO_PROFILE)


def test_headline_numbers(demo):
    """기획서 예시 인물의 3대 지표."""
    # "현재 고객님은 4년간 소득공백기가 예상됩니다"
    assert fmt_months(demo.income_gap_months) == "4년"

    # "현재 자산으로 예상 생활비를 충당할 경우 약 6.2년간 생활이 가능합니다"
    assert fmt_years(demo.expense_coverage_years) == "6.2년"

    # 연금 개시 시점(2030년)에 남는 금융자산
    assert demo.balance_at_pension_start == pytest.approx(118_552_982, rel=1e-6)


def test_depletion_happens_and_is_reported(demo):
    """이 인물은 현재 계획대로면 69세에 금융자산이 고갈된다 — 서비스가 경고해야 할 지점."""
    assert demo.depletion_year == 2035
    assert demo.depletion_age == 69
    assert demo.years_until_depletion == pytest.approx(8.4, abs=0.1)
    assert not demo.survives_horizon


def test_asset_map_headline():
    amap = build_asset_map(DEMO_PROFILE)
    assert fmt_krw(amap.total_assets) == "7억 2,000만원"
    assert fmt_krw(amap.by_key("survival").amount) == "3,600만원"
    assert fmt_krw(amap.by_key("income_gap").amount) == "1억 800만원"
    assert amap.gap_shortfall == 0  # 소득공백기 자체는 넘길 수 있다


def test_the_actual_risk_is_after_the_pension_starts(demo):
    """소득공백기는 넘기지만 장기 소득이 부족하다 — 이 대비가 데모의 핵심 메시지."""
    amap = build_asset_map(DEMO_PROFILE)
    assert amap.gap_coverage_ratio == 1.0  # 단기: 문제 없음
    assert demo.depletion_year is not None  # 장기: 고갈
    assert demo.depletion_year > DEMO_PROFILE.pension_start_year
