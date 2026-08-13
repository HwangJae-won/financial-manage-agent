from __future__ import annotations

import pytest

from core.asset_map import build_asset_map
from core.assumptions import Assumptions
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


def test_survival_bucket_is_one_year_of_expenses(base_assumptions: Assumptions):
    amap = build_asset_map(DEMO_PROFILE, base_assumptions)
    months = base_assumptions.asset_map.survival_months
    assert amap.survival_need == DEMO_PROFILE.monthly_expense * months
    assert amap.by_key("survival").amount == amap.survival_need  # 충분히 보유


def test_income_gap_need_uses_net_monthly_need():
    """기타소득이 있으면 소득공백기 필요액이 그만큼 줄어든다."""
    amap = build_asset_map(DIVERSIFIED_PROFILE)
    expected_net = DIVERSIFIED_PROFILE.monthly_expense - DIVERSIFIED_PROFILE.other_monthly_income
    assert amap.net_monthly_need == expected_net
    assert amap.income_gap_need == expected_net * DIVERSIFIED_PROFILE.income_gap_months


def test_survival_and_gap_do_not_double_count():
    """생존자산 + 소득공백기 자산이 소득공백기 총 필요액을 넘지 않아야 한다."""
    amap = build_asset_map(DEMO_PROFILE)
    allocated = amap.by_key("survival").amount + amap.by_key("income_gap").amount
    assert allocated <= amap.income_gap_need + 1  # 반올림 여유


def test_financial_buckets_sum_to_financial_assets():
    """실물자산을 제외한 버킷 합계는 금융자산 총액과 같아야 한다."""
    for profile in (DEMO_PROFILE, DIVERSIFIED_PROFILE):
        amap = build_asset_map(profile)
        financial_sum = sum(
            b.amount for b in amap.buckets if b.key != "real_asset"
        )
        assert financial_sum == profile.financial_assets


def test_real_asset_bucket_is_real_estate():
    amap = build_asset_map(DEMO_PROFILE)
    assert amap.by_key("real_asset").amount == DEMO_PROFILE.real_estate


def test_shortfall_is_detected_when_liquid_assets_are_insufficient():
    """유동자산이 소득공백기 필요액에 못 미치면 부족액이 잡혀야 한다."""
    poor = DEMO_PROFILE.model_copy(
        update={"severance_pay": 30_000_000, "cash_savings": 0}
    )
    amap = build_asset_map(poor)

    assert amap.liquid_assets == 30_000_000
    assert amap.gap_shortfall > 0
    assert amap.gap_shortfall == amap.income_gap_need - amap.liquid_assets
    assert amap.gap_coverage_ratio < 1.0


def test_no_shortfall_when_assets_are_ample():
    rich = DEMO_PROFILE.model_copy(update={"cash_savings": 2_000_000_000})
    amap = build_asset_map(rich)
    assert amap.gap_shortfall == 0
    assert amap.gap_coverage_ratio == 1.0


def test_zero_income_gap_needs_nothing():
    """소득공백기가 없으면 해당 버킷의 필요액은 0."""
    profile = DEMO_PROFILE.model_copy(
        update={"birth_year": 1960, "national_pension_start_age": 63}
    )
    assert profile.income_gap_months == 0
    amap = build_asset_map(profile)
    assert amap.income_gap_need == 0
    assert amap.by_key("income_gap").required == 0
    assert amap.gap_coverage_ratio == 1.0


def test_waterfall_priority_survival_first():
    """유동자산이 생존자산 필요액에도 못 미치면 전부 생존자산에 배정된다."""
    broke = DEMO_PROFILE.model_copy(
        update={"severance_pay": 10_000_000, "cash_savings": 0}
    )
    amap = build_asset_map(broke)
    assert amap.by_key("survival").amount == 10_000_000
    assert amap.by_key("income_gap").amount == 0
    assert amap.by_key("survival").shortfall > 0


def test_bucket_labels_are_present_for_ui():
    amap = build_asset_map(DEMO_PROFILE)
    assert [b.key for b in amap.buckets] == [
        "survival",
        "income_gap",
        "long_term",
        "growth",
        "real_asset",
    ]
    for bucket in amap.buckets:
        assert bucket.label and bucket.purpose and bucket.horizon


def test_by_key_raises_for_unknown_bucket():
    with pytest.raises(KeyError):
        build_asset_map(DEMO_PROFILE).by_key("nope")
