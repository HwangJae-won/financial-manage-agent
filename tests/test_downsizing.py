"""주택 다운사이징 테스트.

이 기능이 존재하는 이유는 하나다. **"집 팔면 되죠"와 실제로 손에 남는 돈 사이에
세금과 수수료가 있다.** 그래서 여기서 지켜야 할 것은 차액이 아니라 순 유입이 화면에
나가는 것이고, 거래비용이 항목별로 보이는 것이다.

취득세는 유일하게 원문(지방세법 제11조 제1항 제8호)을 대조한 값이므로 경계값을
직접 확인한다. 6~9억 구간이 계산식이라 여기를 뭉개면 수백만원이 틀린다.
"""

from __future__ import annotations

import pytest

from core.assumptions import load_assumptions
from core.downsizing import (
    DEFAULT_RATIOS,
    analyze_downsizing,
    transaction_cost,
)
from core.samples import DEMO_PROFILE


@pytest.fixture(scope="module")
def spec():
    return load_assumptions().housing


@pytest.fixture(scope="module")
def report():
    return analyze_downsizing(DEMO_PROFILE)


# --------------------------------------------------------------------------- #
# 취득세 — 원문 대조한 값이다
# --------------------------------------------------------------------------- #


def test_acquisition_tax_low_bracket(spec):
    """6억원 이하는 1%. 경계값 자체도 1%다 ('6억원 이하')."""
    assert spec.acquisition_tax.rate_for(400_000_000) == pytest.approx(0.010)
    assert spec.acquisition_tax.rate_for(600_000_000) == pytest.approx(0.010)


def test_acquisition_tax_high_bracket(spec):
    """9억원 **초과**가 3%. 9억원 정확히는 계산식 구간의 끝이다."""
    assert spec.acquisition_tax.rate_for(1_200_000_000) == pytest.approx(0.030)
    assert spec.acquisition_tax.rate_for(900_000_001) == pytest.approx(0.030)


def test_acquisition_tax_middle_is_a_formula_not_a_flat_rate(spec):
    """6~9억은 (가액 ÷ 3억 × 2 - 3) × 1/100. 뭉개면 안 되는 구간이다."""
    rate = spec.acquisition_tax.rate_for
    # 7.5억 → (7.5/3 × 2 - 3) / 100 = 2/100
    assert rate(750_000_000) == pytest.approx(0.020)
    # 구간 양 끝이 아래·위 구간과 이어진다 (계단이 아니라 경사다)
    assert rate(600_000_001) == pytest.approx(0.010, abs=1e-4)
    assert rate(900_000_000) == pytest.approx(0.030, abs=1e-4)
    # 단조증가
    prices = list(range(600_000_000, 900_000_001, 25_000_000))
    rates = [rate(p) for p in prices]
    assert rates == sorted(rates)


def test_brokerage_uses_the_band_the_price_falls_in(spec):
    """중개보수는 구간별 요율이다. 구간이 올라가면 요율도 올라간다."""
    assert spec.brokerage_fee(400_000_000) == 400_000_000 * 0.004
    assert spec.brokerage_fee(1_000_000_000) == 1_000_000_000 * 0.005
    assert spec.brokerage_fee(0) == 0


# --------------------------------------------------------------------------- #
# 거래비용 — 이 기능의 존재 이유
# --------------------------------------------------------------------------- #


def test_costs_include_every_item(spec):
    """항목이 하나라도 빠지면 '집 팔면 되죠'로 되돌아간다."""
    costs = transaction_cost(500_000_000, 300_000_000, spec)
    assert costs.acquisition_tax > 0
    assert costs.brokerage_sell > 0  # 파는 쪽에도 중개보수가 있다
    assert costs.brokerage_buy > 0
    assert costs.registration > 0
    assert costs.moving > 0
    assert costs.total == (
        costs.acquisition_tax
        + costs.brokerage_sell
        + costs.brokerage_buy
        + costs.registration
        + costs.moving
    )


def test_net_proceeds_are_smaller_than_the_headline_difference(report):
    """차액 그대로 들어오지 않는다. 이 한 줄이 기능 전체의 요지다."""
    assert report.options
    for option in report.options:
        assert option.net_proceeds < option.gross_difference
        assert option.costs.total == option.gross_difference - option.net_proceeds


def test_screen_shows_the_cost_not_just_the_net(report):
    """비용을 숨기고 순액만 보여주면 사용자가 검증할 수 없다 (규칙 6)."""
    first = report.options[0]
    assert first.costs.total_label in report.headline
    assert first.net_proceeds_label in report.headline
    assert first.costs.total_label in first.cost_share_label


# --------------------------------------------------------------------------- #
# 계획에 미치는 영향
# --------------------------------------------------------------------------- #


def test_bigger_downsizing_leaves_more_and_lasts_longer(report):
    """많이 줄일수록 남는 돈이 많고 고갈이 늦다 — 순서가 뒤집히면 계산이 틀린 것이다."""
    nets = [o.net_proceeds for o in report.options]
    assert nets == sorted(nets)
    gained = [o.years_gained for o in report.options]
    assert gained == sorted(gained)


def test_demo_persona_actually_gains_years(report):
    """데모 인물은 부동산이 총자산의 69%다. 이 레버가 실제로 움직여야 한다."""
    assert report.baseline_depletion_age == 69
    assert report.options[-1].years_gained > 0
    assert "69%" in report.home_share_label


def test_explicit_price_computes_only_that_one():
    """가고 싶은 집이 정해진 사용자에게 사다리를 보여줄 이유가 없다."""
    report = analyze_downsizing(DEMO_PROFILE, new_home_price=350_000_000)
    assert len(report.options) == 1
    assert report.options[0].new_home_price == 350_000_000


def test_ladder_never_suggests_a_bigger_house():
    """다운사이징인데 더 비싼 집을 제안하면 안 된다."""
    report = analyze_downsizing(DEMO_PROFILE)
    assert len(report.options) == len(DEFAULT_RATIOS)
    assert all(o.new_home_price < DEMO_PROFILE.real_estate for o in report.options)


# --------------------------------------------------------------------------- #
# 건강보험 재산요건 — 집을 줄이는 두 번째 효과
# --------------------------------------------------------------------------- #


def test_property_tax_base_falls_with_the_house(report):
    """재산이 줄면 과세표준도 준다. 피부양자 재산요건이 여기에 걸려 있다."""
    bases = [o.property_tax_base for o in report.options]
    assert bases == sorted(bases, reverse=True)
    assert all(b < report.baseline_property_tax_base for b in bases)


def test_health_effect_is_always_stated(report):
    """효과가 없으면 없다고 말한다. 비워 두면 화면이 침묵한다."""
    assert all(o.health_effect for o in report.options)


def test_relief_is_reported_when_it_crosses_the_property_test():
    """과세표준 한도(5.4억)를 넘던 사람이 내려오면 그 사실이 나가야 한다.

    시가 15억이면 과세표준은 6.3억이라 소득과 무관하게 탈락한다. 집을 줄이면
    보험료가 0원이 되는 것이라 현금 이상의 효과다.
    """
    rich = DEMO_PROFILE.model_copy(update={"real_estate": 1_500_000_000})
    report = analyze_downsizing(rich)
    assert report.baseline_property_tax_base > 540_000_000
    assert any(o.relieves_property_test for o in report.options)
    relieved = next(o for o in report.options if o.relieves_property_test)
    assert "피부양자" in relieved.health_effect


# --------------------------------------------------------------------------- #
# 모르면 물어본다
# --------------------------------------------------------------------------- #


def test_without_a_house_it_asks_instead_of_guessing():
    """집값을 추정하지 않는다 (규칙 5). 시세를 지어내면 전부가 틀어진다."""
    renter = DEMO_PROFILE.model_copy(update={"real_estate": 0})
    report = analyze_downsizing(renter)
    assert not report.computable
    assert not report.options
    assert "알려주셔야" in report.reason


def test_capital_gains_tax_is_disclosed_as_missing(report):
    """넣지 않은 세금은 넣지 않았다고 말해야 한다. 이게 빠지면 과소추정이다."""
    assert any("양도소득세" in note for note in report.notes)
    assert any("지방교육세" in note for note in report.notes)


def test_brokerage_is_labelled_as_a_ceiling(report):
    """상한요율이라 실제는 더 낮을 수 있다."""
    assert any("상한요율" in note for note in report.notes)
