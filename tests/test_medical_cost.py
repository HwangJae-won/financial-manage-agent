"""의료비·본인부담상한제 테스트.

이 기능이 지켜야 하는 것은 **모르는 것을 지어내지 않는 것**이다. 상한액 표가
시행령 별표의 첨부파일(hwp/pdf)이라 기계로 읽을 수 없다. 그래서 표를 들고 있지
않고, 대신 알려주시면 계산한다 — 근속연수·가입월수와 같은 규약이다.

그리고 상한액을 몰라도 **의료비가 계획에 미치는 영향은 계산한다.** 사용자가
실제로 알고 싶은 것이 그것이기 때문이다.
"""

from __future__ import annotations

import pytest

from core.medical_cost import COVERED_SHARE, analyze_medical_cost
from core.samples import DEMO_PROFILE


@pytest.fixture(scope="module")
def unknown():
    return analyze_medical_cost(DEMO_PROFILE)


@pytest.fixture(scope="module")
def known():
    return analyze_medical_cost(DEMO_PROFILE, annual_ceiling=4_000_000)


# --------------------------------------------------------------------------- #
# 모르면 지어내지 않는다
# --------------------------------------------------------------------------- #


def test_without_a_ceiling_nothing_is_refunded(unknown):
    """상한액을 모르면 상한제를 적용하지 않는다. 추측한 표를 쓰지 않는다."""
    assert not unknown.ceiling_known
    assert all(s.refund == 0 for s in unknown.scenarios)
    assert all(not s.ceiling_applied for s in unknown.scenarios)


def test_the_missing_table_is_disclosed(unknown):
    """왜 계산하지 않았는지, 어디서 확인하는지까지 말해야 한다."""
    assert any("들고 있지 않습니다" in note for note in unknown.notes)
    assert "1577-1000" in unknown.verdict or "공단" in unknown.verdict


def test_the_mechanism_is_explained_even_without_the_number(unknown):
    """금액을 몰라도 '상한제가 있다'는 사실 자체가 이 기능의 값어치다."""
    assert "본인부담상한제" in unknown.headline
    assert "돌려줍니다" in unknown.headline


def test_the_unknown_case_says_the_numbers_are_overstated(unknown):
    """상한제를 안 뺀 값이라 실제보다 크다는 것을 말해야 한다."""
    assert "실제보다 큽니다" in unknown.verdict


# --------------------------------------------------------------------------- #
# 알려주면 계산한다
# --------------------------------------------------------------------------- #


def test_the_ceiling_caps_the_covered_portion(known):
    """급여 본인부담이 상한을 넘으면 초과분이 환급된다."""
    big = known.scenarios[-1]
    covered = int(big.total_cost * COVERED_SHARE)

    assert big.ceiling_applied
    assert big.refund == covered - known.ceiling
    assert big.out_of_pocket == big.total_cost - big.refund


def test_a_small_bill_is_not_capped(known):
    """상한에 못 미치면 환급이 없다."""
    small = known.scenarios[0]
    assert not small.ceiling_applied
    assert small.out_of_pocket == small.total_cost


def test_the_ceiling_reduces_the_damage(known, unknown):
    """상한제가 있으면 같은 병원비라도 계획이 덜 흔들린다."""
    with_ceiling = known.scenarios[-1]
    without = unknown.scenarios[-1]

    assert with_ceiling.out_of_pocket < without.out_of_pocket
    assert with_ceiling.years_lost < without.years_lost


def test_the_headline_leads_with_the_real_burden(known):
    """'수억' 이 아니라 실제로 내는 금액이 먼저 나와야 한다."""
    big = known.scenarios[-1]
    assert big.out_of_pocket_label in known.headline
    assert big.refund_label in known.headline


# --------------------------------------------------------------------------- #
# 한계를 함께 낸다
# --------------------------------------------------------------------------- #


def test_non_covered_costs_are_disclosed(known):
    """비급여·간병비는 상한에 안 들어간다. 이게 실제 부담의 큰 몫이다."""
    assert any("비급여" in note and "간병비" in note for note in known.notes)


def test_the_refund_timing_is_disclosed(known):
    """돌려받는 것은 다음 해다. 현금흐름 관점에서 시점이 중요하다."""
    assert any("다음 해" in note for note in known.notes)


def test_the_covered_share_is_called_an_assumption(known):
    """급여 비중은 질환마다 다르다. 개인의 실제 비율이 아니라고 말해야 한다."""
    assert any("실제 비율이 아닙니다" in note for note in known.notes)


def test_the_analysis_does_not_mutate_the_profile():
    analyze_medical_cost(DEMO_PROFILE, annual_ceiling=4_000_000)
    assert DEMO_PROFILE.life_events == []
