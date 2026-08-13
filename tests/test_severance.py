"""퇴직금 수령 방식 테스트.

이 서비스에서 유일하게 실효세율 근사가 아니라 **제도의 계산 구조를 그대로**
따라가는 세금이다. 평균 세율 하나로 뭉개면 "장기근속자는 퇴직소득세가 거의 없다"는
사실이 사라지는데, 타겟(30년 안팎 근속 후 퇴직)에게는 그게 핵심이기 때문이다.

그래서 손계산 대조를 고정한다. 세율표를 손대면 여기가 먼저 깨진다.
"""

from __future__ import annotations

import json

import pytest

from agents.llm import ToolCall
from agents.tools import Toolbox
from core.assumptions import load_assumptions
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.severance import (
    compare_severance_options,
    pension_income_tax,
    retirement_income_tax,
    service_deduction,
)


# --------------------------------------------------------------------------- #
# 손계산 대조 — 세율표를 손대면 여기가 먼저 깨진다
# --------------------------------------------------------------------------- #


def test_the_demo_persona_tax_matches_hand_calculation():
    """퇴직금 2억 · 근속 32년.

        근속연수공제 = 4,000만 + 300만 × 12          = 7,600만원
        환산급여     = (2억 - 7,600만) ÷ 32 × 12     = 4,650만원
        환산급여공제 = 800만 + (4,650만-800만) × 60% = 3,110만원
        과세표준     = 4,650만 - 3,110만            = 1,540만원
        산출세액     = (1,400만×6% + 140만×15%)      =   105만원
        연분연승     = 105만 ÷ 12 × 32              =   280만원
        지방소득세 포함                              =   308만원
    """
    assert retirement_income_tax(200_000_000, 32) == 3_080_000


def test_service_deduction_bands():
    assert service_deduction(5, load_assumptions().retirement_income_tax) == 5_000_000
    assert service_deduction(32, load_assumptions().retirement_income_tax) == 76_000_000


def test_long_service_pays_far_less_tax():
    """근속연수공제가 커서 장기근속자는 세금이 훨씬 적다 — 화면에서 말할 사실."""
    same_amount = 200_000_000
    short = retirement_income_tax(same_amount, 5)
    long = retirement_income_tax(same_amount, 40)

    assert short > long * 10
    assert long / same_amount < 0.02  # 40년 근속이면 실효세율 2% 미만


def test_tax_rises_with_the_amount():
    for years in (10, 30):
        assert retirement_income_tax(300_000_000, years) > retirement_income_tax(
            100_000_000, years
        )


def test_no_severance_or_no_service_means_no_tax():
    """입력이 불완전해도 화면이 멈추면 안 된다."""
    assert retirement_income_tax(0, 30) == 0
    assert retirement_income_tax(200_000_000, 0) == 0


def test_deduction_larger_than_severance_means_no_tax():
    """장기근속에 소액 퇴직금이면 공제가 퇴직금보다 크다."""
    assert retirement_income_tax(30_000_000, 30) == 0


# --------------------------------------------------------------------------- #
# 연금 감면
# --------------------------------------------------------------------------- #


def test_pension_receipt_is_discounted():
    """10년 이내 수령분은 30% 감면된다."""
    full = retirement_income_tax(200_000_000, 32)
    assert pension_income_tax(200_000_000, 32, 10) == pytest.approx(full * 0.7, rel=1e-3)


def test_receiving_longer_discounts_more():
    """11년차부터 40% 감면이라 오래 나눠 받을수록 총액이 줄어든다."""
    ten = pension_income_tax(200_000_000, 32, 10)
    twenty = pension_income_tax(200_000_000, 32, 20)
    assert twenty < ten


def test_pension_is_never_worse_than_a_lump_sum():
    """감면 제도이므로 세액이 늘어날 수는 없다 (부호 불변식)."""
    for severance, years in ((200_000_000, 32), (500_000_000, 10), (50_000_000, 25)):
        full = retirement_income_tax(severance, years)
        assert pension_income_tax(severance, years, 10) <= full


# --------------------------------------------------------------------------- #
# 비교 — 세액만이 아니라 납부 시점까지 본다
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo_comparison():
    return compare_severance_options(DEMO_PROFILE)


def test_the_comparison_simulates_both_paths(demo_comparison):
    assert demo_comparison.lump_sum.depletion_label
    assert demo_comparison.pension.depletion_label
    assert demo_comparison.tax_saved > 0


def test_the_timing_difference_is_stated(demo_comparison):
    """세율표만 봐서는 안 보이는 부분 — 언제 내느냐가 결과를 가른다."""
    assert "한 번에" in demo_comparison.lump_sum.when_paid
    assert "나눠" in demo_comparison.pension.when_paid


def test_paying_later_leaves_more_money():
    """세금을 나눠 내면 그동안 그 돈이 운용된다."""
    result = compare_severance_options(DIVERSIFIED_PROFILE)
    assert result.pension.final_balance > result.lump_sum.final_balance


def test_the_effective_rate_is_shown(demo_comparison):
    """'퇴직금 2억에 세금 308만원'은 실효세율로 봐야 의미가 잡힌다."""
    assert "%" in demo_comparison.lump_sum.effective_rate_label
    assert demo_comparison.lump_sum.total_tax > demo_comparison.pension.total_tax


def test_the_limitation_is_disclosed(demo_comparison):
    """IRP 과세이연을 반영하지 않았다는 사실을 숨기지 않는다."""
    assert any("과세이연" in note for note in demo_comparison.notes)
    assert any("참고용" in note for note in demo_comparison.notes)


def test_a_profile_without_severance_says_so():
    no_severance = DEMO_PROFILE.model_copy(update={"severance_pay": 0})
    result = compare_severance_options(no_severance)
    assert "퇴직금이 없으" in result.headline


def test_the_comparison_does_not_mutate_the_profile():
    """세금을 일회성 지출로 표현하지만, 사용자의 실제 프로파일은 그대로여야 한다."""
    compare_severance_options(DEMO_PROFILE)
    assert DEMO_PROFILE.life_events == []


def test_longer_pension_period_saves_more_tax():
    ten = compare_severance_options(DEMO_PROFILE, pension_years=10)
    twenty = compare_severance_options(DEMO_PROFILE, pension_years=20)
    assert twenty.tax_saved > ten.tax_saved


# --------------------------------------------------------------------------- #
# 에이전트 도구
# --------------------------------------------------------------------------- #


def test_the_agent_can_compare_severance_options():
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(ToolCall(id="c1", name="severance_options", arguments={}))
    payload = json.loads(result.content)

    assert not result.is_error
    assert payload["일시금"]["세금"] == "308만원"
    assert payload["연금"]["수령기간"] == "10년"
    assert payload["주의"]


def test_the_agent_can_change_the_pension_period():
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(
        ToolCall(id="c1", name="severance_options", arguments={"pension_years": 20})
    )
    assert json.loads(result.content)["연금"]["수령기간"] == "20년"


# --------------------------------------------------------------------------- #
# 근속연수를 모를 때 — 추측하지 않는다
# --------------------------------------------------------------------------- #


def test_unknown_service_years_asks_instead_of_guessing():
    """기본값으로 채우면 실효세율이 몇 배씩 어긋난다. 0원이라고 답해서도 안 된다."""
    unknown = DEMO_PROFILE.model_copy(update={"years_employed": 0})
    result = compare_severance_options(unknown)

    assert not result.computable
    assert "몇 년 근무" in result.headline
    assert "추측해서 알려드리지 않" in result.verdict


def test_a_complete_profile_is_computable():
    assert compare_severance_options(DEMO_PROFILE).computable
