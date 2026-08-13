"""퇴직 후 건강보험료 절벽 테스트.

여기서 지켜야 하는 것은 **계단**이다. 금융소득 999만원과 1,001만원은 연속적으로
다르지 않다. 앞쪽은 합산소득에서 아예 빠지고 뒤쪽은 전액이 들어간다. 이 계단을
비례식으로 뭉개는 순간 이 기능이 존재할 이유가 사라지므로, 1원 차이를 테스트로
고정한다.
"""

from __future__ import annotations

import json

import pytest

from agents.llm import ToolCall
from agents.tools import Toolbox
from core.assumptions import load_assumptions
from core.cashflow import annual_tax
from core.health_insurance import (
    DEPENDENT,
    LOCAL,
    VOLUNTARY,
    analyze_health_insurance,
    counted_financial_income,
    dependent_income,
    financial_income_tax_only,
    local_monthly_premium,
    voluntary_monthly_premium,
)
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture(scope="module")
def spec():
    return load_assumptions().health_insurance


# --------------------------------------------------------------------------- #
# 계단 — 이 기능이 존재하는 이유
# --------------------------------------------------------------------------- #


def test_financial_income_is_all_or_nothing(spec):
    """1원 차이로 합산소득이 1,000만원 넘게 뛴다."""
    assert counted_financial_income(9_999_999, spec) == 0
    assert counted_financial_income(10_000_001, spec) == 10_000_001


def test_one_won_more_interest_triples_the_premium(spec):
    """'금리 높은 상품으로 갈아타세요'가 이 구간의 고객에게 실제로 얼마인가."""
    under = local_monthly_premium(0, 9_999_999, 0, spec)
    over = local_monthly_premium(0, 10_000_001, 0, spec)

    assert over > under * 2


def test_private_pension_is_not_counted(spec):
    """사적연금(IRP·연금저축)은 합산소득에 들어가지 않는다 — 세지 않는다."""
    assert dependent_income(0, 0, 0, spec) == 0


# --------------------------------------------------------------------------- #
# 손계산 대조 — 요율을 손대면 여기가 먼저 깨진다
# --------------------------------------------------------------------------- #


def test_local_premium_matches_hand_calculation(spec):
    """공적연금 연 2,000만원만 있는 지역가입자.

        부과 대상 = 2,000만 × 50%            = 1,000만원
        건강보험료 = 1,000만 × 7.09% ÷ 12    =  59,083원
        장기요양 포함 (× 1.1295)             =  66,735원
    """
    assert local_monthly_premium(20_000_000, 0, 0, spec) == 66_735


def test_the_minimum_premium_applies(spec):
    """소득이 없어도 최저보험료는 나온다."""
    assert local_monthly_premium(0, 0, 0, spec) == 22_342


def test_voluntary_premium_matches_hand_calculation(spec):
    """퇴직 전 월 급여 500만원 → 5,000,000 × 7.09% × 50% × 1.1295."""
    assert voluntary_monthly_premium(5_000_000, spec) == 200_204


def test_pension_counts_double_for_eligibility_but_half_for_the_bill(spec):
    """같은 연금소득인데 자격 판정은 전액, 보험료 부과는 절반이다."""
    pension = 20_000_000
    assert dependent_income(pension, 0, 0, spec) == pension
    assert local_monthly_premium(pension, 0, 0, spec) == local_monthly_premium(
        0, 0, pension // 2, spec
    )


# --------------------------------------------------------------------------- #
# 절벽은 가만히 있어도 다가온다
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo():
    return analyze_health_insurance(DEMO_PROFILE, monthly_salary=5_000_000)


def test_the_demo_persona_is_free_now_but_not_later(demo):
    """국민연금이 개시되면 합산소득이 한 번에 올라가 2,000만원을 넘는다.

    1966년생·국민연금 월 150만원. 2030년 12월 개시라 첫 온전한 해가 2031년이고,
    물가연동으로 연 2,016만원이 되어 한도를 넘는다. 지금은 0원인데 5년 뒤 절벽이다.
    """
    assert demo.qualifies_at_retirement
    assert demo.cliff_year == 2031
    assert demo.cliff_age == 65
    assert "합산소득" in demo.cliff_reason


def test_the_cliff_year_is_stated_with_money(demo):
    assert demo.local.monthly > 0
    assert "2031년" in demo.headline
    assert demo.total_premiums > 0


def test_the_headroom_uses_a_full_year(demo):
    """12월 퇴직이면 퇴직 연도 소득이 12분의 1이라, 그 해로 말하면 여유가 부풀려진다."""
    assert demo.financial_income > 5_000_000
    assert 0 < demo.financial_headroom < 5_000_000
    assert demo.headroom_rate_label.endswith("p")


def test_a_lower_pension_delays_the_cliff():
    """소득이 적으면 절벽이 뒤로 밀리거나 사라진다 — '고객님껜 해당하지 않습니다'."""
    modest = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 700_000})
    result = analyze_health_insurance(modest)

    assert result.cliff_year is None
    assert "0원입니다" in result.headline


def test_a_qualifying_person_still_hears_the_step():
    """지금 통과해도 계단이 어디인지는 말해 준다."""
    modest = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 700_000})
    result = analyze_health_insurance(modest)

    assert "전액이 합산소득에 들어가" in result.headline
    assert result.financial_headroom > 0


# --------------------------------------------------------------------------- #
# 재산요건
# --------------------------------------------------------------------------- #


def test_property_alone_can_disqualify():
    rich = DEMO_PROFILE.model_copy(update={"real_estate": 2_000_000_000})
    result = analyze_health_insurance(rich)

    assert not result.qualifies_at_retirement
    assert "재산세 과세표준" in result.cliff_reason


def test_the_property_estimate_is_disclosed(demo):
    """시가 → 공시가격 → 과세표준으로 근사가 두 번 겹쳐 있다. 숨기지 않는다."""
    assert demo.property_assumed
    assert any("과세표준을 부동산 시가의" in note for note in demo.notes)


def test_a_real_tax_base_replaces_the_estimate():
    result = analyze_health_insurance(DEMO_PROFILE, property_tax_base_override=100_000_000)

    assert not result.property_assumed
    assert result.property_tax_base == 100_000_000


# --------------------------------------------------------------------------- #
# 모르면 추측하지 않는다
# --------------------------------------------------------------------------- #


def test_without_a_salary_the_voluntary_premium_is_not_guessed():
    """급여에 따라 금액이 달라진다. 기본값으로 채우면 사용자가 그 숫자를 믿는다."""
    unknown = DEMO_PROFILE.model_copy(update={"last_monthly_salary": 0})
    result = analyze_health_insurance(unknown)

    assert not result.voluntary.available
    assert "알려주시면" in result.voluntary.basis
    assert any("퇴직 전 월 급여를 알려주시면" in note for note in result.notes)


def test_an_unknown_family_situation_is_stated_as_a_condition(demo):
    """피부양자는 직장 다니는 가족이 있어야 가능하다. 모르면 조건을 붙여 말한다."""
    assert demo.headline.startswith("직장 다니는 배우자나 자녀가 계시다면")
    assert any("직장에 다니는 배우자·자녀가 있어야" in note for note in demo.notes)


def test_no_employed_family_means_the_bill_starts_at_retirement():
    result = analyze_health_insurance(DEMO_PROFILE, has_employed_family=False)

    assert not result.qualifies_at_retirement
    assert result.cliff_year == DEMO_PROFILE.retirement_year
    assert "즉시 지역가입자" in result.headline


def test_the_missing_property_component_is_disclosed(demo):
    """지역가입자 보험료에서 재산분을 뺐다는 사실과 **오차의 방향**을 함께 말한다."""
    assert "재산분은 빠져" in demo.local.basis
    assert any("실제 부담은 여기 나온 금액보다 큽니다" in note for note in demo.notes)


# --------------------------------------------------------------------------- #
# 임의계속가입 — 퇴직 직후에만 열리는 창
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def high_income_profile():
    """기타소득이 커서 퇴직 즉시 지역가입자가 되는 사람."""
    return DEMO_PROFILE.model_copy(
        update={"other_monthly_income": 5_000_000, "retirement_month": 1}
    )


def test_the_cheaper_path_is_used_first(high_income_profile):
    result = analyze_health_insurance(high_income_profile, monthly_salary=3_000_000)

    assert result.voluntary.monthly < result.local.monthly
    assert VOLUNTARY in result.years[0].method
    assert result.years[0].method != DEPENDENT


def test_the_window_closes_after_three_years(high_income_profile):
    result = analyze_health_insurance(high_income_profile, monthly_salary=3_000_000)
    methods = [row.method for row in result.years]

    assert VOLUNTARY in methods[0]
    assert methods[4] == LOCAL  # 36개월이 지나면 더 이상 못 쓴다


def test_an_expensive_salary_keeps_the_local_premium(high_income_profile):
    """재직 때 보험료가 더 비쌌다면 임의계속가입은 쓸 이유가 없다."""
    result = analyze_health_insurance(high_income_profile, monthly_salary=30_000_000)

    assert result.years[0].method == LOCAL


# --------------------------------------------------------------------------- #
# 계획에 미치는 영향 — 두 번 세지 않는다
# --------------------------------------------------------------------------- #


def test_the_engine_tax_model_drops_the_approximation(base_assumptions):
    """기본 엔진은 금융소득에 건보료 근사를 얹는다. 이 모듈은 그것을 빼고 돌린다."""
    taxable = 30_000_000
    assert financial_income_tax_only(taxable, base_assumptions) < annual_tax(
        taxable, base_assumptions
    )


def test_premiums_can_move_the_depletion_year():
    """보험료는 지출이다. 엔진을 고치지 않고 일회성 지출로 얹어 그대로 반영한다."""
    result = analyze_health_insurance(DIVERSIFIED_PROFILE, monthly_salary=4_000_000)

    assert result.depletion_advanced_years > 0
    assert result.total_premiums > 0


def test_the_analysis_does_not_mutate_the_profile():
    analyze_health_insurance(DEMO_PROFILE)
    assert DEMO_PROFILE.life_events == []


def test_premiums_are_deflated_before_they_enter_the_engine():
    """엔진은 `LifeEvent` 금액을 퇴직 시점 기준으로 보고 물가만큼 키운다.

    보험료는 이미 그 해의 명목 소득에서 계산했으므로, 되돌려 넣지 않으면 물가가
    두 번 곱해진다. 뒤로 갈수록 실질 부담이 커지는 것은 기준금액이 고정이기
    때문이지 물가를 두 번 곱해서가 아니어야 한다.
    """
    result = analyze_health_insurance(DEMO_PROFILE)
    paying = [row for row in result.years if row.annual_premium > 0]
    total_nominal = sum(row.annual_premium for row in paying)

    assert result.total_premiums < total_nominal


# --------------------------------------------------------------------------- #
# 에이전트 도구
# --------------------------------------------------------------------------- #


def test_the_agent_can_ask_about_the_cliff():
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(
        ToolCall(id="c1", name="health_insurance_cliff", arguments={"monthly_salary": 5_000_000})
    )
    payload = json.loads(result.content)

    assert not result.is_error
    assert payload["지금_피부양자인가"] is True
    assert "2031년" in payload["탈락_시점"]
    assert payload["임의계속가입"]["가능"] is True
    assert payload["주의"]


def test_the_agent_gets_the_uncomputed_parts_as_questions():
    """모르는 값은 도구 출력에서도 '물어보라'고 나온다 — 모델이 지어내지 않게."""
    box = Toolbox(DEMO_PROFILE.model_copy(update={"last_monthly_salary": 0}))
    result = box.execute(ToolCall(id="c1", name="health_insurance_cliff", arguments={}))
    payload = json.loads(result.content)

    assert payload["임의계속가입"]["가능"] is False
    assert "알려주시면" in payload["임의계속가입"]["근거"]
