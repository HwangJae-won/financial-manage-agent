"""국민연금 임의계속가입·추납 테스트.

여기서 지켜야 하는 것이 세 가지 있다.

1. **가입 119개월과 120개월의 계단.** 앞쪽은 노령연금이 평생 0원이고 뒤쪽은
   죽을 때까지 나온다. 비례식으로 이어 버리면 "119개월도 절반은 받는다"는
   거짓말이 된다. 건강보험 금융소득 1,000만원 계단과 같은 이유로 1개월 차이를
   테스트로 고정한다.

2. **임의계속가입 창은 연금을 받기 시작하면 닫힌다.** 65세가 상한이지만 수급
   개시가 64세면 64세까지다. 조기수령을 고른 사람에게는 아예 열리지 않는다.

3. **연금을 늘리면 건강보험료가 따라온다.** 이 항목이 다른 기능과 붙는 지점이고,
   한쪽만 계산해서 "연금 늘리세요"라고 말하지 않기 위한 안전장치다.
"""

from __future__ import annotations

import json

import pytest

from agents.llm import ToolCall
from agents.tools import Toolbox
from core.assumptions import load_assumptions
from core.formatting import fmt_eul, fmt_euro, has_final_consonant
from core.national_pension import (
    BOTH,
    CATCHUP,
    VOLUNTARY,
    analyze_national_pension,
    catchup_window_months,
    contribution_cost,
    monthly_pension_for,
    voluntary_window_months,
)
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture(scope="module")
def spec():
    return load_assumptions().national_pension


# --------------------------------------------------------------------------- #
# 계단 — 119개월과 120개월
# --------------------------------------------------------------------------- #


def test_one_month_short_means_no_pension_at_all(spec):
    """119개월이면 0원, 120개월이면 나온다. 그 사이에 중간값은 없다."""
    assert spec.period_factor(119) == 0.0
    assert spec.period_factor(120) > 0.0


def test_the_amount_is_not_prorated_below_the_minimum(spec):
    """가입기간에 비례해 조금씩 나오는 것이 아니다 — 자격이 없으면 아예 없다."""
    anchor = dict(anchor_monthly=1_500_000, anchor_months=384, spec=spec)

    assert monthly_pension_for(119, **anchor) == 0
    assert monthly_pension_for(120, **anchor) > 0


def test_the_period_factor_matches_the_statute(spec):
    """20년(240개월)이 1.0, 초과 1년당 5% 가산, 미달 1년당 5% 감액.

    10년 가입은 0.5, 40년 가입은 2.0 — 국민연금법 별표 1의 구조 그대로다.
    """
    assert spec.period_factor(240) == pytest.approx(1.0)
    assert spec.period_factor(120) == pytest.approx(0.5)
    assert spec.period_factor(480) == pytest.approx(2.0)


def test_the_pension_scales_from_the_users_own_estimate(spec):
    """A값·B값을 지어내지 않는다. 공단 예상액을 기준점으로 계수의 비율만 쓴다.

    가입 384개월(계수 1.6)에 150만원을 받는 사람이 48개월을 더 채우면
    계수가 1.8이 되어 150만 × 1.8 ÷ 1.6 = 168만 7,500원이 된다.
    """
    assert (
        monthly_pension_for(
            432, anchor_monthly=1_500_000, anchor_months=384, spec=spec
        )
        == 1_687_500
    )


# --------------------------------------------------------------------------- #
# 보험료 — 전액 본인 부담
# --------------------------------------------------------------------------- #


def test_contribution_matches_hand_calculation(spec):
    """기준소득월액 500만원 × 9% × 48개월 = 2,160만원."""
    assert contribution_cost(5_000_000, 48, spec) == 21_600_000


def test_the_income_base_is_capped(spec):
    """급여가 상한을 넘어도 보험료는 상한까지만 매겨진다."""
    assert contribution_cost(50_000_000, 12, spec) == contribution_cost(
        spec.income_base_max, 12, spec
    )


def test_the_demo_persona_hits_the_cap():
    """월 625만원을 받던 사람의 기준소득월액은 617만원에서 잘린다."""
    report = analyze_national_pension(DEMO_PROFILE)

    assert report.income_base_capped
    assert report.income_base == load_assumptions().national_pension.income_base_max
    assert any("상한" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# 임의계속가입 창 — 연금을 받으면 닫힌다
# --------------------------------------------------------------------------- #


def test_the_window_closes_at_the_pension_start_age(spec):
    """상한은 65세지만 수급 개시가 64세면 64세까지다 (60→64세 = 48개월)."""
    assert voluntary_window_months(DEMO_PROFILE, spec) == 48


def test_taking_the_pension_early_closes_the_window_entirely(spec):
    """60세부터 받기로 하면 그 순간 자격이 사라져 창이 열리지 않는다."""
    early = DEMO_PROFILE.model_copy(update={"national_pension_start_age": 60})

    assert voluntary_window_months(early, spec) == 0

    report = analyze_national_pension(early)
    assert not report.voluntary.available
    assert "자격이 사라져" in report.voluntary.unavailable_reason


def test_deferring_does_not_extend_past_the_legal_ceiling(spec):
    """68세로 미뤄도 임의계속가입은 65세까지다 (60→65세 = 60개월)."""
    late = DEMO_PROFILE.model_copy(update={"national_pension_start_age": 68})

    assert voluntary_window_months(late, spec) == 60


def test_the_catchup_window_is_capped_at_119_months(spec):
    """2020년 개정으로 추납 상한이 생겼다. 수십 년치를 한 번에 넣을 수 없다."""
    long_gap = DEMO_PROFILE.model_copy(update={"pension_catchup_months": 240})

    assert catchup_window_months(long_gap, spec) == spec.max_catchup_months


def test_an_unavailable_option_carries_no_numbers():
    """쓸 수 없는 수단에 숫자가 남으면 화면도 에이전트도 선택지로 인용한다."""
    report = analyze_national_pension(DEMO_PROFILE)  # 추납 여지가 없는 사람

    assert not report.catchup.available
    assert report.catchup.cost == 0
    assert report.catchup.monthly_gain == 0
    assert report.catchup.unavailable_reason


# --------------------------------------------------------------------------- #
# 건강보험과의 상충 — 이 기능이 존재하는 이유
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def demo():
    return analyze_national_pension(DEMO_PROFILE)


def test_more_pension_creates_more_health_premium(demo):
    """공적연금소득은 피부양자 판정에서 전액 반영된다. 늘리면 건보료가 따라온다."""
    assert demo.voluntary.extra_premium > 0
    assert demo.voluntary.net_gain < demo.voluntary.lifetime_gain - demo.voluntary.cost + 1


def test_the_headline_subtracts_the_premium_from_the_gain(demo):
    """'연금이 늘어납니다'로 끝내지 않는다. 건보료를 빼고 남는 것으로 말한다."""
    assert "건강보험료" in demo.headline
    assert demo.voluntary.net_gain_label in demo.headline


def test_more_pension_can_pull_the_health_cliff_forward():
    """가만히 있으면 2034년에 올 절벽이 연금을 늘리면 2031년으로 당겨진다.

    이 상충을 한 화면에서 말하는 것이 이 항목의 값어치다. 연금만 계산해서
    "늘리세요"라고 말하면 이 3년이 보이지 않는다.
    """
    near_line = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 1_400_000})
    report = analyze_national_pension(near_line)

    assert report.current.cliff_year == 2034
    assert report.voluntary.cliff_year == 2031
    assert report.voluntary.cliff_shift_years == 3
    assert "3년 앞당겨집니다" in report.headline


def test_a_thin_gain_is_called_a_loss():
    """연금이 적은 사람에게는 낸 돈이 회수되지 않는다. 그대로 '손해'라고 말한다."""
    thin = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 400_000})
    report = analyze_national_pension(thin)

    assert report.voluntary.net_gain < 0
    assert report.best_key == "none"
    assert "손해" in report.headline
    assert "하실 이유가 없습니다" in report.verdict


def test_the_loss_reason_matches_the_cause():
    """건보료가 잡아먹은 경우와 연금 자체가 얇은 경우는 할 일이 다르다."""
    thin = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 400_000})
    report = analyze_national_pension(thin)

    assert report.voluntary.extra_premium == 0
    assert "평생 더 받는 연금이" in report.headline
    assert "건강보험료가 평생" not in report.headline


# --------------------------------------------------------------------------- #
# 가입기간이 모자란 사람
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def short_profile():
    """가입 110개월 — 10개월이 모자라 노령연금이 평생 0원인 사람."""
    return DEMO_PROFILE.model_copy(
        update={"national_pension_months": 110, "pension_catchup_months": 24}
    )


def test_the_zero_pension_is_stated_plainly(short_profile):
    report = analyze_national_pension(short_profile)

    assert not report.qualifies_now
    assert report.months_to_qualify == 10
    assert report.current.monthly_pension == 0
    assert "평생 0원" in report.headline


def test_the_cost_to_just_qualify_is_separated_from_the_full_option(short_profile):
    """'10개월만 더 내면 227만원'처럼 읽히면 안 된다.

    자격을 채우는 값(120개월)과 수단을 끝까지 쓴 값은 다른 숫자다.
    """
    report = analyze_national_pension(short_profile)

    assert report.cost_to_qualify > 0
    assert report.monthly_at_minimum > 0
    assert report.monthly_at_minimum < report.both.monthly_pension
    assert report.cost_to_qualify_label in report.headline
    assert report.monthly_at_minimum_label in report.headline


def test_filling_the_gap_is_worth_it(short_profile):
    """0원과 평생 연금 사이의 선택이라 회수 시점이 압도적으로 빠르다."""
    report = analyze_national_pension(short_profile)

    assert report.both.breakeven_years is not None
    assert report.both.breakeven_years < 5
    assert "고민하실 필요가 없습니다" in report.verdict


# --------------------------------------------------------------------------- #
# 모르면 추측하지 않는다
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "update, missing",
    [
        ({"national_pension_months": 0}, "가입월수"),
        ({"national_pension_monthly": 0}, "예상 월 수령액"),
        ({"last_monthly_salary": 0}, "기준소득월액"),
    ],
)
def test_missing_inputs_stop_the_calculation(update, missing):
    """기준소득월액을 0으로 두면 보험료가 0원이 되어 **공짜처럼** 보인다."""
    report = analyze_national_pension(DEMO_PROFILE.model_copy(update=update))

    assert not report.computable
    assert missing in report.headline
    assert "알려주시면" in report.headline
    assert not report.voluntary.available


def test_the_unknown_case_points_at_the_source():
    """어디서 확인하는지까지 말해야 사용자가 실제로 값을 가져온다."""
    report = analyze_national_pension(
        DEMO_PROFILE.model_copy(update={"national_pension_months": 0})
    )

    assert "1355" in report.verdict
    assert "119개월과 120개월" in report.verdict


# --------------------------------------------------------------------------- #
# 계획에 미치는 영향
# --------------------------------------------------------------------------- #


def test_the_cost_is_charged_to_the_plan(demo):
    """목돈이 나가면 고갈이 앞당겨질 수 있다. 연금이 느는 것과 별개의 사실이다."""
    assert demo.voluntary.depletion_age is not None
    assert demo.voluntary.depletion_age <= demo.current.depletion_age


def test_the_catchup_and_voluntary_add_up():
    """둘 다 쓰면 개월 수와 비용이 각각의 합이다."""
    report = analyze_national_pension(DIVERSIFIED_PROFILE)

    assert report.both.months_added == (
        report.catchup.months_added + report.voluntary.months_added
    )
    assert report.both.cost == report.catchup.cost + report.voluntary.cost
    assert report.both.monthly_gain > report.voluntary.monthly_gain


def test_the_analysis_does_not_mutate_the_profile():
    analyze_national_pension(DEMO_PROFILE)

    assert DEMO_PROFILE.life_events == []
    assert DEMO_PROFILE.national_pension_monthly == 1_500_000


# --------------------------------------------------------------------------- #
# 조사 — 화면에 '둘 다으로'가 나가지 않게
# --------------------------------------------------------------------------- #


def test_the_particle_follows_the_final_consonant():
    assert fmt_euro("임의계속가입") == "임의계속가입으로"
    assert fmt_euro("둘 다") == "둘 다로"
    assert fmt_eul("가입월수") == "가입월수를"
    assert fmt_eul("예상 월 수령액") == "예상 월 수령액을"


def test_trailing_punctuation_does_not_confuse_the_particle():
    """'퇴직 전 월 급여(기준소득월액)' 처럼 괄호로 끝나는 말이 실제로 쓰인다."""
    assert has_final_consonant("퇴직 전 월 급여(기준소득월액)")
    assert fmt_eul("퇴직 전 월 급여(기준소득월액)").endswith("을")


def test_every_option_label_gets_a_readable_particle():
    for label in (CATCHUP, VOLUNTARY, BOTH):
        assert "다으로" not in fmt_euro(label)


# --------------------------------------------------------------------------- #
# 에이전트 도구
# --------------------------------------------------------------------------- #


def test_the_agent_can_ask_whether_to_pay_more():
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(ToolCall(id="c1", name="national_pension_options", arguments={}))
    payload = json.loads(result.content)

    assert not result.is_error
    assert payload["가입월수"] == 384
    assert payload["수급자격"] == "있음"
    assert payload["임의계속가입"]["가능"] is True
    assert payload["임의계속가입"]["추가_건강보험료"]
    assert payload["주의"]


def test_the_agent_is_told_what_it_may_not_guess():
    """가입월수를 모르면 도구 출력에서도 '물어보라'고 나온다."""
    box = Toolbox(DEMO_PROFILE.model_copy(update={"national_pension_months": 0}))
    result = box.execute(ToolCall(id="c1", name="national_pension_options", arguments={}))
    payload = json.loads(result.content)

    assert payload["계산_가능"] is False
    assert "알려주시면" in payload["한_문장"]
