"""처방 엔진 테스트.

가장 중요한 것은 **왕복 검증**이다. "월 187만원으로 줄이면 95세까지 유지됩니다"
라고 답했으면, 그 값을 실제로 엔진에 넣었을 때 정말 95세까지 유지되어야 한다.
처방이 틀리면 진단이 틀린 것보다 나쁘다 — 사용자가 그 말을 믿고 생활을 바꾼다.

경계도 함께 본다. 답이 '가능한 최대 생활비'라면 거기서 한 칸 더 올린 값은
실패해야 한다. 그렇지 않으면 필요 이상으로 허리띠를 졸라매게 만드는 답이다.
"""

from __future__ import annotations

import pytest

from core.assumptions import load_assumptions
from core.cashflow import simulate
from core.models import RiskTolerance, UserProfile
from core.prescribe import STEP, Lever, Unit, prescribe
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture(scope="module")
def demo_plan():
    return prescribe(DEMO_PROFILE)


def _option(plan, lever: Lever):
    return next(o for o in plan.options if o.lever is lever)


def _depletion(profile, **overrides):
    return simulate(profile.model_copy(update=overrides)).depletion_age


# --------------------------------------------------------------------------- #
# 수급 시기 조정 계수
# --------------------------------------------------------------------------- #


def test_normal_start_age_is_unadjusted(base_assumptions):
    """법정 연령에 받으면 계수는 정확히 1.0 이어야 한다 (기존 결과 불변의 근거)."""
    normal = base_assumptions.national_pension_start_age(1966)
    assert base_assumptions.pension_amount_factor(1966, normal) == 1.0


def test_deferral_adds_and_early_claim_subtracts(base_assumptions):
    normal = base_assumptions.national_pension_start_age(1966)

    assert base_assumptions.pension_amount_factor(1966, normal + 5) == pytest.approx(1.36)
    assert base_assumptions.pension_amount_factor(1966, normal - 5) == pytest.approx(0.70)
    assert base_assumptions.pension_amount_factor(1966, normal + 1) == pytest.approx(1.072)


def test_adjustment_is_capped_at_the_legal_limit(base_assumptions):
    """제도 한도를 넘겨 물어도 계산을 멈추지 않고 한도까지만 반영한다."""
    normal = base_assumptions.national_pension_start_age(1966)
    capped = base_assumptions.pension_amount_factor(1966, normal + 5)
    assert base_assumptions.pension_amount_factor(1966, normal + 12) == capped


def test_existing_profiles_are_unaffected():
    """샘플 두 명은 법정 연령을 쓰므로 W1~W7 숫자가 하나도 바뀌면 안 된다."""
    assert simulate(DEMO_PROFILE).depletion_year == 2035
    assert simulate(DEMO_PROFILE).balance_at_pension_start == pytest.approx(
        118_552_982, rel=1e-6
    )


# --------------------------------------------------------------------------- #
# 왕복 검증 — 처방대로 하면 정말 되는가
# --------------------------------------------------------------------------- #


def test_expense_prescription_actually_works(demo_plan):
    """답한 생활비를 엔진에 그대로 넣으면 목표 나이까지 버텨야 한다.

    이 테스트가 이 모듈에서 가장 중요하다. 처방이 틀리면 사용자가 그 말을 믿고
    실제로 생활을 바꾼다.
    """
    option = _option(demo_plan, Lever.EXPENSE)
    assert option.feasible
    assert option.required_value is not None
    assert option.required_value < option.current_value

    depletion = _depletion(DEMO_PROFILE, monthly_expense=option.required_value)
    assert depletion is None or depletion >= demo_plan.target_age


def _highest_expense_that_survives(profile, target_age: int) -> int:
    """테스트용 독립 구현 — 처방 엔진과 같은 답이 나와야 한다."""
    value = STEP
    best = STEP
    while value <= profile.monthly_expense:
        depletion = _depletion(profile, monthly_expense=value)
        if depletion is None or depletion >= target_age:
            best = value
        value += STEP
    return best


def test_expense_answer_is_the_highest_that_works(demo_plan):
    """한 칸(만원) 더 쓰면 실패해야 한다. 아니면 과하게 졸라매게 하는 답이다."""
    option = _option(demo_plan, Lever.EXPENSE)

    # 완전탐색으로 구한 답과 이분 탐색의 답이 같아야 한다.
    brute = _highest_expense_that_survives(DEMO_PROFILE, demo_plan.target_age)
    assert option.required_value == brute

    over = _depletion(DEMO_PROFILE, monthly_expense=brute + STEP)
    assert over is not None and over < demo_plan.target_age


def test_answers_land_on_clean_amounts(demo_plan):
    """'2,413,271원으로 줄이세요'는 사람이 쓸 수 없는 답이다."""
    for option in demo_plan.options:
        if option.unit is not Unit.WON_PER_MONTH:
            continue
        for value in (option.required_value, option.change_value):
            if value is not None:
                assert value % STEP == 0, f"{option.lever}: {value}"


def test_income_prescription_actually_works(demo_plan):
    option = _option(demo_plan, Lever.OTHER_INCOME)
    assert option.feasible
    assert option.change_value and option.change_value > 0  # 더 벌어야 하는 인물이다

    depletion = _depletion(DEMO_PROFILE, other_monthly_income=option.required_value)
    assert depletion is None or depletion >= demo_plan.target_age


# --------------------------------------------------------------------------- #
# 연금 레버 — 단조가 아니라는 점이 핵심
# --------------------------------------------------------------------------- #


def test_deferring_the_pension_can_make_things_worse():
    """이 레버에 이분 탐색을 쓰면 안 되는 이유.

    DEMO 인물은 자산이 빠르게 줄어서, 연금을 미루면 그때까지 자산을 더 헐어야 해
    고갈이 오히려 앞당겨진다. 전 구간을 계산해야 옳은 답이 나온다.
    """
    normal = _depletion(DEMO_PROFILE, national_pension_start_age=64)
    deferred = _depletion(DEMO_PROFILE, national_pension_start_age=69)
    assert deferred is not None and normal is not None
    assert deferred < normal  # 미룰수록 나빠진다


def test_pension_option_picks_the_best_age(demo_plan):
    option = _option(demo_plan, Lever.PENSION_START)
    assert option.improves
    assert option.gained_years >= 1
    # 앞당기는 처방에는 평생 감액 경고가 반드시 붙는다.
    assert "평생" in option.caution


def test_early_claim_is_not_offered_before_retirement():
    """소득이 있는 동안에는 조기노령연금을 받을 수 없다."""
    option = _option(prescribe(DEMO_PROFILE), Lever.PENSION_START)
    assert option.unit is Unit.AGE
    assert option.required_value >= DEMO_PROFILE.retirement_age

    ages = [int(part.split("만 ")[1].split("세")[0]) for part in option.detail.split(" / ")]
    assert min(ages) >= DEMO_PROFILE.retirement_age


def test_profile_without_a_pension_skips_the_lever():
    no_pension = DEMO_PROFILE.model_copy(update={"national_pension_monthly": 0})
    option = _option(prescribe(no_pension), Lever.PENSION_START)
    assert not option.feasible
    assert "계산하지 않았" in option.headline


# --------------------------------------------------------------------------- #
# 묶음 동작
# --------------------------------------------------------------------------- #


def test_plan_reports_the_baseline_problem(demo_plan):
    assert demo_plan.baseline_depletion_age == 69
    assert not demo_plan.already_safe
    assert "69세" in demo_plan.summary
    assert demo_plan.feasible_options


def test_feasible_options_come_first(demo_plan):
    feasible = [o.feasible for o in demo_plan.options]
    assert feasible == sorted(feasible, reverse=True)


def test_a_safe_plan_is_told_how_much_room_it_has():
    """이미 목표를 넘기는 사람에게 '그대로 유지하세요'는 아무 정보가 아니다."""
    plan = prescribe(DIVERSIFIED_PROFILE)
    assert plan.already_safe
    assert plan.baseline_depletion_age is None

    expense = _option(plan, Lever.EXPENSE)
    assert expense.feasible
    assert "까지" in expense.headline  # 여유 상한을 말해 준다


def test_lower_target_needs_a_smaller_change():
    """목표를 낮추면 요구되는 절감폭도 줄어야 한다."""
    strict = _highest_expense_that_survives(DEMO_PROFILE, 95)
    relaxed = _highest_expense_that_survives(DEMO_PROFILE, 85)
    assert relaxed > strict

    assert "85세" in prescribe(DEMO_PROFILE, target_age=85).summary


def test_everything_is_explainable():
    """모든 처방에 근거 문장이 붙어야 한다 — 숫자만 던지지 않는다."""
    for profile in (DEMO_PROFILE, DIVERSIFIED_PROFILE):
        for option in prescribe(profile).options:
            assert option.headline.strip()
            assert option.label.strip()


def test_profile_that_cannot_be_saved_says_so():
    """생활비를 줄여도 답이 없는 경우를 정직하게 말해야 한다."""
    hopeless = UserProfile(
        birth_year=1966,
        retirement_year=2026,
        retirement_month=12,
        severance_pay=0,
        cash_savings=1_000_000,
        monthly_expense=3_000_000,
        national_pension_monthly=0,
        risk_tolerance=RiskTolerance.CONSERVATIVE,
    )
    plan = prescribe(hopeless)
    assert not plan.already_safe
    expense = _option(plan, Lever.EXPENSE)
    assert not expense.feasible
    assert "어렵습니다" in expense.headline


def test_target_age_defaults_to_the_simulation_horizon():
    assert prescribe(DEMO_PROFILE).target_age == load_assumptions().macro.horizon_age
