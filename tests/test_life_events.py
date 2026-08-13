"""일회성 큰 지출 테스트.

이 연령대의 계획을 실제로 무너뜨리는 것은 매달의 생활비가 아니라 **한 번에 나가는
큰돈**이다. 자녀 결혼자금, 목돈 의료비, 주택 수리.

두 가지를 지킨다:
  - 지출이 없으면 기존 숫자가 **하나도** 바뀌면 안 된다 (W1~A3 회귀 방지).
  - 두 엔진(결정론적·몬테카를로)이 같은 규약을 써야 한다. 한쪽만 반영하면
    "고갈은 66세인데 생존 확률은 그대로"인 화면이 나온다.
"""

from __future__ import annotations

import json

import pytest

from agents.llm import ToolCall
from agents.tools import Toolbox
from core.cashflow import simulate
from core.models import LifeEvent
from core.montecarlo import run_monte_carlo
from core.prescribe import prescribe
from core.samples import DEMO_PROFILE
from core.schedule import build_schedule

WEDDING = LifeEvent(year=2029, amount=50_000_000, label="자녀 결혼자금")


@pytest.fixture(scope="module")
def with_event():
    return DEMO_PROFILE.model_copy(update={"life_events": [WEDDING]})


# --------------------------------------------------------------------------- #
# 회귀 — 지출이 없으면 아무것도 바뀌지 않는다
# --------------------------------------------------------------------------- #


def test_no_events_means_no_change():
    sim = simulate(DEMO_PROFILE)
    assert sim.depletion_year == 2035
    assert sim.balance_at_pension_start == pytest.approx(118_552_982, rel=1e-6)
    assert all(row.event_expense == 0 for row in sim.rows)


# --------------------------------------------------------------------------- #
# 반영
# --------------------------------------------------------------------------- #


def test_the_event_lands_in_the_right_year(with_event):
    plans = {plan.year: plan for plan in build_schedule(with_event)}

    assert plans[2029].event_expense > 0
    assert plans[2028].event_expense == 0
    assert plans[2030].event_expense == 0


def test_the_amount_is_inflated_to_that_year(with_event):
    """금액은 퇴직 시점 기준으로 본다 — 월 생활비와 같은 규약이라 물가만큼 커진다."""
    plan = next(p for p in build_schedule(with_event) if p.year == 2029)
    assert plan.event_expense > WEDDING.amount
    assert plan.event_expense == pytest.approx(WEDDING.amount * plan.inflation_factor)


def test_a_big_expense_pulls_the_depletion_forward(with_event):
    """5,000만원이 나가면 고갈이 3년 앞당겨진다 — 이 기능을 만든 이유."""
    before = simulate(DEMO_PROFILE).depletion_age
    after = simulate(with_event).depletion_age

    assert after < before
    assert after == 66


def test_the_row_shows_the_event_separately(with_event):
    """화면에서 '그 해에 왜 이렇게 줄었나'를 설명할 수 있어야 한다."""
    row = next(r for r in simulate(with_event).rows if r.year == 2029)
    assert row.event_expense > 0
    assert row.expense > 0  # 생활비와 섞이지 않는다


def test_events_outside_the_horizon_are_ignored():
    """입력이 이상해도 계산을 멈추지 않는다."""
    weird = DEMO_PROFILE.model_copy(
        update={"life_events": [LifeEvent(year=1990, amount=10_000_000)]}
    )
    assert simulate(weird).depletion_age == simulate(DEMO_PROFILE).depletion_age


def test_several_events_in_one_year_are_summed():
    profile = DEMO_PROFILE.model_copy(
        update={
            "life_events": [
                LifeEvent(year=2029, amount=30_000_000, label="결혼자금"),
                LifeEvent(year=2029, amount=20_000_000, label="의료비"),
            ]
        }
    )
    plan = next(p for p in build_schedule(profile) if p.year == 2029)
    assert plan.event_expense == pytest.approx(50_000_000 * plan.inflation_factor)


# --------------------------------------------------------------------------- #
# 두 엔진이 어긋나지 않는다
# --------------------------------------------------------------------------- #


def test_monte_carlo_sees_the_same_expense(with_event):
    """한쪽만 반영하면 '고갈은 66세인데 생존 확률은 그대로'인 화면이 나온다."""
    before = run_monte_carlo(DEMO_PROFILE, n_paths=500)
    after = run_monte_carlo(with_event, n_paths=500)

    # 만 70세면 양쪽 다 이미 0% 다. 차이가 드러나는 구간에서 비교한다.
    assert after.depletion_age_median < before.depletion_age_median
    assert after.survival_at_age(67) < before.survival_at_age(67)


# --------------------------------------------------------------------------- #
# 처방·도구에 자동으로 얹힌다
# --------------------------------------------------------------------------- #


def test_the_prescription_gets_harder_with_a_big_expense(with_event):
    """지출이 늘면 필요한 절감폭도 커져야 한다."""
    lever = "expense"
    plain = next(o for o in prescribe(DEMO_PROFILE).options if o.lever.value == lever)
    harder = next(o for o in prescribe(with_event).options if o.lever.value == lever)

    assert harder.required_value < plain.required_value


def test_the_agent_can_try_an_expense_without_it_being_in_the_profile():
    """'내후년에 결혼자금 5천만원이 나가면요?' — 대화로 넣어볼 수 있어야 한다."""
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(
        ToolCall(
            id="c1",
            name="simulate_plan",
            arguments={
                "one_off_expense_amount": 50_000_000,
                "one_off_expense_year": 2029,
                "one_off_expense_label": "자녀 결혼자금",
            },
        )
    )
    payload = json.loads(result.content)

    assert not result.is_error
    assert "자녀 결혼자금" in payload["바꾼_것"]
    assert payload["바꾼_계획"]["금융자산_고갈_나이"] == 66
    assert payload["고갈_시점_변화"] == "-3년"


def test_an_amount_without_a_year_is_rejected():
    """조용히 기본 연도를 넣으면 모델이 엉뚱한 답을 그대로 말하게 된다."""
    box = Toolbox(DEMO_PROFILE)
    result = box.execute(
        ToolCall(id="c1", name="simulate_plan", arguments={"one_off_expense_amount": 10_000_000})
    )
    assert result.is_error
    assert "함께 지정" in result.content


def test_the_agent_does_not_mutate_the_stored_profile():
    """도구로 가정을 넣어봐도 사용자의 실제 프로파일은 그대로여야 한다."""
    box = Toolbox(DEMO_PROFILE)
    box.execute(
        ToolCall(
            id="c1",
            name="simulate_plan",
            arguments={"one_off_expense_amount": 50_000_000, "one_off_expense_year": 2029},
        )
    )
    assert box.profile.life_events == []
    assert DEMO_PROFILE.life_events == []
