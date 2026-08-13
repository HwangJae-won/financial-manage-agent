"""도구 계층 테스트.

두 가지가 핵심이다:

  - **어떤 입력에도 예외가 밖으로 나오면 안 된다.** 모델이 도구 이름을 틀리든,
    인자를 빼먹든, 말도 안 되는 값을 넣든 대화는 계속되어야 한다. 무엇이
    잘못됐는지 결과로 돌려주면 모델이 다시 시도할 수 있다.
  - **모델은 인자만 고른다.** 프로파일은 도구가 들고 있어야 하고, 출력의 숫자는
    전부 `core/` 에서 나온 것이어야 한다.
"""

from __future__ import annotations

import json

import pytest

from agents.llm import ToolCall
from agents.tools import TOOLS, SimulateArgs, Toolbox, policy_keys
from core.cashflow import simulate
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture
def box():
    return Toolbox(DEMO_PROFILE)


def _run(box, name, **arguments):
    result = box.execute(ToolCall(id="call-1", name=name, arguments=arguments))
    return result, json.loads(result.content)


# --------------------------------------------------------------------------- #
# 정의 — 모델이 읽을 수 있는 형태인가
# --------------------------------------------------------------------------- #


def test_every_tool_publishes_an_object_schema(box):
    """두 프로바이더 모두 최상위가 object 인 JSON 스키마를 요구한다."""
    assert len(box.specs) == len(TOOLS)
    for spec in box.specs:
        assert spec.name and spec.description
        assert spec.input_schema["type"] == "object"


def test_tool_descriptions_say_when_to_use_them(box):
    """설명이 부실하면 모델이 엉뚱한 도구를 고른다."""
    for spec in box.specs:
        assert len(spec.description) > 30


def test_the_first_tool_is_safe_to_call_with_no_arguments(box):
    """mock 의 기본 동작이 첫 도구를 빈 인자로 부른다 — 그것이 오류가 되면 안 된다."""
    result, payload = _run(box, box.specs[0].name)
    assert not result.is_error
    assert payload["현재_계획"]["금융자산_고갈_나이"] == 69


# --------------------------------------------------------------------------- #
# 실행 — 숫자는 엔진에서만 나온다
# --------------------------------------------------------------------------- #


def test_simulate_reports_the_baseline_from_the_engine(box):
    _, payload = _run(box, "simulate_plan")
    assert payload["현재_계획"]["금융자산_고갈_나이"] == simulate(DEMO_PROFILE).depletion_age


def test_simulate_applies_the_override_and_shows_the_change(box):
    _, payload = _run(box, "simulate_plan", monthly_expense=2_500_000)

    expected = simulate(
        DEMO_PROFILE.model_copy(update={"monthly_expense": 2_500_000})
    ).depletion_age
    assert payload["바꾼_계획"]["금융자산_고갈_나이"] == expected
    assert payload["고갈_시점_변화"] == "+4년"
    assert "300만원 → 250만원" in payload["바꾼_것"]


def test_simulate_can_change_the_pension_start_age(box):
    _, payload = _run(box, "simulate_plan", national_pension_start_age=62)
    assert "만 64세 → 만 62세" in payload["바꾼_것"]
    # 연기가 불리한 인물이라 앞당기면 고갈이 늦춰진다
    assert payload["바꾼_계획"]["금융자산_고갈_나이"] == 70


def test_prescribe_returns_every_lever(box):
    _, payload = _run(box, "prescribe", target_age=90)
    assert payload["목표_나이"] == 90
    assert len(payload["선택지"]) == 3
    assert any(o["목표_달성_가능"] for o in payload["선택지"])


def test_sensitivity_keeps_its_caveat(box):
    _, payload = _run(box, "sensitivity")
    assert payload["가정별_흔들림"][0]["가정"] == "월 생활비"
    assert payload["한계"]


def test_policy_impact_carries_its_source_and_stage(box):
    """숫자만 있고 출처·확정여부가 없는 정책 답변은 만들지 않는다."""
    _, payload = _run(box, "policy_impact")
    assert "미확정" in payload["확정여부"]
    assert payload["출처"]
    assert payload["미확정_고지"]


def test_check_message_links_to_the_users_assets():
    box = Toolbox(DIVERSIFIED_PROFILE)
    _, payload = _run(
        box, "check_message", text="ISA 비과세가 폐지됩니다. 원금 보장에 월 3% 확정 수익입니다."
    )
    assert payload["위험도"] == "위험"
    assert payload["걸린_신호"]
    assert payload["고객님_자산과의_관계"]


# --------------------------------------------------------------------------- #
# 오류 — 예외가 아니라 결과로 돌려준다
# --------------------------------------------------------------------------- #


def test_unknown_tool_lists_the_available_ones(box):
    result, payload = _run(box, "없는도구")
    assert result.is_error
    assert "simulate_plan" in payload["사용가능"]


def test_out_of_range_argument_explains_itself(box):
    result, payload = _run(box, "simulate_plan", monthly_expense=-5)
    assert result.is_error
    assert any("monthly_expense" in problem for problem in payload["문제"])


def test_unknown_argument_is_rejected(box):
    """모델이 지어낸 인자명을 조용히 무시하면 엉뚱한 계산이 그대로 나간다."""
    result, payload = _run(box, "simulate_plan", 생활비=1_000_000)
    assert result.is_error
    assert any("생활비" in problem for problem in payload["문제"])


def test_missing_required_argument_says_which(box):
    result, payload = _run(box, "check_message")
    assert result.is_error
    assert any("text" in problem for problem in payload["문제"])


def test_engine_failure_does_not_escape(box):
    """엔진이 터져도 대화는 끊기지 않는다."""
    result, payload = _run(box, "policy_impact", policy_key="없는키")
    assert result.is_error
    assert "오류" in payload


def test_failures_are_not_recorded_as_outputs(box):
    """실패한 호출의 값은 '계산된 사실'이 아니다 — 인용 가능 범위에 들어가면 안 된다."""
    _run(box, "없는도구")
    _run(box, "simulate_plan", monthly_expense=-5)
    assert box.outputs == []

    _run(box, "simulate_plan")
    assert len(box.outputs) == 1


def test_the_call_id_always_comes_back(box):
    """id 가 어긋나면 다음 턴에서 프로바이더가 요청을 거부한다."""
    for name, arguments in (
        ("simulate_plan", {}),
        ("없는도구", {}),
        ("simulate_plan", {"monthly_expense": -5}),
    ):
        result = box.execute(ToolCall(id="xyz", name=name, arguments=arguments))
        assert result.call_id == "xyz"


# --------------------------------------------------------------------------- #
# 프로파일은 도구가 들고 있는다
# --------------------------------------------------------------------------- #


def test_no_tool_accepts_a_profile_argument():
    """모델이 자산을 인자로 넘기게 하면 거기서 숫자를 지어낼 여지가 생긴다."""
    for tool in TOOLS:
        fields = set(tool.args_model.model_fields)
        assert not fields & {"profile", "severance_pay", "cash_savings", "birth_year"}


def test_the_same_tool_answers_differently_per_profile():
    demo = Toolbox(DEMO_PROFILE)
    diversified = Toolbox(DIVERSIFIED_PROFILE)

    _, a = _run(demo, "simulate_plan")
    _, b = _run(diversified, "simulate_plan")
    assert a["현재_계획"]["금융자산_고갈_나이"] != b["현재_계획"]["금융자산_고갈_나이"]


def test_output_is_compact_enough_to_read(box):
    """40년치 시뮬레이션 행을 통째로 돌려주면 토큰만 먹고 모델이 길을 잃는다."""
    for name in ("simulate_plan", "prescribe", "sensitivity", "policy_impact"):
        result, _ = _run(box, name, **({"target_age": 95} if name == "prescribe" else {}))
        assert len(result.content) < 4000, name


def test_argument_model_rejects_absurd_amounts():
    with pytest.raises(Exception):
        SimulateArgs(monthly_expense=10**15)


def test_policy_keys_are_available_for_the_prompt():
    assert "isa_reform" in policy_keys()
