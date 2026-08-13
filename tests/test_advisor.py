"""도구를 쥔 상담 에이전트 테스트.

실제 API 는 부르지 않는다. mock 클라이언트에 시나리오를 심어 **루프가 어떻게
도는지**를 검증한다. 여기서 확인해야 하는 것은 모델의 답변 품질이 아니라
"무한히 돌지 않는가 / 실패해도 끊기지 않는가 / 한 일이 남는가" 세 가지다.
"""

from __future__ import annotations

import pytest

from agents.advisor import MAX_ROUNDS, Advisor, AdvisorReply
from agents.llm import MockClient, ToolCall, ToolTurn
from core.samples import DEMO_PROFILE


def scripted(*rounds):
    """도구 호출을 미리 정해 둔 핸들러. 대본이 끝나면 말로 마무리한다."""

    def handler(turns, tools):
        done = sum(1 for turn in turns if turn.tool_results)
        if tools and done < len(rounds):
            return ToolTurn(tool_calls=list(rounds[done]), stop_reason="tool_use")
        return ToolTurn(text="확인해 보았습니다.", stop_reason="end_turn")

    return handler


def advisor(handler=None, **kwargs):
    client = MockClient(tool_handler=handler) if handler else MockClient()
    return Advisor(DEMO_PROFILE, client=client, **kwargs)


# --------------------------------------------------------------------------- #
# 기본 루프
# --------------------------------------------------------------------------- #


def test_the_default_mock_completes_a_loop_without_a_key():
    """키가 없어도 도구를 부르고 답까지 나와야 한다."""
    reply = advisor().ask("제 계획은 어떤가요?")

    assert reply.stop_reason == "answered"
    assert reply.rounds == 1
    assert reply.used_tools
    assert reply.trace[0].tool == "simulate_plan"
    assert reply.trace[0].ok


def test_the_model_can_call_several_tools_in_sequence():
    """한 질문에 계산이 두 번 필요할 수 있다 — 그게 루프를 만든 이유다."""
    handler = scripted(
        [ToolCall(id="c1", name="simulate_plan", arguments={"monthly_expense": 2_500_000})],
        [ToolCall(id="c2", name="prescribe", arguments={"target_age": 95})],
    )
    reply = advisor(handler).ask("생활비를 50만원 줄이면 어떻게 되나요?")

    assert reply.rounds == 2
    assert [step.tool for step in reply.trace] == ["simulate_plan", "prescribe"]
    assert reply.stop_reason == "answered"


def test_parallel_tool_calls_in_one_turn_are_all_executed():
    handler = scripted(
        [
            ToolCall(id="c1", name="simulate_plan", arguments={}),
            ToolCall(id="c2", name="sensitivity", arguments={}),
        ]
    )
    reply = advisor(handler).ask("전체적으로 봐 주세요")

    assert reply.rounds == 1
    assert len(reply.trace) == 2


# --------------------------------------------------------------------------- #
# 반복 상한 — 비용과 대기 시간을 통제한다
# --------------------------------------------------------------------------- #


def test_the_loop_stops_at_the_round_limit():
    """도구를 계속 부르려 해도 상한에서 멈춘다."""
    def greedy(turns, tools):
        if tools:
            return ToolTurn(
                tool_calls=[ToolCall(id="x", name="simulate_plan", arguments={})],
                stop_reason="tool_use",
            )
        return ToolTurn(text="정리하면 이렇습니다.", stop_reason="end_turn")

    reply = advisor(greedy, max_rounds=3).ask("계속 계산해 주세요")

    assert reply.stop_reason == "max_rounds"
    assert reply.rounds == 3
    assert len(reply.trace) == 3


def test_the_wrap_up_call_offers_no_tools():
    """상한에 닿으면 도구를 빼고 물어야 모델이 말로 마무리한다."""
    seen: list[list[str]] = []

    def greedy(turns, tools):
        seen.append([t.name for t in tools])
        if tools:
            return ToolTurn(
                tool_calls=[ToolCall(id="x", name="simulate_plan", arguments={})],
                stop_reason="tool_use",
            )
        return ToolTurn(text="여기까지 확인했습니다.", stop_reason="end_turn")

    reply = advisor(greedy, max_rounds=2).ask("계속 계산해 주세요")

    assert seen[-1] == []  # 마지막 호출에는 도구를 주지 않는다
    assert reply.text == "여기까지 확인했습니다."


def test_hitting_the_limit_still_produces_an_answer():
    """상한에 닿았다고 빈손으로 끝내지 않는다."""
    reply = advisor(
        lambda turns, tools: ToolTurn(
            tool_calls=[ToolCall(id="x", name="simulate_plan", arguments={})]
        ),
        max_rounds=2,
    ).ask("무한히 계산해 주세요")

    assert reply.text.strip()
    assert reply.trace


# --------------------------------------------------------------------------- #
# 실패 — 대화를 끊지 않는다
# --------------------------------------------------------------------------- #


def test_an_llm_failure_does_not_raise():
    """네트워크가 끊겨도 발표 화면이 죽으면 안 된다."""
    def broken(turns, tools):
        raise RuntimeError("네트워크 오류")

    reply = advisor(broken).ask("계획을 봐 주세요")

    assert reply.stop_reason == "llm_error"
    assert reply.text.strip()


def test_a_failure_after_some_work_still_shows_what_was_calculated():
    """계산까지는 됐는데 정리에 실패했다면, 계산 결과라도 보여준다."""
    state = {"calls": 0}

    def flaky(turns, tools):
        state["calls"] += 1
        if state["calls"] == 1:
            return ToolTurn(
                tool_calls=[ToolCall(id="c1", name="simulate_plan", arguments={})],
                stop_reason="tool_use",
            )
        raise RuntimeError("두 번째 호출에서 끊김")

    reply = advisor(flaky).ask("계획을 봐 주세요")

    assert reply.stop_reason == "llm_error"
    assert reply.trace  # 계산한 것은 남아 있다
    assert "만 69세 고갈" in reply.text  # 도구 결과로 만든 문장


def test_a_tool_error_does_not_end_the_conversation():
    """모델이 인자를 틀려도 대화는 계속된다. 그 사실이 trace 에 남는다."""
    handler = scripted(
        [ToolCall(id="c1", name="simulate_plan", arguments={"monthly_expense": -1})]
    )
    reply = advisor(handler).ask("계산해 주세요")

    assert reply.stop_reason == "answered"
    assert reply.trace[0].ok is False
    assert "인자" in reply.trace[0].summary


def test_an_empty_answer_is_replaced_by_the_trace():
    """모델이 도구만 부르고 말을 안 하는 경우가 있다. 빈 화면을 내보내지 않는다."""
    def silent(turns, tools):
        if tools and not any(t.tool_results for t in turns):
            return ToolTurn(
                tool_calls=[ToolCall(id="c1", name="simulate_plan", arguments={})],
                stop_reason="tool_use",
            )
        return ToolTurn(text="   ", stop_reason="end_turn")

    reply = advisor(silent).ask("계획을 봐 주세요")

    assert reply.text.strip()
    assert "만 69세 고갈" in reply.text


# --------------------------------------------------------------------------- #
# 기록 — 심사에서 보여줄 부분
# --------------------------------------------------------------------------- #


def test_the_trace_keeps_arguments_and_output():
    """무엇을 어떤 인자로 불러 무슨 값이 나왔는지가 전부 남아야 한다."""
    handler = scripted(
        [ToolCall(id="c1", name="simulate_plan", arguments={"monthly_expense": 2_000_000})]
    )
    step = advisor(handler).ask("200만원으로 줄이면요?").trace[0]

    assert step.arguments == {"monthly_expense": 2_000_000}
    assert step.output["바꾼_계획"]["금융자산_고갈_나이"]
    assert "300만원 → 200만원" in step.summary


def test_each_question_only_reports_its_own_trace():
    """후속 질문의 trace 에 앞 질문의 계산이 섞이면 안 된다."""
    agent = advisor()
    first = agent.ask("제 계획은요?")
    second = agent.ask("그럼 생활비를 줄이면요?")

    assert len(first.trace) == 1
    assert len(second.trace) == 1


def test_history_is_kept_for_follow_up_questions():
    agent = advisor()
    agent.ask("제 계획은요?")
    before = len(agent.turns)
    agent.ask("그럼 어떻게 하면 되나요?")

    assert len(agent.turns) > before
    assert agent.turns[0].text == "제 계획은요?"


# --------------------------------------------------------------------------- #
# 프롬프트
# --------------------------------------------------------------------------- #


def test_the_system_prompt_forbids_making_numbers_up():
    from agents.advisor import SYSTEM

    assert "직접 만들지 마세요" in SYSTEM
    assert "isa_reform" in SYSTEM  # 고를 수 있는 정책 키를 알려준다
    assert "자산을 인자로 넘기지 마세요" in SYSTEM


def test_default_round_limit_is_small():
    """실제 질문은 두세 번이면 끝난다. 상한이 크면 비용만 늘어난다."""
    assert MAX_ROUNDS <= 5


# --------------------------------------------------------------------------- #
# A7 — 도구 출력에 없는 숫자는 내보내지 않는다
# --------------------------------------------------------------------------- #


def _says(*texts):
    """도구를 한 번 부른 뒤 정해진 문장을 차례로 내놓는 핸들러."""
    said = {"count": 0}

    def handler(turns, tools):
        if tools and not any(turn.tool_results for turn in turns):
            return ToolTurn(
                tool_calls=[ToolCall(id="c1", name="simulate_plan", arguments={})],
                stop_reason="tool_use",
            )
        index = min(said["count"], len(texts) - 1)
        said["count"] += 1
        return ToolTurn(text=texts[index], stop_reason="end_turn")

    return handler


def test_an_invented_number_is_blocked():
    """계산에 없는 숫자를 말하면 그 답변은 나가지 않는다.

    브리핑은 표시만 하고 내보내지만 상담 답변은 막는다. 브리핑은 화면의 계산
    결과와 대조할 수 있지만, 상담 답변은 대조할 대상이 화면에 없기 때문이다.
    """
    reply = advisor(_says("연 7.4% 수익률이면 걱정 없으십니다.")).ask("어떤가요?")

    assert reply.blocked
    assert reply.stop_reason == "blocked"
    assert "7.4" in reply.unverified_numbers
    assert "7.4" not in reply.text
    assert "확인되지 않은 숫자" in reply.text


def test_the_model_gets_one_chance_to_fix_it():
    """지어낸 숫자를 스스로 빼면 그 답변은 그대로 나간다."""
    reply = advisor(
        _says("연 7.4% 수익률이면 걱정 없으십니다.", "지금 계획으로는 만 69세에 고갈됩니다.")
    ).ask("어떤가요?")

    assert not reply.blocked
    assert reply.stop_reason == "answered"
    assert "69세" in reply.text


def test_numbers_from_the_tool_output_pass():
    """계산에서 나온 숫자는 막히지 않는다 — 과잉 차단이면 기능이 죽는다."""
    reply = advisor(_says("만 69세에 금융자산이 바닥납니다.")).ask("언제 고갈되나요?")

    assert reply.unverified_numbers == []
    assert reply.stop_reason == "answered"


def test_numbers_the_user_said_are_quotable():
    """'생활비 50만원 줄이면요?'에 답하며 50만원을 되뇌는 것은 지어낸 것이 아니다."""
    reply = advisor(_says("50만원을 줄이시면 도움이 됩니다.")).ask(
        "생활비를 50만원 줄이면 어떻게 되나요?"
    )

    assert reply.unverified_numbers == []


def test_earlier_results_stay_quotable_in_follow_ups():
    """후속 질문에서 앞 턴의 계산 결과를 다시 인용하는 것은 정상이다."""
    agent = advisor()
    first = agent.ask("제 계획은요?")
    depletion = str(first.trace[0].output["현재_계획"]["금융자산_고갈_나이"])

    agent.client.tool_handler = _says(f"앞서 말씀드린 만 {depletion}세 그대로입니다.")
    second = agent.ask("다시 정리해 주세요")

    assert second.unverified_numbers == []
    assert second.stop_reason == "answered"


def test_nested_tool_output_counts_as_quotable():
    """출력이 겹쳐 있어도 안쪽 금액까지 인용 가능 범위에 들어가야 한다."""
    from agents.advisor import _flatten

    flat = _flatten({"임의계속가입": {"내는_돈": "2,665만원", "주의": ["평생 6,993만원"]}})

    assert "2,665만원" in flat
    assert "6,993만원" in flat


def test_the_mock_keeps_calculating_across_follow_up_questions():
    """키 없이 도는 데모에서 두 번째 질문부터 계산이 멈추면 안 된다.

    이력 전체를 보고 '이미 도구를 썼다'고 판단하면 그런 화면이 된다.
    """
    agent = advisor()
    first = agent.ask("제 계획은요?")
    second = agent.ask("그럼 생활비를 줄이면요?")
    third = agent.ask("한 번 더 봐 주세요")

    assert len(first.trace) == 1
    assert len(second.trace) == 1
    assert len(third.trace) == 1
