"""프로파일링 에이전트 테스트 — LLM 호출 없이 전체 대화 흐름을 검증한다."""

from __future__ import annotations

import datetime as _dt

import pytest

from agents.llm import MockClient
from agents.mocks import korean_slot_extractor
from agents.parsing import parse_korean_amount, parse_month, parse_year
from agents.profiling import MAX_RETRIES, ProfilingAgent, build_profile
from agents.slots import (
    ALL_SLOTS,
    DEFAULTS,
    MANDATORY_SLOTS,
    QUESTIONS,
    Priority,
    all_important_answered,
    extraction_schema,
    is_ready,
    next_question,
)
from core.models import RiskTolerance, UserProfile

TODAY = _dt.date(2026, 8, 13)


@pytest.fixture
def agent() -> ProfilingAgent:
    client = MockClient(structured_handler=korean_slot_extractor(TODAY.year))
    return ProfilingAgent(client=client, today=TODAY)


# --------------------------------------------------------------------------- #
# 한국어 금액 파서
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2억", 200_000_000),
        ("2억 5천", 250_000_000),  # 구어체 만 단위 생략
        ("300만원", 3_000_000),
        ("300만", 3_000_000),
        ("5천만원", 50_000_000),
        ("5천", 50_000_000),
        ("1억2천3백만", 123_000_000),
        ("3000만원", 30_000_000),
        ("200000000", 200_000_000),
        ("5천원", 5_000),  # '원'이 명시되면 액면 그대로
        ("2억 정도요", 200_000_000),
        ("한 1억쯤", 100_000_000),
    ],
)
def test_parse_korean_amount(text: str, expected: int):
    assert parse_korean_amount(text) == expected


@pytest.mark.parametrize("text", ["", "없어요", "잘 모르겠는데요", "글쎄요"])
def test_parse_korean_amount_returns_none_when_unparseable(text: str):
    assert parse_korean_amount(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1966년생입니다", 1966),
        ("66년생이요", 1966),
        ("올해요", 2026),
        ("내년입니다", 2027),
        ("작년에 그만뒀어요", 2025),
        ("2030년", 2030),
    ],
)
def test_parse_year(text: str, expected: int):
    assert parse_year(text, current_year=2026) == expected


@pytest.mark.parametrize(
    "text,expected", [("12월이요", 12), ("올해 말", 12), ("3월", 3), ("연초", 1)]
)
def test_parse_month(text: str, expected: int):
    assert parse_month(text) == expected


# --------------------------------------------------------------------------- #
# 슬롯 정의
# --------------------------------------------------------------------------- #


def test_every_slot_maps_to_a_real_profile_field():
    """질문이 채우는 필드가 UserProfile 에 실제로 있어야 한다."""
    fields = set(UserProfile.model_fields)
    for slot in ALL_SLOTS:
        assert slot in fields, slot


def test_defaults_cover_every_non_mandatory_slot():
    """필수가 아닌 슬롯은 전부 기본값이 있어야 프로파일을 만들 수 있다."""
    for slot in ALL_SLOTS:
        if slot not in MANDATORY_SLOTS:
            assert slot in DEFAULTS, slot


def test_questions_are_asked_in_priority_order():
    slots: dict = {}
    asked: set[str] = set()
    priorities = []
    while (question := next_question(slots, asked)) is not None:
        priorities.append(question.priority)
        asked.add(question.key)
    assert priorities == sorted(priorities, reverse=True)


def test_next_question_skips_answered_slots():
    slots = {"birth_year": 1966}
    question = next_question(slots, asked=set())
    assert question is not None
    assert question.key != "birth"


def test_extraction_schema_is_all_nullable():
    """말하지 않은 항목이 0 으로 오면 '없다'는 뜻이 되어 잘못된 결과가 나간다."""
    schema = extraction_schema()
    for name, spec in schema["properties"].items():
        assert "null" in spec["type"], name
    assert schema["additionalProperties"] is False


def test_is_ready_requires_all_mandatory_slots():
    assert not is_ready({})
    assert not is_ready({"birth_year": 1966, "retirement_year": 2026})
    assert is_ready({"birth_year": 1966, "retirement_year": 2026, "monthly_expense": 3_000_000})


# --------------------------------------------------------------------------- #
# 대화 흐름
# --------------------------------------------------------------------------- #


def test_conversation_starts_with_the_first_question(agent: ProfilingAgent):
    first = agent.start()
    assert "몇 년생" in first
    assert not agent.done


def test_full_conversation_produces_a_valid_profile(agent: ProfilingAgent):
    """기획서 예시 인물을 대화만으로 재현한다."""
    agent.start()
    script = [
        "1966년 12월생입니다",
        "올해 12월에 퇴직할 예정이에요",
        "생활비는 한 300만원 정도 씁니다",
        "퇴직금은 2억 정도 나올 것 같아요, 32년 다녔습니다",
        "예금이 2천만원 있습니다",
        "국민연금은 월 150만원 정도 나온다고 하더라고요, 32년 넣었습니다",
        "주식이나 펀드는 없어요",
        "퇴직연금도 없습니다",
        "아파트가 한 채 있는데 5억 정도 합니다",
        "다른 수입은 없어요",
        "대출은 없습니다",
        "원금을 지키는 쪽이 편합니다",
    ]
    for line in script:
        agent.respond(line)

    assert agent.done
    profile = agent.build_profile()

    assert profile.birth_year == 1966
    assert profile.birth_month == 12
    assert profile.retirement_year == 2026
    assert profile.retirement_month == 12
    assert profile.monthly_expense == 3_000_000
    assert profile.severance_pay == 200_000_000
    assert profile.cash_savings == 20_000_000
    assert profile.national_pension_monthly == 1_500_000
    assert profile.national_pension_months == 384  # "32년 넣었습니다"
    assert profile.real_estate == 500_000_000
    assert profile.risk_tolerance is RiskTolerance.CONSERVATIVE

    # 기획서 예시와 같은 결과가 나와야 한다
    assert profile.financial_assets == 220_000_000
    assert profile.total_assets == 720_000_000
    assert profile.income_gap_months == 48


def test_the_pension_question_also_collects_the_months():
    """질문을 하나 더 늘리지 않고 가입월수를 받는다 — 퇴직금 질문과 같은 방식."""
    pension = next(q for q in QUESTIONS if q.key == "pension")

    assert pension.fills == ("national_pension_monthly", "national_pension_months")
    assert not any("가입월수" in q.text for q in QUESTIONS)  # 별도 질문을 만들지 않았다


def test_monthly_salary_is_derived_not_asked():
    """퇴직금은 '30일분 평균임금 × 근속연수'다. 2억 ÷ 32년 = 월 625만원.

    질문을 늘리는 대신 이미 받은 두 값에서 끌어낸다. 다만 명예퇴직금이 섞이면
    어긋나므로 '가정한 값'으로 표시한다.
    """
    assert not any("급여" in q.text for q in QUESTIONS)

    profile = build_profile(
        {
            "birth_year": 1966,
            "retirement_year": 2026,
            "monthly_expense": 3_000_000,
            "severance_pay": 200_000_000,
            "years_employed": 32,
        }
    )
    assert profile.last_monthly_salary == 6_250_000
    assert "last_monthly_salary" in profile.assumed_fields


def test_a_stated_salary_is_not_overwritten_by_the_derivation():
    """사용자가 직접 말한 값을 유도값으로 덮으면 안 된다."""
    profile = build_profile(
        {
            "birth_year": 1966,
            "retirement_year": 2026,
            "monthly_expense": 3_000_000,
            "severance_pay": 200_000_000,
            "years_employed": 32,
            "last_monthly_salary": 4_000_000,
        }
    )
    assert profile.last_monthly_salary == 4_000_000
    assert "last_monthly_salary" not in profile.assumed_fields


def test_the_salary_is_left_unknown_when_it_cannot_be_derived():
    """근속연수를 모르면 끌어낼 수 없다. 0 으로 두고 화면이 물어보게 한다."""
    profile = build_profile(
        {
            "birth_year": 1966,
            "retirement_year": 2026,
            "monthly_expense": 3_000_000,
            "severance_pay": 200_000_000,
        }
    )
    assert profile.last_monthly_salary == 0


def test_pension_start_age_is_derived_not_asked(agent: ProfilingAgent):
    """묻지 않은 국민연금 개시 연령이 출생연도에서 법정 스케줄로 채워져야 한다."""
    assert not any("개시 연령" in q.text for q in QUESTIONS)
    profile = build_profile(
        {"birth_year": 1966, "retirement_year": 2026, "monthly_expense": 3_000_000}
    )
    assert profile.national_pension_start_age == 64  # 1965~1968년생


def test_volunteered_information_is_captured_in_one_turn(agent: ProfilingAgent):
    """묻지 않은 것을 먼저 말해도 반영되어야 한다."""
    agent.start()
    agent.respond("1966년생이고요, 퇴직금은 2억, 예금은 5천 있습니다")

    slots = agent.state["slots"]
    assert slots["birth_year"] == 1966
    assert slots["severance_pay"] == 200_000_000
    assert slots["cash_savings"] == 50_000_000


def test_no_question_repeats_in_a_normal_conversation(agent: ProfilingAgent):
    """정상적으로 답하는 동안에는 같은 질문이 두 번 나오면 안 된다."""
    agent.start()
    answers = [
        "1966년 12월생입니다",
        "올해 12월 퇴직이요",
        "생활비 300만원",
        "퇴직금 2억",
        "예금 2천만원",
        "국민연금 150만원",
    ]
    asked = []
    for answer in answers:
        asked.append(agent.state.get("pending_question"))
        agent.respond(answer)
    assert len(asked) == len(set(asked))


def test_unresponsive_user_does_not_hang_the_conversation(agent: ProfilingAgent):
    """필수 정보를 끝내 안 주면 되묻다가 안내하고 멈춰야 한다.

    되묻기가 없으면 '질문도 없고 완료도 안 되는' 상태로 앱이 멎는다.
    """
    agent.start()
    replies = []
    for _ in range(len(QUESTIONS) + MAX_RETRIES + 3):
        if agent.done:
            break
        replies.append(agent.respond("모르겠어요"))

    assert not agent.done
    assert agent.state["error"]  # 왜 못 끝냈는지 남아야 한다
    assert "필요한 정보를 아직 확인하지 못했습니다" in replies[-1]
    assert agent.state["retries"] == MAX_RETRIES


def test_mandatory_questions_are_re_asked_when_unanswered(agent: ProfilingAgent):
    """필수 항목은 한 번 놓쳤다고 포기하지 않고 되물어야 한다."""
    agent.start()
    for _ in range(len(QUESTIONS)):
        agent.respond("모르겠어요")

    # 모든 질문을 한 바퀴 돈 뒤, 필수 항목으로 되돌아온다
    assert agent.state["pending_question"] == "birth"
    assert agent.state["retries"] >= 1

    # 이제 답하면 정상 진행된다
    agent.respond("1966년생입니다")
    assert agent.state["slots"]["birth_year"] == 1966


def test_unanswered_questions_do_not_block_completion(agent: ProfilingAgent):
    """다 모른다고 해도 필수만 채워지면 결과를 만들 수 있어야 한다."""
    agent.start()
    agent.respond("1966년생이요")
    agent.respond("올해 12월에 퇴직합니다")
    agent.respond("생활비는 300만원 씁니다")
    for _ in range(len(QUESTIONS)):
        if agent.done:
            break
        agent.respond("잘 모르겠습니다")

    assert agent.done
    profile = agent.build_profile()
    assert profile.monthly_expense == 3_000_000


def test_assumed_fields_are_recorded(agent: ProfilingAgent):
    """기본값으로 채운 항목은 화면에 '가정한 값'으로 표시할 수 있어야 한다."""
    profile = build_profile(
        {"birth_year": 1966, "retirement_year": 2026, "monthly_expense": 3_000_000}
    )
    assert "severance_pay" in profile.assumed_fields
    assert "real_estate" in profile.assumed_fields
    assert "birth_year" not in profile.assumed_fields
    assert profile.assumed_fields == sorted(profile.assumed_fields)


def test_answered_fields_are_not_marked_assumed():
    profile = build_profile(
        {
            "birth_year": 1966,
            "retirement_year": 2026,
            "monthly_expense": 3_000_000,
            "severance_pay": 200_000_000,
        }
    )
    assert "severance_pay" not in profile.assumed_fields


def test_build_profile_rejects_missing_mandatory_slots():
    with pytest.raises(ValueError, match="필수 정보"):
        build_profile({"birth_year": 1966})


def test_ready_flag_tracks_important_questions(agent: ProfilingAgent):
    agent.start()
    agent.respond("1966년생이요")
    agent.respond("올해 12월 퇴직입니다")
    agent.respond("생활비 300만원이요")
    assert not agent.ready  # 필수는 찼지만 중요 항목이 남았다

    agent.respond("퇴직금은 2억이요")
    agent.respond("예금은 2천만원 있습니다")
    agent.respond("국민연금은 150만원이요")
    assert all_important_answered(agent.state["slots"])


def test_progress_is_reported(agent: ProfilingAgent):
    agent.start()
    answered, total = agent.progress
    assert (answered, total) == (0, len(QUESTIONS))

    agent.respond("1966년 12월생입니다")
    answered, _ = agent.progress
    assert answered == 1


def test_denial_is_recorded_as_zero_not_unknown(agent: ProfilingAgent):
    """'없어요'는 0 이고, '모르겠어요'는 미상이다 — 이 둘은 달라야 한다."""
    agent.start()
    agent.respond("1966년생이요")
    agent.respond("올해 12월 퇴직입니다")
    agent.respond("생활비 300만원이요")
    agent.respond("퇴직금은 없습니다")
    assert agent.state["slots"]["severance_pay"] == 0


def test_mock_client_records_every_extraction_call(agent: ProfilingAgent):
    agent.start()
    agent.respond("1966년생이요")
    calls = agent.client.calls
    assert calls and all(c["kind"] == "structured" for c in calls)
    assert "대화 내용" in calls[-1]["prompt"]


# --------------------------------------------------------------------------- #
# 자릿수 교차검증 — 작은 로컬 모델이 "5억"을 50억으로 적는 것을 막는다
# --------------------------------------------------------------------------- #

from agents.profiling import reconcile_amounts  # noqa: E402


def test_a_ten_fold_slip_is_corrected():
    """실제로 관측된 오류다. 총자산이 7억에서 57억이 되어도 화면은 멀쩡해 보인다."""
    fixed = reconcile_amounts(
        {"real_estate": 5_000_000_000, "monthly_expense": 30_000_000},
        {"real_estate": 500_000_000, "monthly_expense": 3_000_000},
    )

    assert fixed["real_estate"] == 500_000_000
    assert fixed["monthly_expense"] == 3_000_000


def test_a_genuine_disagreement_is_left_alone():
    """10의 거듭제곱이 아닌 차이는 서로 다른 것을 읽은 것이다. 덮으면 안 된다."""
    fixed = reconcile_amounts({"cash_savings": 30_000_000}, {"cash_savings": 20_000_000})

    assert fixed["cash_savings"] == 30_000_000


def test_slots_the_parser_did_not_see_are_untouched():
    """규칙 파서가 못 읽은 항목까지 건드리면 LLM 을 쓰는 의미가 없다."""
    fixed = reconcile_amounts({"isa": 50_000_000}, {"cash_savings": 20_000_000})

    assert fixed["isa"] == 50_000_000


def test_non_money_slots_are_not_reconciled():
    """가입월수·연도는 자릿수 규칙이 다르다. 금액 슬롯만 본다."""
    fixed = reconcile_amounts(
        {"national_pension_months": 384, "birth_year": 1966},
        {"national_pension_months": 38, "birth_year": 196},
    )

    assert fixed["national_pension_months"] == 384
    assert fixed["birth_year"] == 1966


def test_the_mock_path_is_unaffected():
    """mock 은 규칙 파서와 같은 값을 내므로 아무 일도 일어나지 않아야 한다."""
    same = {"severance_pay": 200_000_000}
    assert reconcile_amounts(same, same) == same
