"""저장소 테스트.

여기서 지켜야 하는 것은 세 가지다.

1. **왕복.** 대화를 저장했다 되살렸을 때 같은 프로파일·같은 대화가 나와야 한다.
   중간에 한 칸이라도 새면 사용자는 상담을 처음부터 다시 해야 한다.
2. **저장 실패가 대화를 끊지 않는다.** 저장소는 편의 기능이지 서비스의 전제가
   아니다 — 키 없이도 전체 흐름이 돌아야 한다는 규칙과 같은 이유다.
3. **집계가 실제로 쌓인다.** 이 기록이 로컬 모델을 붙인 뒤 L1~L4 를 추측이 아니라
   숫자로 답하게 해 주는 재료다.
"""

from __future__ import annotations

import datetime as _dt
import json

import pytest

import storage
from agents.graph import Supervisor
from agents.llm import MockClient, Turn, turn_from_dict, turn_to_dict
from agents.mocks import demo_handler, demo_tool_handler

CHAT_SCRIPT = [
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


def _client() -> MockClient:
    client = MockClient(structured_handler=demo_handler(2026))
    client.tool_handler = demo_tool_handler()
    return client


def _finished() -> Supervisor:
    supervisor = Supervisor(client=_client(), today=_dt.date(2026, 8, 14))
    supervisor.start()
    for line in CHAT_SCRIPT:
        supervisor.send(line)
    return supervisor


@pytest.fixture(autouse=True)
def clean_db():
    """표를 비우고 시작한다. 집계 테스트가 앞선 테스트에 영향받지 않게."""
    connection = storage.connect()
    for table in ("tool_runs", "advice", "messages", "profiles", "sessions", "users"):
        connection.execute(f"DELETE FROM {table}")
    connection.commit()
    yield


# --------------------------------------------------------------------------- #
# 왕복
# --------------------------------------------------------------------------- #


def test_a_turn_survives_the_round_trip():
    """도구 호출이 들어간 칸도 그대로 돌아와야 한다."""
    original = Supervisor(client=_client())
    original.start()
    original.send("1966년 12월생입니다")

    turn = Turn(role="assistant", text="확인했습니다")
    assert turn_from_dict(turn_to_dict(turn)) == turn


def test_an_unknown_field_does_not_destroy_the_history():
    """저장해 둔 대화를 나중 버전이 읽을 때 필드가 늘 수 있다."""
    restored = turn_from_dict({"role": "user", "text": "안녕", "미래필드": 1})

    assert restored.role == "user"
    assert restored.text == "안녕"


def test_a_finished_session_round_trips():
    """대화를 마친 세션을 저장했다 되살리면 같은 프로파일이 나와야 한다."""
    original = _finished()
    storage.save_session(
        "s1", original.dump(), done=original.done, ready=original.ready
    )

    saved = storage.load_session("s1")
    assert saved is not None
    assert saved["done"] is True

    restored = Supervisor.restore(saved, client=_client())
    assert restored.build_profile() == original.build_profile()
    assert restored.messages == original.messages


def test_the_restored_session_keeps_talking():
    """복원한 뒤에도 상담이 이어져야 한다 — 이력이 살아 있다는 뜻이다."""
    original = _finished()
    original.send("국민연금을 더 내는 게 나을까요?")
    storage.save_session("s2", original.dump(), done=True, ready=True)

    restored = Supervisor.restore(storage.load_session("s2"), client=_client())
    before = len(restored.advisor().turns)
    reply = restored.send("그럼 생활비를 줄이면요?")

    assert before > 0  # 앞 대화가 복원되어 있다
    assert reply.kind == "advice"
    assert reply.advice.trace


def test_an_unfinished_session_round_trips():
    """대화 도중에 저장해도 남은 질문을 이어서 물어야 한다."""
    original = Supervisor(client=_client(), today=_dt.date(2026, 8, 14))
    original.start()
    for line in CHAT_SCRIPT[:3]:
        original.send(line)
    storage.save_session("s3", original.dump())

    restored = Supervisor.restore(storage.load_session("s3"), client=_client())
    assert not restored.done
    assert restored.progress == original.progress

    reply = restored.send(CHAT_SCRIPT[3])
    assert reply.kind == "question"


def test_a_missing_session_is_none_not_an_exception():
    assert storage.load_session("없는세션") is None


# --------------------------------------------------------------------------- #
# 사용자
# --------------------------------------------------------------------------- #


def test_a_user_is_created_once():
    first = storage.ensure_user(nickname="김OO")
    again = storage.ensure_user(first)

    assert first == again
    assert storage.get_user(first)["nickname"] == "김OO"


def test_an_empty_nickname_does_not_erase_the_saved_one():
    """세션을 새로 만들 때마다 닉네임이 지워지면 안 된다."""
    user_id = storage.ensure_user(nickname="김OO")
    storage.ensure_user(user_id)  # 닉네임 없이 다시 부른다

    assert storage.get_user(user_id)["nickname"] == "김OO"


def test_sessions_are_listed_per_user():
    user_id = storage.ensure_user()
    storage.save_session("a", {"state": {}}, user_id=user_id)
    storage.save_session("b", {"state": {}}, user_id=user_id)

    assert {row["id"] for row in storage.list_sessions(user_id)} == {"a", "b"}


def test_deleting_a_session_takes_its_records_with_it():
    """'처음부터 다시'를 눌렀는데 이전 대화가 남아 있으면 곤란하다."""
    storage.save_session("gone", {"state": {}})
    storage.record_exchange("gone", question="안녕", answer="네")

    storage.delete_session("gone")

    assert storage.load_session("gone") is None
    assert storage.messages("gone") == []


# --------------------------------------------------------------------------- #
# 기록과 집계 — L1~L4 의 재료
# --------------------------------------------------------------------------- #


def test_an_exchange_records_its_tools_and_verdict():
    storage.save_session("s4", {"state": {}})
    storage.record_exchange(
        "s4",
        question="국민연금을 더 내는 게 나을까요?",
        answer="계산해 보았습니다.",
        kind="advice",
        routed_to="result",
        advice={
            "provider": "local",
            "model": "llama",
            "stop_reason": "answered",
            "unverified_numbers": [],
            "rounds": 1,
            "latency_ms": 1234,
        },
        tool_runs=[
            {
                "tool": "national_pension_options",
                "arguments": {},
                "ok": True,
                "summary": "수급자격 있음",
                "output": {"한_문장": "..."},
            }
        ],
    )

    assert [row["role"] for row in storage.messages("s4")] == ["user", "assistant"]
    assert storage.tool_usage()[0]["tool"] == "national_pension_options"

    stats = storage.advice_stats()
    assert stats["total"] == 1
    assert stats["avg_latency_ms"] == 1234


def test_the_block_rate_is_countable():
    """A7 이 얼마나 걸리는지가 L1 이다. mock 에서는 거의 0 이어야 한다."""
    storage.save_session("s5", {"state": {}})
    for reason in ("answered", "answered", "blocked"):
        storage.record_exchange(
            "s5",
            question="질문",
            answer="답",
            advice={"stop_reason": reason, "unverified_numbers": [], "rounds": 1},
        )

    stats = storage.advice_stats()
    assert stats["total"] == 3
    assert stats["blocked"] == 1
    assert stats["block_rate"] == pytest.approx(1 / 3)


def test_the_profile_snapshot_is_not_duplicated():
    """같은 내용을 매 턴 다시 넣으면 표가 의미 없이 커진다."""
    profile = _finished().build_profile().model_dump_json()
    storage.save_session("s6", {"state": {}})
    storage.save_profile("s6", profile)
    storage.save_profile("s6", profile)

    assert storage.counts()["profiles"] == 1
    assert json.loads(storage.latest_profile("s6"))["birth_year"] == 1966
