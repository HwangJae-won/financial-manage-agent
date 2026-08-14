"""저장·복원. 애플리케이션이 호출하는 유일한 자리.

**여기는 평범한 dict 만 주고받는다.** `Supervisor` 나 `UserProfile` 같은 타입을
알지 못한다. 직렬화는 부르는 쪽(`Supervisor.dump()`)이 하고 저장소는 그것을
받아 넣기만 한다. 이렇게 나눠 두면 `agents/` 가 저장 방식을 모르고, `storage/`
가 에이전트 구조를 몰라도 된다 — 한쪽을 바꿔도 다른 쪽이 흔들리지 않는다.

기록의 두 번째 용도가 있다. `advice` 와 `tool_runs` 는 화면에 그리려고 남기는
것이기도 하지만, **로컬 모델을 붙인 뒤 "실제로 어땠나"를 집계로 답하기 위한
재료**다. 지금까지 그 기록은 메모리에만 있다가 프로세스와 함께 사라졌다.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from storage.db import connect, transaction


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(raw: Optional[str], fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------- #
# 사용자
# --------------------------------------------------------------------------- #


def ensure_user(
    user_id: Optional[str] = None,
    *,
    nickname: str = "",
    path: Optional[Path] = None,
) -> str:
    """사용자를 만들거나 이미 있으면 그대로 둔다. 사용자 id 를 돌려준다.

    인증이 없으므로 **id 를 아는 것이 곧 신원**이다. 브라우저가 uuid 를 들고
    있고 서버는 그것을 믿는다. 데모 범위의 결정이며 화면에 그렇게 적는다.
    """
    user_id = user_id or uuid.uuid4().hex
    with transaction(path) as connection:
        connection.execute(
            "INSERT INTO users (id, nickname, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET nickname = "
            "  CASE WHEN excluded.nickname != '' THEN excluded.nickname "
            "       ELSE users.nickname END",
            (user_id, nickname, _now()),
        )
    return user_id


def get_user(user_id: str, *, path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    row = connect(path).execute(
        "SELECT id, nickname, created_at FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# 세션
# --------------------------------------------------------------------------- #


def save_session(
    session_id: str,
    payload: dict[str, Any],
    *,
    user_id: Optional[str] = None,
    done: bool = False,
    ready: bool = False,
    path: Optional[Path] = None,
) -> None:
    """세션 상태를 통째로 덮어쓴다.

    `payload` 는 `Supervisor.dump()` 의 결과 — state / after / advisor 세 칸이다.
    매 턴 전체를 다시 쓰는 것이 낭비로 보일 수 있지만, 대화 하나가 수십 KB 라
    문제가 되지 않고 부분 갱신보다 어긋날 여지가 훨씬 적다.
    """
    now = _now()
    with transaction(path) as connection:
        connection.execute(
            """
            INSERT INTO sessions
                (id, user_id, created_at, updated_at, done, ready,
                 state_json, after_json, advisor_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id      = COALESCE(excluded.user_id, sessions.user_id),
                updated_at   = excluded.updated_at,
                done         = excluded.done,
                ready        = excluded.ready,
                state_json   = excluded.state_json,
                after_json   = excluded.after_json,
                advisor_json = excluded.advisor_json
            """,
            (
                session_id,
                user_id,
                now,
                now,
                int(done),
                int(ready),
                _dumps(payload.get("state", {})),
                _dumps(payload.get("after", [])),
                _dumps(payload.get("advisor", [])),
            ),
        )


def load_session(
    session_id: str, *, path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """세션 상태. 없으면 None — 예외로 흐름을 끊지 않는다."""
    row = connect(path).execute(
        "SELECT id, user_id, done, ready, state_json, after_json, advisor_json, "
        "       created_at, updated_at "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["id"],
        "user_id": row["user_id"],
        "done": bool(row["done"]),
        "ready": bool(row["ready"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "state": _loads(row["state_json"], {}),
        "after": _loads(row["after_json"], []),
        "advisor": _loads(row["advisor_json"], []),
    }


def list_sessions(
    user_id: str, *, limit: int = 20, path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """한 사용자의 세션 목록. 최근 것부터."""
    rows = connect(path).execute(
        "SELECT id, done, ready, created_at, updated_at FROM sessions "
        "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str, *, path: Optional[Path] = None) -> None:
    """세션과 딸린 기록을 지운다 (외래키 CASCADE).

    사용자가 '처음부터 다시'를 눌렀을 때 이전 대화가 남아 있으면 곤란하다.
    """
    with transaction(path) as connection:
        connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def save_profile(
    session_id: str, profile_json: str, *, path: Optional[Path] = None
) -> None:
    """완성된 프로파일 스냅샷. 같은 내용이면 다시 넣지 않는다."""
    connection = connect(path)
    latest = connection.execute(
        "SELECT profile_json FROM profiles WHERE session_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if latest and latest["profile_json"] == profile_json:
        return
    with transaction(path) as conn:
        conn.execute(
            "INSERT INTO profiles (session_id, profile_json, created_at) "
            "VALUES (?, ?, ?)",
            (session_id, profile_json, _now()),
        )


def latest_profile(
    session_id: str, *, path: Optional[Path] = None
) -> Optional[str]:
    row = connect(path).execute(
        "SELECT profile_json FROM profiles WHERE session_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    return row["profile_json"] if row else None


# --------------------------------------------------------------------------- #
# 대화와 실행 기록
# --------------------------------------------------------------------------- #


def _next_seq(connection: sqlite3.Connection, session_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["next"])


def record_exchange(
    session_id: str,
    *,
    question: str,
    answer: str,
    kind: str = "question",
    routed_to: Optional[str] = None,
    advice: Optional[dict[str, Any]] = None,
    tool_runs: Optional[list[dict[str, Any]]] = None,
    path: Optional[Path] = None,
) -> int:
    """사용자 입력 한 번과 그에 대한 답을 한 묶음으로 남긴다.

    답변 메시지의 id 를 돌려준다. 실행 기록(`tool_runs`)과 상담 메타(`advice`)가
    그 id 에 매달리므로, 중간에 끊기면 답변만 있고 근거가 없는 상태가 된다 —
    그래서 한 트랜잭션이다.
    """
    now = _now()
    with transaction(path) as connection:
        seq = _next_seq(connection, session_id)
        connection.execute(
            "INSERT INTO messages (session_id, seq, role, content, kind, routed_to, created_at) "
            "VALUES (?, ?, 'user', ?, ?, ?, ?)",
            (session_id, seq, question, kind, routed_to, now),
        )
        cursor = connection.execute(
            "INSERT INTO messages (session_id, seq, role, content, kind, routed_to, created_at) "
            "VALUES (?, ?, 'assistant', ?, ?, ?, ?)",
            (session_id, seq + 1, answer, kind, routed_to, now),
        )
        message_id = int(cursor.lastrowid)

        if advice is not None:
            connection.execute(
                "INSERT INTO advice (message_id, provider, model, stop_reason, "
                "                    unverified_json, rounds, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    advice.get("provider", ""),
                    advice.get("model", ""),
                    advice.get("stop_reason", "answered"),
                    _dumps(advice.get("unverified_numbers", [])),
                    int(advice.get("rounds", 0)),
                    int(advice.get("latency_ms", 0)),
                ),
            )

        for index, run in enumerate(tool_runs or []):
            connection.execute(
                "INSERT INTO tool_runs (message_id, seq, tool, arguments_json, ok, "
                "                       summary, output_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    index,
                    run.get("tool", ""),
                    _dumps(run.get("arguments", {})),
                    int(bool(run.get("ok", True))),
                    run.get("summary", ""),
                    _dumps(run.get("output", {})),
                    now,
                ),
            )

    return message_id


def messages(session_id: str, *, path: Optional[Path] = None) -> list[dict[str, Any]]:
    rows = connect(path).execute(
        "SELECT seq, role, content, kind, routed_to, created_at FROM messages "
        "WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# 집계 — 로컬 모델을 붙인 뒤 L1~L4 를 추측이 아니라 숫자로 답하기 위한 것
# --------------------------------------------------------------------------- #


def tool_usage(*, path: Optional[Path] = None) -> list[dict[str, Any]]:
    """도구별 호출 횟수와 실패 수 (L2 의 재료)."""
    rows = connect(path).execute(
        "SELECT tool, COUNT(*) AS calls, SUM(1 - ok) AS failures "
        "FROM tool_runs GROUP BY tool ORDER BY calls DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def advice_stats(*, path: Optional[Path] = None) -> dict[str, Any]:
    """상담 답변 집계 — 차단 비율(L1)과 지연(L4).

    `blocked` 는 A7 이 도구 출력에 없는 숫자를 발견해 답변을 버린 경우다.
    mock 은 도구 출력을 그대로 옮겨 적어 거의 0 이 나오고, 실제 모델에서
    이 비율이 얼마나 되는지가 A7 이 과한지 모자란지를 가른다.
    """
    row = connect(path).execute(
        "SELECT COUNT(*) AS total, "
        "       SUM(CASE WHEN stop_reason = 'blocked' THEN 1 ELSE 0 END) AS blocked, "
        "       SUM(CASE WHEN stop_reason = 'llm_error' THEN 1 ELSE 0 END) AS errors, "
        "       AVG(rounds) AS avg_rounds, "
        "       AVG(NULLIF(latency_ms, 0)) AS avg_latency_ms "
        "FROM advice"
    ).fetchone()
    total = int(row["total"] or 0)
    blocked = int(row["blocked"] or 0)
    return {
        "total": total,
        "blocked": blocked,
        "errors": int(row["errors"] or 0),
        "block_rate": (blocked / total) if total else 0.0,
        "avg_rounds": float(row["avg_rounds"] or 0.0),
        "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
    }


def counts(*, path: Optional[Path] = None) -> dict[str, int]:
    """표별 행 수. 저장이 실제로 되고 있는지 한눈에 보는 용도."""
    connection = connect(path)
    tables = ("users", "sessions", "profiles", "messages", "advice", "tool_runs")
    return {
        table: int(
            connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        )
        for table in tables
    }
