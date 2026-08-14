"""sqlite 연결과 스키마 적용.

이 패키지는 `core/` 도 `agents/` 도 import 하지 않는 방향으로 만들었다. 반대로
`core/` 와 `agents/` 는 저장소를 모른다 — 계산과 LLM 계층이 저장 방식에 묶이면
나중에 DB 를 바꿀 때 그것들까지 흔들린다.

스키마는 `schema.sql` 한 파일이고 `CREATE TABLE IF NOT EXISTS` 로만 되어 있어서
연결할 때마다 그냥 다시 실행한다. 마이그레이션 도구를 붙이지 않은 이유는 아직
바꿀 스키마 이력이 없기 때문이다. 컬럼을 지우거나 이름을 바꿀 일이 생기면 그때
버전 테이블을 넣는다.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# 기본 위치. 자산·소득이 들어가므로 var/ 는 .gitignore 에 있다.
DEFAULT_DB_PATH = REPO_ROOT / "var" / "finagent.db"

# 연결은 스레드마다 따로 만든다. sqlite3 연결 객체는 스레드 간 공유가 안 되고,
# FastAPI 는 요청을 스레드풀에서 처리한다.
_local = threading.local()
_schema_applied: set[str] = set()
_schema_lock = threading.Lock()


def db_path() -> Path:
    """DB 파일 경로. `FINAGENT_DB_PATH` 로 덮을 수 있다.

    테스트는 임시 경로를 넣어 격리하고, 배포에서는 마운트된 볼륨을 가리키게 한다.
    """
    override = os.environ.get("FINAGENT_DB_PATH", "").strip()
    return Path(override) if override else DEFAULT_DB_PATH


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """현재 스레드의 연결. 없으면 만들고 스키마를 적용한다."""
    target = Path(path) if path else db_path()
    key = str(target)

    cached = getattr(_local, "connections", None)
    if cached is None:
        cached = _local.connections = {}
    if key in cached:
        return cached[key]

    if key != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(key, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    _apply_schema(connection, key)
    cached[key] = connection
    return connection


def _apply_schema(connection: sqlite3.Connection, key: str) -> None:
    """스키마를 적용한다. 같은 파일에는 프로세스당 한 번만."""
    with _schema_lock:
        if key in _schema_applied:
            return
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.commit()
        _schema_applied.add(key)


@contextmanager
def transaction(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """쓰기 묶음. 예외가 나면 되돌린다.

    한 번의 사용자 입력이 messages·advice·tool_runs 여러 줄을 만든다. 중간에
    끊기면 답변은 남았는데 실행 기록은 없는 상태가 되므로 묶어서 커밋한다.
    """
    connection = connect(path)
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    connection.commit()


def reset(path: Optional[Path] = None) -> None:
    """이 프로세스의 연결을 닫는다. 테스트에서 파일을 갈아 끼울 때 쓴다."""
    cached = getattr(_local, "connections", None) or {}
    key = str(Path(path) if path else db_path())
    connection = cached.pop(key, None)
    if connection is not None:
        connection.close()
    with _schema_lock:
        _schema_applied.discard(key)
