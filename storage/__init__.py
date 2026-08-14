"""저장소 계층.

`core/` 는 순수 계산이고 `agents/` 는 LLM 계층이다. **둘 다 저장소를 모른다.**
반대로 저장소도 그것들을 모른다 — 평범한 dict 만 주고받는다. 화면
(`web/api.py`, `app/streamlit_app.py`)만 양쪽을 안다.
"""

from storage.db import connect, db_path, reset, transaction
from storage.repository import (
    advice_stats,
    counts,
    delete_session,
    ensure_user,
    get_user,
    latest_profile,
    list_sessions,
    load_session,
    messages,
    record_exchange,
    save_profile,
    save_session,
    tool_usage,
)

__all__ = [
    "advice_stats",
    "connect",
    "counts",
    "db_path",
    "delete_session",
    "ensure_user",
    "get_user",
    "latest_profile",
    "list_sessions",
    "load_session",
    "messages",
    "record_exchange",
    "reset",
    "save_profile",
    "save_session",
    "tool_usage",
    "transaction",
]
