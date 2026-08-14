-- 은퇴 자산 네비게이터 저장소
--
-- sqlite 를 고른 이유는 서버 프로세스가 필요 없어서다. 발표 당일 DB 가 안 떠서
-- 데모가 죽는 경로를 만들지 않는다 — 키 없이도 전체 흐름이 돌아야 한다는 규칙과
-- 같은 이유다.
--
-- ⚠️ 이 파일에는 사용자의 자산·소득·연금이 들어간다. DB 파일(var/)은 커밋하지 않는다.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 사용자 --------------------------------------------------------------------
--
-- 인증은 만들지 않는다. 브라우저가 들고 있는 uuid 하나가 신원의 전부이고,
-- 비밀번호도 이메일도 받지 않는다. 데모 범위에서 인증을 흉내 내면 안전하지
-- 않은 것을 안전한 것처럼 보이게 만들 뿐이다.
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    nickname   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

-- 세션 ----------------------------------------------------------------------
--
-- Supervisor 하나를 통째로 담는다. 프로파일링 상태·상담 이력·그 뒤의 주고받음이
-- 전부 평범한 dict/dataclass 라 JSON 한 덩어리로 왕복할 수 있다.
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    done         INTEGER NOT NULL DEFAULT 0,
    ready        INTEGER NOT NULL DEFAULT 0,
    state_json   TEXT NOT NULL DEFAULT '{}',  -- ProfilingState
    after_json   TEXT NOT NULL DEFAULT '[]',  -- 프로파일링 이후의 말풍선
    advisor_json TEXT NOT NULL DEFAULT '[]'   -- Advisor.turns
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);

-- 완성된 프로파일 ------------------------------------------------------------
--
-- 세션 상태에서 다시 만들 수 있지만 따로 남긴다. 나중에 "이 사람의 계획이
-- 언제 어떻게 바뀌었나"를 보려면 스냅샷이 필요하고, ⑨ 능동 알림의 선행 조건이다.
CREATE TABLE IF NOT EXISTS profiles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    profile_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_session ON profiles(session_id, created_at DESC);

-- 대화 ----------------------------------------------------------------------
--
-- kind / routed_to 는 Supervisor 가 어디로 보냈는지다. 화면이 어떻게 그릴지도
-- 정하지만, 나중에 "질문이 엉뚱한 데로 갔다"를 찾는 단서이기도 하다.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'question',
    routed_to  TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

-- 상담 답변 한 건 ------------------------------------------------------------
--
-- 여기부터 아래 두 표가 **L1~L4 의 재료**다. 지금까지는 이 기록이 메모리에만
-- 있다가 사라져서, "실제 모델이 A7 에 얼마나 걸리나" 같은 질문에 추측으로만
-- 답할 수 있었다.
CREATE TABLE IF NOT EXISTS advice (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    stop_reason     TEXT NOT NULL,               -- answered / max_rounds / llm_error / blocked
    unverified_json TEXT NOT NULL DEFAULT '[]',  -- 걸러낸 숫자 (A7)
    rounds          INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_advice_message ON advice(message_id);

-- 도구 실행 기록 -------------------------------------------------------------
--
-- 화면의 trace(A4)와 같은 내용이다. 어떤 도구를 어떤 인자로 불러 무슨 값이
-- 나왔는지. tool 분포가 곧 도구 선택 정확도(L2)의 집계 대상이 된다.
CREATE TABLE IF NOT EXISTS tool_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id     INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    tool           TEXT NOT NULL,
    arguments_json TEXT NOT NULL DEFAULT '{}',
    ok             INTEGER NOT NULL DEFAULT 1,
    summary        TEXT NOT NULL DEFAULT '',
    output_json    TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_runs_message ON tool_runs(message_id, seq);
CREATE INDEX IF NOT EXISTS idx_tool_runs_tool ON tool_runs(tool);
