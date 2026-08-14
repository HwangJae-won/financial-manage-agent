"""FastAPI 백엔드.

W1 의 설계 원칙(`core/` 는 LLM·UI 를 모른다)이 여기서 배당금을 준다.
Streamlit 에서 웹으로 넘어오면서 `core/` 와 `agents/` 는 **한 줄도 고치지 않았다.**
이 파일은 그것들을 HTTP 로 감싸는 얇은 어댑터일 뿐이다.

세션은 **sqlite 에 남는다**(`storage/`). 메모리 딕셔너리는 매 요청마다 객체를 다시
만들지 않으려는 캐시일 뿐이고, 캐시에 없으면 저장소에서 되살린다. 프로세스를
재시작해도 진행 중이던 상담이 이어진다.

저장이 실패해도 응답은 나간다. 저장소는 편의 기능이지 서비스의 전제가 아니다 —
키 없이도 전체 흐름이 돌아야 한다는 규칙과 같은 이유다.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.config import describe as describe_llm
from agents.explain import Briefing, explain
from agents.fraud import FraudAssessment, analyze_message
from agents.llm import MockClient, get_client
from agents.mocks import demo_handler, demo_tool_handler
from agents.advisor import AdvisorReply
from agents.graph import Supervisor, SupervisorReply
import storage
from core.asset_map import AssetMap, build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_years
from core.models import SimulationResult, UserProfile
from core.montecarlo import MonteCarloResult, run_monte_carlo
from core.prescribe import PrescriptionSet, prescribe
from core.sensitivity import SensitivityReport, analyze_sensitivity
from core.family_report import FamilyReport, build_family_report
from core.health_insurance import HealthInsuranceReport, analyze_health_insurance
from core.national_pension import NationalPensionReport, analyze_national_pension
from core.severance import SeveranceComparison, compare_severance_options
from core.policy import (
    DEFAULT_POLICY_KEY,
    PolicyFact,
    PolicyImpact,
    analyze_policy_impact,
    load_policy_catalog,
)
from core.risk_score import RiskScore, compute_risk_score
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.scenarios import Scenario, build_scenarios

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="내 자산 AI 네비게이터",
    description="은퇴 시니어를 위한 금융 의사결정 지원 API",
    version="0.1.0",
)

# 살아 있는 Supervisor 캐시. **이제 진실의 출처는 sqlite 다** — 여기는 매 요청마다
# 객체를 다시 만들지 않으려는 캐시일 뿐이고, 없으면 저장소에서 되살린다.
_SESSIONS: dict[str, Supervisor] = {}
_SESSION_USERS: dict[str, str] = {}
_MAX_SESSIONS = 200


def _client():
    """LLM 클라이언트. 실패해도 서버를 죽이지 않고 mock 으로 떨어진다."""
    try:
        client = get_client()
    except Exception:
        client = MockClient()
    if isinstance(client, MockClient):
        client.structured_handler = demo_handler(_dt.date.today().year)
        # 질문에 맞는 도구를 고르게 한다. 기본 동작은 언제나 첫 도구라, 화면에
        # trace 를 띄우는 지금은 에이전트가 질문을 못 알아듣는 것처럼 보인다.
        client.tool_handler = demo_tool_handler()
    return client


# --------------------------------------------------------------------------- #
# 응답 모델
# --------------------------------------------------------------------------- #


class Health(BaseModel):
    status: str
    llm: str
    active_sessions: int


class Message(BaseModel):
    role: str
    content: str


class ConversationState(BaseModel):
    session_id: str
    messages: list[Message]
    done: bool
    ready: bool = Field(description="필수·중요 질문에 다 답해 결과를 봐도 되는 상태")
    answered: int
    total: int
    error: Optional[str] = None

    # 결과가 나온 뒤의 답변에만 붙는다. 프로파일링 중에는 전부 None 이다.
    kind: str = Field(
        default="question", description="question / advice / fraud — 화면이 어떻게 그릴지"
    )
    routed_to: Optional[str] = Field(
        default=None, description="어느 에이전트로 갔는지 (Supervisor 분류 결과)"
    )
    routing_reason: str = ""
    advice: Optional[AdvisorReply] = Field(
        default=None, description="상담 에이전트의 답변과 실행 기록(trace)"
    )
    fraud: Optional[FraudAssessment] = None


class UserMessage(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class UserRequest(BaseModel):
    """사용자 식별. 인증이 아니다 — id 를 아는 것이 곧 신원이다."""

    user_id: Optional[str] = Field(
        default=None, description="브라우저가 들고 있던 id. 없으면 새로 만든다"
    )
    nickname: str = Field(default="", max_length=40)


class UserInfo(BaseModel):
    user_id: str
    nickname: str = ""
    sessions: list[dict[str, Any]] = Field(
        default_factory=list, description="이 사용자의 지난 상담 목록 (최근 순)"
    )


class SessionRequest(BaseModel):
    user_id: Optional[str] = Field(
        default=None, description="있으면 이 사용자의 상담으로 묶는다"
    )


class Headline(BaseModel):
    """화면 맨 위에 크게 띄우는 값. 서버에서 미리 포맷해 보낸다.

    프런트엔드가 금액 표기 규칙을 다시 구현하면 화면마다 달라진다.
    포맷은 core/formatting.py 한 곳에서만 한다.
    """

    total_assets: str
    financial_assets: str
    income_gap: str
    expense_coverage: str
    depletion: str
    risk_score: str
    risk_status: str


class AnalysisResponse(BaseModel):
    profile: UserProfile
    headline: Headline
    simulation: SimulationResult
    asset_map: AssetMap
    monte_carlo: MonteCarloResult
    risk_score: RiskScore
    scenarios: list[Scenario]
    briefing: Optional[Briefing] = None
    assumed_fields: list[str] = Field(
        default_factory=list, description="사용자가 답하지 않아 가정한 항목"
    )


class FraudRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    session_id: Optional[str] = Field(
        default=None, description="있으면 사용자 보유 자산과 엮어 설명한다"
    )


class PrescriptionRequest(BaseModel):
    """처방 요청. 프로파일을 찾는 규칙은 정책 영향 요청과 같다."""

    target_age: Optional[int] = Field(
        default=None, ge=60, le=110, description="목표 나이. 생략하면 시뮬레이션 종료 나이"
    )
    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


class SeveranceRequest(BaseModel):
    """퇴직금 수령 방식 비교 요청."""

    pension_years: Optional[int] = Field(default=None, ge=1, le=40)
    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


class HealthInsuranceRequest(BaseModel):
    """퇴직 후 건강보험료 요청.

    세 값은 전부 선택이다. 없으면 없는 대로 계산하고 **모르는 부분은 모른다고
    화면에 적어 보낸다** — 기본값으로 채우면 사용자가 그 숫자를 믿게 된다.
    """

    monthly_salary: Optional[int] = Field(
        default=None, ge=0, description="퇴직 전 월 급여(원). 임의계속가입 보험료의 기준"
    )
    property_tax_base: Optional[int] = Field(
        default=None, ge=0, description="재산세 고지서의 과세표준(원)"
    )
    has_employed_family: Optional[bool] = Field(
        default=None, description="직장 다니는 배우자·자녀가 있는지"
    )
    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


class NationalPensionRequest(BaseModel):
    """국민연금 임의계속가입·추납 요청.

    가입월수는 화면에서 따로 받는다. 프로파일에 0으로 들어와도 사용자가 여기서
    알려주면 계산할 수 있어야 하기 때문이다 — 건강보험료의 월 급여와 같은 규약이다.
    """

    contributed_months: Optional[int] = Field(
        default=None, ge=0, le=600, description="국민연금 가입월수"
    )
    catchup_months: Optional[int] = Field(
        default=None, ge=0, le=600, description="추납 가능 개월수(납부예외·적용제외 기간)"
    )
    monthly_income_base: Optional[int] = Field(
        default=None, ge=0, description="기준소득월액(원). 생략하면 퇴직 전 월 급여"
    )
    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


class SensitivityRequest(BaseModel):
    """민감도 분석 요청. 프로파일을 찾는 규칙은 다른 요청과 같다."""

    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


class PolicyImpactRequest(BaseModel):
    """정책 영향 계산 요청.

    프로파일은 세 가지 경로로 온다: 직접 입력(profile) / 대화 세션(session_id) /
    예시 인물(sample). 화면에서 어느 경로로 들어와도 같은 답을 주기 위한 것이다.
    """

    policy_key: str = DEFAULT_POLICY_KEY
    profile: Optional[UserProfile] = None
    session_id: Optional[str] = None
    sample: Optional[str] = None


# --------------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------------- #


def _state_of(
    session_id: str,
    supervisor: Supervisor,
    reply: Optional[SupervisorReply] = None,
) -> ConversationState:
    answered, total = supervisor.progress
    state = ConversationState(
        session_id=session_id,
        messages=[Message(**m) for m in supervisor.messages],
        done=supervisor.done,
        ready=supervisor.ready,
        answered=answered,
        total=total,
        error=supervisor.profiler.state.get("error"),
    )
    if reply is not None:
        state.kind = reply.kind
        state.routed_to = reply.intent.value
        state.routing_reason = reply.reason
        state.advice = reply.advice
        state.fraud = reply.fraud
    return state


def _get_agent(session_id: str) -> Supervisor:
    """세션을 찾는다. 메모리에 없으면 **저장소에서 되살린다.**

    이 한 줄이 서버 재시작을 견디게 한다. 예전에는 프로세스가 죽으면 진행 중이던
    상담이 통째로 사라졌다.
    """
    agent = _SESSIONS.get(session_id)
    if agent is not None:
        return agent

    saved = storage.load_session(session_id)
    if saved is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다. 다시 시작해 주세요.")

    agent = Supervisor.restore(saved, client=_client())
    _SESSIONS[session_id] = agent
    return agent


def _persist(session_id: str, agent: Supervisor, reply: Optional[SupervisorReply]) -> None:
    """세션과 이번 주고받음을 남긴다.

    **저장이 실패해도 응답은 나간다.** 디스크가 가득 찼다고 상담이 멈추면 안 된다.
    저장소는 편의 기능이지 서비스의 전제가 아니다.
    """
    try:
        storage.save_session(
            session_id,
            agent.dump(),
            user_id=_SESSION_USERS.get(session_id),
            done=agent.done,
            ready=agent.ready,
        )
        # 프로파일 스냅샷은 **대화가 끝난 뒤에만.** 진행 중에는 슬롯이 하나씩
        # 차면서 매 턴 다른 값이 되어, 남겨도 쓸 데 없는 중간 상태만 쌓인다.
        if agent.done:
            profile = agent.profile()
            if profile is not None:
                storage.save_profile(session_id, profile.model_dump_json())

        if reply is not None:
            advice = reply.advice
            storage.record_exchange(
                session_id,
                question=reply.question,
                answer=reply.text,
                kind=reply.kind,
                routed_to=reply.intent.value,
                advice=(
                    {
                        "provider": getattr(agent.client, "provider", ""),
                        "model": getattr(agent.client, "model", ""),
                        "stop_reason": advice.stop_reason,
                        "unverified_numbers": advice.unverified_numbers,
                        "rounds": advice.rounds,
                        "latency_ms": reply.latency_ms,
                    }
                    if advice is not None
                    else None
                ),
                tool_runs=(
                    [
                        {
                            "tool": step.tool,
                            "arguments": step.arguments,
                            "ok": step.ok,
                            "summary": step.summary,
                            "output": step.output,
                        }
                        for step in advice.trace
                    ]
                    if advice is not None
                    else None
                ),
            )
    except Exception:  # pragma: no cover - 저장 실패로 대화를 끊지 않는다
        pass


def _headline(
    profile: UserProfile, sim: SimulationResult, score: RiskScore
) -> Headline:
    if sim.depletion_age is not None:
        depletion = f"만 {sim.depletion_age}세 ({sim.depletion_year}년)"
    else:
        depletion = f"만 {sim.horizon_age}세까지 유지"

    return Headline(
        total_assets=fmt_krw(profile.total_assets),
        financial_assets=fmt_krw(profile.financial_assets),
        income_gap=fmt_months(sim.income_gap_months),
        expense_coverage=fmt_years(sim.expense_coverage_years),
        depletion=depletion,
        risk_score=f"{score.total} / 100",
        risk_status=score.status,
    )


def _analyze(profile: UserProfile, *, n_paths: int, with_briefing: bool) -> AnalysisResponse:
    sim = simulate(profile)
    amap = build_asset_map(profile)
    mc = run_monte_carlo(profile, n_paths=n_paths)
    score = compute_risk_score(profile)
    scenarios = build_scenarios(profile, n_paths=n_paths)

    briefing = None
    if with_briefing:
        briefing = explain(
            profile, sim, amap, score=score, monte_carlo=mc, client=_client()
        )

    return AnalysisResponse(
        profile=profile,
        headline=_headline(profile, sim, score),
        simulation=sim,
        asset_map=amap,
        monte_carlo=mc,
        risk_score=score,
        scenarios=scenarios,
        briefing=briefing,
        assumed_fields=profile.assumed_fields,
    )


# --------------------------------------------------------------------------- #
# 엔드포인트
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=Health)
def health() -> Health:
    try:
        llm = describe_llm()
    except Exception as exc:
        llm = f"설정 확인 필요 ({exc})"
    return Health(status="ok", llm=llm, active_sessions=len(_SESSIONS))


@app.post("/api/users", response_model=UserInfo, status_code=201)
def create_or_get_user(request: UserRequest) -> UserInfo:
    """사용자를 만들거나 이미 있으면 그대로 돌려준다.

    **인증이 아니다.** 브라우저가 들고 있는 uuid 하나가 신원의 전부이고 비밀번호도
    이메일도 받지 않는다. 데모 범위의 결정이며 화면에도 그렇게 적는다.
    """
    user_id = storage.ensure_user(request.user_id, nickname=request.nickname)
    saved = storage.get_user(user_id) or {}
    return UserInfo(
        user_id=user_id,
        nickname=saved.get("nickname", ""),
        sessions=storage.list_sessions(user_id),
    )


@app.post("/api/sessions", response_model=ConversationState, status_code=201)
def create_session(request: Optional[SessionRequest] = None) -> ConversationState:
    """상담을 시작하고 첫 질문을 돌려준다."""
    if len(_SESSIONS) >= _MAX_SESSIONS:
        # 메모리 캐시 상한. 버려도 저장소에 남아 있어 다시 부르면 되살아난다.
        for stale in list(_SESSIONS)[: _MAX_SESSIONS // 2]:
            _SESSIONS.pop(stale, None)
            _SESSION_USERS.pop(stale, None)

    session_id = uuid.uuid4().hex
    agent = Supervisor(client=_client())
    agent.start()
    _SESSIONS[session_id] = agent

    user_id = (request.user_id if request else None) or None
    if user_id:
        _SESSION_USERS[session_id] = storage.ensure_user(user_id)
    _persist(session_id, agent, None)
    return _state_of(session_id, agent)


@app.get("/api/sessions/{session_id}", response_model=ConversationState)
def get_session(session_id: str) -> ConversationState:
    return _state_of(session_id, _get_agent(session_id))


@app.delete("/api/sessions/{session_id}", status_code=204)
def drop_session(session_id: str) -> None:
    """세션과 딸린 기록을 지운다. '처음부터 다시'가 여기로 온다."""
    _SESSIONS.pop(session_id, None)
    _SESSION_USERS.pop(session_id, None)
    storage.delete_session(session_id)


@app.post("/api/sessions/{session_id}/messages", response_model=ConversationState)
def send_message(session_id: str, message: UserMessage) -> ConversationState:
    """대화의 단일 창구.

    Supervisor 가 분류해서 보낸다 — 프로파일링 답변이면 다음 질문을, 결과가 나온
    뒤의 질문이면 도구를 돌린 상담 답변을, 붙여넣은 문자면 사기 확인 결과를
    돌려준다. 화면은 `kind` 만 보고 어떻게 그릴지 정한다.
    """
    agent = _get_agent(session_id)
    reply = agent.send(message.text)
    _persist(session_id, agent, reply)
    return _state_of(session_id, agent, reply)


@app.get("/api/sessions/{session_id}/analysis", response_model=AnalysisResponse)
def get_analysis(
    session_id: str, n_paths: int = 3000, briefing: bool = True
) -> AnalysisResponse:
    """대화로 모은 정보로 전체 분석을 돌린다."""
    agent = _get_agent(session_id)
    try:
        profile = agent.build_profile()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _analyze(profile, n_paths=n_paths, with_briefing=briefing)


@app.post("/api/analysis", response_model=AnalysisResponse)
def analyze_profile(
    profile: UserProfile, n_paths: int = 3000, briefing: bool = False
) -> AnalysisResponse:
    """대화 없이 프로파일을 직접 넣어 분석한다 (직접 입력 화면·테스트용)."""
    return _analyze(profile, n_paths=n_paths, with_briefing=briefing)


@app.get("/api/samples/{key}", response_model=UserProfile)
def get_sample(key: str) -> UserProfile:
    samples = {"demo": DEMO_PROFILE, "diversified": DIVERSIFIED_PROFILE}
    if key not in samples:
        raise HTTPException(404, f"알 수 없는 예시입니다: {key}")
    return samples[key]


def _resolve_profile(
    profile: Optional[UserProfile], session_id: Optional[str], sample: Optional[str]
) -> UserProfile:
    """직접 입력 / 대화 세션 / 예시 인물 — 어느 경로로 들어와도 같은 답을 준다."""
    if profile is not None:
        return profile

    if session_id:
        agent = _SESSIONS.get(session_id)
        if agent is not None:
            try:
                return agent.build_profile()
            except ValueError:
                pass  # 프로파일이 아직 완성되지 않았으면 다음 경로로 넘어간다

    if sample:
        samples = {"demo": DEMO_PROFILE, "diversified": DIVERSIFIED_PROFILE}
        if sample in samples:
            return samples[sample]

    raise HTTPException(400, "먼저 상담을 마치시거나 예시 인물을 선택해 주세요.")


@app.post("/api/prescriptions", response_model=PrescriptionSet)
def prescriptions(request: PrescriptionRequest) -> PrescriptionSet:
    """목표 나이까지 유지하려면 무엇을 얼마나 바꿔야 하는지 (기능 ⑤ 확장)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return prescribe(profile, target_age=request.target_age)


@app.post("/api/severance", response_model=SeveranceComparison)
def severance(request: SeveranceRequest) -> SeveranceComparison:
    """퇴직금을 일시금으로 받을지 연금으로 받을지 (기능 ⑤)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return compare_severance_options(profile, pension_years=request.pension_years)


@app.post("/api/health-insurance", response_model=HealthInsuranceReport)
def health_insurance(request: HealthInsuranceRequest) -> HealthInsuranceReport:
    """퇴직하면 건강보험료가 언제 얼마나 생기는지 (피부양자 절벽)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return analyze_health_insurance(
        profile,
        monthly_salary=request.monthly_salary,
        property_tax_base_override=request.property_tax_base,
        has_employed_family=request.has_employed_family,
    )


@app.post("/api/national-pension", response_model=NationalPensionReport)
def national_pension(request: NationalPensionRequest) -> NationalPensionReport:
    """국민연금을 더 내는 게 이득인지 (임의계속가입·추납)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return analyze_national_pension(
        profile,
        contributed_months=request.contributed_months,
        catchup_months=request.catchup_months,
        monthly_income_base=request.monthly_income_base,
    )


@app.post("/api/family-report", response_model=FamilyReport)
def family_report(request: SensitivityRequest) -> FamilyReport:
    """가족에게 보여줄 한 장 (기능 5).

    요청 형태가 민감도 분석과 같다(프로파일만 있으면 된다). 모델을 하나 더
    만들지 않고 같은 것을 쓴다.
    """
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return build_family_report(profile)


@app.post("/api/sensitivity", response_model=SensitivityReport)
def sensitivity(request: SensitivityRequest) -> SensitivityReport:
    """가정이 틀렸을 때 결과가 얼마나 흔들리는지 (자기검증)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    return analyze_sensitivity(profile)


@app.get("/api/policies", response_model=list[PolicyFact])
def policies() -> list[PolicyFact]:
    """정책 팩트시트 목록.

    출처·확정여부·시행일·확인일이 모델에 이미 들어 있다(기능 ⑧). 프런트엔드가
    뱃지 문구를 다시 만들지 않게 서버가 완성해서 보낸다.
    """
    return load_policy_catalog()


@app.post("/api/policy-impact", response_model=PolicyImpact)
def policy_impact(request: PolicyImpactRequest) -> PolicyImpact:
    """정책 한 건이 이 사용자의 은퇴 계획에 미치는 영향을 계산한다 (기능 ④)."""
    profile = _resolve_profile(request.profile, request.session_id, request.sample)
    try:
        return analyze_policy_impact(profile, policy_key=request.policy_key)
    except KeyError as exc:
        raise HTTPException(404, f"알 수 없는 정책입니다: {request.policy_key}") from exc


@app.post("/api/fraud-check", response_model=FraudAssessment)
def fraud_check(request: FraudRequest) -> FraudAssessment:
    """받은 문자·카톡의 위험 신호를 확인한다."""
    profile: Optional[UserProfile] = None
    if request.session_id:
        agent = _SESSIONS.get(request.session_id)
        if agent is not None:
            try:
                profile = agent.build_profile()
            except ValueError:
                profile = None  # 아직 프로파일이 완성되지 않았으면 연결만 생략한다
    return analyze_message(request.text, profile=profile)


# --------------------------------------------------------------------------- #
# 정적 파일 (프런트엔드)
# --------------------------------------------------------------------------- #

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")
