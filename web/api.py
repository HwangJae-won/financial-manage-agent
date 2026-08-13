"""FastAPI 백엔드.

W1 의 설계 원칙(`core/` 는 LLM·UI 를 모른다)이 여기서 배당금을 준다.
Streamlit 에서 웹으로 넘어오면서 `core/` 와 `agents/` 는 **한 줄도 고치지 않았다.**
이 파일은 그것들을 HTTP 로 감싸는 얇은 어댑터일 뿐이다.

세션 저장은 메모리 딕셔너리다. 데모 범위에서는 충분하지만 프로세스를 재시작하면
사라지고 여러 워커로 확장할 수 없다 — 실서비스로 가면 Redis 나 DB 로 옮겨야 한다.
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
from agents.mocks import demo_handler
from agents.profiling import ProfilingAgent
from core.asset_map import AssetMap, build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_years
from core.models import SimulationResult, UserProfile
from core.montecarlo import MonteCarloResult, run_monte_carlo
from core.prescribe import PrescriptionSet, prescribe
from core.sensitivity import SensitivityReport, analyze_sensitivity
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

# 세션 저장소 — 데모용 인메모리. 실서비스에서는 Redis/DB 로 교체할 것.
_SESSIONS: dict[str, ProfilingAgent] = {}
_MAX_SESSIONS = 200


def _client():
    """LLM 클라이언트. 실패해도 서버를 죽이지 않고 mock 으로 떨어진다."""
    try:
        client = get_client()
    except Exception:
        client = MockClient()
    if isinstance(client, MockClient):
        client.structured_handler = demo_handler(_dt.date.today().year)
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


class UserMessage(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


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


def _state_of(session_id: str, agent: ProfilingAgent) -> ConversationState:
    answered, total = agent.progress
    return ConversationState(
        session_id=session_id,
        messages=[Message(**m) for m in agent.state.get("messages", [])],
        done=agent.done,
        ready=agent.ready,
        answered=answered,
        total=total,
        error=agent.state.get("error"),
    )


def _get_agent(session_id: str) -> ProfilingAgent:
    agent = _SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다. 다시 시작해 주세요.")
    return agent


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


@app.post("/api/sessions", response_model=ConversationState, status_code=201)
def create_session() -> ConversationState:
    """상담을 시작하고 첫 질문을 돌려준다."""
    if len(_SESSIONS) >= _MAX_SESSIONS:
        # 데모용 상한. 오래된 것부터 버린다.
        for stale in list(_SESSIONS)[: _MAX_SESSIONS // 2]:
            _SESSIONS.pop(stale, None)

    session_id = uuid.uuid4().hex
    agent = ProfilingAgent(client=_client())
    agent.start()
    _SESSIONS[session_id] = agent
    return _state_of(session_id, agent)


@app.get("/api/sessions/{session_id}", response_model=ConversationState)
def get_session(session_id: str) -> ConversationState:
    return _state_of(session_id, _get_agent(session_id))


@app.post("/api/sessions/{session_id}/messages", response_model=ConversationState)
def send_message(session_id: str, message: UserMessage) -> ConversationState:
    """사용자 답변을 보내고 다음 질문을 받는다."""
    agent = _get_agent(session_id)
    agent.respond(message.text)
    return _state_of(session_id, agent)


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
