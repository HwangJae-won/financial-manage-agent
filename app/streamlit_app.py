"""은퇴 자산 네비게이터 — Streamlit 데모.

두 가지 입력 방식을 제공한다:
  - 대화로 알아보기 (기본): 상담사가 12가지를 여쭙고 프로파일을 채운다
  - 직접 입력: 폼으로 값을 넣는다. 데모 중 대화가 꼬였을 때의 안전망이므로
    대화형이 완성된 뒤에도 절대 지우지 않는다.

LLM 설정이 잘못돼도 앱은 죽지 않고 mock 으로 떨어진다. 발표 중에 키가
만료되거나 프로바이더 이름에 오타가 있어도 데모는 끝까지 돌아가야 한다.

실행:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` 은 스크립트 디렉터리를 sys.path 에 넣으므로 저장소 루트를 직접 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime as _dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.config import describe as describe_llm
from agents.explain import Briefing, explain
from agents.fraud import RiskLevel, Severity, analyze_message
from agents.llm import MockClient, get_client
from agents.mocks import demo_handler, demo_tool_handler
from agents.advisor import AdvisorReply
from agents.graph import Supervisor
from core.asset_map import AssetMap, build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_pct, fmt_years
from core.models import LifeEvent, RiskTolerance, UserProfile
from core.montecarlo import run_monte_carlo
from core.policy import PolicyImpact, analyze_policy_impact
from core.prescribe import PrescriptionSet, prescribe
from core.sensitivity import SensitivityReport, analyze_sensitivity
from core.family_report import FamilyReport, build_family_report
from core.health_insurance import HealthInsuranceReport, analyze_health_insurance
from core.national_pension import NationalPensionReport, analyze_national_pension
from core.severance import SeveranceComparison, compare_severance_options
from core.risk_score import RiskScore, compute_risk_score
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE
from core.scenarios import build_scenarios, comparison_table

MAN = 10_000  # 입력은 만원 단위로 받는다 — 시니어에게 자연스럽고 오타가 줄어든다

PALETTE = {
    "survival": "#1f6f8b",
    "income_gap": "#e08214",
    "long_term": "#4d9078",
    "growth": "#8b5fa8",
    "real_asset": "#8c8c8c",
}

st.set_page_config(page_title="내 자산 AI 네비게이터", page_icon="🧭", layout="wide")

# 시니어 접근성: 기본 글씨를 키운다. W6 웹 전환 때 제대로 된 디자인 시스템으로 대체한다.
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 17px; }
      [data-testid="stMetricValue"] { font-size: 1.9rem; }
      [data-testid="stMetricLabel"] { font-size: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# 입력
# --------------------------------------------------------------------------- #


def profile_form() -> UserProfile:
    """사이드바 입력 폼. 금액은 만원 단위로 받아 원으로 변환한다."""
    st.sidebar.header("내 정보")

    preset = st.sidebar.selectbox(
        "예시 불러오기",
        ["기획서 예시 (김OO, 60세 퇴직)", "자산 분산형 (59세 퇴직)"],
        help="값을 바꾸면 아래 결과가 바로 다시 계산됩니다.",
    )
    base = DEMO_PROFILE if preset.startswith("기획서") else DIVERSIFIED_PROFILE

    with st.sidebar.expander("기본 정보", expanded=True):
        birth_year = st.number_input("출생연도", 1940, 2000, base.birth_year, step=1)
        birth_month = st.number_input("출생월", 1, 12, base.birth_month, step=1)
        retirement_year = st.number_input(
            "퇴직(예정) 연도", 2020, 2045, base.retirement_year, step=1
        )
        retirement_month = st.number_input(
            "퇴직(예정) 월", 1, 12, base.retirement_month, step=1
        )
        # 근속연수와 퇴직 전 급여는 세금·건강보험료의 핵심 변수다. 여기가 비어 있으면
        # 그 두 화면은 계산하지 않고 물어본다 — 기본값으로 채우지 않는다.
        years_employed = st.number_input(
            "근속연수", 0, 50, base.years_employed, step=1, help="0이면 퇴직소득세를 계산하지 않습니다"
        )
        last_salary = st.number_input(
            "퇴직 전 월 급여(만원)",
            0,
            5_000,
            base.last_monthly_salary // MAN,
            step=10,
            help="0이면 임의계속가입 보험료를 계산하지 않습니다",
        )
        risk = st.select_slider(
            "투자성향",
            options=[r.value for r in RiskTolerance],
            value=base.risk_tolerance.value,
        )

    with st.sidebar.expander("자산 (만원)", expanded=True):
        severance = st.number_input("퇴직금", 0, 500_000, base.severance_pay // MAN, step=100)
        cash = st.number_input("예금·적금", 0, 500_000, base.cash_savings // MAN, step=100)
        equity = st.number_input("주식·ETF·펀드", 0, 500_000, base.equity // MAN, step=100)
        isa = st.number_input("ISA", 0, 500_000, base.isa // MAN, step=100)
        pension_dc = st.number_input("퇴직·개인연금(DC/IRP)", 0, 500_000, base.pension_dc // MAN, step=100)
        real_estate = st.number_input("부동산", 0, 2_000_000, base.real_estate // MAN, step=1_000)
        debt = st.number_input("부채", 0, 500_000, base.debt // MAN, step=100)

    with st.sidebar.expander("현금흐름 · 연금 (만원)", expanded=True):
        monthly_expense = st.number_input(
            "월 생활비", 10, 2_000, base.monthly_expense // MAN, step=10
        )
        other_income = st.number_input(
            "월 기타소득 (임대·근로)", 0, 2_000, base.other_monthly_income // MAN, step=10
        )
        pension_monthly = st.number_input(
            "국민연금 예상 월 수령액", 0, 500, base.national_pension_monthly // MAN, step=5
        )
        pension_age = st.number_input(
            "국민연금 개시 연령", 55, 75, base.national_pension_start_age, step=1
        )
        # 가입월수는 119개월과 120개월이 '연금 0원'과 '평생 연금'을 가르는 값이다.
        # 근속연수와 같은 규약으로, 0이면 계산하지 않고 물어본다.
        pension_months = st.number_input(
            "국민연금 가입월수",
            0,
            600,
            base.national_pension_months,
            step=12,
            help="0이면 임의계속가입·추납을 계산하지 않습니다. 공단 앱 '가입내역 조회'에 나옵니다",
        )
        catchup_months = st.number_input(
            "추납 가능 개월수",
            0,
            600,
            base.pension_catchup_months,
            step=12,
            help="실직·휴직으로 납부예외였거나 전업주부로 적용제외였던 기간 (최대 119개월)",
        )

    # 이 연령대의 계획을 실제로 흔드는 것은 매달의 생활비가 아니라 한 번에 나가는 큰돈이다.
    with st.expander("예정된 큰 지출이 있으신가요? (자녀 결혼자금·의료비 등)"):
        event_label = st.text_input("지출 이름", value="자녀 결혼자금")
        event_col1, event_col2 = st.columns(2)
        with event_col1:
            event_year = st.number_input(
                "연도", 0, 2100, 0, step=1, help="0이면 반영하지 않습니다"
            )
        with event_col2:
            event_amount = st.number_input("금액(만원)", 0, 100_000, 0, step=100)

    life_events = (
        [LifeEvent(year=int(event_year), amount=int(event_amount) * MAN, label=event_label)]
        if event_year and event_amount
        else []
    )

    return UserProfile(
        life_events=life_events,
        birth_year=birth_year,
        birth_month=birth_month,
        retirement_year=retirement_year,
        retirement_month=retirement_month,
        years_employed=years_employed,
        last_monthly_salary=last_salary * MAN,
        risk_tolerance=RiskTolerance(risk),
        severance_pay=severance * MAN,
        cash_savings=cash * MAN,
        equity=equity * MAN,
        isa=isa * MAN,
        pension_dc=pension_dc * MAN,
        real_estate=real_estate * MAN,
        debt=debt * MAN,
        monthly_expense=monthly_expense * MAN,
        other_monthly_income=other_income * MAN,
        national_pension_monthly=pension_monthly * MAN,
        national_pension_start_age=pension_age,
        national_pension_months=pension_months,
        pension_catchup_months=catchup_months,
    )


# --------------------------------------------------------------------------- #
# 계산 (캐시)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="계산 중...")
def analyze(profile_json: str, n_paths: int):
    profile = UserProfile.model_validate_json(profile_json)
    return (
        simulate(profile),
        build_asset_map(profile),
        run_monte_carlo(profile, n_paths=n_paths),
        compute_risk_score(profile),
        build_scenarios(profile, n_paths=n_paths),
    )


@st.cache_resource
def _resolve_client():
    """LLM 클라이언트를 만든다. 실패하면 mock 으로 떨어진다.

    설정이 잘못됐다고 앱이 통째로 죽으면 안 된다. 발표 중에 키가 만료됐거나
    프로바이더 이름에 오타가 있어도 데모는 계속되어야 한다.
    """
    error: str | None = None
    try:
        client = get_client()
    except Exception as exc:  # 설정 오류·SDK 미설치·키 문제 전부
        client, error = MockClient(), f"{type(exc).__name__}: {exc}"

    if isinstance(client, MockClient):
        client.structured_handler = demo_handler(_dt.date.today().year)
        # 질문에 맞는 도구를 고르게 한다. 기본 동작은 언제나 첫 도구라, 화면에
        # trace 를 띄우는 지금은 에이전트가 질문을 못 알아듣는 것처럼 보인다.
        client.tool_handler = demo_tool_handler()
    return client, error


def llm_client():
    return _resolve_client()[0]


def llm_status() -> str:
    """사이드바에 띄울 LLM 상태. 이것도 실패로 앱을 죽이지 않는다."""
    _, error = _resolve_client()
    if error:
        return f"LLM: mock 으로 대체됨 ({error})"
    try:
        return describe_llm()
    except Exception as exc:
        return f"LLM: 설정 확인 필요 ({exc})"


# --------------------------------------------------------------------------- #
# 대화형 프로파일링
# --------------------------------------------------------------------------- #


def render_conversation() -> UserProfile | None:
    """대화로 프로파일을 채운다. 완료되면 UserProfile 을 돌려준다.

    프로파일링이 끝나도 대화는 끝나지 않는다. Supervisor 가 그 뒤의 질문을 상담
    에이전트로 넘기므로, 입력창은 계속 열어 둔다.
    """
    if "agent" not in st.session_state:
        agent = Supervisor(client=llm_client())
        agent.start()
        st.session_state.agent = agent

    agent: Supervisor = st.session_state.agent

    st.subheader("상담")
    answered, total = agent.progress
    st.progress(answered / total, text=f"{answered} / {total}개 확인")

    # 여기에는 프로파일링 대화만 그린다. 그 뒤의 상담은 결과 아래
    # (render_followup)에 붙는다 — 결과를 보고 묻는 질문이라 결과 옆에 있어야 한다.
    for message in agent.profiler.state.get("messages", []):
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.markdown(message["content"])

    if not agent.done:
        question_key = agent.profiler.state.get("pending_question")
        placeholder = "말씀해 주세요"
        if question_key:
            from agents.slots import QUESTION_BY_KEY

            hint = QUESTION_BY_KEY[question_key].hint
            if hint:
                placeholder = hint

        if answer := st.chat_input(placeholder):
            agent.send(answer)
            st.rerun()

        if agent.ready:
            st.info("필요한 정보는 모두 확인했습니다. 남은 질문을 건너뛰고 결과를 보셔도 됩니다.")
            if st.button("지금 결과 보기", type="primary"):
                st.session_state.skip_rest = True
                st.rerun()

        if not st.session_state.get("skip_rest"):
            return None

    try:
        return agent.build_profile()
    except ValueError as exc:
        st.error(str(exc))
        return None


EXAMPLE_QUESTIONS = (
    "생활비를 50만원 줄이면 어떻게 되나요?",
    "국민연금을 더 내는 게 나을까요?",
    "95세까지 유지하려면 어떻게 해야 하나요?",
)


def render_followup(agent: Supervisor) -> None:
    """결과를 보고 이어서 묻는 자리 (A6).

    지금까지 이 서비스는 한 번 계산하고 끝이었다. "생활비를 50만원 줄이면요?"는
    계산을 다시 돌려야 답할 수 있는 질문이고, 그 루프가 상담 에이전트다.
    여기가 그 에이전트에 닿는 유일한 입구다.

    시니어 사용자에게는 "무엇이든 물어보세요" 보다 **눌러볼 수 있는 문장 세 개**가
    낫다. 빈 입력창 앞에서 무엇을 물어야 할지 몰라 멈추는 것이 가장 흔한 이탈이다.
    """
    st.divider()
    st.subheader("여기까지 보시고, 궁금한 점을 물어보세요")
    st.caption(
        "화면에 없는 것도 물어보실 수 있습니다. 상담사가 계산을 다시 돌려서 답해 드립니다."
    )

    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, question in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(question, use_container_width=True):
            reply = agent.send(question)
            st.session_state.last_advice = reply.advice
            st.rerun()

    for message in agent.followups:
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.markdown(message["content"])

    render_trace(st.session_state.get("last_advice"))

    if asked := st.chat_input("궁금한 점을 물어보세요"):
        reply = agent.send(asked)
        st.session_state.last_advice = reply.advice
        st.rerun()


def render_trace(advice: AdvisorReply | None) -> None:
    """무엇을 어떤 값으로 계산했는지 (A4).

    이 서비스의 주장은 "숫자는 전부 계산에서 나온다"는 것이다. 그 주장을 검증
    가능하게 만드는 화면이다. 에이전트가 자율적으로 도구를 골라도 **어떤 도구를
    어떤 인자로 불러 무슨 값이 나왔는지**가 그대로 남는다.

    기본은 접어 둔다. 시니어 사용자에게 먼저 보여야 하는 것은 답이고,
    근거는 펼쳐볼 수 있으면 충분하다.
    """
    if advice is None:
        return

    if advice.blocked:
        # 지어낸 숫자를 걸러냈다는 사실은 답변보다 먼저 알려야 한다.
        st.warning(
            "계산 결과에 없는 숫자"
            f"({', '.join(advice.unverified_numbers)})가 있어 원래 답변을 "
            "내보내지 않았습니다. 계산된 값만 위에 정리했습니다."
        )

    count = len(advice.trace)
    label = (
        f"이 답을 만들려고 계산 {count}번을 돌렸습니다 — 눌러서 확인"
        if count
        else "계산 없이 답했습니다"
    )
    with st.expander(label):
        if not count:
            st.caption("도구를 부르지 않았습니다.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "순서": index + 1,
                        "계산": step.tool + ("" if step.ok else " (실패)"),
                        "넣은 값": (
                            " · ".join(f"{k} = {v}" for k, v in step.arguments.items())
                            or "인자 없이 — 현재 계획 그대로"
                        ),
                        "나온 값": step.summary,
                    }
                    for index, step in enumerate(advice.trace)
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "답변의 숫자는 전부 위 계산에서 나온 값입니다. "
            "계산에 없는 숫자는 답변에서 걸러집니다."
        )


SAMPLE_SCAM = """[특별안내] 고객님만 드리는 기회입니다.
정부 세법이 바뀌면서 ISA 비과세 혜택이 곧 폐지됩니다.
지금 갈아타지 않으시면 손해입니다.
원금 보장되면서 월 3% 확정 수익 나오는 상품이고요,
오늘까지만 선착순으로 받습니다.
자세한 내용은 텔레그램으로 연락 주세요. 가족한테는 비밀로 해주세요."""

RISK_STYLE = {
    RiskLevel.HIGH: ("🔴", st.error),
    RiskLevel.CAUTION: ("🟠", st.warning),
    RiskLevel.LOW: ("🟢", st.success),
}


def render_fraud_check() -> None:
    """받은 문자·카톡을 붙여넣어 위험 신호를 확인한다 (기능 ⑦)."""
    st.subheader("받은 연락 확인하기")
    st.markdown(
        "투자를 권유하는 문자나 카카오톡을 받으셨나요? "
        "**내용을 그대로 붙여넣으시면** 확인이 필요한 부분을 짚어드립니다."
    )

    if st.button("예시 문자로 확인해보기"):
        st.session_state.fraud_text = SAMPLE_SCAM

    text = st.text_area(
        "받으신 내용",
        value=st.session_state.get("fraud_text", ""),
        height=180,
        placeholder="받으신 문자나 메시지를 그대로 붙여넣어 주세요.",
    )

    if not text.strip():
        st.info(
            "붙여넣기가 어려우시면 내용을 직접 입력하셔도 됩니다. "
            "위의 '예시 문자로 확인해보기'를 눌러 어떻게 동작하는지 먼저 보셔도 좋습니다."
        )
        return

    profile = st.session_state.get("profile_for_fraud")
    result = analyze_message(text, profile=profile)

    icon, banner = RISK_STYLE[result.risk_level]
    banner(f"{icon} **{result.risk_level.value}** — {result.summary}")

    if result.signals:
        st.markdown("#### 확인이 필요한 부분")
        for signal in result.signals:
            mark = "🔴" if signal.severity is Severity.HIGH else "🟠"
            with st.expander(f"{mark} {signal.label}", expanded=signal.severity is Severity.HIGH):
                if signal.evidence:
                    st.caption("걸린 표현: " + ", ".join(f"`{e}`" for e in signal.evidence))
                st.markdown(signal.why)
                st.info(f"**이렇게 하세요** — {signal.advice}")

    if result.policy_checks:
        st.markdown("#### 언급된 제도의 실제 상태")
        st.caption(
            "뉴스는 '발표'와 '시행'을 구분하지 않습니다. "
            "발표된 개편안은 국회 논의에서 바뀌거나 무산될 수 있습니다."
        )
        for check in result.policy_checks:
            with st.expander(f"{check.badge} · {check.title}"):
                st.markdown(check.summary)
                st.warning(check.caution)
                st.caption(check.trust_note)

    if result.profile_notes:
        st.markdown("#### 고객님 자산과의 관계")
        for note in result.profile_notes:
            st.markdown(f"- {note}")

    st.caption(
        "⚠️ 이 확인은 사기 여부를 확정하지 않습니다. 확인이 필요한 신호를 알려드릴 뿐입니다. "
        "신호가 없어도 안전이 보장되지 않으며, 가입 전에는 반드시 거래하시는 금융회사에 "
        "직접 문의하세요. 금융감독원 파인(fine.fss.or.kr)에서 제도권 금융회사인지 조회하실 수 있습니다."
    )


def render_briefing(briefing: Briefing) -> None:
    st.subheader("상담사의 정리")
    st.markdown(f"### {briefing.headline}")
    st.markdown(briefing.situation)
    st.info(briefing.priority)
    st.markdown("**이렇게 해보세요**")
    for step in briefing.next_steps:
        st.markdown(f"- {step}")

    if briefing.is_verified:
        st.caption(
            "✅ 위 설명에 쓰인 숫자는 모두 계산 결과에서 나온 값입니다 "
            "(AI가 임의로 만든 숫자가 없는지 자동 검사했습니다)."
        )
    else:
        st.warning(
            "⚠️ 아래 숫자는 계산 결과에서 확인되지 않았습니다: "
            f"{', '.join(briefing.unverified_numbers)}. 참고만 해주세요."
        )


# --------------------------------------------------------------------------- #
# 화면 구성
# --------------------------------------------------------------------------- #


def render_headline(profile: UserProfile, sim, amap: AssetMap) -> None:
    st.subheader("한눈에 보기")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총자산", fmt_krw(profile.total_assets), f"금융자산 {fmt_krw(profile.financial_assets)}")
    c2.metric(
        "소득공백기",
        fmt_months(sim.income_gap_months),
        f"국민연금 {sim.pension_start_year}년 개시",
    )
    c3.metric(
        "생활비 충당 가능",
        fmt_years(sim.expense_coverage_years),
        "연금·기타소득 제외 기준",
        delta_color="off",
    )
    if sim.depletion_age is not None:
        c4.metric("금융자산 고갈", f"만 {sim.depletion_age}세", f"{sim.depletion_year}년", delta_color="inverse")
    else:
        c4.metric("금융자산 고갈", "없음", f"만 {sim.horizon_age}세까지 유지")

    if amap.gap_shortfall > 0:
        st.error(
            f"소득공백기 {fmt_months(amap.income_gap_months)} 동안 필요한 생활비 "
            f"{fmt_krw(amap.income_gap_need)} 중 **{fmt_krw(amap.gap_shortfall)}이 부족**합니다. "
            "지금 가장 중요한 것은 유동자산 확보입니다."
        )
    elif sim.depletion_age is not None:
        st.warning(
            f"소득공백기는 넘길 수 있지만, 현재 계획대로면 **만 {sim.depletion_age}세에 "
            "금융자산이 고갈**됩니다. 장기 소득 대비가 필요합니다."
        )
    else:
        st.success(f"현재 계획으로 만 {sim.horizon_age}세까지 금융자산이 유지됩니다.")


def render_asset_map(amap: AssetMap) -> None:
    st.subheader("나의 자산지도")
    st.caption("내 돈이 얼마 있는지가 아니라, 이 돈을 어떤 목적으로 얼마 동안 쓸 것인지를 봅니다.")

    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure()
        for bucket in amap.buckets:
            if bucket.amount <= 0:
                continue
            fig.add_bar(
                y=["자산"],
                x=[bucket.amount],
                name=bucket.label,
                orientation="h",
                marker_color=PALETTE.get(bucket.key),
                hovertemplate=f"<b>{bucket.label}</b><br>{fmt_krw(bucket.amount)}<br>{bucket.purpose}<extra></extra>",
            )
        fig.update_layout(
            barmode="stack",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.3),
            xaxis=dict(title=None, showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rows = []
        for bucket in amap.buckets:
            rows.append(
                {
                    "구분": bucket.label,
                    "금액": fmt_krw(bucket.amount),
                    "필요액": fmt_krw(bucket.required) if bucket.required is not None else "—",
                    "부족": fmt_krw(bucket.shortfall) if bucket.shortfall else "",
                    "사용 시기": bucket.horizon,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_balance_chart(sim) -> None:
    st.subheader("시간에 따른 자산 변화")
    df = pd.DataFrame([r.model_dump() for r in sim.rows])
    live = df[df["start_balance"] > 0]

    fig = go.Figure()
    fig.add_scatter(
        x=df["age"],
        y=df["start_balance"],
        mode="lines",
        name="금융자산 잔액",
        line=dict(width=3, color=PALETTE["survival"]),
        hovertemplate="만 %{x}세<br>%{y:,.0f}원<extra></extra>",
    )
    fig.add_vline(
        x=sim.pension_start_year - (sim.rows[0].year - sim.rows[0].age),
        line_dash="dash",
        line_color=PALETTE["long_term"],
        annotation_text="국민연금 개시",
    )
    if sim.depletion_age is not None:
        fig.add_vline(
            x=sim.depletion_age,
            line_dash="dot",
            line_color="#c0392b",
            annotation_text=f"자산 고갈 (만 {sim.depletion_age}세)",
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="나이",
        yaxis_title="금융자산 잔액(원)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("연도별 상세 내역 보기"):
        table = live[
            ["year", "age", "start_balance", "investment_return", "pension_income", "expense", "tax", "end_balance"]
        ].copy()
        table.columns = ["연도", "나이", "기초잔액", "투자수익", "국민연금", "생활비", "세금", "기말잔액"]
        for col in ["기초잔액", "투자수익", "국민연금", "생활비", "세금", "기말잔액"]:
            table[col] = table[col].map(fmt_krw)
        st.dataframe(table, hide_index=True, use_container_width=True)


def render_monte_carlo(mc) -> None:
    st.subheader("자산이 남아있을 확률")
    st.caption(
        f"수익률의 불확실성을 반영해 {mc.n_paths:,}가지 미래를 시뮬레이션한 결과입니다."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("10년 후 자산이 남아있을 확률", fmt_pct(mc.survival_after_years(10), precision=0))
    c2.metric("20년 후 자산이 남아있을 확률", fmt_pct(mc.survival_after_years(20), precision=0))
    c3.metric("80세 이전 고갈 가능성", fmt_pct(mc.prob_depleted_before_age(80), precision=0))

    ages = sorted(mc.survival_by_age)
    fig = go.Figure()
    fig.add_scatter(
        x=ages,
        y=[mc.survival_by_age[a] * 100 for a in ages],
        mode="lines",
        line=dict(width=3, color=PALETTE["income_gap"]),
        fill="tozeroy",
        hovertemplate="만 %{x}세<br>자산 잔존 확률 %{y:.0f}%<extra></extra>",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="나이",
        yaxis_title="자산이 남아있을 확률(%)",
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_score(score: RiskScore) -> None:
    st.subheader("은퇴 재무 안정도")
    left, right = st.columns([1, 3])

    with left:
        st.metric("종합", f"{score.total} / 100", score.status)
        st.caption(f"가장 취약한 항목: **{score.weakest.label}**")
        if score.cap is not None:
            st.caption(
                f"현재 상태 점수는 {score.raw_total}점이지만, {score.cap_reason} "
                f"계획을 유지할 수 없으므로 종합 점수를 {score.cap}점으로 제한했습니다."
            )

    with right:
        for component in score.components:
            cols = st.columns([2, 1, 6])
            cols[0].markdown(f"{component.icon} **{component.label}**")
            cols[1].markdown(f"{component.score:.0f}점")
            cols[2].caption(component.detail)


def render_scenarios(scenarios) -> None:
    st.subheader("대응 시나리오 비교")
    st.caption(
        "하나의 정답을 권하지 않습니다. 선택지별 예상 결과를 비교하고 직접 판단하세요. "
        "소득공백기 생활비로 반드시 필요한 금액은 어떤 선택에서도 현금으로 남겨둡니다."
    )

    cols = st.columns(len(scenarios))
    for col, scenario in zip(cols, scenarios):
        with col:
            st.markdown(f"### {scenario.label}")
            st.caption(scenario.description)
            for key, label in [("cash", "현금성"), ("bond", "채권·중위험"), ("equity", "주식·투자")]:
                st.markdown(
                    f"- {label} **{fmt_krw(scenario.amounts[key])}** "
                    f"({scenario.allocation.as_dict()[key] * 100:.0f}%)"
                )

    st.markdown("#### 비교표")
    st.dataframe(pd.DataFrame(comparison_table(scenarios)), hide_index=True, use_container_width=True)

    st.markdown("#### 만 95세 시점 예상 잔액")
    rows = [
        {
            "시나리오": s.label,
            "나쁜 시장(하위 10%)": fmt_krw(s.downside_terminal_balance),
            "중간": fmt_krw(s.median_terminal_balance),
            "좋은 시장(상위 10%)": fmt_krw(s.monte_carlo.terminal_balance_p90),
        }
        for s in scenarios
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_severance(profile: UserProfile) -> None:
    """퇴직금을 일시금으로 받을지 연금으로 받을지 (기능 ⑤)."""
    st.subheader("퇴직금, 일시금과 연금 중 어느 쪽이 나은가")
    st.caption(
        "퇴직 시점에 내려야 하는 가장 큰 결정입니다. "
        "연금으로 받으시면 퇴직소득세가 줄고, 내는 시점도 나뉩니다."
    )

    pension_years = st.select_slider(
        "연금 수령 기간", options=[5, 10, 15, 20], value=10
    )
    result: SeveranceComparison = compare_severance_options(
        profile, pension_years=pension_years
    )
    if not result.computable:
        # 근속연수를 모르면 세금이 전부 0으로 나온다. 표를 보여주면 '세금이 없다'는
        # 잘못된 인상을 준다.
        st.warning(result.headline)
        st.info(result.verdict)
        return

    st.success(result.headline)

    st.dataframe(
        pd.DataFrame(
            [
                {"항목": "세금", result.lump_sum.method: result.lump_sum.total_tax_label,
                 result.pension.method: result.pension.total_tax_label},
                {"항목": "실효세율", result.lump_sum.method: result.lump_sum.effective_rate_label,
                 result.pension.method: result.pension.effective_rate_label},
                {"항목": "언제 내나", result.lump_sum.method: result.lump_sum.when_paid,
                 result.pension.method: result.pension.when_paid},
                {"항목": "자산 고갈", result.lump_sum.method: result.lump_sum.depletion_label,
                 result.pension.method: result.pension.depletion_label},
                {"항목": "마지막 잔액", result.lump_sum.method: result.lump_sum.final_balance_label,
                 result.pension.method: result.pension.final_balance_label},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(f"**이렇게 하세요** — {result.verdict}")
    for note in result.notes:
        st.caption(note)


def render_health_insurance(profile: UserProfile) -> None:
    """퇴직 후 건강보험료 절벽."""
    st.subheader("퇴직하면 건강보험료가 얼마나 나오나")
    st.caption(
        "직장에 다니실 땐 회사가 절반을 냈습니다. 퇴직하면 그 자격이 없어집니다. "
        "직장 다니는 배우자·자녀의 피부양자가 되면 0원이지만, 소득이 기준을 넘으면 "
        "연 수백만원이 새로 생깁니다."
    )

    col1, col2 = st.columns(2)
    with col1:
        family = st.selectbox(
            "직장 다니는 배우자·자녀", ["모르겠어요", "있습니다", "없습니다"]
        )
    with col2:
        property_base = st.number_input(
            "재산세 과세표준(만원)",
            0,
            500_000,
            0,
            step=1_000,
            help="0이면 부동산 시가에서 추정합니다",
        )

    report: HealthInsuranceReport = analyze_health_insurance(
        profile,
        property_tax_base_override=property_base * MAN if property_base else None,
        has_employed_family=None if family == "모르겠어요" else family == "있습니다",
    )

    if report.cliff_year is None:
        st.success(report.headline)
    else:
        st.warning(report.headline)

    cols = st.columns(4)
    cols[0].metric("합산소득", report.counted_income_label)
    cols[1].metric("소득 한도까지", report.income_headroom_label)
    cols[2].metric("이자·배당 계단까지", report.financial_headroom_label)
    cols[3].metric(
        "탈락 시점",
        "없음" if report.cliff_year is None else f"{report.cliff_year}년",
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "가입 방법": path.method if path.available else f"{path.method} (해당 없음)",
                    "매달": path.monthly_label,
                    "1년": path.annual_label,
                    "근거": path.basis.replace("**", ""),
                }
                for path in (report.dependent, report.local, report.voluntary)
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    if report.depletion_advanced_years > 0:
        st.warning(
            f"보험료를 넣으면 금융자산이 바닥나는 시점이 "
            f"{report.depletion_advanced_years}년 앞당겨집니다 "
            f"(평생 {report.total_premiums_label})."
        )

    st.info(f"**이렇게 하세요** — {report.verdict}")
    for note in report.notes:
        st.caption(note.replace("**", ""))


def render_national_pension(profile: UserProfile) -> None:
    """국민연금을 더 낼까 — 임의계속가입과 추납.

    바로 위의 건강보험료 화면과 **붙여 놓는다.** 연금을 늘리면 피부양자에서 더 빨리
    탈락한다는 것이 이 항목의 핵심이라, 두 화면이 떨어져 있으면 상충이 보이지 않는다.
    """
    st.subheader("국민연금을 더 내는 게 이득일까")
    st.caption(
        "60세가 지나도 65세까지 계속 낼 수 있고(임의계속가입), 실직·휴직으로 못 냈던 "
        "기간의 보험료를 나중에 낼 수도 있습니다(추납). 다만 연금이 늘면 바로 위의 "
        "건강보험 피부양자 기준을 더 빨리 넘습니다. 두 가지를 함께 계산했습니다."
    )

    report: NationalPensionReport = analyze_national_pension(profile)

    if not report.computable:
        st.warning(report.headline)
        st.info(report.verdict)
        for note in report.notes:
            st.caption(note.replace("**", ""))
        return

    if not report.qualifies_now:
        st.error(report.headline.replace("**", ""))
    elif report.best_key == "none":
        st.success(report.headline.replace("**", ""))
    else:
        st.warning(report.headline.replace("**", ""))

    cols = st.columns(4)
    cols[0].metric("가입기간", report.contributed_label)
    cols[1].metric(
        "수급자격",
        "있음" if report.qualifies_now else f"{report.months_to_qualify}개월 부족",
    )
    cols[2].metric("지금 월 연금", report.current.monthly_pension_label)
    cols[3].metric("가장 나은 선택", report.best_label)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "수단": option.label if option.available else f"{option.label} (해당 없음)",
                    "더 내는 기간": f"{option.months_added}개월" if option.available else "—",
                    "내는 돈": option.cost_label if option.available else "—",
                    "월 연금 증가": option.monthly_gain_label if option.available else "—",
                    "평생 더 받음": option.lifetime_gain_label if option.available else "—",
                    "추가 건보료": option.extra_premium_label if option.available else "—",
                    "빼고 남는 것": option.net_gain_label if option.available else "—",
                    "본전": option.breakeven_label or "—",
                }
                for option in report.options
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    for option in report.options:
        if not option.available and option.unavailable_reason:
            st.caption(f"{option.label} — {option.unavailable_reason}")

    # 상충을 문장으로 한 번 더 짚는다. 표만 보면 '건보료' 칸이 그냥 비용으로 읽힌다.
    shifted = [o for o in report.options if o.available and o.cliff_shift_years > 0]
    if shifted:
        worst = max(shifted, key=lambda o: o.cliff_shift_years)
        st.warning(
            f"{worst.label}을 선택하시면 건강보험 피부양자 탈락이 "
            f"{worst.cliff_shift_years}년 앞당겨집니다 "
            f"({report.current.cliff_year}년 → {worst.cliff_year}년)."
        )

    st.info(f"**이렇게 하세요** — {report.verdict}")
    for note in report.notes:
        st.caption(note.replace("**", ""))


def render_prescriptions(profile: UserProfile) -> None:
    """진단 다음에 오는 것 — 그래서 무엇을 하면 되는가."""
    st.subheader("그래서 무엇을 하면 되나")

    target_age = st.select_slider(
        "목표 나이", options=[85, 90, 95, 100], value=95,
        help="이 나이까지 금융자산이 유지되는 것을 목표로 계산합니다.",
    )
    plan: PrescriptionSet = prescribe(profile, target_age=target_age)

    (st.success if plan.already_safe else st.warning)(plan.summary)

    for option in plan.options:
        mark = "✅" if option.feasible else ("🟠" if option.improves else "⬜")
        with st.expander(f"{mark} {option.label}", expanded=option.feasible):
            st.markdown(f"**{option.headline}**")
            if option.change_label:
                st.info(f"**해야 할 일** — {option.change_label}")
            if option.gained_years > 0:
                st.caption(f"자산이 버티는 기간이 {option.gained_years}년 늘어납니다.")
            if option.detail:
                st.caption(option.detail)
            if option.caution:
                st.warning(f"⚠️ {option.caution}")


def render_sensitivity(profile: UserProfile) -> None:
    """가정이 틀렸을 때 결과가 얼마나 흔들리는지 — 우리 숫자의 자기검증."""
    st.subheader("이 결과는 얼마나 믿을 수 있나")
    st.caption(
        "여기 나온 숫자는 모두 가정 위에 있습니다. "
        "가정이 틀렸을 때 결과가 얼마나 달라지는지 함께 봅니다."
    )

    report: SensitivityReport = analyze_sensitivity(profile)
    st.warning(report.summary)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "가정": case.label,
                    "지금 값": case.baseline_label,
                    "낮을 때": f"{case.low_label} → {_age_text(case.low_depletion_age, report.horizon_age)}",
                    "높을 때": f"{case.high_label} → {_age_text(case.high_depletion_age, report.horizon_age)}",
                    "흔들림": f"{case.swing_years}년",
                }
                for case in report.cases
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    for case in report.cases[:2]:
        st.caption(case.detail)
    st.caption(report.caveat)


def _age_text(depletion_age, horizon: int) -> str:
    return f"{horizon}세까지 유지" if depletion_age is None else f"만 {depletion_age}세 고갈"


def render_policy_impact(profile: UserProfile) -> None:
    """발표된 제도 변화가 이 사람에게 실제로 얼마인지 (기능 ④·⑧)."""
    impact: PolicyImpact = analyze_policy_impact(profile)
    fact = impact.policy

    st.subheader("제도가 바뀌면 나는 얼마나 달라지나")
    st.markdown(f"**{fact.badge} · {fact.title}**")

    # 확정되지 않았다는 사실을 숫자보다 먼저 보여준다.
    if not fact.is_confirmed:
        st.warning(f"⚠️ {fact.notice}")

    (st.warning if impact.material else st.success)(impact.headline)
    st.markdown(impact.reason)

    if impact.comparison:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "항목": row.label,
                        impact.current.label: row.current,
                        impact.proposed.label: row.proposed,
                        "차이": row.delta,
                    }
                    for row in impact.comparison
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.info(f"**이렇게 하세요** — {impact.verdict}")
    for note in impact.notes:
        st.caption(note)
    st.caption(fact.trust_note)


def render_family_report(profile: UserProfile) -> None:
    """가족에게 보여줄 한 장 (기능 5).

    시니어의 금융 결정은 혼자 내려지지 않는다. 지금까지 만든 것은 전부 본인
    화면이었고, 가족은 그것을 보지 못한다. 여기가 나머지 기능의 결과를 담는
    그릇이다.
    """
    st.subheader("가족에게 보여줄 한 장")
    st.caption(
        "자녀나 배우자와 함께 보시라고 정리했습니다. "
        "기한이 있는 것을 맨 위에 두었고, 하지 않으셔도 되는 것도 함께 적었습니다."
    )

    report: FamilyReport = build_family_report(profile)

    st.markdown(f"**{report.subject}**")
    if report.now:
        st.warning(report.headline)
    else:
        st.success(report.headline)
    st.caption(report.summary)

    def block(title: str, items, tone) -> None:
        if not items:
            return
        st.markdown(f"**{title}**")
        for item in items:
            with st.container(border=True):
                st.markdown(f"**{item.title}** — {item.value}" if item.value else f"**{item.title}**")
                if item.detail:
                    st.caption(item.detail)
                if item.deadline:
                    tone(f"⏰ {item.deadline}")

    block("기한이 있는 것 — 놓치면 되돌릴 수 없습니다", report.now, st.warning)
    block("천천히 보셔도 되는 것", report.later, st.info)
    block("하지 않으셔도 되는 것", report.not_needed, st.info)

    if report.watch_outs:
        st.markdown("**가족이 함께 봐 주실 것**")
        for note in report.watch_outs:
            st.caption(f"· {note}")

    for note in report.assumptions:
        st.caption(f"※ {note}")

    # 실제 공유 경로는 카톡이다. 화면을 캡처해 보내면 숫자만 남고 단서가 사라진다.
    with st.expander("문자·카톡으로 보낼 수 있게 글로 보기"):
        st.code(report.as_text, language=None)


def main() -> None:
    st.title("🧭 내 자산 AI 네비게이터")
    st.markdown("**퇴직 후, 내 자산 어떻게 관리해야 할까요?**")

    mode = st.sidebar.radio(
        "무엇을 하시겠어요?",
        ["대화로 알아보기", "직접 입력", "받은 연락 확인하기"],
        help="대화가 막히면 언제든 직접 입력으로 바꾸실 수 있습니다.",
    )
    n_paths = st.sidebar.select_slider(
        "시뮬레이션 횟수", options=[500, 1_000, 3_000, 5_000], value=3_000
    )
    status = llm_status()
    st.sidebar.caption(status)
    if "대체됨" in status or "확인 필요" in status:
        st.sidebar.warning("LLM 설정에 문제가 있어 기본 응답으로 동작합니다. .env 를 확인해 주세요.")

    if mode == "받은 연락 확인하기":
        render_fraud_check()
        return

    if mode == "대화로 알아보기":
        profile = render_conversation()
        if profile is None:
            return
        st.divider()
    else:
        profile = profile_form()

    st.session_state.profile_for_fraud = profile
    sim, amap, mc, score, scenarios = analyze(profile.model_dump_json(), n_paths)

    if mode == "대화로 알아보기":
        briefing = explain(
            profile, sim, amap, score=score, monte_carlo=mc, client=llm_client()
        )
        render_briefing(briefing)
        if profile.assumed_fields:
            st.caption(
                f"말씀하지 않으신 {len(profile.assumed_fields)}개 항목은 "
                "기본값으로 가정했습니다. 왼쪽에서 '직접 입력'으로 바꾸면 수정하실 수 있습니다."
            )
        st.divider()

    if profile.life_events:
        st.info(
            "예정된 큰 지출이 반영되었습니다: "
            + ", ".join(
                f"{e.year}년 {e.label} {fmt_krw(e.amount)}" for e in profile.life_events
            )
        )

    render_headline(profile, sim, amap)
    st.divider()
    render_asset_map(amap)
    st.divider()
    render_balance_chart(sim)
    st.divider()
    render_monte_carlo(mc)
    st.divider()
    render_risk_score(score)
    st.divider()
    render_scenarios(scenarios)
    st.divider()
    render_severance(profile)
    st.divider()
    render_health_insurance(profile)
    render_national_pension(profile)
    st.divider()
    render_prescriptions(profile)
    st.divider()
    render_sensitivity(profile)
    st.divider()
    render_policy_impact(profile)
    st.divider()
    render_family_report(profile)

    # 결과를 다 본 뒤에 온다. 여기서부터는 사용자가 묻고 에이전트가 계산한다.
    if mode == "대화로 알아보기" and isinstance(st.session_state.get("agent"), Supervisor):
        render_followup(st.session_state.agent)

    st.divider()
    st.caption(
        "⚠️ **참고용 안내입니다.** 세금과 건강보험료는 실효세율 근사치이며, "
        "실제 부담액은 개인의 소득·재산 상황에 따라 달라집니다. "
        "투자 수익률은 가정값이며 미래 수익을 보장하지 않습니다. "
        "이 서비스는 특정 금융상품의 가입을 권유하지 않습니다."
    )
    st.caption("가정값의 근거는 `data/assumptions.yaml` 에 출처와 함께 기록되어 있습니다.")


if __name__ == "__main__":
    main()
