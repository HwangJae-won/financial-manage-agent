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
from agents.mocks import demo_handler
from agents.profiling import ProfilingAgent
from core.asset_map import AssetMap, build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_pct, fmt_years
from core.models import RiskTolerance, UserProfile
from core.montecarlo import run_monte_carlo
from core.policy import PolicyImpact, analyze_policy_impact
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

    return UserProfile(
        birth_year=birth_year,
        birth_month=birth_month,
        retirement_year=retirement_year,
        retirement_month=retirement_month,
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
    """대화로 프로파일을 채운다. 완료되면 UserProfile 을 돌려준다."""
    if "agent" not in st.session_state:
        agent = ProfilingAgent(client=llm_client())
        agent.start()
        st.session_state.agent = agent

    agent: ProfilingAgent = st.session_state.agent

    st.subheader("상담")
    answered, total = agent.progress
    st.progress(answered / total, text=f"{answered} / {total}개 확인")

    for message in agent.state.get("messages", []):
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.markdown(message["content"])

    if not agent.done:
        question_key = agent.state.get("pending_question")
        placeholder = "말씀해 주세요"
        if question_key:
            from agents.slots import QUESTION_BY_KEY

            hint = QUESTION_BY_KEY[question_key].hint
            if hint:
                placeholder = hint

        if answer := st.chat_input(placeholder):
            agent.respond(answer)
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
    render_policy_impact(profile)

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
