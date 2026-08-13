"""은퇴 자산 네비게이터 — Streamlit 데모 (W2).

아직 대화형 프로파일링(W3)은 붙지 않았다. 폼으로 직접 입력받아
core/ 의 계산 결과를 그대로 보여주는 것이 이 단계의 목표다.

폼 입력은 W3 이후에도 남겨둔다 — 데모 중 대화가 꼬였을 때의 안전망이다.

실행:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` 은 스크립트 디렉터리를 sys.path 에 넣으므로 저장소 루트를 직접 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.asset_map import AssetMap, build_asset_map
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_months, fmt_pct, fmt_years
from core.models import RiskTolerance, UserProfile
from core.montecarlo import run_monte_carlo
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


def main() -> None:
    st.title("🧭 내 자산 AI 네비게이터")
    st.markdown("**퇴직 후, 내 자산 어떻게 관리해야 할까요?**")

    profile = profile_form()

    n_paths = st.sidebar.select_slider(
        "시뮬레이션 횟수", options=[500, 1_000, 3_000, 5_000], value=3_000
    )

    sim, amap, mc, score, scenarios = analyze(profile.model_dump_json(), n_paths)

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
    st.caption(
        "⚠️ **참고용 안내입니다.** 세금과 건강보험료는 실효세율 근사치이며, "
        "실제 부담액은 개인의 소득·재산 상황에 따라 달라집니다. "
        "투자 수익률은 가정값이며 미래 수익을 보장하지 않습니다. "
        "이 서비스는 특정 금융상품의 가입을 권유하지 않습니다."
    )
    st.caption("가정값의 근거는 `data/assumptions.yaml` 에 출처와 함께 기록되어 있습니다.")


if __name__ == "__main__":
    main()
