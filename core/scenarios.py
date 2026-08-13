"""대응 시나리오 비교 (기능 ⑥).

하나의 정답을 강요하지 않고 선택지와 예상 결과를 나란히 놓는다.

핵심 설계: 세 시나리오의 **현금 비중에 바닥(floor)을 둔다.** 생존자산과
소득공백기 자산으로 반드시 필요한 금액은 어떤 성향이든 위험자산에 넣을 수 없다.
그래서 소득공백기가 길고 자산이 빠듯한 사용자에게는 세 시나리오가 자연스럽게
수렴하는데, 이건 버그가 아니라 "지금은 위험을 질 여력이 없다"는 진단이다.

비교 지표는 임의로 만든 점수가 아니라 전부 몬테카를로에서 나온 실제 확률이다.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field

from core.asset_map import build_asset_map
from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.models import Allocation, SimulationResult, UserProfile
from core.montecarlo import DEFAULT_PATHS, DEFAULT_SEED, MonteCarloResult, run_monte_carlo

# 시나리오별 목표 주식 비중과 최소 현금 비중
_TEMPLATES: list[tuple[str, str, str, float, float]] = [
    (
        "stable",
        "안정형",
        "생활비를 최우선으로 확보합니다. 시장이 나빠도 계획이 흔들리지 않지만, "
        "물가상승을 이기기는 어렵습니다.",
        0.50,  # 최소 현금 비중
        0.10,  # 목표 주식 비중
    ),
    (
        "balanced",
        "균형형",
        "단기 생활비를 확보하면서 남는 자산은 일부 투자로 운용합니다. "
        "대부분의 은퇴자에게 무난한 출발점입니다.",
        0.35,
        0.25,
    ),
    (
        "growth",
        "성장형",
        "장기 물가상승에 대비해 투자 비중을 높입니다. 자산이 오래 버틸 가능성도 "
        "커지지만, 나쁜 시장을 만나면 계획이 크게 흔들릴 수 있습니다.",
        0.25,
        0.45,
    ),
]


def volatility_label(volatility: float) -> str:
    """포트폴리오 변동성을 사용자가 읽을 수 있는 말로 바꾼다."""
    if volatility < 0.05:
        return "낮음"
    if volatility < 0.10:
        return "중간"
    return "높음"


class Scenario(BaseModel):
    """하나의 대응 시나리오."""

    key: str
    label: str
    description: str

    allocation: Allocation
    amounts: dict[str, int] = Field(description="자산군별 금액(원)")

    simulation: SimulationResult
    monte_carlo: MonteCarloResult

    volatility: float

    @property
    def volatility_label(self) -> str:
        return volatility_label(self.volatility)

    @property
    def prob_survive_income_gap(self) -> float:
        """소득공백기 안정성 — 국민연금 개시까지 자산이 버틸 확률."""
        return self.monte_carlo.prob_survive_income_gap

    @property
    def prob_deplete_before_85(self) -> float:
        """85세 이전 자산 고갈 가능성."""
        return self.monte_carlo.prob_depleted_before_age(85)

    @property
    def median_terminal_balance(self) -> int:
        """장기 성장성 — 시뮬레이션 종료 시점 잔액의 중앙값."""
        return self.monte_carlo.terminal_balance_median

    @property
    def downside_terminal_balance(self) -> int:
        """나쁜 시장(하위 10%)에서의 종료 시점 잔액."""
        return self.monte_carlo.terminal_balance_p10


def _floor6(value: float) -> float:
    """소수점 6자리에서 내림. 반올림이 현금 바닥을 깎아먹지 않게 한다."""
    return math.floor(value * 1_000_000) / 1_000_000


def _build_allocation(min_cash: float, target_equity: float, cash_floor: float) -> Allocation:
    """현금 바닥을 지키면서 목표 주식 비중에 최대한 근접시킨다.

    현금 바닥은 '소득공백기 생활비'라는 안전 제약이므로 반올림으로도 침범해선 안 된다.
    그래서 위험자산 쪽을 내림 처리하고 남는 우수리를 전부 현금에 몰아준다.
    """
    cash = min(1.0, max(min_cash, cash_floor))
    remaining = 1.0 - cash
    equity = _floor6(min(target_equity, remaining))
    bond = _floor6(max(0.0, remaining - equity))
    return Allocation(cash=1.0 - equity - bond, bond=bond, equity=equity)


def cash_floor_ratio(profile: UserProfile, assumptions: Optional[Assumptions] = None) -> float:
    """어떤 성향이든 위험자산에 넣으면 안 되는 금융자산 비율.

    생존자산 + 소득공백기 생활비를 금융자산 총액으로 나눈 값.
    """
    assumptions = assumptions or load_assumptions()
    amap = build_asset_map(profile, assumptions)
    if profile.financial_assets <= 0:
        return 1.0
    needed = amap.survival_need + max(0, amap.income_gap_need - amap.survival_need)
    return min(1.0, needed / profile.financial_assets)


def build_scenarios(
    profile: UserProfile,
    *,
    assumptions: Optional[Assumptions] = None,
    n_paths: int = DEFAULT_PATHS,
    seed: int = DEFAULT_SEED,
) -> list[Scenario]:
    """안정형 / 균형형 / 성장형 시나리오를 만들고 각각 시뮬레이션한다."""
    assumptions = assumptions or load_assumptions()
    floor = cash_floor_ratio(profile, assumptions)
    financial = profile.financial_assets

    scenarios: list[Scenario] = []
    for key, label, description, min_cash, target_equity in _TEMPLATES:
        allocation = _build_allocation(min_cash, target_equity, floor)
        weights = allocation.as_dict()

        scenarios.append(
            Scenario(
                key=key,
                label=label,
                description=description,
                allocation=allocation,
                amounts={k: int(round(financial * w)) for k, w in weights.items()},
                simulation=simulate(profile, allocation=allocation, assumptions=assumptions),
                monte_carlo=run_monte_carlo(
                    profile,
                    allocation=allocation,
                    assumptions=assumptions,
                    n_paths=n_paths,
                    seed=seed,
                ),
                volatility=assumptions.volatility(weights),
            )
        )

    return scenarios


def comparison_table(scenarios: list[Scenario]) -> list[dict[str, object]]:
    """기획서의 시나리오 비교표를 그대로 만든다. UI와 LLM 설명이 같은 표를 쓴다."""
    rows: list[dict[str, object]] = [
        {
            "지표": "소득공백기 안정성",
            **{s.label: f"{s.prob_survive_income_gap * 100:.0f}%" for s in scenarios},
        },
        {
            "지표": "85세 이전 자산 고갈 가능성",
            **{s.label: f"{s.prob_deplete_before_85 * 100:.0f}%" for s in scenarios},
        },
        {
            "지표": "변동성",
            **{s.label: s.volatility_label for s in scenarios},
        },
    ]
    return rows
