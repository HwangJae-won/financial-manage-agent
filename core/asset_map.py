"""은퇴 자산지도 (기능 ②).

"총자산 7.2억"이 아니라 "이 돈은 어떤 목적으로 얼마 동안 쓰는 돈인가"를 보여준다.
목적별 **필요액**을 먼저 계산하고, 보유 자산을 우선순위대로 흘려 담는(waterfall)
방식이라 "소득공백기 자산이 4천만 원 부족합니다" 같은 진단이 바로 나온다.

우선순위: 생존자산 → 소득공백기 자산 → 장기자산 → 성장자산 → 실물자산
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.models import UserProfile


class Bucket(BaseModel):
    """자산지도의 한 칸."""

    key: str
    label: str
    purpose: str = Field(description="이 돈의 용도 — 화면과 LLM 설명에 그대로 쓰인다")
    horizon: str = Field(description="사용 시기")
    amount: int = Field(description="배분된 금액(원)")
    required: Optional[int] = Field(
        default=None, description="필요액(원). None이면 필요액 개념이 없는 버킷"
    )

    @property
    def shortfall(self) -> int:
        """필요액 대비 부족액. 필요액 개념이 없으면 0."""
        if self.required is None:
            return 0
        return max(0, self.required - self.amount)


class AssetMap(BaseModel):
    """자산지도 전체."""

    buckets: list[Bucket]

    total_assets: int
    financial_assets: int
    liquid_assets: int

    survival_need: int = Field(description="비상 현금 필요액 (월 생활비 x survival_months)")
    income_gap_need: int = Field(description="소득공백기 전체 생활비 부족액")
    income_gap_months: int

    net_monthly_need: int = Field(description="월 생활비에서 기타소득을 뺀 순 필요액")

    @property
    def gap_shortfall(self) -> int:
        """소득공백기를 넘기기 위해 유동자산이 부족한 금액."""
        return max(0, self.income_gap_need - self.liquid_assets)

    @property
    def gap_coverage_ratio(self) -> float:
        """소득공백기 필요액 대비 유동자산 충당 비율. 필요액이 0이면 1.0."""
        if self.income_gap_need <= 0:
            return 1.0
        return min(1.0, self.liquid_assets / self.income_gap_need)

    def by_key(self, key: str) -> Bucket:
        for bucket in self.buckets:
            if bucket.key == key:
                return bucket
        raise KeyError(key)


def build_asset_map(
    profile: UserProfile, assumptions: Optional[Assumptions] = None
) -> AssetMap:
    """프로파일에서 자산지도를 만든다."""
    assumptions = assumptions or load_assumptions()

    gap_months = profile.income_gap_months
    net_monthly_need = max(0, profile.monthly_expense - profile.other_monthly_income)

    survival_months = assumptions.asset_map.survival_months
    survival_need = profile.monthly_expense * survival_months
    income_gap_need = net_monthly_need * gap_months

    # --- waterfall: 유동자산을 우선순위대로 흘려 담는다 ---
    pool = profile.liquid_assets

    survival_amount = min(pool, survival_need)
    pool -= survival_amount

    # 생존자산이 소득공백기 필요액의 앞부분을 이미 덮으므로 중복 계상하지 않는다.
    gap_remaining_need = max(0, income_gap_need - survival_need)
    gap_amount = min(pool, gap_remaining_need)
    pool -= gap_amount

    long_term_amount = pool + profile.pension_dc

    buckets = [
        Bucket(
            key="survival",
            label="생존자산",
            purpose="당장의 생활비. 어떤 상황에서도 손대지 않고 남겨두는 비상 현금",
            horizon=f"퇴직 직후 {survival_months}개월",
            amount=survival_amount,
            required=survival_need,
        ),
        Bucket(
            key="income_gap",
            label="소득공백기 자산",
            purpose="국민연금을 받기 전까지 생활비로 쓸 돈. 변동성이 낮고 언제든 찾을 수 있어야 한다",
            horizon=f"퇴직 후 ~ 국민연금 개시({profile.pension_start_year}년)",
            amount=gap_amount,
            required=gap_remaining_need,
        ),
        Bucket(
            key="long_term",
            label="장기자산",
            purpose="연금을 받기 시작한 뒤 부족한 생활비를 메우는 돈",
            horizon=f"{profile.pension_start_year}년 이후",
            amount=long_term_amount,
            required=None,
        ),
        Bucket(
            key="growth",
            label="성장자산",
            purpose="당장 쓰지 않는 돈. 물가상승을 이기기 위해 투자로 운용",
            horizon="10년 이상",
            amount=profile.equity,
            required=None,
        ),
        Bucket(
            key="real_asset",
            label="실물자산",
            purpose="거주 또는 임대 중인 부동산. 즉시 현금화가 어렵다",
            horizon="처분 또는 상속 시점",
            amount=profile.real_estate,
            required=None,
        ),
    ]

    return AssetMap(
        buckets=buckets,
        total_assets=profile.total_assets,
        financial_assets=profile.financial_assets,
        liquid_assets=profile.liquid_assets,
        survival_need=survival_need,
        income_gap_need=income_gap_need,
        income_gap_months=gap_months,
        net_monthly_need=net_monthly_need,
    )
