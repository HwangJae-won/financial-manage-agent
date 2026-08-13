"""데이터 모델 — core 패키지 전체의 계약.

설계 원칙: core 하위 모듈은 LLM이나 UI를 절대 import 하지 않는다.
여기 정의된 타입만이 agents/ 및 app/ 과 주고받는 유일한 인터페이스다.

금액 단위는 전부 원(KRW) 정수다. 표시용 변환은 core.formatting 참조.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class RiskTolerance(str, Enum):
    """투자성향."""

    CONSERVATIVE = "안정형"
    MODERATE = "중립형"
    AGGRESSIVE = "공격형"


# ISA 계좌 안에서 주식성 자산이 차지한다고 보는 비중. 투자성향에 따라 달라진다.
# 자산군 배분(current_allocation)과 ISA 세제 영향 계산(core/policy.py)이 **같은 값**을
# 써야 하므로 모듈 상수로 뺐다. 한쪽만 바뀌면 두 계산이 조용히 어긋난다.
ISA_EQUITY_SHARE: dict["RiskTolerance", float] = {
    RiskTolerance.CONSERVATIVE: 0.2,
    RiskTolerance.MODERATE: 0.5,
    RiskTolerance.AGGRESSIVE: 0.8,
}


class AssetClass(str, Enum):
    """시뮬레이션에서 사용하는 자산군. assumptions.yaml 의 키와 일치해야 한다."""

    CASH = "cash"
    BOND = "bond"
    EQUITY = "equity"


class Allocation(BaseModel):
    """금융자산의 자산군별 배분 비중. 합이 1.0 이어야 한다."""

    cash: float = Field(ge=0.0, le=1.0)
    bond: float = Field(ge=0.0, le=1.0)
    equity: float = Field(ge=0.0, le=1.0)

    def model_post_init(self, __context) -> None:
        total = self.cash + self.bond + self.equity
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"배분 비중의 합이 1.0이어야 합니다 (현재 {total:.4f})")

    def as_dict(self) -> dict[str, float]:
        return {"cash": self.cash, "bond": self.bond, "equity": self.equity}


class LifeEvent(BaseModel):
    """예정된 일회성 큰 지출.

    이 연령대의 계획을 실제로 무너뜨리는 것은 매달의 생활비가 아니라 **한 번에 나가는
    큰돈**이다. 자녀 결혼자금, 목돈 의료비, 주택 수리. 매년 일정한 생활비만 가정하면
    "그래서 결혼자금 5천만원은 어디에 반영됐나요"라는 질문에 답할 수 없다.

    금액은 **퇴직 시점 기준**으로 본다. 월 생활비와 같은 규약이라, 발생 연도까지
    물가상승률만큼 커진다(보수적).
    """

    year: int = Field(description="지출이 발생하는 연도")
    amount: int = Field(gt=0, description="퇴직 시점 기준 금액(원)")
    label: str = Field(default="큰 지출", description="자녀 결혼자금 등 화면에 쓸 이름")


class UserProfile(BaseModel):
    """AI 금융 프로파일링(기능 ①)의 산출물이자 모든 계산의 입력."""

    # --- 인적사항 ---
    birth_year: int = Field(description="출생연도")
    birth_month: int = Field(default=1, ge=1, le=12, description="출생월")
    dependents: int = Field(default=0, ge=0, description="부양가족 수")
    years_employed: int = Field(default=0, ge=0, description="재직 기간(년)")
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE

    # --- 퇴직 시점 ---
    retirement_year: int = Field(description="퇴직(예정) 연도")
    retirement_month: int = Field(default=12, ge=1, le=12, description="퇴직(예정) 월")

    # --- 자산 (원) ---
    severance_pay: int = Field(default=0, ge=0, description="퇴직금·명예퇴직금")
    cash_savings: int = Field(default=0, ge=0, description="예금·적금 등 현금성 자산")
    equity: int = Field(default=0, ge=0, description="주식·ETF·펀드")
    isa: int = Field(default=0, ge=0, description="ISA 계좌 평가액")
    pension_dc: int = Field(default=0, ge=0, description="DC/IRP 등 사적연금 적립금")
    real_estate: int = Field(default=0, ge=0, description="부동산 평가액")
    debt: int = Field(default=0, ge=0, description="총 부채")

    # --- 현금흐름 (원/월) ---
    monthly_expense: int = Field(gt=0, description="월 생활비")
    other_monthly_income: int = Field(default=0, ge=0, description="임대·근로 등 기타 월소득")

    # --- 국민연금 ---
    national_pension_monthly: int = Field(default=0, ge=0, description="국민연금 예상 월 수령액")
    national_pension_start_age: int = Field(default=63, ge=55, le=75, description="국민연금 수령 개시 연령")

    # --- 예정된 큰 지출 ---
    life_events: list[LifeEvent] = Field(
        default_factory=list,
        description="자녀 결혼자금·의료비 등 일회성 큰 지출. 비어 있으면 기존 계산과 동일하다.",
    )

    # --- 메타 ---
    assumed_fields: list[str] = Field(
        default_factory=list,
        description="사용자가 직접 답하지 않아 기본값으로 가정한 필드명. UI에서 '가정한 값'으로 표시한다.",
    )

    # ------------------------------------------------------------------ #
    # 파생값
    # ------------------------------------------------------------------ #

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retirement_age(self) -> int:
        """퇴직 시점의 만 나이."""
        return self.retirement_year - self.birth_year

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pension_start_year(self) -> int:
        """국민연금 수령이 시작되는 연도."""
        return self.birth_year + self.national_pension_start_age

    @computed_field  # type: ignore[prop-decorator]
    @property
    def income_gap_months(self) -> int:
        """소득공백기 개월 수 — 퇴직 시점부터 국민연금 개시까지."""
        retire_abs = self.retirement_year * 12 + self.retirement_month
        pension_abs = self.pension_start_year * 12 + self.birth_month
        return max(0, pension_abs - retire_abs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def liquid_assets(self) -> int:
        """퇴직 직후 즉시 운용 가능한 금융자산 (퇴직금 포함, 부동산 제외)."""
        return self.severance_pay + self.cash_savings + self.isa

    @computed_field  # type: ignore[prop-decorator]
    @property
    def financial_assets(self) -> int:
        """전체 금융자산 (부동산 제외)."""
        return self.liquid_assets + self.equity + self.pension_dc

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_assets(self) -> int:
        """총자산 (부동산 포함, 부채 차감 전)."""
        return self.financial_assets + self.real_estate

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_worth(self) -> int:
        """순자산 (부채 차감 후)."""
        return self.total_assets - self.debt

    @computed_field  # type: ignore[prop-decorator]
    @property
    def annual_expense(self) -> int:
        """연간 생활비 (퇴직 시점 기준, 물가상승 반영 전)."""
        return self.monthly_expense * 12

    def current_allocation(self) -> Allocation:
        """현재 보유 상태에서 추정한 자산군 배분.

        현금성(예적금·퇴직금) → cash, 사적연금 → bond(중립적 운용 가정),
        주식·ETF·펀드 → equity, ISA는 성향에 따라 bond/equity로 나눈다.
        """
        base = self.financial_assets
        if base <= 0:
            return Allocation(cash=1.0, bond=0.0, equity=0.0)

        isa_equity_share = ISA_EQUITY_SHARE[self.risk_tolerance]

        cash = self.severance_pay + self.cash_savings
        bond = self.pension_dc + self.isa * (1 - isa_equity_share)
        equity = self.equity + self.isa * isa_equity_share

        return Allocation(cash=cash / base, bond=bond / base, equity=equity / base)


class YearRow(BaseModel):
    """시뮬레이션 1개 연도의 상세 내역."""

    year: int
    age: int
    active_months: int = Field(description="해당 연도에 시뮬레이션이 적용되는 개월 수")
    start_balance: int
    investment_return: int
    pension_income: int
    other_income: int
    expense: int
    event_expense: int = Field(default=0, description="그 해의 일회성 큰 지출")
    tax: int
    end_balance: int

    @property
    def total_income(self) -> int:
        return self.pension_income + self.other_income


class SimulationResult(BaseModel):
    """캐시플로우 시뮬레이션(기능 ③)의 결과.

    이 안의 숫자만이 LLM 설명 계층이 인용할 수 있는 유일한 출처다.
    """

    rows: list[YearRow]
    allocation: Allocation

    income_gap_months: int = Field(description="소득공백기 개월 수")
    pension_start_year: int
    balance_at_pension_start: int = Field(description="국민연금 개시 시점의 금융자산 잔액")

    depletion_year: Optional[int] = Field(
        default=None, description="자산이 고갈되는 연도. None이면 시뮬레이션 기간 내 고갈되지 않음"
    )
    depletion_age: Optional[int] = None
    years_until_depletion: Optional[float] = Field(
        default=None, description="퇴직 시점부터 고갈까지의 연수 (소수점 포함)"
    )

    expense_coverage_years: float = Field(
        description="연금·기타소득을 전혀 고려하지 않고 현재 금융자산만으로 생활비를 충당할 수 있는 연수"
    )

    starting_balance: int
    final_balance: int
    horizon_age: int

    @property
    def survives_horizon(self) -> bool:
        return self.depletion_year is None
