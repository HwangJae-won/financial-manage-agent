"""data/assumptions.yaml 로더.

가정값은 코드에 하드코딩하지 않고 전부 이 모듈을 통해 읽는다.
심사·발표에서 "이 숫자 근거가 뭐냐"는 질문이 나오면 YAML 한 파일만 보여주면 된다.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "assumptions.yaml"


class AssetClassAssumption(BaseModel):
    label: str
    expected_return: float
    volatility: float


class MacroAssumption(BaseModel):
    inflation_rate: float
    horizon_age: int


class TaxAssumption(BaseModel):
    financial_income_rate: float
    comprehensive_threshold: int


class DependentAssumption(BaseModel):
    """피부양자 자격 요건. 하나라도 넘기면 보험료가 0원에서 연 수백만원이 된다."""

    income_limit: int = 20_000_000
    financial_income_limit: int = 10_000_000
    property_limit: int = 540_000_000
    property_soft_limit: int = 360_000_000
    soft_limit_income_limit: int = 10_000_000


class LocalPremiumAssumption(BaseModel):
    """지역가입자 보험료 산정. 요율은 매년 바뀐다."""

    health_rate: float = 0.0709
    long_term_care_rate: float = 0.1295
    pension_income_share: float = 0.50
    minimum_monthly: int = 19_780


class VoluntaryEnrollmentAssumption(BaseModel):
    """임의계속가입 — 퇴직 직후의 완충장치."""

    max_months: int = 36
    employee_share: float = 0.50
    min_service_years: int = 1


class HealthInsuranceAssumption(BaseModel):
    effective_rate_on_financial_income: float
    exemption_threshold: int

    # 아래 항목들은 기본값을 둔다. 이 항목이 없는 예전 YAML 도 계속 읽혀야 한다.
    dependent: DependentAssumption = Field(default_factory=DependentAssumption)
    local: LocalPremiumAssumption = Field(default_factory=LocalPremiumAssumption)
    voluntary: VoluntaryEnrollmentAssumption = Field(
        default_factory=VoluntaryEnrollmentAssumption
    )
    property_tax_base_ratio: float = Field(default=0.42, gt=0.0, le=1.0)


class PensionStartAgeBand(BaseModel):
    from_birth_year: int
    age: int


class PensionAssumption(BaseModel):
    start_age_schedule: list[PensionStartAgeBand]
    indexed_to_inflation: bool = True

    # 수급 시기 조정. 기본값을 둔 이유는 이 항목이 없는 예전 YAML 도 계속 읽히게 하려는 것이다.
    deferral_rate_per_year: float = Field(default=0.072, ge=0.0)
    early_rate_per_year: float = Field(default=0.060, ge=0.0)
    max_adjust_years: int = Field(default=5, ge=0)


class AssetMapAssumption(BaseModel):
    survival_months: int


class ServiceDeductionBand(BaseModel):
    over_years: int
    base: int
    per_year: int


class DeductionBand(BaseModel):
    over: int
    base: int
    rate: float


class TaxBracket(BaseModel):
    over: int
    rate: float


class RetirementIncomeTaxAssumption(BaseModel):
    """퇴직소득세 계산 파라미터. 실효세율 근사가 아니라 제도 구조를 그대로 따른다."""

    service_deduction: list[ServiceDeductionBand]
    converted_deduction: list[DeductionBand]
    brackets: list[TaxBracket]
    local_tax_rate: float = 0.10
    pension_discount: float = 0.30
    pension_discount_after: float = 0.40
    pension_discount_after_years: int = 10
    default_pension_years: int = 10


class SensitivityAssumption(BaseModel):
    """민감도 분석에서 각 가정을 흔들어 볼 폭. 기본값은 YAML 이 없을 때만 쓰인다."""

    inflation_delta: float = Field(default=0.007, ge=0.0)
    return_delta: float = Field(default=0.010, ge=0.0)
    expense_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    pension_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    assets_ratio: float = Field(default=0.10, ge=0.0, le=1.0)


class Assumptions(BaseModel):
    """시뮬레이션 가정값 전체."""

    version: str
    macro: MacroAssumption
    asset_classes: dict[str, AssetClassAssumption]
    tax: TaxAssumption
    health_insurance: HealthInsuranceAssumption
    pension: PensionAssumption
    asset_map: AssetMapAssumption
    retirement_income_tax: RetirementIncomeTaxAssumption
    sensitivity: SensitivityAssumption = Field(default_factory=SensitivityAssumption)

    def expected_return(self, allocation: dict[str, float]) -> float:
        """자산군 배분에 대한 가중평균 기대수익률."""
        return sum(
            weight * self.asset_classes[key].expected_return
            for key, weight in allocation.items()
        )

    def volatility(self, allocation: dict[str, float]) -> float:
        """자산군 배분에 대한 포트폴리오 변동성.

        자산군 간 상관계수를 0으로 가정한 단순 근사다. 실제로는 주식-채권 상관이
        0이 아니므로 이 값은 변동성을 과소평가할 수 있다 (MVP 한계로 명시).
        """
        variance = sum(
            (weight * self.asset_classes[key].volatility) ** 2
            for key, weight in allocation.items()
        )
        return variance**0.5

    def national_pension_start_age(self, birth_year: int) -> int:
        """출생연도에 해당하는 국민연금 수급 개시 연령 (법정 스케줄)."""
        applicable = [
            band.age
            for band in self.pension.start_age_schedule
            if birth_year >= band.from_birth_year
        ]
        if not applicable:
            raise ValueError(f"수급 개시 연령을 결정할 수 없습니다: birth_year={birth_year}")
        return max(applicable)

    def pension_start_age_range(self, birth_year: int) -> tuple[int, int]:
        """선택 가능한 수급 개시 연령 구간 (조기 최대 ~ 연기 최대)."""
        normal = self.national_pension_start_age(birth_year)
        span = self.pension.max_adjust_years
        return normal - span, normal + span

    def pension_amount_factor(self, birth_year: int, start_age: int) -> float:
        """수급 시기를 조정했을 때 월 수령액에 곱할 계수.

        법정 개시연령에 받으면 1.0. 미루면 연 7.2% 가산, 앞당기면 연 6% 감액이며
        양쪽 모두 최대 5년까지다. 조정 한도를 넘겨 요청해도 한도까지만 반영한다 —
        입력이 잘못되었다고 계산을 멈추는 것보다, 제도상 가능한 범위로 잘라서
        계산하고 그 사실을 화면에서 말하는 편이 낫다.

        소득공백기(퇴직~수급 개시)는 이 계수와 별개로 core/schedule.py 가 처리한다.
        여기서는 '얼마를 받느냐'만 정한다.
        """
        normal = self.national_pension_start_age(birth_year)
        span = self.pension.max_adjust_years
        delta = max(-span, min(span, start_age - normal))

        if delta >= 0:
            return 1.0 + delta * self.pension.deferral_rate_per_year
        return max(0.0, 1.0 + delta * self.pension.early_rate_per_year)


@functools.lru_cache(maxsize=4)
def load_assumptions(path: str | Path = DEFAULT_PATH) -> Assumptions:
    """가정값을 로드한다. 동일 경로는 캐시된다."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Assumptions.model_validate(raw)
