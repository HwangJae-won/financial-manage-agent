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


class HealthInsuranceAssumption(BaseModel):
    effective_rate_on_financial_income: float
    exemption_threshold: int


class PensionStartAgeBand(BaseModel):
    from_birth_year: int
    age: int


class PensionAssumption(BaseModel):
    start_age_schedule: list[PensionStartAgeBand]
    indexed_to_inflation: bool = True


class AssetMapAssumption(BaseModel):
    survival_months: int


class Assumptions(BaseModel):
    """시뮬레이션 가정값 전체."""

    version: str
    macro: MacroAssumption
    asset_classes: dict[str, AssetClassAssumption]
    tax: TaxAssumption
    health_insurance: HealthInsuranceAssumption
    pension: PensionAssumption
    asset_map: AssetMapAssumption

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


@functools.lru_cache(maxsize=4)
def load_assumptions(path: str | Path = DEFAULT_PATH) -> Assumptions:
    """가정값을 로드한다. 동일 경로는 캐시된다."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Assumptions.model_validate(raw)
