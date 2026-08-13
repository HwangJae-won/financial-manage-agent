"""데모용 샘플 프로파일.

기획서의 예시 인물을 그대로 코드로 옮긴 것. 테스트와 Streamlit 초기값에서 쓴다.
"""

from __future__ import annotations

from core.models import RiskTolerance, UserProfile

# 기획서 예시:
#   총자산 7.2억 (금융자산 2.2억 / 부동산 5억)
#   월 생활비 300만 원, 국민연금 예상 월 150만 원, 연금 수령까지 4년
#   2026년 12월 퇴직
#
# 1966년 12월생 → 국민연금 개시 64세(2030년 12월) → 소득공백기 정확히 48개월.
DEMO_PROFILE = UserProfile(
    birth_year=1966,
    birth_month=12,
    dependents=1,
    years_employed=32,
    # 퇴직금은 '30일분 평균임금 × 근속연수'다. 2억 ÷ 32년 = 월 625만원 —
    # 급여를 따로 지어내지 않고 이미 정한 두 값에서 끌어낸다.
    last_monthly_salary=6_250_000,
    risk_tolerance=RiskTolerance.MODERATE,
    retirement_year=2026,
    retirement_month=12,
    severance_pay=200_000_000,
    cash_savings=20_000_000,
    equity=0,
    isa=0,
    pension_dc=0,
    real_estate=500_000_000,
    debt=0,
    monthly_expense=3_000_000,
    other_monthly_income=0,
    national_pension_monthly=1_500_000,
    national_pension_start_age=64,
)

# 자산 구성이 더 다양한 프로파일 — 자산지도와 시나리오 비교를 보여주기 좋다.
DIVERSIFIED_PROFILE = UserProfile(
    birth_year=1968,
    birth_month=5,
    dependents=0,
    years_employed=28,
    last_monthly_salary=5_350_000,  # 1.5억 ÷ 28년
    risk_tolerance=RiskTolerance.MODERATE,
    retirement_year=2027,
    retirement_month=6,
    severance_pay=150_000_000,
    cash_savings=80_000_000,
    equity=120_000_000,
    isa=50_000_000,
    pension_dc=100_000_000,
    real_estate=600_000_000,
    debt=50_000_000,
    monthly_expense=3_500_000,
    other_monthly_income=800_000,
    national_pension_monthly=1_700_000,
    national_pension_start_age=64,
)
