"""정책 영향 분석 + 신뢰 표시 테스트 (기능 ④·⑧).

두 가지를 지킨다:

  - **세제 seam 이 기존 결과를 건드리지 않는다.** tax_model 을 주지 않으면
    W1~W6 의 숫자가 그대로여야 한다. 정책 기능을 붙이면서 기본 계산이 조용히
    바뀌면 발표에서 말할 숫자가 전부 흔들린다.
  - **영향이 작다는 결론도 정확해야 한다.** "제도가 바뀌니 갈아타라"는 권유에
    맞서려면 "고객님께는 연 ○○원입니다"가 정확해야 한다. 과장도 축소도 안 된다.
"""

from __future__ import annotations

import pytest

from core.assumptions import load_assumptions
from core.cashflow import annual_tax, simulate
from core.models import RiskTolerance, UserProfile
from core.policy import (
    STATUS_SEQUENCE,
    UNCONFIRMED_NOTICE,
    IsaTaxModel,
    IsaTaxRule,
    PolicyStatus,
    analyze_policy_impact,
    get_policy_fact,
    isa_share_of_taxable_return,
    load_policy_catalog,
)
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE

# ISA 비중이 커서 개편안의 영향이 실제로 체감되는 인물.
# 샘플 두 명(DEMO=ISA 없음, DIVERSIFIED=ISA 5,000만원)만으로는
# "영향이 크다" 쪽 분기가 한 번도 실행되지 않는다.
BIG_ISA_PROFILE = DIVERSIFIED_PROFILE.model_copy(update={"isa": 400_000_000})


@pytest.fixture(scope="module")
def isa_fact():
    return get_policy_fact("isa_reform")


# --------------------------------------------------------------------------- #
# ⑧ 신뢰 표시
# --------------------------------------------------------------------------- #


def test_every_fact_declares_a_valid_stage():
    """오타 하나로 '미확정'이 '시행 중'으로 표시되면 이 기능의 의미가 사라진다."""
    facts = load_policy_catalog()
    assert facts
    for fact in facts:
        assert fact.status in STATUS_SEQUENCE
        assert fact.source.strip()
        assert fact.checked_at.strip()
        assert 1 <= fact.stage <= fact.stage_total


def test_unconfirmed_policy_carries_the_forced_notice(isa_fact):
    """발표 단계 제도에는 '아직 확정되지 않았습니다' 문구가 강제로 붙는다."""
    assert isa_fact.status is PolicyStatus.ANNOUNCED
    assert not isa_fact.is_confirmed
    assert isa_fact.notice == UNCONFIRMED_NOTICE
    assert "미확정" in isa_fact.badge
    assert isa_fact.effective_label == "시행일 미정"


def test_enacted_policy_has_no_notice_and_shows_its_date():
    pension = get_policy_fact("pension_start_age")
    assert pension.is_confirmed
    assert pension.notice == ""
    assert "시행 중" in pension.badge
    assert "2013-01-01" in pension.effective_label


def test_trust_note_carries_source_stage_and_check_date(isa_fact):
    """출처 / 확정여부 / 시행일 / 확인일 — 네 가지가 한 줄에 다 있어야 한다."""
    note = isa_fact.trust_note
    assert isa_fact.source in note
    assert "발표" in note and "1/5단계" in note
    assert "시행일 미정" in note
    assert isa_fact.checked_at in note


def test_unknown_policy_key_is_an_error():
    with pytest.raises(KeyError):
        get_policy_fact("존재하지_않는_정책")


# --------------------------------------------------------------------------- #
# 세제 seam — 기존 결과를 건드리지 않는다
# --------------------------------------------------------------------------- #


def test_default_tax_model_changes_nothing():
    """tax_model 을 주지 않은 시뮬레이션은 W1 이후와 완전히 동일해야 한다."""
    base = simulate(DEMO_PROFILE)
    explicit = simulate(DEMO_PROFILE, tax_model=annual_tax)
    assert base.model_dump() == explicit.model_dump()


def test_zero_isa_share_reduces_to_the_default_tax_model(base_assumptions):
    """ISA 비중이 0이면 새 세금 계산기는 기존 annual_tax 와 같은 값을 내야 한다."""
    rule = IsaTaxRule(label="현행", exempt_limit_total=2_000_000, separate_rate=0.099)
    model = IsaTaxModel(isa_share_of_taxable=0.0, rule=rule)
    for income in (0.0, 1_000_000.0, 30_000_000.0):
        assert model(income, base_assumptions) == annual_tax(income, base_assumptions)


def test_isa_tax_rule_only_taxes_the_excess():
    rule = IsaTaxRule(
        label="현행", exempt_limit_total=2_000_000, contract_years=3, separate_rate=0.099
    )
    limit = rule.annual_exempt_limit
    assert limit == pytest.approx(666_666.67, abs=1)

    assert rule.tax_on(limit) == 0.0  # 한도까지는 비과세
    assert rule.tax_on(-500_000) == 0.0  # 손실 난 해에는 0
    assert rule.tax_on(limit + 1_000_000) == pytest.approx(1_000_000 * 0.099)


def test_isa_share_is_zero_without_an_isa_account():
    assert isa_share_of_taxable_return(DEMO_PROFILE) == 0.0
    assert isa_share_of_taxable_return(DIVERSIFIED_PROFILE) > 0.0


def test_isa_share_follows_the_allocation_convention():
    """ISA 안의 주식 비중은 자산군 배분과 같은 상수를 써야 한다.

    안정형은 ISA 를 채권 쪽으로 더 많이 담으므로 과세대상 수익 기여가 더 커진다.
    두 계산이 다른 상수를 쓰기 시작하면 여기가 깨진다.
    """
    conservative = DIVERSIFIED_PROFILE.model_copy(
        update={"risk_tolerance": RiskTolerance.CONSERVATIVE}
    )
    aggressive = DIVERSIFIED_PROFILE.model_copy(
        update={"risk_tolerance": RiskTolerance.AGGRESSIVE}
    )
    assert isa_share_of_taxable_return(conservative) > isa_share_of_taxable_return(
        aggressive
    )


# --------------------------------------------------------------------------- #
# ④ 영향 계산
# --------------------------------------------------------------------------- #


def test_user_without_an_isa_is_told_it_does_not_apply():
    """빈 화면 대신 '영향 없음'을 분명히 말한다. 기획서 예시 인물이 이 경우다."""
    impact = analyze_policy_impact(DEMO_PROFILE)

    assert not impact.applicable
    assert impact.current is None and impact.proposed is None
    assert "ISA" in impact.reason
    assert impact.annual_tax_saving == 0
    # 미확정 고지는 계산 여부와 무관하게 항상 따라붙는다.
    assert any(UNCONFIRMED_NOTICE in note for note in impact.notes)


def test_isa_holder_gets_a_number_not_a_news_summary():
    impact = analyze_policy_impact(DIVERSIFIED_PROFILE)

    assert impact.applicable
    assert impact.current is not None and impact.proposed is not None
    assert impact.annual_tax_saving > 0
    assert impact.lifetime_tax_saving >= impact.annual_tax_saving
    # 세금을 덜 내면 잔액은 늘어난다 — 부호가 뒤집히면 계산이 잘못된 것이다.
    assert impact.final_balance_delta > 0
    assert "연" in impact.headline


def test_raising_the_exempt_limit_never_increases_the_tax():
    """비과세 한도를 올리는 개편안이 불리해질 수는 없다 (부호 방향 불변식)."""
    for profile in (DIVERSIFIED_PROFILE, BIG_ISA_PROFILE):
        impact = analyze_policy_impact(profile)
        assert impact.proposed.lifetime_tax <= impact.current.lifetime_tax
        assert impact.proposed.annual_tax_first_full_year <= (
            impact.current.annual_tax_first_full_year
        )


def test_small_impact_is_called_small():
    """이 서비스의 존재 이유 — '제도 바뀌니 갈아타라'에 대한 반박.

    ISA 5,000만원 보유자에게 이 개편안은 연 몇 만원 수준이다.
    그걸 크게 말하면 사기 문자와 다를 게 없다.
    """
    impact = analyze_policy_impact(DIVERSIFIED_PROFILE)
    assert not impact.material
    assert "이유가 되지 않는" in impact.verdict


def test_the_saving_has_a_ceiling_no_portfolio_can_exceed():
    """비과세 한도만 올리는 개편의 절감액에는 구조적 상한이 있다.

        절감액 ≤ (한도 증가분 ÷ 의무가입기간) × 분리과세율

    개인의 자산을 몰라도 "누구에게든 최대 연 ○○원"이라고 말할 수 있다는 뜻이고,
    이게 "지금 갈아타라"는 권유에 대한 가장 강한 반박이다.
    """
    ceiling = analyze_policy_impact(DIVERSIFIED_PROFILE).max_annual_saving
    assert ceiling == pytest.approx((5_000_000 - 2_000_000) / 3 * 0.099, abs=1)

    # ISA 를 8배로 늘려도 상한을 넘지 못한다.
    big = analyze_policy_impact(BIG_ISA_PROFILE)
    assert big.max_annual_saving == ceiling
    assert big.annual_tax_saving <= ceiling
    assert not big.material  # 상한 자체가 판단 기준(연 50만원)보다 작다
    assert "넘지 않습니다" in big.verdict


def test_large_impact_is_called_large(tmp_path):
    """세율까지 바꾸는 개편이라면 상한이 없어지고 '체감되는 크기'로 말해야 한다.

    현행 팩트시트로는 이 분기가 실행되지 않는다. 팩트시트가 갱신되어 영향이
    커졌을 때 문구가 제대로 바뀌는지 확인하려고 파라미터를 갈아끼워 본다.
    """
    fixture = tmp_path / "policy_facts.yaml"
    fixture.write_text(
        """
facts:
  - key: isa_reform
    topic: ISA
    title: 대폭 개편안
    status: 발표
    effective_date: null
    source: 테스트 픽스처
    checked_at: "2026-08-13"
    summary: 시험용
    caution: 시험용
    impact:
      kind: isa_tax
      materiality_threshold: 500000
      current:
        label: 현행 제도
        exempt_limit_total: 2000000
        contract_years: 3
        separate_rate: 0.154
      proposed:
        label: 개편안 (미확정)
        exempt_limit_total: 60000000
        contract_years: 3
        separate_rate: 0.0
""",
        encoding="utf-8",
    )

    impact = analyze_policy_impact(BIG_ISA_PROFILE, path=fixture)
    assert impact.material
    assert impact.max_annual_saving is None  # 세율이 바뀌면 상한이 없다
    assert "확정되면 실제로 체감되는" in impact.verdict


def test_comparison_table_is_ready_to_render():
    """금액 표기는 서버가 끝낸다. 프런트엔드가 다시 포맷하면 화면마다 달라진다."""
    impact = analyze_policy_impact(DIVERSIFIED_PROFILE)
    labels = [row.label for row in impact.comparison]

    assert "연간 세금 (첫 온전한 1년)" in labels
    assert "은퇴 재무 안정도" in labels
    for row in impact.comparison:
        assert row.current and row.proposed and row.delta
        assert not row.current.isdigit()  # 원시 숫자가 그대로 나가면 안 된다


def test_impact_always_ships_with_its_source():
    """숫자만 있고 출처·확정여부가 없는 정책 안내는 만들지 않는다."""
    for profile in (DEMO_PROFILE, DIVERSIFIED_PROFILE):
        impact = analyze_policy_impact(profile)
        assert impact.policy.source
        assert impact.policy.trust_note
        assert not impact.is_confirmed


def test_impact_uses_the_same_engine_conventions():
    """비교 대상 두 시뮬레이션은 세제만 달라야 한다.

    ISA 를 갖지 않은 사람에게는 두 세제가 같은 값을 내므로, 기본 시뮬레이션과
    잔액이 정확히 일치해야 한다.
    """
    assumptions = load_assumptions()
    rule = IsaTaxRule(label="현행", exempt_limit_total=2_000_000, separate_rate=0.099)
    model = IsaTaxModel(isa_share_of_taxable=0.0, rule=rule)

    base = simulate(DEMO_PROFILE, assumptions=assumptions)
    with_model = simulate(DEMO_PROFILE, assumptions=assumptions, tax_model=model)
    assert with_model.final_balance == base.final_balance
    assert with_model.depletion_year == base.depletion_year


def test_policy_without_an_impact_model_still_answers():
    """계산이 연결되지 않은 제도도 안내는 한다 — 화면이 비면 안 된다."""
    impact = analyze_policy_impact(DEMO_PROFILE, policy_key="deposit_protection")
    assert not impact.applicable
    assert impact.policy.is_confirmed
    assert impact.verdict


def test_unknown_policy_key_raises():
    with pytest.raises(KeyError):
        analyze_policy_impact(DIVERSIFIED_PROFILE, policy_key="없는키")


def test_profile_without_financial_assets_is_handled():
    """자산이 없으면 나눗셈이 터지지 않고 '영향 없음'으로 떨어져야 한다."""
    broke = UserProfile(
        birth_year=1965,
        retirement_year=2026,
        monthly_expense=2_000_000,
    )
    impact = analyze_policy_impact(broke)
    assert not impact.applicable
