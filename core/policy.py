"""정책 영향 분석 축소판 + 신뢰 표시 (기능 ④·⑧).

기획서의 정책 RAG(크롤링 + Vector DB + 시점 관리)는 문서 수집과 시점 메타데이터
관리가 별도 프로젝트급이라 MVP 범위 밖이다. 대신 **한 케이스(ISA 세제 개편안)만**
골라 계산 끝까지 연결한다. 뉴스 요약은 포털에도 있다. "고객님 자산 기준으로
연 ○○원" 이 나와야 이 기능이 존재할 이유가 생긴다.

두 축:

  ④ 영향 계산 — 팩트시트의 세제 파라미터를 캐시플로우 엔진에 주입해 **현행 제도와
     개편안을 같은 기준으로** 재시뮬레이션하고 차이만 본다. 세제 외의 규약(수익
     시점, 물가연동, 퇴직월 안분)은 두 시뮬레이션이 공유하므로, 결과 차이는 오직
     세제 차이다.

  ⑧ 신뢰 표시 — 모든 정책성 출력에 출처·확정여부·시행일·확인일을 붙인다.
     **발표 → 입법예고 → 국회통과 → 공포 → 시행** 중 어디인지 밝히고, 시행되지
     않은 건에는 "아직 확정되지 않았습니다" 문구를 강제한다.

결론이 "영향이 거의 없습니다"로 나오는 경우가 많다는 점이 오히려 이 기능의 핵심이다.
사기 문자는 늘 "제도가 바뀌니 지금 갈아타라"고 말한다. 실제 영향을 원 단위로 계산해
보여주는 것이 그 주장에 대한 가장 값싸고 확실한 반박이다.

한계 (발표에서 질문받을 항목):
  - ISA 가 전체 금융자산에서 차지하는 비중이 시뮬레이션 기간 내내 유지된다고 본다.
    계좌 간 이동이나 만기 재가입은 반영하지 않는다.
  - 기본 분석 화면의 세금은 ISA 비과세 혜택을 반영하지 않은 보수적 값이다.
    이 비교는 현행·개편안 **양쪽 모두에** ISA 세제를 적용해 차이만 본다.
"""

from __future__ import annotations

import functools
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, computed_field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import annual_tax, simulate
from core.formatting import fmt_krw
from core.models import ISA_EQUITY_SHARE, UserProfile
from core.risk_score import compute_risk_score

POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "policy_facts.yaml"

DEFAULT_POLICY_KEY = "isa_reform"

# 시행되지 않은 제도에 강제로 붙는 문구. 화면·LLM 설명 어디를 거치든 이 문장은 남는다.
UNCONFIRMED_NOTICE = (
    "아직 확정되지 않았습니다. 국회 논의 과정에서 내용이 바뀌거나 무산될 수 있으므로, "
    "이것을 근거로 지금 상품을 옮기실 필요는 없습니다."
)


class PolicyStatus(str, Enum):
    """입법 단계. 이 구분이 기능 ⑧의 전부다.

    일반 뉴스는 '발표'와 '시행'을 같은 문장으로 쓴다. 발표된 개편안이 그대로
    확정되지 않는 경우가 많고, 그 틈이 곧 "제도 바뀌니 갈아타라"는 권유가
    파고드는 자리다.
    """

    ANNOUNCED = "발표"
    LEGISLATIVE_NOTICE = "입법예고"
    PASSED = "국회통과"
    PROMULGATED = "공포"
    IN_FORCE = "시행"


# 진행 순서. 화면에서 "5단계 중 1단계"로 보여준다.
STATUS_SEQUENCE: tuple[PolicyStatus, ...] = (
    PolicyStatus.ANNOUNCED,
    PolicyStatus.LEGISLATIVE_NOTICE,
    PolicyStatus.PASSED,
    PolicyStatus.PROMULGATED,
    PolicyStatus.IN_FORCE,
)


# --------------------------------------------------------------------------- #
# 팩트시트 (⑧ 신뢰 표시)
# --------------------------------------------------------------------------- #


class PolicyFact(BaseModel):
    """정책 팩트시트 한 건 + 신뢰 표시에 필요한 메타데이터 전부.

    출처(source)와 확인일(checked_at)이 없는 항목은 만들 수 없게 필수로 뒀다.
    근거를 댈 수 없는 정책 안내는 이 서비스에서 하지 않는다.
    """

    key: str
    topic: str
    title: str
    status: PolicyStatus
    effective_date: Optional[str] = None
    source: str
    checked_at: str
    summary: str
    caution: str
    impact: Optional[dict[str, Any]] = Field(
        default=None, description="영향 계산 파라미터. 없으면 안내만 하고 계산은 하지 않는다."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_confirmed(self) -> bool:
        return self.status is PolicyStatus.IN_FORCE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def stage(self) -> int:
        """입법 단계 번호 (1~5)."""
        return STATUS_SEQUENCE.index(self.status) + 1

    @property
    def stage_total(self) -> int:
        return len(STATUS_SEQUENCE)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def badge(self) -> str:
        """화면 뱃지. 색이 아니라 글자로 구분한다 (시니어 접근성)."""
        if self.is_confirmed:
            return "✅ 시행 중"
        return f"⚠️ {self.status.value} (미확정)"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_label(self) -> str:
        if self.effective_date:
            return f"{self.effective_date} 시행"
        return "시행 중 (시행일 별도 확인)" if self.is_confirmed else "시행일 미정"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def notice(self) -> str:
        """미확정 건에 강제되는 문구. 확정된 제도면 빈 문자열."""
        return "" if self.is_confirmed else UNCONFIRMED_NOTICE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trust_note(self) -> str:
        """출처 · 확정여부 · 시행일 · 확인일 한 줄. 모든 정책성 답변에 붙는다."""
        return (
            f"출처: {self.source} · 확정여부: {self.status.value}"
            f"({self.stage}/{self.stage_total}단계) · {self.effective_label}"
            f" · 확인일: {self.checked_at}"
        )

    @property
    def has_impact_model(self) -> bool:
        return bool(self.impact)


@functools.lru_cache(maxsize=2)
def load_raw_facts(path: str | Path = POLICY_PATH) -> list[dict[str, Any]]:
    """YAML 원본을 그대로 읽는다. 타입이 필요한 쪽은 load_policy_catalog 를 쓴다."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("facts", [])


def load_policy_catalog(path: str | Path = POLICY_PATH) -> list[PolicyFact]:
    """팩트시트 전체를 타입 있는 모델로 읽는다.

    status 가 5단계 중 하나가 아니면 여기서 검증 오류가 난다 — 오타 하나로
    "미확정"이 "시행 중"으로 표시되는 사고를 막는다.
    """
    return [PolicyFact.model_validate(fact) for fact in load_raw_facts(path)]


def get_policy_fact(
    key: str = DEFAULT_POLICY_KEY, path: str | Path = POLICY_PATH
) -> PolicyFact:
    for fact in load_policy_catalog(path):
        if fact.key == key:
            return fact
    raise KeyError(f"알 수 없는 정책 키입니다: {key}")


# --------------------------------------------------------------------------- #
# ISA 세제 모델 (④ 영향 계산)
# --------------------------------------------------------------------------- #


class IsaTaxRule(BaseModel):
    """ISA 계좌의 이자·배당 소득에 적용되는 세제 한 벌."""

    label: str
    exempt_limit_total: int = Field(ge=0, description="계약기간 전체 비과세 한도")
    contract_years: int = Field(default=3, gt=0, description="의무가입기간(년)")
    separate_rate: float = Field(ge=0.0, le=1.0, description="한도 초과분 분리과세율")

    @property
    def annual_exempt_limit(self) -> float:
        """연 환산 비과세 한도.

        총액 한도를 의무가입기간으로 나눈다. 해마다 한도가 새로 채워진다고 보면
        실제보다 유리해지므로 이쪽이 보수적이다.
        """
        return self.exempt_limit_total / self.contract_years

    def tax_on(self, isa_income: float) -> float:
        """ISA 계좌에서 발생한 이자·배당 소득에 붙는 세금.

        분리과세된 소득은 금융소득종합과세 대상이 아니므로 건강보험료도 붙지 않는다.
        """
        if isa_income <= 0:
            return 0.0
        return max(0.0, isa_income - self.annual_exempt_limit) * self.separate_rate


class IsaTaxModel(BaseModel):
    """ISA 세제를 반영한 연간 세금 계산기. `simulate(tax_model=...)` 에 넣는다.

    포트폴리오 전체의 과세대상 수익 중 ISA 계좌 몫만 떼어 ISA 세제를 적용하고,
    나머지는 기존 규약(`annual_tax`)을 그대로 쓴다. 기본 엔진의 계산 규약을
    복제하지 않고 재사용하므로 두 경로가 어긋날 수 없다.
    """

    isa_share_of_taxable: float = Field(ge=0.0, le=1.0)
    rule: IsaTaxRule

    def __call__(self, taxable_return: float, assumptions: Assumptions) -> float:
        isa_income = taxable_return * self.isa_share_of_taxable
        other_income = taxable_return - isa_income
        return annual_tax(other_income, assumptions) + self.rule.tax_on(isa_income)


def isa_share_of_taxable_return(
    profile: UserProfile, assumptions: Optional[Assumptions] = None
) -> float:
    """포트폴리오의 과세대상(이자·배당) 수익 중 ISA 계좌에서 나오는 비율.

    ISA 안의 주식성 자산은 어차피 이 엔진에서 비과세(국내 소액주주 양도차익)이므로,
    ISA 의 채권성 부분만 과세대상 수익에 기여한다. 주식/채권 배분 비율은
    `core.models.ISA_EQUITY_SHARE` 를 자산군 배분 계산과 공유한다.
    """
    assumptions = assumptions or load_assumptions()
    base = profile.financial_assets
    if base <= 0 or profile.isa <= 0:
        return 0.0

    allocation = profile.current_allocation()
    classes = assumptions.asset_classes
    taxable_rate = (
        allocation.cash * classes["cash"].expected_return
        + allocation.bond * classes["bond"].expected_return
    )
    if taxable_rate <= 0:
        return 0.0

    isa_bond_weight = profile.isa * (1.0 - ISA_EQUITY_SHARE[profile.risk_tolerance]) / base
    isa_taxable_rate = isa_bond_weight * classes["bond"].expected_return
    return min(1.0, isa_taxable_rate / taxable_rate)


# --------------------------------------------------------------------------- #
# 영향 분석 결과
# --------------------------------------------------------------------------- #


class PlanOutcome(BaseModel):
    """한 세제 아래에서의 은퇴 계획 결과 요약."""

    label: str
    annual_tax_first_full_year: int = Field(
        description="퇴직 다음 온전한 1년의 세금 (퇴직 연도는 개월 안분이라 비교에 부적합)"
    )
    lifetime_tax: int = Field(description="시뮬레이션 기간 전체 세금 합계")
    balance_at_pension_start: int
    depletion_year: Optional[int] = None
    depletion_age: Optional[int] = None
    final_balance: int
    risk_score: int
    risk_status: str


class ComparisonRow(BaseModel):
    """현행 제도 vs 개편안 비교표 한 줄. 표시 문자열까지 서버가 만든다.

    프런트엔드가 금액 표기를 다시 구현하면 화면마다 표기가 달라진다.
    포맷은 core/formatting.py 한 곳에서만 한다 (Headline 과 같은 규칙).
    """

    label: str
    current: str
    proposed: str
    delta: str


def _delta_label(delta: int) -> str:
    if delta == 0:
        return "변화 없음"
    sign = "+" if delta > 0 else "−"
    return f"{sign}{fmt_krw(abs(delta))}"


class PolicyImpact(BaseModel):
    """정책 한 건이 이 사용자의 은퇴 계획에 미치는 영향.

    `policy` 가 항상 들어 있다는 점이 중요하다 — 숫자만 있고 출처·확정여부가 없는
    정책 안내는 이 서비스에서 만들지 않는다 (기능 ⑧).
    """

    policy: PolicyFact
    applicable: bool = Field(description="이 사용자에게 계산 가능한 영향이 있는가")
    reason: str = Field(description="영향이 있다/없다고 본 이유")

    current: Optional[PlanOutcome] = None
    proposed: Optional[PlanOutcome] = None
    comparison: list[ComparisonRow] = Field(
        default_factory=list, description="화면에 그대로 그릴 수 있는 비교표"
    )

    annual_tax_saving: int = 0
    lifetime_tax_saving: int = 0
    balance_delta_at_pension_start: int = 0
    final_balance_delta: int = 0
    depletion_age_delta: Optional[int] = None
    risk_score_delta: int = 0

    material: bool = Field(
        default=False, description="의사결정을 바꿀 만한 크기인가 (materiality_threshold 기준)"
    )
    max_annual_saving: Optional[int] = Field(
        default=None,
        description=(
            "보유액과 무관하게 이 개편안으로 줄어들 수 있는 연간 세금의 상한. "
            "분리과세율이 그대로일 때만 정의된다."
        ),
    )
    headline: str = ""
    verdict: str = Field(default="", description="'그래서 어떻게 하면 되는가' 한 문장")
    notes: list[str] = Field(default_factory=list, description="계산 가정과 강제 고지")

    @property
    def is_confirmed(self) -> bool:
        return self.policy.is_confirmed


# --------------------------------------------------------------------------- #
# 진입점
# --------------------------------------------------------------------------- #


def _outcome(
    label: str,
    profile: UserProfile,
    assumptions: Assumptions,
    tax_model: IsaTaxModel,
) -> PlanOutcome:
    sim = simulate(profile, assumptions=assumptions, tax_model=tax_model)
    score = compute_risk_score(profile, assumptions, tax_model=tax_model)

    full_years = [row for row in sim.rows if row.active_months == 12]
    return PlanOutcome(
        label=label,
        annual_tax_first_full_year=full_years[0].tax if full_years else 0,
        lifetime_tax=sum(row.tax for row in sim.rows),
        balance_at_pension_start=sim.balance_at_pension_start,
        depletion_year=sim.depletion_year,
        depletion_age=sim.depletion_age,
        final_balance=sim.final_balance,
        risk_score=score.total,
        risk_status=score.status,
    )


def _max_annual_saving(
    current: IsaTaxRule, proposed: IsaTaxRule
) -> Optional[int]:
    """이 개편안으로 줄어들 수 있는 연간 세금의 상한.

    비과세 한도만 올리는 개편이라면 절감액은 **한도 증가분 × 분리과세율**을 넘을 수
    없다. 보유액이 아무리 커도 그렇다. 이 상한을 계산해 두는 이유는 "그래서 지금
    갈아타야 하나"에 대한 가장 강한 답이기 때문이다 — 개인의 자산을 몰라도
    "누구에게든 최대 연 ○○원"이라고 말할 수 있다.

    분리과세율까지 바뀌면 절감액이 소득에 비례해 커져 상한이 없어지므로 None 이다.
    """
    if current.separate_rate != proposed.separate_rate:
        return None
    gain = proposed.annual_exempt_limit - current.annual_exempt_limit
    if gain <= 0:
        return 0
    return int(round(gain * current.separate_rate))


def _not_applicable(fact: PolicyFact, reason: str) -> PolicyImpact:
    notes = [reason]
    if fact.notice:
        notes.append(fact.notice)
    return PolicyImpact(
        policy=fact,
        applicable=False,
        reason=reason,
        headline="고객님의 은퇴 계획에 미치는 영향은 계산되지 않았습니다.",
        verdict=(
            "지금 하실 일은 없습니다. 이 제도를 근거로 상품 이동을 권유받으셨다면 "
            "'받은 연락 확인하기'에서 그 내용을 먼저 확인해 보세요."
        ),
        notes=notes,
    )


def analyze_policy_impact(
    profile: UserProfile,
    *,
    policy_key: str = DEFAULT_POLICY_KEY,
    assumptions: Optional[Assumptions] = None,
    path: str | Path = POLICY_PATH,
) -> PolicyImpact:
    """정책 한 건이 이 사용자의 은퇴 계획에 미치는 영향을 계산한다.

    현행 제도와 개편안을 **같은 엔진·같은 가정**으로 각각 시뮬레이션하고 차이를 본다.
    영향을 계산할 수 없는 경우(해당 자산 미보유 등)에도 결과를 돌려주되
    `applicable=False` 로 표시한다 — 화면이 빈칸이 되는 것보다 "영향 없음"이 낫다.
    """
    assumptions = assumptions or load_assumptions()
    fact = get_policy_fact(policy_key, path)

    if not fact.has_impact_model:
        return _not_applicable(
            fact, "이 항목은 안내만 제공하며 금액 영향 계산은 연결되어 있지 않습니다."
        )

    impact = fact.impact or {}
    if impact.get("kind") != "isa_tax":
        return _not_applicable(
            fact, f"아직 계산 방법이 준비되지 않은 유형입니다: {impact.get('kind')}"
        )

    share = isa_share_of_taxable_return(profile, assumptions)
    if profile.isa <= 0 or share <= 0:
        return _not_applicable(
            fact,
            "고객님은 ISA 계좌를 보유하고 계시지 않습니다. 이 개편안은 ISA 계좌에서 "
            "발생하는 이자·배당 소득의 세금에 관한 것이라, 지금 구성으로는 직접적인 "
            "영향이 없습니다.",
        )

    current_rule = IsaTaxRule.model_validate(impact["current"])
    proposed_rule = IsaTaxRule.model_validate(impact["proposed"])

    current = _outcome(
        current_rule.label,
        profile,
        assumptions,
        IsaTaxModel(isa_share_of_taxable=share, rule=current_rule),
    )
    proposed = _outcome(
        proposed_rule.label,
        profile,
        assumptions,
        IsaTaxModel(isa_share_of_taxable=share, rule=proposed_rule),
    )

    annual_saving = current.annual_tax_first_full_year - proposed.annual_tax_first_full_year
    lifetime_saving = current.lifetime_tax - proposed.lifetime_tax
    threshold = int(impact.get("materiality_threshold", 0))
    material = annual_saving >= threshold > 0
    ceiling = _max_annual_saving(current_rule, proposed_rule)

    depletion_delta: Optional[int] = None
    if current.depletion_age is not None and proposed.depletion_age is not None:
        depletion_delta = proposed.depletion_age - current.depletion_age

    if annual_saving > 0:
        headline = (
            f"개편안이 확정되면 세부담이 연 {fmt_krw(annual_saving)}, "
            f"은퇴 기간 전체로 {fmt_krw(lifetime_saving)} 줄어듭니다."
        )
    elif annual_saving < 0:
        headline = (
            f"개편안이 확정되면 세부담이 연 {fmt_krw(-annual_saving)} 늘어납니다."
        )
    else:
        headline = (
            "개편안이 확정되어도 고객님의 세부담은 달라지지 않습니다. "
            f"ISA 에서 발생하는 이자·배당 소득이 현행 비과세 한도"
            f"(연 {fmt_krw(current_rule.annual_exempt_limit)}) 안에 들어오기 때문입니다."
        )

    if material:
        verdict = (
            "확정되면 실제로 체감되는 크기입니다. 다만 아직 확정 전이므로, 확정 여부를 "
            "확인하신 뒤에 움직이셔도 늦지 않습니다."
        )
    else:
        verdict = (
            "지금 상품을 옮기실 이유가 되지 않는 크기입니다. "
            "'제도가 바뀌니 지금 갈아타라'는 권유를 받으셨다면 특히 주의하세요."
        )
        if ceiling:
            verdict = (
                f"이 개편안으로 줄어드는 세금은 보유액이 아무리 크셔도 연 "
                f"{fmt_krw(ceiling)}을 넘지 않습니다. 상품을 옮기실 이유가 되지 않는 "
                "크기입니다. '제도가 바뀌니 지금 갈아타라'는 권유를 받으셨다면 특히 주의하세요."
            )

    money_rows = (
        ("연간 세금 (첫 온전한 1년)", "annual_tax_first_full_year"),
        ("은퇴 기간 전체 세금", "lifetime_tax"),
        ("연금 개시 시점 잔액", "balance_at_pension_start"),
        ("시뮬레이션 종료 시점 잔액", "final_balance"),
    )
    comparison = [
        ComparisonRow(
            label=label,
            current=fmt_krw(getattr(current, field)),
            proposed=fmt_krw(getattr(proposed, field)),
            delta=_delta_label(getattr(proposed, field) - getattr(current, field)),
        )
        for label, field in money_rows
    ]
    score_delta = proposed.risk_score - current.risk_score
    comparison.append(
        ComparisonRow(
            label="은퇴 재무 안정도",
            current=f"{current.risk_score}점 ({current.risk_status})",
            proposed=f"{proposed.risk_score}점 ({proposed.risk_status})",
            delta="변화 없음" if score_delta == 0 else f"{score_delta:+d}점",
        )
    )

    notes = [
        "현행 제도와 개편안 모두에 ISA 세제를 적용해 **차이만** 비교했습니다. "
        "기본 분석 화면의 세금은 ISA 비과세 혜택을 반영하지 않은 보수적 값이라 "
        "여기 금액과 다를 수 있습니다.",
        "ISA 가 전체 금융자산에서 차지하는 비중이 은퇴 기간 내내 유지된다고 가정했습니다.",
        "세금은 실효세율 근사치입니다. 실제 부담액은 개인의 소득·재산 상황에 따라 달라집니다.",
    ]
    if ceiling:
        notes.insert(
            0,
            f"이 개편안은 비과세 한도만 올리는 내용이라, 절감액이 연 {fmt_krw(ceiling)}을 "
            "넘을 수 없습니다. ISA 를 아무리 많이 보유하셔도 마찬가지입니다.",
        )
    if fact.notice:
        notes.insert(0, fact.notice)

    return PolicyImpact(
        policy=fact,
        applicable=True,
        reason=(
            f"고객님은 ISA 에 {fmt_krw(profile.isa)}을(를) 보유하고 계셔서 "
            "이 개편안의 영향 범위 안에 있습니다."
        ),
        current=current,
        proposed=proposed,
        comparison=comparison,
        annual_tax_saving=annual_saving,
        lifetime_tax_saving=lifetime_saving,
        balance_delta_at_pension_start=(
            proposed.balance_at_pension_start - current.balance_at_pension_start
        ),
        final_balance_delta=proposed.final_balance - current.final_balance,
        depletion_age_delta=depletion_delta,
        risk_score_delta=score_delta,
        material=material,
        max_annual_saving=ceiling,
        headline=headline,
        verdict=verdict,
        notes=notes,
    )
