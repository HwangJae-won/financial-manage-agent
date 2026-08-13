"""금융사기·불완전판매 탐지 (기능 ⑦).

사용자가 받은 투자 권유 문자·카톡·상품 설명을 붙여넣으면 위험 신호를 짚어준다.

세 겹으로 본다:
  1. **규칙** — 레드플래그를 낱말로 잡는다. LLM 없이 동작하고, 왜 걸렸는지 그대로
     보여줄 수 있다. 이 서비스에서 근거를 댈 수 있다는 건 타협 불가다.
  2. **정책 대조** — "제도가 바뀌었으니 갈아타라"는 주장을 공식 팩트시트와 맞춰본다.
     발표된 개편안과 시행된 제도를 구분하는 것이 기획서의 핵심 차별점이다.
  3. **프로필 연결** — 사용자가 실제로 보유한 자산과 엮어 설명한다.
     "ISA에 5,000만원을 운용 중이시라 이 주장의 사실 여부가 중요합니다."

**중요**: 이 기능은 사기 여부를 확정하지 않는다. "확인이 필요한 신호"를 알릴 뿐이다.
정상적인 상품 안내도 일부 표현이 걸릴 수 있으므로 문구를 단정적으로 쓰지 않는다.
"""

from __future__ import annotations

import functools
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from agents.llm import LLMClient, get_client
from core.formatting import fmt_krw
from core.models import UserProfile

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud_rules.yaml"
POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "policy_facts.yaml"


class RiskLevel(str, Enum):
    LOW = "낮음"
    CAUTION = "주의"
    HIGH = "위험"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Signal(BaseModel):
    """탐지된 위험 신호 하나."""

    key: str
    label: str
    severity: Severity
    why: str
    advice: str
    evidence: list[str] = Field(default_factory=list, description="문장에서 걸린 표현")


class PolicyCheck(BaseModel):
    """메시지가 언급한 제도와 공식 정보의 대조 결과."""

    topic: str
    title: str
    status: str = Field(description="발표 / 입법예고 / 국회통과 / 공포 / 시행")
    effective_date: Optional[str]
    source: str
    summary: str
    caution: str

    @property
    def is_confirmed(self) -> bool:
        return self.status == "시행"


class FraudAssessment(BaseModel):
    """탐지 결과."""

    risk_level: RiskLevel
    signals: list[Signal]
    policy_checks: list[PolicyCheck] = Field(default_factory=list)
    profile_notes: list[str] = Field(
        default_factory=list, description="사용자 보유 자산과 연결한 설명"
    )
    summary: str = ""
    checked_text_length: int = 0

    @property
    def has_high_severity(self) -> bool:
        return any(s.severity is Severity.HIGH for s in self.signals)

    @property
    def unconfirmed_policies(self) -> list[PolicyCheck]:
        """아직 시행되지 않은 제도 — 이걸 근거로 든 권유는 특히 주의해야 한다."""
        return [p for p in self.policy_checks if not p.is_confirmed]


# --------------------------------------------------------------------------- #
# 규칙 로딩
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=2)
def load_rules(path: str | Path = RULES_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=2)
def load_policy_facts(path: str | Path = POLICY_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("facts", [])


def _normalize(text: str) -> str:
    """띄어쓰기 차이를 흡수한다. '원금 보장'과 '원금보장'을 다르게 보면 안 된다."""
    return re.sub(r"\s+", "", text)


def _hits(text: str, keywords: list[str]) -> list[str]:
    """걸린 표현을 돌려준다.

    낱말은 띄어쓰기를 무시하고 비교한다("원금 보장" == "원금보장").
    `re:` 로 시작하면 정규식으로 본다 — 수익률처럼 숫자가 끼는 표현은
    낱말 목록으로는 잡을 수 없기 때문이다("월 %"로는 "월 3%"를 못 잡는다).
    """
    flat = _normalize(text)
    found: list[str] = []

    for keyword in keywords:
        if keyword.startswith("re:"):
            for match in re.finditer(keyword[3:], text):
                found.append(match.group(0).strip())
        elif _normalize(keyword) and _normalize(keyword) in flat:
            found.append(keyword)

    return found


# --------------------------------------------------------------------------- #
# 1) 규칙 기반 탐지
# --------------------------------------------------------------------------- #


def detect_signals(text: str, rules: Optional[dict[str, Any]] = None) -> list[Signal]:
    """레드플래그 규칙을 적용한다. LLM 없이 동작한다."""
    rules = rules or load_rules()
    signals: list[Signal] = []

    for rule in rules.get("rules", []):
        match = rule.get("match") or {}
        evidence: list[str] = []

        if "all" in match:
            groups = match["all"]
            group_hits = [_hits(text, group) for group in groups]
            if not all(group_hits):
                continue
            for hits in group_hits:
                evidence.extend(hits)
        elif "any" in match:
            hits = _hits(text, match["any"])
            if not hits:
                continue
            evidence.extend(hits)
        else:
            continue

        signals.append(
            Signal(
                key=rule["key"],
                label=rule["label"],
                severity=Severity(rule.get("severity", "medium")),
                why=rule["why"].strip(),
                advice=rule["advice"].strip(),
                evidence=sorted(set(evidence)),
            )
        )

    return signals


def assess_risk(signals: list[Signal]) -> RiskLevel:
    """신호를 종합해 위험도를 매긴다.

    high 하나면 위험. medium 은 둘 이상 겹칠 때 위험으로 본다 —
    정상적인 상품 안내도 표현 하나쯤은 걸릴 수 있기 때문이다.
    """
    high = sum(1 for s in signals if s.severity is Severity.HIGH)
    medium = sum(1 for s in signals if s.severity is Severity.MEDIUM)

    if high >= 1 or medium >= 2:
        return RiskLevel.HIGH
    if medium == 1 or signals:
        return RiskLevel.CAUTION
    return RiskLevel.LOW


# --------------------------------------------------------------------------- #
# 2) 정책 대조
# --------------------------------------------------------------------------- #


def check_policies(text: str, rules: Optional[dict[str, Any]] = None) -> list[PolicyCheck]:
    """메시지가 언급한 제도를 공식 팩트시트와 맞춰본다."""
    rules = rules or load_rules()
    topics = rules.get("policy_topics", {})

    mentioned = {
        topic for topic, keywords in topics.items() if _hits(text, keywords)
    }
    if not mentioned:
        return []

    return [
        PolicyCheck(
            topic=fact["topic"],
            title=fact["title"],
            status=fact["status"],
            effective_date=fact.get("effective_date"),
            source=fact["source"],
            summary=fact["summary"].strip(),
            caution=fact["caution"].strip(),
        )
        for fact in load_policy_facts()
        if fact["topic"] in mentioned
    ]


# --------------------------------------------------------------------------- #
# 3) 프로필 연결
# --------------------------------------------------------------------------- #

# 어떤 낱말이 나오면 사용자의 어떤 자산과 엮어 설명할지
_PROFILE_LINKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("isa", ("ISA", "아이에스에이"), "ISA"),
    ("pension_dc", ("IRP", "퇴직연금", "개인연금", "연금저축"), "퇴직·개인연금"),
    ("equity", ("주식", "펀드", "ETF"), "주식·펀드"),
    ("cash_savings", ("예금", "적금", "정기예금"), "예금·적금"),
    ("severance_pay", ("퇴직금", "명예퇴직금"), "퇴직금"),
)


def link_to_profile(text: str, profile: UserProfile) -> list[str]:
    """메시지가 언급한 상품을 사용자의 실제 보유 자산과 엮는다.

    "고객님은 ISA에 5,000만원을 운용하고 계셔서 이 주장이 사실인지 중요합니다."
    — 일반 사기 경보와 이 서비스를 가르는 지점이다.
    """
    notes: list[str] = []

    for field, keywords, label in _PROFILE_LINKS:
        if not _hits(text, list(keywords)):
            continue
        amount = getattr(profile, field, 0) or 0
        if amount > 0:
            notes.append(
                f"고객님은 {label}에 {fmt_krw(amount)}을(를) 보유하고 계십니다. "
                "이 권유가 사실인지에 따라 실제로 영향을 받으실 수 있으므로 "
                "반드시 공식 창구에서 확인하세요."
            )
        else:
            notes.append(
                f"고객님은 현재 {label}을(를) 보유하고 계시지 않습니다. "
                "보유하지 않은 상품을 근거로 한 권유는 특히 주의가 필요합니다."
            )

    if profile.liquid_assets > 0 and _hits(text, ["이체", "송금", "입금", "계좌"]):
        notes.append(
            f"바로 인출 가능한 자산이 {fmt_krw(profile.liquid_assets)} 있으십니다. "
            "이체를 요구하는 연락은 금액과 관계없이 먼저 은행에 확인하세요."
        )

    return notes


# --------------------------------------------------------------------------- #
# 요약
# --------------------------------------------------------------------------- #


def build_summary(
    risk: RiskLevel, signals: list[Signal], policies: list[PolicyCheck]
) -> str:
    """LLM 없이도 화면에 띄울 수 있는 요약."""
    if not signals:
        return (
            "눈에 띄는 위험 신호는 발견되지 않았습니다. "
            "다만 신호가 없다고 안전이 보장되는 것은 아닙니다. "
            "가입 전에는 거래하시는 금융회사에 직접 확인하세요."
        )

    labels = ", ".join(s.label for s in signals[:3])
    head = {
        RiskLevel.HIGH: "금융사기나 불완전판매가 의심됩니다.",
        RiskLevel.CAUTION: "확인이 필요한 신호가 있습니다.",
        RiskLevel.LOW: "가볍게 확인해 보시면 좋겠습니다.",
    }[risk]

    parts = [f"{head} 확인된 신호는 {labels}입니다."]

    unconfirmed = [p for p in policies if not p.is_confirmed]
    if unconfirmed:
        titles = ", ".join(p.title for p in unconfirmed)
        parts.append(
            f"언급된 제도({titles})는 아직 확정되지 않은 상태입니다. "
            "확정되지 않은 제도를 근거로 한 권유는 사실 확인이 먼저입니다."
        )

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# 진입점
# --------------------------------------------------------------------------- #


def analyze_message(
    text: str,
    *,
    profile: Optional[UserProfile] = None,
    client: Optional[LLMClient] = None,
    rules: Optional[dict[str, Any]] = None,
) -> FraudAssessment:
    """받은 메시지를 분석한다.

    client 는 현재 쓰이지 않는다 — 규칙과 팩트시트만으로 근거 있는 판단이 나오고,
    LLM 을 끼우면 "왜 위험한지"를 설명할 수 없게 되기 때문이다. 문맥 이해가 필요한
    확장(우회 표현 탐지 등)은 이 자리에 덧붙인다.
    """
    signals = detect_signals(text, rules)
    policies = check_policies(text, rules)
    risk = assess_risk(signals)

    notes = link_to_profile(text, profile) if profile is not None else []

    return FraudAssessment(
        risk_level=risk,
        signals=signals,
        policy_checks=policies,
        profile_notes=notes,
        summary=build_summary(risk, signals, policies),
        checked_text_length=len(text),
    )
