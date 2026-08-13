"""프로파일링 질문·슬롯 정의 (기능 ① 의 뼈대).

**흐름은 결정론적으로, 이해만 LLM 으로.** 질문 문장과 순서는 여기 고정되어 있고,
LLM 은 사용자의 답을 구조화하는 일만 한다. 그래야 대화가 예측 가능하고,
데모 중에 엉뚱한 질문으로 새지 않으며, 테스트할 수 있다.

질문 하나가 슬롯 여러 개를 채울 수 있다("퇴직 시점" → 연도 + 월).
사용자가 묻지 않은 정보를 먼저 말해도(“퇴직금 2억이고 예금 5천”) 한 번에 반영된다.

국민연금 수급 개시 연령은 묻지 않는다 — 출생연도에서 법정 스케줄로 결정된다.
질문을 하나라도 줄이는 것이 시니어 대상 서비스에서는 그 자체로 가치다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from core.models import RiskTolerance


class Priority(IntEnum):
    """질문의 중요도. 값이 클수록 먼저 묻는다."""

    REQUIRED = 3  # 없으면 시뮬레이션 자체가 불가능
    IMPORTANT = 2  # 결과가 크게 달라짐
    OPTIONAL = 1  # 기본값으로 둬도 무방


@dataclass(frozen=True)
class Question:
    """사용자에게 던지는 질문 하나."""

    key: str
    text: str
    fills: tuple[str, ...]
    priority: Priority
    hint: str = ""

    def is_answered(self, slots: dict[str, Any]) -> bool:
        """이 질문이 채우려는 슬롯 중 하나라도 값이 있으면 답변된 것으로 본다.

        '퇴직 시점'에서 월을 안 밝혀도 연도만 알면 넘어갈 수 있어야 한다
        (월은 기본값으로 채운다).
        """
        return any(slots.get(field) is not None for field in self.fills)


QUESTIONS: tuple[Question, ...] = (
    Question(
        key="birth",
        text="먼저 연세를 확인하겠습니다. 몇 년생이신가요?",
        fills=("birth_year", "birth_month"),
        priority=Priority.REQUIRED,
        hint="예: 1966년생이요 / 66년 12월생입니다",
    ),
    Question(
        key="retirement",
        text="퇴직은 언제 하셨거나, 언제 하실 예정인가요?",
        fills=("retirement_year", "retirement_month"),
        priority=Priority.REQUIRED,
        hint="예: 올해 12월이요 / 작년에 그만뒀습니다",
    ),
    Question(
        key="expense",
        text="한 달 생활비로 어느 정도 쓰고 계신가요?",
        fills=("monthly_expense",),
        priority=Priority.REQUIRED,
        hint="예: 300만원 정도요",
    ),
    Question(
        key="severance",
        text=(
            "퇴직금은 얼마나 받으셨거나 받으실 것 같으세요?\n"
            "몇 년 근무하셨는지도 함께 말씀해 주시면 세금까지 계산해 드립니다."
        ),
        # 질문 하나로 슬롯 두 개를 채운다. 근속연수는 퇴직소득세의 핵심 변수인데
        # (근속연수공제) 질문을 하나 더 늘리는 것은 시니어 대상 서비스에서 비용이 크다.
        fills=("severance_pay", "years_employed"),
        priority=Priority.IMPORTANT,
        hint="예: 2억 정도 나올 것 같아요, 32년 다녔습니다 / 없습니다",
    ),
    Question(
        key="savings",
        text="예금이나 적금은 얼마나 가지고 계신가요?",
        fills=("cash_savings",),
        priority=Priority.IMPORTANT,
        hint="예: 5천만원쯤 있습니다",
    ),
    Question(
        key="pension",
        text=(
            "국민연금은 매달 얼마나 받으실 것 같으세요? "
            "국민연금공단 홈페이지나 앱에서 예상 수령액을 확인하실 수 있습니다.\n"
            "몇 년 동안 납부하셨는지도 함께 말씀해 주시면 더 내실지까지 계산해 드립니다."
        ),
        # 퇴직금 질문과 같은 방식이다. 가입월수는 임의계속가입·추납 계산의 핵심 변수인데
        # (119개월과 120개월이 '연금 0원'과 '평생 연금'을 가른다) 질문을 하나 더
        # 늘리는 것은 시니어 대상 서비스에서 비용이 크다.
        fills=("national_pension_monthly", "national_pension_months"),
        priority=Priority.IMPORTANT,
        hint="예: 월 150만원 정도라고 나왔어요, 32년 넣었습니다",
    ),
    Question(
        key="investments",
        text="주식이나 펀드, ISA 계좌도 가지고 계신가요?",
        fills=("equity", "isa"),
        priority=Priority.OPTIONAL,
        hint="예: 주식 1억이랑 ISA 3천 있습니다 / 없어요",
    ),
    Question(
        key="retirement_pension",
        text="퇴직연금이나 개인연금(IRP)에 쌓아두신 돈이 있으신가요?",
        fills=("pension_dc",),
        priority=Priority.OPTIONAL,
        hint="예: IRP에 1억 있습니다 / 없습니다",
    ),
    Question(
        key="real_estate",
        text="살고 계신 집이나 다른 부동산이 있으신가요? 대략적인 시세로 말씀해 주세요.",
        fills=("real_estate",),
        priority=Priority.OPTIONAL,
        hint="예: 아파트 한 채 5억 정도 합니다",
    ),
    Question(
        key="other_income",
        text="임대료나 일하시는 수입 등, 매달 들어오는 다른 돈이 있으신가요?",
        fills=("other_monthly_income",),
        priority=Priority.OPTIONAL,
        hint="예: 월세로 80만원 받습니다 / 없어요",
    ),
    Question(
        key="debt",
        text="대출이나 갚아야 할 빚이 있으신가요?",
        fills=("debt",),
        priority=Priority.OPTIONAL,
        hint="예: 주택담보대출 5천만원 남았습니다 / 없습니다",
    ),
    Question(
        key="risk",
        text="투자에 대해서는 어떤 편이신가요? "
        "원금을 지키는 쪽이 편하신지, 손실을 감수하더라도 수익을 노리시는 편인지요.",
        fills=("risk_tolerance",),
        priority=Priority.OPTIONAL,
        hint="예: 원금 지키는 게 우선입니다",
    ),
)

QUESTION_BY_KEY = {q.key: q for q in QUESTIONS}

# 모든 슬롯 (UserProfile 필드명)
ALL_SLOTS: tuple[str, ...] = tuple(
    field for question in QUESTIONS for field in question.fills
)

# 사용자가 답하지 않았을 때 채워 넣는 값.
# 이 값으로 채워진 슬롯은 UserProfile.assumed_fields 에 기록되어 화면에 '가정한 값'으로 표시된다.
DEFAULTS: dict[str, Any] = {
    "birth_month": 1,
    "retirement_month": 12,
    "severance_pay": 0,
    "years_employed": 0,  # 0 이면 '모름' — 퇴직소득세는 추측하지 않고 계산을 보류한다
    "cash_savings": 0,
    "national_pension_monthly": 0,
    "national_pension_months": 0,  # 0 이면 '모름' — 임의계속가입·추납은 계산을 보류한다
    "equity": 0,
    "isa": 0,
    "pension_dc": 0,
    "real_estate": 0,
    "other_monthly_income": 0,
    "debt": 0,
    "risk_tolerance": RiskTolerance.MODERATE.value,
}

# 이 슬롯들이 없으면 시뮬레이션을 만들 수 없다.
MANDATORY_SLOTS: tuple[str, ...] = ("birth_year", "retirement_year", "monthly_expense")


def next_question(slots: dict[str, Any], asked: set[str]) -> Question | None:
    """다음에 물을 질문. 우선순위 높은 순, 같은 우선순위면 정의된 순서.

    이미 답을 얻었거나 이미 물어본 질문은 건너뛴다(같은 질문 반복 방지).
    """
    candidates = [
        q
        for q in QUESTIONS
        if q.key not in asked and not q.is_answered(slots)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda q: (q.priority, -QUESTIONS.index(q)))


def missing_mandatory(slots: dict[str, Any]) -> list[str]:
    """아직 없는 필수 슬롯."""
    return [key for key in MANDATORY_SLOTS if slots.get(key) is None]


def question_for_slot(slot: str) -> Question | None:
    """해당 슬롯을 채우는 질문."""
    return next((q for q in QUESTIONS if slot in q.fills), None)


def is_ready(slots: dict[str, Any]) -> bool:
    """시뮬레이션을 돌릴 수 있는 최소 조건을 갖췄는가."""
    return not missing_mandatory(slots)


def all_important_answered(slots: dict[str, Any]) -> bool:
    """필수와 중요 질문에 모두 답했는가 — '이제 결과를 봐도 됩니다'의 기준."""
    return all(
        q.is_answered(slots)
        for q in QUESTIONS
        if q.priority >= Priority.IMPORTANT
    )


def extraction_schema() -> dict[str, Any]:
    """LLM 구조화 추출용 JSON 스키마.

    모든 필드가 nullable 이다. 사용자가 말하지 않은 것은 null 로 와야 하며,
    0 으로 오면 '없다'는 뜻이 되어 잘못된 시뮬레이션이 나간다.
    """
    money = {"type": ["integer", "null"], "description": "원 단위 정수. 언급 없으면 null"}
    properties: dict[str, Any] = {
        "birth_year": {"type": ["integer", "null"], "description": "4자리 출생연도"},
        "birth_month": {"type": ["integer", "null"], "description": "1-12"},
        "retirement_year": {"type": ["integer", "null"], "description": "4자리 퇴직 연도"},
        "retirement_month": {"type": ["integer", "null"], "description": "1-12"},
        "monthly_expense": dict(money, description="월 생활비(원)"),
        "severance_pay": dict(money, description="퇴직금(원)"),
        "years_employed": {
            "type": ["integer", "null"],
            "description": "재직(근속) 기간(년). 언급 없으면 null",
        },
        "cash_savings": dict(money, description="예금·적금(원)"),
        "national_pension_monthly": dict(money, description="국민연금 예상 월 수령액(원)"),
        "national_pension_months": {
            "type": ["integer", "null"],
            "description": (
                "국민연금 가입(납부) 기간을 **개월 수**로. '32년 넣었다' 면 384, "
                "'10년' 이면 120. 언급 없으면 null"
            ),
        },
        "last_monthly_salary": dict(
            money, description="퇴직 전 월 급여·보수월액(원). 언급 없으면 null"
        ),
        "equity": dict(money, description="주식·ETF·펀드 평가액(원)"),
        "isa": dict(money, description="ISA 계좌 평가액(원)"),
        "pension_dc": dict(money, description="퇴직연금·IRP 적립금(원)"),
        "real_estate": dict(money, description="부동산 시세(원)"),
        "other_monthly_income": dict(money, description="임대·근로 등 월 기타소득(원)"),
        "debt": dict(money, description="총 부채(원)"),
        "risk_tolerance": {
            "type": ["string", "null"],
            "enum": [*(r.value for r in RiskTolerance), None],
            "description": "투자성향",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
