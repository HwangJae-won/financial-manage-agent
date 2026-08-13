"""처방 엔진 — "그래서 무엇을 하면 되는가" (기능 ⑤ 확장).

지금까지의 엔진은 진단만 했다. "만 69세에 금융자산이 고갈됩니다"까지 말하고
멈춘다. 사용자가 정작 알고 싶은 것은 그다음이다.

    "월 생활비를 241만원으로 줄이시면 만 95세까지 유지됩니다."

이 모듈은 기존 `simulate()` 를 **역방향으로** 쓴다. 레버를 하나씩 움직이며
목표(기본값: 시뮬레이션 종료 나이)를 달성하는 값을 찾는다. 엔진을 고치는 게
아니라 위에 얹는 구조라, 기존 계산 규약과 회귀 테스트가 그대로 유지된다.

레버 세 개:
  1. 월 생활비        — 줄일수록 유리. 단조라서 이분 탐색.
  2. 국민연금 개시 시기 — **단조가 아니다.** 미루면 수령액은 늘지만 그때까지
     자산을 더 헐어야 한다. 어느 쪽이 유리한지는 사람마다 뒤집히므로
     가능한 연령을 전부 계산해서 비교한다.
  3. 기타 월소득      — 늘릴수록 유리. 단조라서 이분 탐색.

조합 탐색(생활비도 줄이고 연금도 미루고)은 여기서 하지 않는다. 경우의 수가
곱으로 늘고, 무엇보다 "무엇을 포기할지"는 사용자가 정할 문제다. 이 모듈은
**레버별로 얼마가 필요한지**만 정확히 답하고, 조합은 대화 에이전트에게 맡긴다.

답은 만원 단위로 내림/올림한다. "2,413,271원으로 줄이세요"는 사람이 쓸 수 없는
답이다. 반올림 방향은 항상 목표를 달성하는 쪽(보수적)으로 잡는다.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from core.assumptions import Assumptions, load_assumptions
from core.cashflow import simulate
from core.formatting import fmt_krw, fmt_pct
from core.models import UserProfile

# 답을 만원 단위로 맞춘다. 시니어 사용자가 그대로 실행에 옮길 수 있어야 한다.
STEP = 10_000


class Lever(str, Enum):
    """움직일 수 있는 손잡이."""

    EXPENSE = "expense"
    PENSION_START = "pension_start"
    OTHER_INCOME = "other_income"


class Unit(str, Enum):
    """레버마다 단위가 다르다. 값만 보고 해석할 수 있어야 한다."""

    WON_PER_MONTH = "won_per_month"
    AGE = "age"


class Prescription(BaseModel):
    """레버 하나에 대한 처방.

    표시용 문자열과 원시 숫자를 함께 싣는다. 문자열은 화면이 그대로 쓰고,
    숫자는 대화 에이전트가 "그럼 생활비를 그만큼 줄이면?" 하고 다시 시뮬레이션을
    돌릴 때 쓴다. 화면이 문자열을 다시 파싱하게 만들면 안 된다.
    """

    lever: Lever
    label: str
    unit: Unit
    feasible: bool = Field(description="이 레버 하나만으로 목표를 달성할 수 있는가")
    improves: bool = Field(default=False, description="목표에 못 미쳐도 고갈 시점이 늦춰지는가")

    current_value: int
    required_value: Optional[int] = None
    change_value: Optional[int] = Field(
        default=None, description="현재값 대비 변화량. 부호가 방향을 나타낸다."
    )

    current_label: str
    required_label: Optional[str] = None
    change_label: Optional[str] = None

    depletion_age: Optional[int] = Field(
        default=None, description="이 처방을 적용했을 때의 고갈 나이. None이면 고갈되지 않음"
    )
    gained_years: int = Field(default=0, description="현재 계획 대비 늘어나는 연수")

    headline: str
    detail: str = ""
    caution: str = ""


class PrescriptionSet(BaseModel):
    """처방 묶음. 화면에 그대로 그릴 수 있는 형태로 만든다."""

    target_age: int
    baseline_depletion_age: Optional[int] = None
    already_safe: bool = False
    summary: str
    options: list[Prescription] = Field(default_factory=list)

    @property
    def feasible_options(self) -> list[Prescription]:
        return [o for o in self.options if o.feasible]


# --------------------------------------------------------------------------- #
# 탐색 도구
# --------------------------------------------------------------------------- #


def _outcome(profile: UserProfile, assumptions: Assumptions, **overrides) -> Optional[int]:
    """레버를 바꿔 시뮬레이션하고 고갈 나이를 돌려준다. None 이면 고갈되지 않음."""
    candidate = profile.model_copy(update=overrides)
    return simulate(candidate, assumptions=assumptions).depletion_age


def _survives(depletion_age: Optional[int], target_age: int) -> bool:
    return depletion_age is None or depletion_age >= target_age


def _highest_passing(
    test: Callable[[int], bool], low: int, high: int, step: int = STEP
) -> Optional[int]:
    """test 가 값이 작을수록 참인 단조 함수일 때, 참이 되는 최대값 (step 단위).

    생활비처럼 "줄일수록 유리한" 레버에 쓴다. 경계에서 반올림 방향은 항상
    참이 되는 쪽이다 — 아슬아슬하게 목표를 못 넘기는 답을 주면 안 된다.
    """
    if test(high):
        return high
    if not test(low):
        return None

    while high - low > step:
        mid = ((low + high) // 2 // step) * step
        if mid <= low:
            break
        if test(mid):
            low = mid
        else:
            high = mid
    return low


def _lowest_passing(
    test: Callable[[int], bool], low: int, high: int, step: int = STEP
) -> Optional[int]:
    """test 가 값이 클수록 참인 단조 함수일 때, 참이 되는 최소값 (step 단위)."""
    if test(low):
        return low
    if not test(high):
        return None

    while high - low > step:
        mid = ((low + high) // 2 // step) * step
        if mid <= low:
            mid = low + step
        if test(mid):
            high = mid
        else:
            low = mid
    return high


# --------------------------------------------------------------------------- #
# 레버별 처방
# --------------------------------------------------------------------------- #


def _prescribe_expense(
    profile: UserProfile, assumptions: Assumptions, target_age: int, baseline: Optional[int]
) -> Prescription:
    """월 생활비를 얼마로 줄이면 목표까지 버티는가."""
    current = profile.monthly_expense

    def test(value: int) -> bool:
        return _survives(_outcome(profile, assumptions, monthly_expense=value), target_age)

    # 현재보다 위쪽까지 탐색한다. 이미 목표를 넘기는 사람에게는 "얼마까지 쓰셔도
    # 되는가"가 답이다. "그대로 유지하세요"는 아무것도 알려주지 않는다.
    required = _highest_passing(test, STEP, max(current * 3, current + STEP))
    base = Prescription(
        lever=Lever.EXPENSE,
        label="월 생활비 줄이기",
        unit=Unit.WON_PER_MONTH,
        feasible=False,
        current_value=current,
        current_label=f"현재 월 {fmt_krw(current)}",
        headline="",
    )

    if required is None:
        base.headline = (
            "생활비를 줄이는 것만으로는 목표를 맞추기 어렵습니다. "
            "다른 방법과 함께 보셔야 합니다."
        )
        base.detail = "생활비를 크게 낮춰도 목표 나이까지 자산이 유지되지 않습니다."
        return base

    if required >= current:
        room = required - current
        base.feasible = True
        base.required_value = required
        base.change_value = room
        base.required_label = f"월 {fmt_krw(required)}까지 가능"
        base.headline = (
            f"지금 생활비로는 만 {target_age}세까지 여유가 있습니다. "
            f"월 {fmt_krw(required)}까지 쓰셔도 유지됩니다."
        )
        base.detail = (
            f"지금보다 월 {fmt_krw(room)}의 여유가 있다는 뜻입니다. "
            "다만 이 여유는 가정한 수익률이 그대로 실현될 때의 값입니다."
            if room > 0
            else "지금 생활비가 유지 가능한 상한에 가깝습니다."
        )
        return base

    cut = current - required
    depletion = _outcome(profile, assumptions, monthly_expense=required)

    base.feasible = True
    base.improves = True
    base.required_value = required
    base.change_value = -cut
    base.required_label = f"월 {fmt_krw(required)}"
    base.change_label = f"월 {fmt_krw(cut)} 줄이기 ({fmt_pct(cut / current)})"
    base.depletion_age = depletion
    base.gained_years = _gain(baseline, depletion, target_age)
    base.headline = (
        f"월 생활비를 {fmt_krw(required)}으로 줄이시면 "
        f"만 {target_age}세까지 유지됩니다."
    )
    base.detail = (
        f"지금보다 월 {fmt_krw(cut)} 적은 금액입니다. "
        "고정비(보험료·통신비·구독)부터 점검하시면 체감이 가장 적습니다."
    )
    return base


def _prescribe_pension_start(
    profile: UserProfile, assumptions: Assumptions, target_age: int, baseline: Optional[int]
) -> Prescription:
    """국민연금을 언제 받기 시작하는 것이 가장 유리한가.

    **이 레버는 단조가 아니다.** 미루면 수령액은 늘지만 그때까지 자산을 더 헐어야
    해서, 자산이 빠듯한 사람에게는 오히려 불리해진다. 이분 탐색을 쓰면 틀린 답이
    나오므로 가능한 연령을 전부 계산해 비교한다.
    """
    current = profile.national_pension_start_age
    low, high = assumptions.pension_start_age_range(profile.birth_year)

    # 조기노령연금은 소득이 있는 동안에는 받을 수 없다. 퇴직 전 연령은 후보에서 뺀다.
    low = max(low, profile.retirement_age)
    normal = assumptions.national_pension_start_age(profile.birth_year)

    base = Prescription(
        lever=Lever.PENSION_START,
        label="국민연금 받는 시기 조정",
        unit=Unit.AGE,
        feasible=False,
        current_value=current,
        current_label=f"현재 만 {current}세 개시 (법정 {normal}세)",
        headline="",
    )

    if profile.national_pension_monthly <= 0 or high < low:
        base.headline = "국민연금 수령 정보가 없어 이 방법은 계산하지 않았습니다."
        return base

    results = {
        age: _outcome(profile, assumptions, national_pension_start_age=age)
        for age in range(low, high + 1)
    }

    # 고갈되지 않는 쪽이 최우선, 그다음 고갈이 늦은 쪽, 같으면 법정 연령에 가까운 쪽.
    def rank(age: int) -> tuple[int, int, int]:
        depletion = results[age]
        return (1 if depletion is None else 0, depletion or 0, -abs(age - normal))

    best = max(results, key=rank)
    best_depletion = results[best]
    factor = assumptions.pension_amount_factor(profile.birth_year, best)
    monthly = int(round(profile.national_pension_monthly * factor))

    base.depletion_age = best_depletion
    base.feasible = _survives(best_depletion, target_age)
    base.gained_years = _gain(baseline, best_depletion, target_age)
    base.improves = base.gained_years > 0
    base.required_value = best
    base.change_value = best - current
    base.required_label = f"만 {best}세 개시 (월 {fmt_krw(monthly)})"

    if best == current:
        base.headline = (
            f"지금 계획(만 {current}세 개시)이 이미 가장 유리합니다. "
            "받는 시기를 바꾸실 이유는 없습니다."
        )
        base.detail = _pension_detail(results, assumptions, profile, normal)
        return base

    direction = "미루면" if best > current else "앞당기면"
    base.change_label = f"만 {current}세 → 만 {best}세"
    base.headline = (
        f"국민연금을 만 {best}세부터 받으시는 편이 유리합니다. "
        f"{abs(best - current)}년 {direction} 고갈 시점이 "
        f"{_depletion_phrase(best_depletion, target_age)}."
    )
    base.detail = _pension_detail(results, assumptions, profile, normal)

    if best < normal:
        cut = int(round((1 - factor) * 100))
        base.caution = (
            f"앞당겨 받으시면 수령액이 {cut}% 줄고, 그 감액은 **평생 유지됩니다.** "
            "오래 사실수록 총 수령액은 불리해지므로 건강 상태와 함께 판단하셔야 합니다."
        )
    elif best > normal:
        base.caution = (
            "미루는 동안에는 연금이 나오지 않습니다. 그 기간의 생활비를 "
            "다른 자산으로 감당할 수 있어야 합니다."
        )
    return base


def _prescribe_other_income(
    profile: UserProfile, assumptions: Assumptions, target_age: int, baseline: Optional[int]
) -> Prescription:
    """월 얼마를 더 벌면 목표까지 버티는가."""
    current = profile.other_monthly_income
    # 기타소득은 물가에 연동하지 않는 보수적 가정이라, 후반 생활비를 덮으려면
    # 생활비보다 큰 금액이 필요할 수 있다. 상한을 넉넉히 잡는다.
    ceiling = max(profile.monthly_expense * 3, current + STEP)

    def test(value: int) -> bool:
        return _survives(
            _outcome(profile, assumptions, other_monthly_income=value), target_age
        )

    required = _lowest_passing(test, current, ceiling)
    base = Prescription(
        lever=Lever.OTHER_INCOME,
        label="추가 수입 만들기",
        unit=Unit.WON_PER_MONTH,
        feasible=False,
        current_value=current,
        current_label=(
            f"현재 월 {fmt_krw(current)}" if current > 0 else "현재 추가 수입 없음"
        ),
        headline="",
    )

    if required is None:
        base.headline = (
            "추가 수입만으로 목표를 맞추려면 현실적이지 않은 금액이 필요합니다."
        )
        return base

    if required <= current:
        base.feasible = True
        base.required_value = current
        base.change_value = 0
        base.required_label = f"월 {fmt_krw(current)}"
        base.headline = "지금 수입을 그대로 유지하셔도 목표를 넘깁니다."
        return base

    need = required - current
    depletion = _outcome(profile, assumptions, other_monthly_income=required)

    base.feasible = True
    base.improves = True
    base.required_value = required
    base.change_value = need
    base.required_label = f"월 {fmt_krw(required)}"
    base.change_label = f"월 {fmt_krw(need)} 더 벌기"
    base.depletion_age = depletion
    base.gained_years = _gain(baseline, depletion, target_age)
    base.headline = (
        f"월 {fmt_krw(need)}의 수입을 더 만드시면 만 {target_age}세까지 유지됩니다."
    )
    base.detail = (
        "임대소득, 재취업, 시간제 근로 모두 해당합니다. "
        "이 계산은 추가 수입이 물가에 연동되지 않는다고 보므로 실제보다 보수적입니다."
    )
    return base


# --------------------------------------------------------------------------- #
# 문구
# --------------------------------------------------------------------------- #


def _gain(baseline: Optional[int], after: Optional[int], target_age: int) -> int:
    """현재 계획 대비 늘어나는 연수. 고갈되지 않으면 목표 나이까지로 본다."""
    before = baseline if baseline is not None else target_age
    now = after if after is not None else target_age
    return max(0, now - before)


def _depletion_phrase(depletion: Optional[int], target_age: int) -> str:
    if depletion is None:
        return f"사라져 만 {target_age}세까지 유지됩니다"
    return f"만 {depletion}세로 늦춰집니다"


def _pension_detail(
    results: dict[int, Optional[int]],
    assumptions: Assumptions,
    profile: UserProfile,
    normal: int,
) -> str:
    """연령별 결과를 한 줄로 풀어 쓴다. 왜 그 나이가 최선인지 보이게 하려는 것이다."""
    parts = []
    for age in sorted(results):
        factor = assumptions.pension_amount_factor(profile.birth_year, age)
        monthly = int(round(profile.national_pension_monthly * factor))
        depletion = results[age]
        outcome = "고갈 없음" if depletion is None else f"만 {depletion}세 고갈"
        mark = " (법정)" if age == normal else ""
        parts.append(f"만 {age}세{mark} 월 {fmt_krw(monthly)} → {outcome}")
    return " / ".join(parts)


# --------------------------------------------------------------------------- #
# 진입점
# --------------------------------------------------------------------------- #


def prescribe(
    profile: UserProfile,
    *,
    target_age: Optional[int] = None,
    assumptions: Optional[Assumptions] = None,
) -> PrescriptionSet:
    """목표 나이까지 자산을 유지하려면 무엇을 얼마나 바꿔야 하는지 계산한다.

    Args:
        profile: 사용자 프로파일.
        target_age: 목표 나이. 생략하면 시뮬레이션 종료 나이(기본 95세).
    """
    assumptions = assumptions or load_assumptions()
    target_age = target_age or assumptions.macro.horizon_age

    baseline = simulate(profile, assumptions=assumptions).depletion_age
    already_safe = _survives(baseline, target_age)

    options = [
        _prescribe_expense(profile, assumptions, target_age, baseline),
        _prescribe_pension_start(profile, assumptions, target_age, baseline),
        _prescribe_other_income(profile, assumptions, target_age, baseline),
    ]
    # 달성 가능한 것부터, 그다음 개선폭이 큰 것부터 보여준다.
    options.sort(key=lambda o: (o.feasible, o.gained_years), reverse=True)

    if already_safe:
        summary = (
            f"지금 계획으로도 만 {target_age}세까지 자산이 유지됩니다. "
            "아래는 여유를 더 확보하고 싶으실 때의 선택지입니다."
        )
    elif any(o.feasible for o in options):
        summary = (
            f"지금 계획으로는 만 {baseline}세에 금융자산이 바닥납니다. "
            f"만 {target_age}세까지 유지하시려면 아래 중 **하나만** 하셔도 됩니다."
        )
    else:
        summary = (
            f"지금 계획으로는 만 {baseline}세에 금융자산이 바닥납니다. "
            "한 가지만 바꿔서는 목표를 맞추기 어렵고, 두 가지 이상을 함께 조정하셔야 합니다."
        )

    return PrescriptionSet(
        target_age=target_age,
        baseline_depletion_age=baseline,
        already_safe=already_safe,
        summary=summary,
        options=options,
    )
