"""한국어 금액 표현 파서.

"2억 5천", "300만원", "1억2천3백만" 같은 표현을 원 단위 정수로 바꾼다.

쓰임새는 두 가지다:
  1. MockClient 가 키 없이도 그럴듯한 추출을 하게 한다 (데모·테스트 현실성).
  2. LLM 추출 결과의 교차검증 — 두 값이 다르면 사용자에게 되물을 수 있다.

**구어체 만 단위 생략**: 자산 대화에서 "예금 5천"은 5,000원이 아니라 5천만원을
뜻한다. 억/만 없이 천·백·십으로 끝나면 만 단위를 생략한 것으로 본다.
단, "5천원"처럼 '원'이 명시되면 액면 그대로 읽는다.

한계: "생활비 300"처럼 단위가 아예 없는 표현은 판단하지 않는다(None 반환).
이런 모호함은 문맥을 아는 LLM 이 처리하고, 이 파서는 검증에만 쓴다.
"""

from __future__ import annotations

import re
from typing import Optional

MAJOR_UNITS = {"억": 100_000_000, "만": 10_000}
SUB_UNITS = {"천": 1_000, "백": 100, "십": 10}

# 한글 수사
_SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}

# 한글 수사는 **단위가 뒤따를 때만** 숫자로 읽는다.
# "예금이 2천만원"의 조사 '이'를 2로 오독하면 20,020,000 같은 값이 나온다.
_TOKEN = re.compile(
    r"(\d+(?:\.\d+)?|[일이삼사오육칠팔구](?=\s*[억만천백십]))?\s*([억만천백십])?"
)


def _normalize(text: str) -> str:
    text = text.strip()
    for noise in ("정도", "쯤", "가량", "약", "한", ",", " "):
        text = text.replace(noise, "")
    return text


def parse_korean_amount(text: str) -> Optional[int]:
    """한국어 금액 표현을 원 단위 정수로 바꾼다. 해석할 수 없으면 None."""
    if not text:
        return None

    explicit_won = "원" in text
    s = _normalize(text).replace("원", "")
    if not s or not re.search(r"[\d일이삼사오육칠팔구]", s):
        return None

    result = 0
    section = 0.0
    leftover_from_subunit = False
    matched_any = False

    pos = 0
    while pos < len(s):
        match = _TOKEN.match(s, pos)
        if not match or match.end() == pos:
            pos += 1
            continue
        pos = match.end()

        raw_num, unit = match.group(1), match.group(2)
        if raw_num is None and unit is None:
            continue
        matched_any = True

        if raw_num is None:
            value = None
        elif raw_num in _SINO:
            value = float(_SINO[raw_num])
        else:
            value = float(raw_num)

        if unit in MAJOR_UNITS:
            # 앞에 누적된 section 이 있으면 그것이 이 단위의 계수다.
            # "1억2천3백만" 에서 만 앞의 2300 이 그 예.
            coefficient = section + (value if value is not None else (0 if section else 1))
            result += coefficient * MAJOR_UNITS[unit]
            section = 0.0
            leftover_from_subunit = False
        elif unit in SUB_UNITS:
            section += (value if value is not None else 1) * SUB_UNITS[unit]
            leftover_from_subunit = True
        else:
            section += value if value is not None else 0
            leftover_from_subunit = False

    if not matched_any:
        return None

    # 구어체 만 단위 생략: "2억 5천" → 2억 5천만
    if section and leftover_from_subunit and not explicit_won:
        section *= 10_000

    return int(round(result + section))


def parse_year(text: str, *, current_year: int) -> Optional[int]:
    """연도 표현을 4자리 연도로 바꾼다. '올해', '내년', '2026년', '66년생' 지원."""
    s = text.strip()

    if "올해" in s or "금년" in s:
        return current_year
    if "내년" in s:
        return current_year + 1
    if "작년" in s or "지난해" in s:
        return current_year - 1
    if "내후년" in s:
        return current_year + 2

    four = re.search(r"(19|20)\d{2}", s)
    if four:
        return int(four.group(0))

    # "66년생" 같은 두 자리 표기 — 생년으로만 쓰인다고 보고 1900년대로 해석
    two = re.search(r"(?<!\d)(\d{2})\s*년\s*생", s)
    if two:
        return 1900 + int(two.group(1))

    return None


def parse_month(text: str) -> Optional[int]:
    """'12월', '올해 말' 같은 표현을 1~12 로 바꾼다."""
    s = text.strip()
    match = re.search(r"(\d{1,2})\s*월", s)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            return month
    if "연말" in s or "말" in s:
        return 12
    if "연초" in s or "초" in s:
        return 1
    return None
