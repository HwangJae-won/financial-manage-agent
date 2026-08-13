"""금액·기간 표시 포맷.

시니어 사용자와 LLM 프롬프트 양쪽에서 쓰이므로, "1200000000" 같은 원시 숫자가
화면이나 프롬프트에 그대로 나가지 않게 여기를 거친다.
"""

from __future__ import annotations

EOK = 100_000_000  # 1억
MAN = 10_000  # 1만


def fmt_krw(amount: int | float) -> str:
    """원 단위 금액을 한국식 '억/만원' 표기로 변환한다.

    >>> fmt_krw(720_000_000)
    '7억 2,000만원'
    >>> fmt_krw(3_000_000)
    '300만원'
    >>> fmt_krw(0)
    '0원'
    """
    amount = int(round(amount))
    if amount == 0:
        return "0원"

    sign = "-" if amount < 0 else ""
    amount = abs(amount)

    eok, rest = divmod(amount, EOK)
    man = rest // MAN
    won = rest % MAN

    parts: list[str] = []
    if eok:
        parts.append(f"{eok}억")
    if man:
        parts.append(f"{man:,}만")
    # 100만 원 이상이면 만원 미만은 읽기만 방해하므로 버린다.
    # (예: 38,541,570 → "3,854만원"). 소액은 그대로 보여준다.
    if won and amount < 1_000_000:
        parts.append(f"{won:,}")

    body = " ".join(parts)
    suffix = "원" if not body.endswith("원") else ""
    return f"{sign}{body}{suffix}"


def fmt_months(months: int) -> str:
    """개월 수를 '4년 3개월' 형태로 변환한다."""
    if months <= 0:
        return "없음"
    years, rest = divmod(months, 12)
    if years and rest:
        return f"{years}년 {rest}개월"
    if years:
        return f"{years}년"
    return f"{rest}개월"


def fmt_years(years: float | None, *, never_text: str = "고갈되지 않음") -> str:
    """연수를 '6.2년' 형태로 변환한다. None이면 never_text."""
    if years is None:
        return never_text
    return f"{years:.1f}년"


def fmt_pct(ratio: float, *, precision: int = 1) -> str:
    """비율(0.18)을 퍼센트 문자열('18.0%')로 변환한다."""
    return f"{ratio * 100:.{precision}f}%"


def has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는지.

    괄호나 숫자로 끝나는 말이 많아서(예: "기준소득월액(퇴직 전 월 급여)") 뒤에서부터
    **한글 음절을 찾아** 판정한다. 마지막 글자만 보면 조사가 엉뚱하게 붙는다.
    """
    for char in reversed(word):
        if "가" <= char <= "힣":
            return (ord(char) - ord("가")) % 28 != 0
    return False


def fmt_euro(word: str) -> str:
    """조사 '으로/로'를 붙인다 — '임의계속가입으로', '둘 다로'.

    문구를 문자열 조립으로 만들다 보면 '둘 다으로' 같은 것이 화면에 나간다.
    시니어 사용자에게 읽히는 문장이라 여기서 한 번에 처리한다.
    """
    return word + ("으로" if has_final_consonant(word) else "로")


def fmt_eul(word: str) -> str:
    """조사 '을/를'을 붙인다 — '가입월수를', '예상 월 수령액을'."""
    return word + ("을" if has_final_consonant(word) else "를")
