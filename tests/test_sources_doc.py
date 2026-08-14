"""출처 문서가 코드와 어긋나지 않는지 지킨다.

`docs/sources.md` 는 "그 숫자 근거가 뭐냐"에 답하려고 만든 문서다. 그런데 문서가
코드보다 낡으면 **없느니만 못하다** — 틀린 근거를 대게 되기 때문이다. 가정값을
바꿨는데 문서를 안 고치면 여기서 걸린다.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "sources.md"
ASSUMPTIONS = pathlib.Path(__file__).resolve().parent.parent / "data" / "assumptions.yaml"


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def values() -> dict:
    return yaml.safe_load(ASSUMPTIONS.read_text(encoding="utf-8"))


def dig(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        node = node[part]
    return node


# 문서에 **글자 그대로** 적힌 값들. 코드가 바뀌면 여기서 깨진다.
QUOTED = [
    ("national_pension.min_months", "120개월"),
    ("national_pension.voluntary_max_age", "65세"),
    ("national_pension.income_base_max", "617만원"),
    ("national_pension.income_base_min", "39만원"),
    ("national_pension.max_catchup_months", "119개월"),
    ("national_pension.earned_income.a_value", "3,193,511원"),
    ("national_pension.earned_income.exempt_threshold", "200만원"),
    ("health_insurance.local.health_rate", "7.09%"),
    ("health_insurance.local.long_term_care_rate", "12.95%"),
    ("health_insurance.local.minimum_monthly", "19,780원"),
    ("health_insurance.dependent.income_limit", "2,000만원"),
    ("macro.inflation_rate", "2.3%"),
    ("macro.horizon_age", "95"),
    ("tax.financial_income_rate", "15.4%"),
]


@pytest.mark.parametrize("path,quoted", QUOTED)
def test_quoted_values_appear_in_the_doc(doc: str, values: dict, path: str, quoted: str):
    """코드에 있는 값이 문서에 그 표기로 적혀 있어야 한다."""
    dig(values, path)  # 경로 자체가 살아 있는지도 함께 확인
    assert quoted in doc, f"{path} 의 값 '{quoted}' 이 문서에 없습니다"


def test_every_referenced_file_exists(doc: str):
    """문서가 가리키는 파일이 사라지면 근거를 못 댄다."""
    import re

    root = DOC.parent.parent
    for match in re.finditer(r"`((?:core|agents|data|scripts|storage|web)/[\w./]+)`", doc):
        assert (root / match.group(1)).exists(), match.group(1)


def test_the_stale_values_are_flagged(doc: str):
    """지금 낡은 것이 확인된 두 값은 문서에서 ❌ 로 표시되어야 한다."""
    assert "❌ **낡음**" in doc
    assert "6,590천원" in doc  # 고시 원문 그대로
    assert "410천원" in doc


def test_the_deliberate_gaps_are_listed(doc: str):
    """계산하지 않기로 한 것들이 이유와 함께 적혀 있어야 한다."""
    for item in ["재산분", "주택연금", "본인부담상한액", "소득세"]:
        assert item in doc, item


def test_example_data_is_called_out(doc: str):
    """policy_facts.yaml 이 예시 데이터라는 사실을 숨기면 안 된다."""
    assert "예시 데이터" in doc
    assert "9만 9,000원" in doc  # 여기 걸려 있는 대표 문장


def test_the_check_script_is_documented(doc: str):
    assert "scripts/check_assumptions.py" in doc
    assert "자동 반영은 하지 않는다" in doc
