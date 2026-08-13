"""Streamlit 앱 스모크 테스트.

데모 중에 앱이 예외로 죽는 것이 가장 치명적이므로, 실제로 스크립트를 실행해
예외 없이 끝까지 렌더링되는지 확인한다. Streamlit 의 AppTest 는 브라우저 없이
스크립트를 돌려주므로 CI 에서도 쓸 수 있다.

두 모드를 모두 검증한다: 대화(기본)와 직접 입력(데모 중 안전망).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.samples import DEMO_PROFILE

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

DEMO_SCRIPT = [
    "1966년 12월생입니다",
    "올해 12월에 퇴직할 예정이에요",
    "생활비는 한 300만원 정도 씁니다",
    "퇴직금은 2억 정도 나올 것 같아요",
    "예금이 2천만원 있습니다",
    "국민연금은 월 150만원 정도 나온다고 하더라고요",
    "주식이나 펀드는 없어요",
    "퇴직연금도 없습니다",
    "아파트가 한 채 있는데 5억 정도 합니다",
    "다른 수입은 없어요",
    "대출은 없습니다",
    "원금을 지키는 쪽이 편합니다",
]


def _fresh() -> "AppTest":
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def _form_mode() -> "AppTest":
    at = _fresh()
    at.radio[0].set_value("직접 입력").run()
    return at


@pytest.fixture(scope="module")
def form_app():
    return _form_mode()


@pytest.fixture(scope="module")
def chat_app():
    """대화를 끝까지 진행한 상태의 앱."""
    at = _fresh()
    for line in DEMO_SCRIPT:
        at.chat_input[0].set_value(line).run()
    return at


# --------------------------------------------------------------------------- #
# 공통
# --------------------------------------------------------------------------- #


def test_app_starts_without_exception():
    at = _fresh()
    assert not at.exception, at.exception


def test_conversation_is_the_default_mode():
    """데모의 첫 장면은 대화여야 한다."""
    at = _fresh()
    assert at.radio[0].value == "대화로 알아보기"
    assert at.chat_input


def test_llm_status_is_visible():
    """키 없이 mock 으로 도는 중이라는 사실이 화면에 드러나야 한다."""
    captions = " ".join(c.value for c in _fresh().caption)
    assert "LLM:" in captions


# --------------------------------------------------------------------------- #
# 직접 입력 모드
# --------------------------------------------------------------------------- #


def test_form_mode_runs_without_exception(form_app):
    assert not form_app.exception, form_app.exception


def test_headline_metrics_are_rendered(form_app):
    labels = [m.label for m in form_app.metric]
    for label in ["총자산", "소득공백기", "생활비 충당 가능", "금융자산 고갈"]:
        assert label in labels


def test_all_sections_are_present(form_app):
    headings = " ".join(h.value for h in form_app.subheader)
    for section in [
        "한눈에 보기",
        "나의 자산지도",
        "시간에 따른 자산 변화",
        "자산이 남아있을 확률",
        "은퇴 재무 안정도",
        "대응 시나리오 비교",
    ]:
        assert section in headings


def test_form_shows_the_demo_persona_numbers(form_app):
    values = {m.label: m.value for m in form_app.metric}
    assert values["총자산"] == "7억 2,000만원"
    assert values["소득공백기"] == "4년"
    assert values["생활비 충당 가능"] == "6.2년"
    assert values["금융자산 고갈"] == "만 69세"


def test_long_term_risk_is_warned_about(form_app):
    warnings = " ".join(w.value for w in form_app.warning)
    assert "금융자산이 고갈" in warnings
    assert not form_app.error


def test_disclaimer_is_shown(form_app):
    captions = " ".join(c.value for c in form_app.caption)
    assert "참고용" in captions
    assert "권유하지 않습니다" in captions


def test_risk_score_shows_the_sustainability_cap(form_app):
    """'양호 91점'과 '69세 고갈'이 나란히 뜨는 모순이 화면에 없어야 한다."""
    values = {m.label: m.value for m in form_app.metric}
    assert values["종합"] == "40 / 100"
    captions = " ".join(c.value for c in form_app.caption)
    assert "종합 점수를 40점으로 제한" in captions


def test_changing_expense_updates_the_result():
    """입력을 바꾸면 결과가 다시 계산되어야 한다."""
    at = _form_mode()
    expense_input = next(i for i in at.number_input if i.label == "월 생활비")
    expense_input.set_value(DEMO_PROFILE.monthly_expense // 10_000 * 2).run()

    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["생활비 충당 가능"] != "6.2년"


# --------------------------------------------------------------------------- #
# 대화 모드
# --------------------------------------------------------------------------- #


def test_conversation_asks_the_first_question():
    at = _fresh()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "몇 년생" in text


def test_results_are_hidden_until_the_conversation_finishes():
    """대화 도중에 결과 화면이 먼저 나오면 안 된다."""
    at = _fresh()
    at.chat_input[0].set_value("1966년 12월생입니다").run()
    labels = [m.label for m in at.metric]
    assert "총자산" not in labels


def test_full_conversation_produces_the_same_numbers(chat_app):
    """대화만으로 폼 입력과 동일한 결과에 도달해야 한다."""
    assert not chat_app.exception, chat_app.exception
    values = {m.label: m.value for m in chat_app.metric}
    assert values["총자산"] == "7억 2,000만원"
    assert values["소득공백기"] == "4년"
    assert values["생활비 충당 가능"] == "6.2년"
    assert values["금융자산 고갈"] == "만 69세"


def test_briefing_is_shown_after_the_conversation(chat_app):
    headings = " ".join(h.value for h in chat_app.subheader)
    assert "상담사의 정리" in headings

    text = " ".join(m.value for m in chat_app.markdown)
    assert "장기" in text  # 이 인물의 진짜 위험은 장기 소득이다


def test_briefing_numbers_are_verified(chat_app):
    """AI 가 지어낸 숫자가 없다는 사실이 화면에 표시되어야 한다."""
    captions = " ".join(c.value for c in chat_app.caption)
    assert "계산 결과에서 나온 값" in captions
    assert not [w for w in chat_app.warning if "확인되지 않았습니다" in w.value]


def test_conversation_advances_to_the_next_question():
    at = _fresh()
    at.chat_input[0].set_value("1966년 12월생입니다").run()

    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "퇴직은 언제" in text  # 다음 질문
    assert "1966년 12월생입니다" in text  # 내 답변이 대화에 남아 있다


def test_chat_placeholder_gives_an_example_answer():
    """시니어 사용자에게는 어떻게 답해야 할지 예시가 필요하다."""
    at = _fresh()
    assert "1966년생" in at.chat_input[0].placeholder


# --------------------------------------------------------------------------- #
# 받은 연락 확인 (기능 ⑦)
# --------------------------------------------------------------------------- #


def _fraud_mode() -> "AppTest":
    at = _fresh()
    at.radio[0].set_value("받은 연락 확인하기").run()
    return at


def test_fraud_mode_runs_without_exception():
    at = _fraud_mode()
    assert not at.exception, at.exception
    headings = " ".join(h.value for h in at.subheader)
    assert "받은 연락 확인하기" in headings


def test_fraud_mode_starts_empty_with_guidance():
    """붙여넣기가 어려운 시니어를 위한 안내가 있어야 한다."""
    at = _fraud_mode()
    infos = " ".join(i.value for i in at.info)
    assert "직접 입력하셔도" in infos


def test_sample_scam_button_fills_and_flags():
    at = _fraud_mode()
    at.button[0].click().run()

    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "위험" in errors
    assert "확정되지 않은" in errors  # 미확정 제도 경고


def test_scam_analysis_shows_evidence_and_policy_status():
    at = _fraud_mode()
    at.text_area[0].set_value(
        "정부 세법이 바뀌어 ISA 비과세가 폐지됩니다. 원금 보장에 월 3% 확정 수익, "
        "오늘까지 선착순입니다. 텔레그램으로 연락 주세요."
    ).run()

    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "확인이 필요한 부분" in body
    assert "언급된 제도의 실제 상태" in body


def test_legitimate_message_is_not_alarmed():
    at = _fraud_mode()
    at.text_area[0].set_value(
        "안녕하세요, ○○은행입니다. 정기예금 연 3.2% 안내드립니다. "
        "예금자보호법에 따라 5,000만원까지 보호됩니다."
    ).run()

    assert not at.exception
    assert not at.error  # 정상 안내를 빨간 경고로 띄우면 안 된다


def test_fraud_disclaimer_is_shown():
    at = _fraud_mode()
    at.text_area[0].set_value("원금 보장 고수익 상품입니다").run()
    captions = " ".join(c.value for c in at.caption)
    assert "사기 여부를 확정하지 않습니다" in captions
    assert "fine.fss.or.kr" in captions
