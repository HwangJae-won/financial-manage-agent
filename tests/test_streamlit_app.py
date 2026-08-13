"""Streamlit 앱 스모크 테스트.

데모 중에 앱이 예외로 죽는 것이 가장 치명적이므로, 실제로 스크립트를 실행해
예외 없이 끝까지 렌더링되는지 확인한다. Streamlit 의 AppTest 는 브라우저 없이
스크립트를 돌려주므로 CI 에서도 쓸 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.samples import DEMO_PROFILE

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception, app.exception


def test_headline_metrics_are_rendered(app):
    labels = [m.label for m in app.metric]
    assert "총자산" in labels
    assert "소득공백기" in labels
    assert "생활비 충당 가능" in labels
    assert "금융자산 고갈" in labels


def test_all_sections_are_present(app):
    headings = " ".join(h.value for h in app.subheader)
    for section in [
        "한눈에 보기",
        "나의 자산지도",
        "시간에 따른 자산 변화",
        "자산이 남아있을 확률",
        "은퇴 재무 안정도",
        "대응 시나리오 비교",
    ]:
        assert section in headings


def test_default_view_shows_the_demo_persona_numbers(app):
    """기본 화면에 기획서 예시의 핵심 숫자가 그대로 보여야 한다."""
    values = {m.label: m.value for m in app.metric}
    assert values["총자산"] == "7억 2,000만원"
    assert values["소득공백기"] == "4년"
    assert values["생활비 충당 가능"] == "6.2년"
    assert values["금융자산 고갈"] == "만 69세"


def test_long_term_risk_is_warned_about(app):
    """소득공백기는 넘기지만 장기 고갈 경고가 떠야 한다."""
    warnings = " ".join(w.value for w in app.warning)
    assert "금융자산이 고갈" in warnings
    assert not app.error  # 소득공백기 자체는 부족하지 않으므로 error 는 없어야 한다


def test_disclaimer_is_shown(app):
    captions = " ".join(c.value for c in app.caption)
    assert "참고용" in captions
    assert "권유하지 않습니다" in captions


def test_changing_expense_updates_the_result():
    """입력을 바꾸면 결과가 다시 계산되어야 한다.

    위젯을 조작하므로 공유 fixture 를 오염시키지 않도록 별도 인스턴스를 쓴다.
    """
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()

    expense_input = next(i for i in at.number_input if i.label == "월 생활비")
    expense_input.set_value(DEMO_PROFILE.monthly_expense // 10_000 * 2).run()

    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["생활비 충당 가능"] != "6.2년"  # 생활비를 두 배로 늘렸으므로


def test_risk_score_shows_the_sustainability_cap(app):
    """'양호 91점'과 '69세 고갈'이 나란히 뜨는 모순이 화면에 없어야 한다."""
    values = {m.label: m.value for m in app.metric}
    assert values["종합"] == "40 / 100"
    captions = " ".join(c.value for c in app.caption)
    assert "종합 점수를 40점으로 제한" in captions
