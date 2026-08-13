"""FastAPI 백엔드 테스트.

W1 의 설계 원칙이 지켜졌는지 확인하는 자리이기도 하다 — 대화로 얻은 결과와
프로파일을 직접 넣은 결과가 같아야 하고, Streamlit 과도 같은 숫자여야 한다.
"""

from __future__ import annotations

import pytest

from core.samples import DEMO_PROFILE

TestClient = pytest.importorskip("fastapi.testclient").TestClient

from web.api import app  # noqa: E402

DEMO_SCRIPT = [
    "1966년 12월생입니다",
    "올해 12월에 퇴직할 예정이에요",
    "생활비는 한 300만원 정도 씁니다",
    "퇴직금은 2억 정도 나올 것 같아요, 32년 다녔습니다",
    "예금이 2천만원 있습니다",
    "국민연금은 월 150만원 정도 나온다고 하더라고요",
    "주식이나 펀드는 없어요",
    "퇴직연금도 없습니다",
    "아파트가 한 채 있는데 5억 정도 합니다",
    "다른 수입은 없어요",
    "대출은 없습니다",
    "원금을 지키는 쪽이 편합니다",
]

SCAM = """[특별안내] 고객님만 드리는 기회입니다.
정부 세법이 바뀌면서 ISA 비과세 혜택이 곧 폐지됩니다.
지금 갈아타지 않으시면 손해입니다.
원금 보장되면서 월 3% 확정 수익 나오는 상품이고요,
오늘까지만 선착순으로 받습니다.
자세한 내용은 텔레그램으로 연락 주세요."""


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def finished_session(client):
    """대화를 끝까지 마친 세션 ID."""
    session_id = client.post("/api/sessions").json()["session_id"]
    for line in DEMO_SCRIPT:
        client.post(f"/api/sessions/{session_id}/messages", json={"text": line})
    return session_id


# --------------------------------------------------------------------------- #
# 기본
# --------------------------------------------------------------------------- #


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"]


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "네비게이터" in response.text


def test_static_assets_are_served(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


# --------------------------------------------------------------------------- #
# 대화
# --------------------------------------------------------------------------- #


def test_creating_a_session_returns_the_first_question(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201

    body = response.json()
    assert body["session_id"]
    assert body["done"] is False
    assert body["answered"] == 0
    assert any("몇 년생" in m["content"] for m in body["messages"])


def test_sending_a_message_advances_the_conversation(client):
    session_id = client.post("/api/sessions").json()["session_id"]
    body = client.post(
        f"/api/sessions/{session_id}/messages", json={"text": "1966년 12월생입니다"}
    ).json()

    assert body["answered"] == 1
    assert any("퇴직은 언제" in m["content"] for m in body["messages"])


def test_session_state_can_be_fetched(client, finished_session):
    body = client.get(f"/api/sessions/{finished_session}").json()
    assert body["done"] is True
    assert body["session_id"] == finished_session


def test_unknown_session_returns_404(client):
    assert client.get("/api/sessions/없는세션").status_code == 404
    assert (
        client.post("/api/sessions/없는세션/messages", json={"text": "안녕"}).status_code
        == 404
    )


def test_empty_message_is_rejected(client):
    session_id = client.post("/api/sessions").json()["session_id"]
    assert (
        client.post(f"/api/sessions/{session_id}/messages", json={"text": ""}).status_code
        == 422
    )


# --------------------------------------------------------------------------- #
# 분석
# --------------------------------------------------------------------------- #


def test_analysis_reproduces_the_demo_persona_numbers(client, finished_session):
    """대화 → 분석이 Streamlit·엔진과 같은 숫자를 내야 한다."""
    body = client.get(f"/api/sessions/{finished_session}/analysis").json()
    head = body["headline"]

    assert head["total_assets"] == "7억 2,000만원"
    assert head["income_gap"] == "4년"
    assert head["expense_coverage"] == "6.2년"
    assert "69세" in head["depletion"]
    assert head["risk_score"] == "40 / 100"
    assert head["risk_status"] == "부족"


def test_analysis_includes_every_section(client, finished_session):
    body = client.get(f"/api/sessions/{finished_session}/analysis").json()
    for key in [
        "profile",
        "headline",
        "simulation",
        "asset_map",
        "monte_carlo",
        "risk_score",
        "scenarios",
    ]:
        assert key in body, key
    assert len(body["scenarios"]) == 3


def test_briefing_is_included_and_verified(client, finished_session):
    body = client.get(f"/api/sessions/{finished_session}/analysis").json()
    briefing = body["briefing"]
    assert briefing["headline"]
    assert briefing["next_steps"]
    assert briefing["unverified_numbers"] == []  # AI 가 지어낸 숫자가 없어야 한다


def test_briefing_can_be_skipped_for_speed(client, finished_session):
    body = client.get(
        f"/api/sessions/{finished_session}/analysis", params={"briefing": False}
    ).json()
    assert body["briefing"] is None


def test_analysis_before_the_conversation_finishes_is_a_400(client):
    session_id = client.post("/api/sessions").json()["session_id"]
    response = client.get(f"/api/sessions/{session_id}/analysis")
    assert response.status_code == 400
    assert "필수 정보" in response.json()["detail"]


def test_direct_profile_analysis_matches_the_conversation(client, finished_session):
    """폼 입력 경로와 대화 경로가 같은 결과를 내야 한다."""
    from_chat = client.get(f"/api/sessions/{finished_session}/analysis").json()
    from_form = client.post(
        "/api/analysis", json=DEMO_PROFILE.model_dump(mode="json")
    ).json()

    assert from_form["headline"]["total_assets"] == from_chat["headline"]["total_assets"]
    assert from_form["headline"]["expense_coverage"] == from_chat["headline"]["expense_coverage"]
    assert from_form["headline"]["depletion"] == from_chat["headline"]["depletion"]


def test_invalid_profile_is_rejected(client):
    bad = DEMO_PROFILE.model_dump(mode="json") | {"monthly_expense": 0}
    assert client.post("/api/analysis", json=bad).status_code == 422


def test_samples_are_available(client):
    demo = client.get("/api/samples/demo").json()
    assert demo["birth_year"] == 1966
    assert client.get("/api/samples/diversified").status_code == 200
    assert client.get("/api/samples/없음").status_code == 404


# --------------------------------------------------------------------------- #
# 사기 확인
# --------------------------------------------------------------------------- #


def test_scam_message_is_flagged(client):
    body = client.post("/api/fraud-check", json={"text": SCAM}).json()
    assert body["risk_level"] == "위험"
    assert body["signals"]
    assert body["policy_checks"]


def test_legitimate_message_is_not_flagged(client):
    body = client.post(
        "/api/fraud-check",
        json={"text": "안녕하세요, 정기예금 연 3.2% 안내드립니다. 예금자보호 대상입니다."},
    ).json()
    assert body["risk_level"] != "위험"


def test_fraud_check_links_to_the_session_profile(client, finished_session):
    """세션이 있으면 사용자의 실제 보유 자산과 엮어 설명해야 한다."""
    body = client.post(
        "/api/fraud-check", json={"text": SCAM, "session_id": finished_session}
    ).json()
    assert body["profile_notes"]


def test_fraud_check_works_without_a_session(client):
    body = client.post("/api/fraud-check", json={"text": SCAM}).json()
    assert body["profile_notes"] == []


def test_fraud_check_tolerates_an_unfinished_session(client):
    """프로파일이 아직 완성되지 않은 세션을 넘겨도 500 이 나면 안 된다."""
    session_id = client.post("/api/sessions").json()["session_id"]
    response = client.post(
        "/api/fraud-check", json={"text": SCAM, "session_id": session_id}
    )
    assert response.status_code == 200
    assert response.json()["profile_notes"] == []


def test_fraud_check_rejects_empty_text(client):
    assert client.post("/api/fraud-check", json={"text": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# 제도 변화 (기능 ④·⑧)
# --------------------------------------------------------------------------- #


def test_policy_list_ships_its_trust_metadata(client):
    """뱃지와 출처 한 줄을 서버가 완성해서 보낸다 — 프런트가 다시 만들지 않게."""
    facts = client.get("/api/policies").json()
    assert facts

    isa = next(f for f in facts if f["key"] == "isa_reform")
    assert isa["status"] == "발표"
    assert isa["is_confirmed"] is False
    assert "미확정" in isa["badge"]
    assert isa["source"] in isa["trust_note"]
    assert isa["notice"]


def test_policy_impact_for_a_sample_persona(client):
    body = client.post(
        "/api/policy-impact", json={"sample": "diversified"}
    ).json()

    assert body["applicable"] is True
    assert body["annual_tax_saving"] > 0
    assert body["comparison"]
    assert body["policy"]["notice"]  # 미확정 고지는 항상 따라붙는다


def test_policy_impact_says_so_when_it_does_not_apply(client, finished_session):
    """기획서 예시 인물은 ISA 가 없다 — 빈 화면 대신 '영향 없음'이 나와야 한다."""
    body = client.post(
        "/api/policy-impact", json={"session_id": finished_session}
    ).json()

    assert body["applicable"] is False
    assert "ISA" in body["reason"]
    assert body["verdict"]


def test_policy_impact_accepts_a_profile_directly(client):
    profile = client.get("/api/samples/diversified").json()
    body = client.post("/api/policy-impact", json={"profile": profile}).json()
    assert body["applicable"] is True


def test_policy_impact_without_a_profile_is_a_400(client):
    assert client.post("/api/policy-impact", json={}).status_code == 400


def test_unknown_policy_key_is_a_404(client):
    response = client.post(
        "/api/policy-impact", json={"sample": "demo", "policy_key": "없는정책"}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# 처방 — "그래서 무엇을 하면 되나"
# --------------------------------------------------------------------------- #


def test_prescriptions_answer_the_demo_persona(client, finished_session):
    body = client.post(
        "/api/prescriptions", json={"session_id": finished_session}
    ).json()

    assert body["baseline_depletion_age"] == 69
    assert body["already_safe"] is False
    assert body["options"]

    expense = next(o for o in body["options"] if o["lever"] == "expense")
    assert expense["feasible"]
    assert expense["required_value"] < expense["current_value"]
    assert expense["required_value"] % 10_000 == 0  # 만원 단위로 답한다


def test_prescriptions_respect_the_target_age(client):
    strict = client.post(
        "/api/prescriptions", json={"sample": "demo", "target_age": 95}
    ).json()
    relaxed = client.post(
        "/api/prescriptions", json={"sample": "demo", "target_age": 85}
    ).json()

    def required_expense(body):
        return next(o for o in body["options"] if o["lever"] == "expense")["required_value"]

    assert required_expense(relaxed) > required_expense(strict)


def test_prescriptions_without_a_profile_is_a_400(client):
    assert client.post("/api/prescriptions", json={}).status_code == 400


def test_prescriptions_reject_an_absurd_target_age(client):
    response = client.post(
        "/api/prescriptions", json={"sample": "demo", "target_age": 200}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 민감도 — 우리 숫자 자체를 흔들어 본다
# --------------------------------------------------------------------------- #


def test_sensitivity_names_the_dominant_assumption(client, finished_session):
    body = client.post("/api/sensitivity", json={"session_id": finished_session}).json()

    assert body["baseline_depletion_age"] == 69
    assert body["cases"]
    assert body["cases"][0]["key"] == "expense"  # 흔들림이 큰 순서로 정렬된다
    assert body["cases"][0]["label"] in body["summary"]
    assert body["caveat"]


def test_sensitivity_without_a_profile_is_a_400(client):
    assert client.post("/api/sensitivity", json={}).status_code == 400


# --------------------------------------------------------------------------- #
# 퇴직금 수령 방식
# --------------------------------------------------------------------------- #


def test_severance_comparison_for_the_demo_persona(client, finished_session):
    body = client.post("/api/severance", json={"session_id": finished_session}).json()

    assert body["lump_sum"]["total_tax"] > body["pension"]["total_tax"]
    assert body["tax_saved"] > 0
    assert body["headline"]
    assert body["notes"]


def test_severance_respects_the_pension_period(client):
    ten = client.post("/api/severance", json={"sample": "demo", "pension_years": 10}).json()
    twenty = client.post("/api/severance", json={"sample": "demo", "pension_years": 20}).json()
    assert twenty["tax_saved"] > ten["tax_saved"]


def test_severance_without_a_profile_is_a_400(client):
    assert client.post("/api/severance", json={}).status_code == 400


# --------------------------------------------------------------------------- #
# 건강보험료 절벽
# --------------------------------------------------------------------------- #


def test_health_insurance_cliff_for_the_demo_persona(client, finished_session):
    body = client.post(
        "/api/health-insurance", json={"session_id": finished_session}
    ).json()

    assert body["qualifies_at_retirement"] is True
    assert body["cliff_year"] == 2031  # 국민연금이 개시되고 나서 한도를 넘는다
    assert body["local"]["monthly"] > 0
    assert body["notes"]


def test_health_insurance_answers_the_unknowns_as_questions(client):
    """급여를 모르면 임의계속가입 보험료를 지어내지 않고 물어본다."""
    profile = client.get("/api/samples/demo").json()
    profile["last_monthly_salary"] = 0

    body = client.post("/api/health-insurance", json={"profile": profile}).json()
    assert body["voluntary"]["available"] is False
    assert "알려주시면" in body["voluntary"]["basis"]

    with_salary = client.post(
        "/api/health-insurance", json={"profile": profile, "monthly_salary": 5_000_000}
    ).json()
    assert with_salary["voluntary"]["available"] is True
    assert with_salary["voluntary"]["monthly"] > 0


def test_health_insurance_takes_a_real_property_tax_base(client):
    body = client.post(
        "/api/health-insurance", json={"sample": "demo", "property_tax_base": 600_000_000}
    ).json()

    assert body["property_assumed"] is False
    assert body["qualifies_at_retirement"] is False  # 재산 한도 초과


def test_health_insurance_without_a_profile_is_a_400(client):
    assert client.post("/api/health-insurance", json={}).status_code == 400
