"""금융사기 탐지 테스트.

두 방향 모두 중요하다:
  - 실제 사기 문구를 놓치지 않는가 (재현율)
  - 정상적인 안내를 사기로 몰지 않는가 (정밀도)

시니어 대상 서비스에서 후자를 놓치면 "맨날 위험하다고만 한다"며 서비스를 믿지 않게 된다.
"""

from __future__ import annotations

import pytest

from agents.fraud import (
    RiskLevel,
    Severity,
    analyze_message,
    assess_risk,
    check_policies,
    detect_signals,
    link_to_profile,
    load_policy_facts,
    load_rules,
)
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE

# 기획서에 나온 형태의 사기 문자
SCAM_TEXT = """[특별안내] 고객님만 드리는 기회입니다.
정부 세법이 바뀌면서 ISA 비과세 혜택이 곧 폐지됩니다.
지금 갈아타지 않으시면 손해입니다.
원금 보장되면서 월 3% 확정 수익 나오는 상품이고요,
오늘까지만 선착순으로 받습니다.
자세한 내용은 텔레그램으로 연락 주세요. 가족한테는 비밀로 해주세요."""

LEGIT_TEXT = """안녕하세요, ○○은행입니다.
고객님께서 문의하신 정기예금 상품 안내드립니다.
1년 만기 기준 연 3.2%이며, 예금자보호법에 따라 5,000만원까지 보호됩니다.
자세한 내용은 영업점이나 공식 앱에서 확인하실 수 있습니다.
궁금하신 점은 고객센터로 연락 주세요."""


# --------------------------------------------------------------------------- #
# 룰 파일 자체
# --------------------------------------------------------------------------- #


def test_rules_file_is_well_formed():
    rules = load_rules()
    assert rules["rules"]
    for rule in rules["rules"]:
        assert rule["key"] and rule["label"]
        assert rule["severity"] in {"high", "medium", "low"}
        assert rule["why"].strip() and rule["advice"].strip()
        match = rule["match"]
        assert ("all" in match) ^ ("any" in match), rule["key"]


def test_rule_keys_are_unique():
    keys = [r["key"] for r in load_rules()["rules"]]
    assert len(keys) == len(set(keys))


def test_policy_facts_declare_their_status():
    """발표와 시행을 구분하지 않으면 이 기능의 의미가 없다."""
    valid = {"발표", "입법예고", "국회통과", "공포", "시행"}
    facts = load_policy_facts()
    assert facts
    for fact in facts:
        assert fact["status"] in valid, fact["key"]
        assert fact["source"] and fact["caution"].strip()


# --------------------------------------------------------------------------- #
# 사기 문구 탐지
# --------------------------------------------------------------------------- #


def test_scam_message_is_flagged_as_high_risk():
    result = analyze_message(SCAM_TEXT)
    assert result.risk_level is RiskLevel.HIGH
    assert result.has_high_severity


@pytest.mark.parametrize(
    "key",
    [
        "guaranteed_high_return",
        "unrealistic_return",
        "urgency",
        "unofficial_channel",
        "policy_pretext",
        "secrecy",
    ],
)
def test_each_expected_signal_is_detected(key: str):
    keys = {s.key for s in detect_signals(SCAM_TEXT)}
    assert key in keys


def test_signals_carry_their_evidence():
    """왜 걸렸는지 사용자에게 보여줄 수 있어야 한다."""
    signals = detect_signals(SCAM_TEXT)
    urgency = next(s for s in signals if s.key == "urgency")
    assert urgency.evidence
    assert any("오늘까지" in e or "선착순" in e for e in urgency.evidence)


def test_spacing_differences_are_absorbed():
    """'원금 보장'과 '원금보장'을 다르게 보면 안 된다."""
    a = detect_signals("원금 보장 되고 고수익 납니다")
    b = detect_signals("원금보장되고 고수익납니다")
    assert {s.key for s in a} == {s.key for s in b}
    assert "guaranteed_high_return" in {s.key for s in a}


def test_impersonation_needs_both_halves():
    """기관명만으로는 사칭이 아니다. 권유가 함께 있어야 한다."""
    assert "impersonation" not in {s.key for s in detect_signals("금융감독원 발표 자료입니다")}
    assert "impersonation" in {
        s.key for s in detect_signals("금융감독원에서 안내드립니다. 이 계좌로 이체하세요")
    }


# --------------------------------------------------------------------------- #
# 정상 안내를 사기로 몰지 않는가
# --------------------------------------------------------------------------- #


def test_legitimate_message_is_not_flagged_as_high_risk():
    result = analyze_message(LEGIT_TEXT)
    assert result.risk_level is not RiskLevel.HIGH


def test_plain_text_produces_no_signals():
    result = analyze_message("어머니 생신 선물 뭐가 좋을까요?")
    assert result.signals == []
    assert result.risk_level is RiskLevel.LOW
    assert "위험 신호는 발견되지 않았습니다" in result.summary


def test_no_signal_does_not_mean_safe():
    """신호가 없다고 안전을 보장한다고 말하면 안 된다."""
    result = analyze_message("안녕하세요")
    assert "안전이 보장되는 것은 아닙니다" in result.summary


def test_single_medium_signal_is_caution_not_high():
    """정상 안내도 표현 하나쯤은 걸릴 수 있으므로 medium 하나로 '위험'을 띄우지 않는다."""
    signals = [s for s in detect_signals("후기 보시면 아실 거예요")]
    assert all(s.severity is Severity.MEDIUM for s in signals)
    assert assess_risk(signals) is RiskLevel.CAUTION


def test_two_medium_signals_become_high():
    signals = detect_signals("제도가 바뀌니 지금 갈아타세요. 다른 분들 후기도 많습니다.")
    assert sum(1 for s in signals if s.severity is Severity.MEDIUM) >= 2
    assert assess_risk(signals) is RiskLevel.HIGH


# --------------------------------------------------------------------------- #
# 정책 대조 — 기획서의 핵심 차별점
# --------------------------------------------------------------------------- #


def test_policy_mention_triggers_a_fact_check():
    checks = check_policies("ISA 비과세가 폐지된다고 하던데요")
    assert checks
    assert any(c.topic == "ISA" for c in checks)


def test_unconfirmed_policy_is_distinguished_from_enacted_one():
    """'발표'와 '시행'을 구분하는 것이 이 기능의 존재 이유다."""
    isa = next(c for c in check_policies("ISA 얘기입니다") if c.topic == "ISA")
    pension = next(c for c in check_policies("국민연금 얘기입니다") if c.topic == "연금")

    assert not isa.is_confirmed
    assert isa.status == "발표"
    assert pension.is_confirmed
    assert pension.status == "시행"


def test_summary_warns_about_unconfirmed_policy():
    result = analyze_message(SCAM_TEXT)
    assert result.unconfirmed_policies
    assert "아직 확정되지 않은" in result.summary


def test_unrelated_message_triggers_no_policy_check():
    assert check_policies("점심 뭐 드셨어요?") == []


# --------------------------------------------------------------------------- #
# 프로필 연결 — 일반 사기 경보와의 차이
# --------------------------------------------------------------------------- #


def test_held_asset_is_named_with_its_amount():
    notes = link_to_profile("ISA 갈아타셔야 합니다", DIVERSIFIED_PROFILE)
    assert notes
    assert "ISA" in notes[0]
    assert "5,000만원" in notes[0]


def test_not_held_asset_is_called_out():
    """보유하지 않은 상품을 근거로 든 권유는 그 자체로 신호다."""
    notes = link_to_profile("ISA 갈아타셔야 합니다", DEMO_PROFILE)
    assert notes
    assert "보유하고 계시지 않습니다" in notes[0]


def test_transfer_request_mentions_liquid_assets():
    notes = link_to_profile("이 계좌로 이체해 주세요", DEMO_PROFILE)
    assert any("바로 인출 가능한 자산" in n for n in notes)


def test_analysis_without_profile_still_works():
    result = analyze_message(SCAM_TEXT)
    assert result.profile_notes == []
    assert result.risk_level is RiskLevel.HIGH


def test_analysis_with_profile_adds_notes():
    result = analyze_message(SCAM_TEXT, profile=DIVERSIFIED_PROFILE)
    assert result.profile_notes
    assert any("ISA" in n for n in result.profile_notes)


# --------------------------------------------------------------------------- #
# 결과 형태
# --------------------------------------------------------------------------- #


def test_every_signal_tells_the_user_what_to_do():
    for signal in detect_signals(SCAM_TEXT):
        assert signal.advice.strip()
        assert signal.why.strip()


def test_assessment_reports_checked_length():
    result = analyze_message(SCAM_TEXT)
    assert result.checked_text_length == len(SCAM_TEXT)


def test_empty_message_is_handled():
    result = analyze_message("")
    assert result.risk_level is RiskLevel.LOW
    assert result.signals == []


# --------------------------------------------------------------------------- #
# Supervisor 라우팅
# --------------------------------------------------------------------------- #

from agents.graph import Intent, classify_intent  # noqa: E402
from agents.llm import MockClient  # noqa: E402


def test_pasted_scam_message_routes_to_fraud_check():
    routing = classify_intent(SCAM_TEXT)
    assert routing.intent is Intent.FRAUD_CHECK
    assert routing.by_rule  # LLM 을 부를 이유가 없다
    assert routing.reason


def test_asking_whether_a_message_is_real_routes_to_fraud_check():
    for question in [
        "이거 사기인가요?",
        "이런 문자 받았는데 믿어도 될까요?",
        "카톡이 왔는데 봐주세요",
    ]:
        assert classify_intent(question).intent is Intent.FRAUD_CHECK, question


@pytest.mark.parametrize(
    "answer",
    ["1966년생입니다", "퇴직금은 2억 정도요", "생활비 300만원 씁니다", "예금 5천 있어요"],
)
def test_profile_answers_route_to_profiling(answer: str):
    assert classify_intent(answer).intent is Intent.PROFILING


def test_result_questions_route_to_result():
    assert classify_intent("결과를 다시 보여주세요").intent is Intent.RESULT
    assert classify_intent("왜 그런지 자세히 설명해 주세요").intent is Intent.RESULT


def test_short_legit_text_does_not_route_to_fraud_check():
    """짧은 정상 답변이 사기 확인으로 새면 대화가 망가진다."""
    assert classify_intent("연 3.2% 예금 들었어요").intent is not Intent.FRAUD_CHECK


def test_empty_input_is_other():
    assert classify_intent("   ").intent is Intent.OTHER


def test_llm_is_only_consulted_when_rules_are_unsure():
    client = MockClient(
        structured_handler=lambda p, s, sys: {"intent": "other", "reason": "테스트"}
    )
    classify_intent(SCAM_TEXT, client=client)
    assert not client.calls  # 규칙으로 확실하므로 호출하지 않는다

    classify_intent("음... 글쎄요 그게 좀", client=client)
    assert client.calls  # 애매하면 물어본다


def test_llm_failure_does_not_break_routing():
    """분류가 실패했다고 대화가 끊기면 안 된다."""

    def explode(prompt, schema, system):
        raise RuntimeError("API 장애")

    routing = classify_intent("애매한 말", client=MockClient(structured_handler=explode))
    assert routing.intent is Intent.PROFILING
