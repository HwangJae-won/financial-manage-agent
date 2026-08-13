"""설명 에이전트 테스트.

가장 중요한 것은 숫자 검증이다. LLM 이 계산 결과에 없는 숫자를 쓰면
반드시 잡아내야 한다 — 금융 서비스에서 이건 타협할 수 없는 부분이다.
"""

from __future__ import annotations

import pytest

from agents.explain import (
    Briefing,
    build_facts,
    explain,
    facts_block,
    verify_numbers,
)
from agents.llm import MockClient
from agents.mocks import briefing_from_facts
from core.asset_map import build_asset_map
from core.cashflow import simulate
from core.montecarlo import run_monte_carlo
from core.risk_score import compute_risk_score
from core.samples import DEMO_PROFILE, DIVERSIFIED_PROFILE


@pytest.fixture(scope="module")
def demo_bundle():
    profile = DEMO_PROFILE
    sim = simulate(profile)
    amap = build_asset_map(profile)
    score = compute_risk_score(profile)
    mc = run_monte_carlo(profile, n_paths=500)
    return profile, sim, amap, score, mc


@pytest.fixture
def mock_client():
    return MockClient(structured_handler=briefing_from_facts)


# --------------------------------------------------------------------------- #
# 숫자 검증 — 이 모듈의 핵심
# --------------------------------------------------------------------------- #


FACTS = {
    "총자산": "7억 2,000만원",
    "소득공백기": "4년",
    "생활비 감당 기간": "6.2년",
    "고갈 시점": "2035년, 만 69세",
    "고갈 가능성": "18%",
}


def test_briefing_using_only_given_numbers_passes():
    text = "총자산 7억 2,000만원 중 소득공백기 4년을 넘기셔야 합니다. 만 69세에 바닥납니다."
    assert verify_numbers(text, FACTS) == []


def test_invented_number_is_caught():
    """계산 결과에 없는 '8억'을 지어내면 잡아야 한다."""
    text = "총자산 8억원이므로 여유가 있습니다."
    assert "8" in verify_numbers(text, FACTS)


def test_llm_doing_arithmetic_is_caught():
    """계산 결과에 없는 값을 LLM 이 직접 계산해 쓰면 잡아야 한다."""
    text = "소득공백기 4년 동안 매달 300만원씩, 총 1억 4,400만원이 필요합니다."
    violations = verify_numbers(text, FACTS)
    assert "300" in violations
    assert "4400" in violations or "1" not in violations


def test_ordinal_numbers_are_not_flagged():
    """'세 가지', '1단계' 같은 순서 표현까지 잡으면 못 쓴다."""
    text = "다음 3가지를 확인하세요. 1) 생활비 2) 연금 3) 부채"
    assert verify_numbers(text, FACTS) == []


def test_comma_formatting_difference_is_absorbed():
    """'7억 2,000만원'과 '7억 2000만원'을 다른 숫자로 보면 안 된다."""
    assert verify_numbers("자산은 7억 2000만원입니다.", FACTS) == []


def test_percentage_from_facts_is_allowed():
    assert verify_numbers("18% 가능성이 있습니다.", FACTS) == []
    assert "42" in verify_numbers("42% 가능성이 있습니다.", FACTS)


# --------------------------------------------------------------------------- #
# 사실 목록
# --------------------------------------------------------------------------- #


def test_facts_contain_the_headline_numbers(demo_bundle):
    profile, sim, amap, score, mc = demo_bundle
    facts = build_facts(profile, sim, amap, score, mc)

    assert facts["총자산"] == "7억 2,000만원"
    assert facts["소득공백기"] == "4년"
    assert facts["연금·기타소득 없이 생활비를 감당할 수 있는 기간"] == "6.2년"
    assert "69세" in facts["금융자산이 바닥나는 시점"]
    assert "40점" in facts["은퇴 재무 안정도 점수"]


def test_facts_work_without_optional_inputs(demo_bundle):
    profile, sim, amap, _, _ = demo_bundle
    facts = build_facts(profile, sim, amap)
    assert "총자산" in facts
    assert "은퇴 재무 안정도 점수" not in facts


def test_surviving_profile_reports_no_depletion():
    profile = DIVERSIFIED_PROFILE
    facts = build_facts(profile, simulate(profile), build_asset_map(profile))
    assert "바닥나지 않음" in facts["금융자산이 바닥나는 시점"]


def test_facts_block_is_readable(demo_bundle):
    profile, sim, amap, _, _ = demo_bundle
    block = facts_block(build_facts(profile, sim, amap))
    assert block.startswith("- 총자산: ")
    assert "\n- 소득공백기: 4년" in block


# --------------------------------------------------------------------------- #
# 브리핑 생성
# --------------------------------------------------------------------------- #


def test_briefing_is_generated_and_verified(demo_bundle, mock_client):
    profile, sim, amap, score, mc = demo_bundle
    briefing = explain(
        profile, sim, amap, score=score, monte_carlo=mc, client=mock_client
    )

    assert isinstance(briefing, Briefing)
    assert briefing.headline
    assert briefing.situation
    assert briefing.priority
    assert briefing.next_steps
    assert briefing.is_verified, briefing.unverified_numbers


def test_demo_briefing_names_the_real_risk(demo_bundle, mock_client):
    """기획서 예시 인물은 단기는 안전하고 장기가 문제다. 브리핑이 그걸 짚어야 한다."""
    profile, sim, amap, score, mc = demo_bundle
    briefing = explain(profile, sim, amap, score=score, monte_carlo=mc, client=mock_client)

    assert "장기" in briefing.headline
    assert "69세" in briefing.priority


def test_shortfall_profile_prioritises_liquidity(mock_client):
    """유동자산이 부족한 사람에게는 생활비 확보를 먼저 말해야 한다."""
    poor = DEMO_PROFILE.model_copy(update={"severance_pay": 30_000_000, "cash_savings": 0})
    briefing = explain(
        poor, simulate(poor), build_asset_map(poor), client=mock_client
    )
    assert "생활비" in briefing.headline
    assert "부족" in briefing.priority
    assert briefing.is_verified, briefing.unverified_numbers


def test_healthy_profile_is_not_alarming(mock_client):
    briefing = explain(
        DIVERSIFIED_PROFILE,
        simulate(DIVERSIFIED_PROFILE),
        build_asset_map(DIVERSIFIED_PROFILE),
        client=mock_client,
    )
    assert "안정" in briefing.headline
    assert briefing.is_verified, briefing.unverified_numbers


def test_hallucinated_briefing_is_flagged_not_silently_accepted(demo_bundle):
    """LLM 이 숫자를 지어내면 결과에 표시되어야 한다 — 조용히 통과하면 안 된다."""
    profile, sim, amap, _, _ = demo_bundle
    liar = MockClient(
        structured_handler=lambda p, s, sys: {
            "headline": "총자산 15억원으로 여유롭습니다.",
            "situation": "매달 999만원을 쓰셔도 됩니다.",
            "priority": "걱정하지 않으셔도 됩니다.",
            "next_steps": ["아무것도 하지 마세요."],
        }
    )
    briefing = explain(profile, sim, amap, client=liar)

    assert not briefing.is_verified
    assert "15" in briefing.unverified_numbers
    assert "999" in briefing.unverified_numbers


def test_full_text_contains_every_section(demo_bundle, mock_client):
    profile, sim, amap, _, _ = demo_bundle
    briefing = explain(profile, sim, amap, client=mock_client)
    text = briefing.full_text()
    assert briefing.headline in text
    assert briefing.situation in text
    assert all(step in text for step in briefing.next_steps)


def test_prompt_carries_the_guardrail(demo_bundle, mock_client):
    profile, sim, amap, _, _ = demo_bundle
    explain(profile, sim, amap, client=mock_client)
    system = mock_client.calls[-1]["system"]
    assert "직접 계산하지 마세요" in system
    assert "권유하지" in system
