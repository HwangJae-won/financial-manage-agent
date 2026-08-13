"""LLM 연결 스모크 테스트 — 실제 API 를 한 번 호출해 본다.

키를 넣은 직후 "제대로 연결됐나"를 확인하는 용도다. pytest 에는 넣지 않는다
(CI 가 키를 요구하게 되고, 돈이 나가고, 네트워크에 의존하게 되므로).

    python scripts/smoke_llm.py

키가 없으면 mock 으로 동작하며 그 사실을 알려준다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.config import Provider, describe, resolve_provider  # noqa: E402
from agents.llm import NUMERIC_GUARDRAIL, get_client  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "retirement_year": {"type": ["integer", "null"]},
        "severance_pay": {"type": ["integer", "null"]},
    },
    "required": ["retirement_year", "severance_pay"],
    "additionalProperties": False,
}


def main() -> int:
    print(describe())
    provider = resolve_provider()

    if provider is Provider.MOCK:
        print()
        print("  API 키가 없어 mock 으로 동작합니다.")
        print("  실제 호출을 보려면 .env 에 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 를 넣으세요.")
        print()

    client = get_client()

    print("\n[1/2] 자연어 응답 테스트")
    resp = client.complete(
        "한 문장으로 인사해줘.",
        system="너는 은퇴 자산관리를 돕는 상담사다. 존댓말을 쓴다.",
        max_tokens=100,
    )
    print(f"  → {resp.text.strip()[:120]}")
    print(f"  토큰: 입력 {resp.input_tokens} / 출력 {resp.output_tokens}")

    print("\n[2/2] 구조화 출력 테스트 (슬롯 추출)")
    extracted = client.structured(
        "다음 대화에서 정보를 추출하세요.\n\n"
        "사용자: 올해 12월에 퇴직할 예정이고요, 퇴직금은 2억 정도 나올 것 같습니다.\n"
        "(올해는 2026년입니다.)\n\n" + NUMERIC_GUARDRAIL,
        SCHEMA,
        max_tokens=200,
    )
    print(f"  → {extracted}")

    if provider is not Provider.MOCK:
        expected = {"retirement_year": 2026, "severance_pay": 200_000_000}
        if extracted == expected:
            print("  ✅ 추출 정확")
        else:
            print(f"  ⚠️  기대값과 다름: {expected}")

    print(f"\n연결 정상 ({client.provider} / {client.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
