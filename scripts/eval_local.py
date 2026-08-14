"""로컬 모델이 실제로 어땠는지 집계한다 (L1~L4).

PROGRESS.md 의 L1~L4 는 "실제 모델을 꽂아야 드러나는 것들"이었다. 지금까지는 그
판단 재료가 메모리에만 있다가 사라져서 추측으로만 답할 수 있었다. 저장소가
생겼으므로 이제 숫자로 답한다.

무엇을 재고 무엇을 재지 않는지:

  L1 차단 비율   — A7 이 도구 출력에 없는 숫자를 발견해 답변을 버린 비율.
                   mock 은 도구 출력을 그대로 옮겨 적어 0 에 가깝다. 실제 모델에서
                   이 값이 **너무 높으면 과잉 차단**(멀쩡한 답이 막힘),
                   **0 이면 검증이 헐거운 것**일 수 있다.
  L2 도구 선택   — mock 의 낱말 매칭(`agents/mocks.py` 의 TOOL_KEYWORDS)을 기준으로
                   같은 도구를 골랐는지 본다. **mock 이 정답이라는 뜻은 아니다.**
                   다르면 어느 쪽이 맞는지 사람이 보라는 신호다.
  L4 응답 시간   — 질문 하나에 걸린 시간. 도구를 여러 번 부르면 그만큼 늘어난다.

  L3 말투        — **자동으로 판정하지 않는다.** 답변을 그대로 출력해 사람이 읽는다.
                   지어낸 지표를 만들지 않는다는 이 프로젝트의 규약과 같다.

실행:
    python scripts/eval_local.py                   # 현재 설정된 프로바이더로
    FINAGENT_LLM_PROVIDER=local python scripts/eval_local.py
    python scripts/eval_local.py --compare-mock    # mock 과 나란히
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import storage
from agents.config import Provider, describe, resolve_provider
from agents.graph import Supervisor
from agents.llm import MockClient, get_client
from agents.mocks import TOOL_KEYWORDS, demo_handler, demo_tool_handler, pick_tool

# 프로파일링을 건너뛰고 상담부터 보기 위한 대화. 예시 인물과 같은 값이 되도록
# 짜여 있어 결과 숫자를 눈으로 대조할 수 있다.
CHAT_SCRIPT = [
    "1966년 12월생입니다",
    "올해 12월에 퇴직할 예정이에요",
    "생활비는 한 300만원 정도 씁니다",
    "퇴직금은 2억 정도 나올 것 같아요, 32년 다녔습니다",
    "예금이 2천만원 있습니다",
    "국민연금은 월 150만원 정도 나온다고 하더라고요, 32년 넣었습니다",
    "주식이나 펀드는 없어요",
    "퇴직연금도 없습니다",
    "아파트가 한 채 있는데 5억 정도 합니다",
    "다른 수입은 없어요",
    "대출은 없습니다",
    "원금을 지키는 쪽이 편합니다",
]

# 각 질문이 어느 도구로 가야 하는지는 mock 의 낱말 매칭이 정한다. 여기 답을 따로
# 적어 두면 두 곳이 조용히 어긋나므로, 기준을 한 곳에서만 가져온다.
QUESTIONS = [
    "국민연금을 더 내는 게 나을까요?",
    "건강보험료는 얼마나 나오나요?",
    "퇴직금은 일시금으로 받을까요?",
    "95세까지 유지하려면 어떻게 해야 하나요?",
    "이 결과 믿어도 되나요?",
    "ISA 세제가 바뀐다던데요",
    "생활비를 50만원 줄이면 어떻게 되나요?",
]

# 예시 인물의 확정 값. 모델이 바뀌어도 이 숫자는 같아야 한다 —
# 계산은 core/ 가 하기 때문이다. 다르면 슬롯 추출이 틀린 것이다.
EXPECTED = {
    "total_assets": "7억 2,000만원",
    "expense_coverage": "6.2년",
    "depletion": "만 69세 (2035년)",
}


def mock_client() -> MockClient:
    client = MockClient(structured_handler=demo_handler(2026))
    client.tool_handler = demo_tool_handler()
    return client


def _record(session_id: str, supervisor: Supervisor, reply) -> None:
    """이번 주고받음을 저장소에 남긴다.

    화면(`web/api.py`)이 하는 것과 같은 일이다. 평가도 같은 표에 쌓여야 나중에
    "그때 어땠더라"를 한 곳에서 볼 수 있다.
    """
    advice = getattr(reply, "advice", None)
    storage.record_exchange(
        session_id,
        question=reply.question,
        answer=reply.text,
        kind=reply.kind,
        routed_to=reply.intent.value,
        advice=(
            {
                "provider": getattr(supervisor.client, "provider", ""),
                "model": getattr(supervisor.client, "model", ""),
                "stop_reason": advice.stop_reason,
                "unverified_numbers": advice.unverified_numbers,
                "rounds": advice.rounds,
                "latency_ms": reply.latency_ms,
            }
            if advice is not None
            else None
        ),
        tool_runs=(
            [
                {
                    "tool": step.tool,
                    "arguments": step.arguments,
                    "ok": step.ok,
                    "summary": step.summary,
                    "output": step.output,
                }
                for step in advice.trace
            ]
            if advice is not None
            else None
        ),
    )


def run(label: str, client, *, session_prefix: str) -> dict:
    """대화 12턴을 진행하고 질문 7개를 던진다. 기록은 저장소에 남는다."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")

    supervisor = Supervisor(client=client)
    supervisor.start()
    session_id = f"eval-{session_prefix}-{int(time.time())}"
    storage.save_session(session_id, supervisor.dump())

    started = time.perf_counter()
    for line in CHAT_SCRIPT:
        supervisor.send(line)
    profiling_seconds = time.perf_counter() - started
    storage.save_session(
        session_id, supervisor.dump(), done=supervisor.done, ready=supervisor.ready
    )

    if not supervisor.done:
        print("  ⚠️  대화가 끝나지 않았습니다. 슬롯 추출이 실패한 것입니다.")
        print(f"     진행: {supervisor.progress[0]}/{supervisor.progress[1]}")

    # 계산이 예시 인물과 같은가 — 여기가 어긋나면 추출이 틀린 것이다.
    profile = supervisor.profile()
    print(f"\n[프로파일] 12턴 {profiling_seconds:.1f}초")
    if profile is None:
        print("  ⚠️  프로파일을 만들지 못했습니다.")
        checks = {}
    else:
        from core.cashflow import simulate
        from core.formatting import fmt_krw, fmt_years

        sim = simulate(profile)
        actual = {
            "total_assets": fmt_krw(profile.total_assets),
            "expense_coverage": fmt_years(sim.expense_coverage_years),
            "depletion": (
                f"만 {sim.depletion_age}세 ({sim.depletion_year}년)"
                if sim.depletion_age
                else "고갈 없음"
            ),
        }
        checks = {k: actual[k] == v for k, v in EXPECTED.items()}
        for key, expected in EXPECTED.items():
            mark = "OK  " if checks[key] else "다름"
            print(f"  {mark} {key:18} {actual[key]:20} (기대 {expected})")

    print(f"\n[질문 {len(QUESTIONS)}개]")
    rows = []
    tool_names = [
        name for name, _ in TOOL_KEYWORDS
    ]  # mock 이 고를 수 있는 도구 이름
    for question in QUESTIONS:
        reply = supervisor.send(question)
        _record(session_id, supervisor, reply)
        advice = reply.advice
        chosen = advice.trace[0].tool if advice and advice.trace else None
        expected_tool = pick_tool(question, tool_names)
        agree = chosen == expected_tool

        rows.append(
            {
                "question": question,
                "tool": chosen,
                "expected": expected_tool,
                "agree": agree,
                "stop_reason": advice.stop_reason if advice else "(도구 없음)",
                "unverified": advice.unverified_numbers if advice else [],
                "latency_ms": reply.latency_ms,
                "text": reply.text,
            }
        )
        mark = "OK  " if agree else "다름"
        print(
            f"  {mark} {question[:26]:28} → {str(chosen):26} "
            f"{reply.latency_ms:>6}ms  {rows[-1]['stop_reason']}"
        )
        if not agree:
            print(f"       (mock 기준: {expected_tool})")

    return {
        "label": label,
        "profile_ok": all(checks.values()) if checks else False,
        "rows": rows,
        "profiling_seconds": profiling_seconds,
    }


def summarize(result: dict) -> None:
    rows = result["rows"]
    agreed = sum(1 for r in rows if r["agree"])
    blocked = sum(1 for r in rows if r["stop_reason"] == "blocked")
    errors = sum(1 for r in rows if r["stop_reason"] == "llm_error")
    latencies = sorted(r["latency_ms"] for r in rows)
    median = latencies[len(latencies) // 2] if latencies else 0

    print(f"\n[요약] {result['label']}")
    print(f"  프로파일 숫자 일치      {'예' if result['profile_ok'] else '아니오'}")
    print(f"  L2 도구 선택 일치       {agreed}/{len(rows)}")
    print(f"  L1 차단(A7)             {blocked}/{len(rows)}")
    print(f"  오류                    {errors}/{len(rows)}")
    print(f"  L4 응답시간 중앙값      {median}ms  (12턴 {result['profiling_seconds']:.1f}초)")

    if blocked:
        print("\n  차단된 답변의 걸린 숫자:")
        for row in rows:
            if row["stop_reason"] == "blocked":
                print(f"    · {row['question'][:30]} → {row['unverified']}")


def print_answers(result: dict) -> None:
    """L3 은 자동 판정하지 않는다. 사람이 읽으라고 그대로 낸다."""
    print(f"\n{'=' * 72}\n[L3] 답변 원문 — 말투는 사람이 판단한다\n{'=' * 72}")
    for row in result["rows"]:
        print(f"\nQ. {row['question']}")
        print(f"A. {row['text'][:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 모델 평가 (L1~L4)")
    parser.add_argument(
        "--compare-mock", action="store_true", help="mock 과 나란히 돌려 비교한다"
    )
    parser.add_argument(
        "--answers", action="store_true", help="답변 원문을 출력한다 (L3)"
    )
    args = parser.parse_args()

    print(describe())
    provider = resolve_provider()

    results = []
    if provider is not Provider.MOCK:
        try:
            results.append(run(f"실제 모델 ({provider.value})", get_client(), session_prefix="live"))
        except Exception as exc:
            print(f"\n⚠️  모델에 연결하지 못했습니다: {type(exc).__name__}: {exc}")
            print("   서버가 떠 있는지 확인하세요: curl $LOCAL_BASE_URL/models")
            return

    if args.compare_mock or provider is Provider.MOCK:
        results.append(run("mock (키 없이 도는 기준선)", mock_client(), session_prefix="mock"))

    for result in results:
        summarize(result)
    if args.answers and results:
        print_answers(results[0])

    print(f"\n[저장소 누적] {storage.counts()}")
    stats = storage.advice_stats()
    if stats["total"]:
        print(
            f"  차단률 {stats['block_rate']:.1%} · 평균 라운드 {stats['avg_rounds']:.1f} "
            f"· 평균 {stats['avg_latency_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
