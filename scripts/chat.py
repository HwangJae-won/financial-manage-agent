"""터미널에서 상담사와 대화한다.

웹을 띄우지 않고 답변만 빠르게 보고 싶을 때 쓴다. 화면이 하는 일과 같은 경로를
탄다 — Supervisor 가 분류하고, 도구를 돌리고, A7 이 숫자를 대조한다.

실행:
    python scripts/chat.py              # 처음부터 대화
    python scripts/chat.py --demo       # 예시 인물 12턴을 자동으로 채우고 시작
    python scripts/chat.py --no-trace   # 실행 기록 숨기기

명령:
    /trace   실행 기록 표시 켜기·끄기
    /state   지금까지 모은 정보
    /quit    끝내기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.config import describe
from agents.graph import Supervisor
from agents.llm import MockClient, get_client
from agents.mocks import demo_handler, demo_tool_handler

DEMO_SCRIPT = [
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

EXAMPLES = [
    "생활비를 50만원 줄이면 어떻게 되나요?",
    "국민연금을 더 내는 게 나을까요?",
    "건강보험료는 얼마나 나오나요?",
    "95세까지 유지하려면 어떻게 해야 하나요?",
    "이 결과 믿어도 되나요?",
]

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def client():
    """설정된 프로바이더. 실패하면 mock 으로 떨어진다 — 대화가 멈추지 않게."""
    try:
        made = get_client()
    except Exception as exc:
        print(f"{DIM}(모델 연결 실패: {exc} — mock 으로 진행합니다){RESET}")
        made = MockClient()
    if isinstance(made, MockClient):
        made.structured_handler = demo_handler(2026)
        made.tool_handler = demo_tool_handler()
    return made


def show_trace(reply) -> None:
    advice = getattr(reply, "advice", None)
    if advice is None:
        return

    if advice.blocked:
        print(
            f"{BOLD}  ⚠ 계산에 없는 숫자({', '.join(advice.unverified_numbers)})가 있어 "
            f"원래 답변을 내보내지 않았습니다.{RESET}"
        )
    for index, step in enumerate(advice.trace, 1):
        args = " · ".join(f"{k}={v}" for k, v in step.arguments.items()) or "인자 없이"
        mark = "" if step.ok else " (실패)"
        print(f"{DIM}  [{index}] {step.tool}{mark}  {args}{RESET}")
        print(f"{DIM}      → {step.summary}{RESET}")
    if advice.trace:
        print(f"{DIM}  ({reply.latency_ms}ms · {reply.intent.value}){RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="터미널 상담")
    parser.add_argument("--demo", action="store_true", help="예시 인물로 12턴을 채우고 시작")
    parser.add_argument("--no-trace", action="store_true", help="실행 기록 숨기기")
    args = parser.parse_args()

    trace_on = not args.no_trace
    print(f"{DIM}{describe()}{RESET}\n")

    agent = Supervisor(client=client())
    print(f"{BOLD}상담사{RESET} {agent.start()}\n")

    if args.demo:
        print(f"{DIM}예시 인물로 12턴을 채웁니다…{RESET}")
        for line in DEMO_SCRIPT:
            agent.send(line)
        profile = agent.profile()
        if profile is not None:
            from core.cashflow import simulate
            from core.formatting import fmt_krw

            sim = simulate(profile)
            print(
                f"{DIM}  총자산 {fmt_krw(profile.total_assets)} · "
                f"고갈 만 {sim.depletion_age}세{RESET}"
            )
        print(f"\n{DIM}이제 물어보세요. 예:{RESET}")
        for example in EXAMPLES:
            print(f"{DIM}  · {example}{RESET}")
        print()

    while True:
        try:
            text = input(f"{BOLD}나{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/q", "quit", "exit"):
            break
        if text == "/trace":
            trace_on = not trace_on
            print(f"{DIM}실행 기록 {'표시' if trace_on else '숨김'}{RESET}\n")
            continue
        if text == "/state":
            slots = agent.profiler.state.get("slots", {})
            for key, value in sorted(slots.items()):
                shown = f"{value:,}" if isinstance(value, int) else value
                print(f"{DIM}  {key:26} {shown}{RESET}")
            print(f"{DIM}  완료: {agent.done}{RESET}\n")
            continue

        reply = agent.send(text)
        print(f"\n{BOLD}상담사{RESET} {reply.text}")
        if trace_on:
            show_trace(reply)
        print()


if __name__ == "__main__":
    main()
