"""가정값이 낡았는지 확인하고, 원문에는 뭐라고 되어 있는지 가져온다.

**이 스크립트는 값을 바꾸지 않는다.** 보고만 한다.

자동 반영을 하지 않는 이유는 이 프로젝트가 다른 데서 내린 결정과 같다. 재산분
점수표를 못 구해서 근사식을 만들지 않았고, 주택연금 산정식도 지어내지 않았다.
그런데 긁어온 숫자를 검증 없이 반영하면 앞뒤가 맞지 않는다. 잘못된 자동 갱신은
보이는 낡음보다 나쁘다 — 낡은 값은 최소한 "낡았다"는 사실이 참이지만, 잘못
갱신된 값은 "최신이다"라고 거짓말을 한다.

그래서 역할을 나눈다.
    기계 — 언제 확인할지 · 어디를 볼지 · 원문에 뭐라고 적혀 있는지 · 반영하면 얼마가 달라지는지
    사람 — "이 값이 맞다" 확정과 커밋

출처는 법제처 국가법령정보 OpenAPI(law.go.kr)다. **키 없이도 동작하지만**
`OC=test` 는 공용 데모 계정이라 스로틀링이 걸려 응답이 잘려 오는 일이 있다.
안정적으로 쓰려면 open.law.go.kr 에서 무료로 발급받아 `LAW_OC` 에 넣는다.

실행:
    python scripts/check_assumptions.py              # 유효기간 지난 것만
    python scripts/check_assumptions.py --all        # 전부 확인
    python scripts/check_assumptions.py --no-fetch   # 원문 조회 없이 목록만
    LAW_OC=myid python scripts/check_assumptions.py  # 발급받은 키로
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_PATH = REPO_ROOT / "data" / "sources.yaml"
ASSUMPTIONS_PATH = REPO_ROOT / "data" / "assumptions.yaml"
CACHE_DIR = REPO_ROOT / "var" / "law_cache"

API = "http://www.law.go.kr/DRF"
# 공용 데모 계정. 스로틀링이 있으므로 발급받은 값을 LAW_OC 로 넣는 편이 낫다.
OC = os.environ.get("LAW_OC", "test")
CACHE_HOURS = 24

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m"
)


# --------------------------------------------------------------------------- #
# 법제처 조회 — 캐시를 반드시 둔다 (데모 계정 스로틀링 때문)
# --------------------------------------------------------------------------- #


def _cached_get(url: str, key: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.xml"
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < CACHE_HOURS:
            return path.read_text(encoding="utf-8")

    request = urllib.request.Request(url, headers={"User-Agent": "finagent-check/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    path.write_text(body, encoding="utf-8")
    return body


def _plain(xml: str) -> str:
    """XML 을 검색 가능한 한 줄 텍스트로. CDATA 와 태그를 걷어낸다."""
    text = re.sub(r"<!\[CDATA\[|\]\]>", " ", xml)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text)


def fetch_law(name: str) -> Optional[str]:
    """법령 본문. 법령명으로 바로 받는다."""
    url = f"{API}/lawService.do?OC={OC}&type=XML&target=law&LM={urllib.parse.quote(name)}"
    try:
        return _plain(_cached_get(url, f"law_{name}"))
    except (urllib.error.URLError, OSError):
        return None


def fetch_admrul(query: str) -> Optional[str]:
    """행정규칙(고시) 본문. 검색으로 일련번호를 얻은 뒤 본문을 받는다.

    고시는 법령명으로 바로 못 받는다. 같은 이름의 옛 고시가 여럿이라 검색 결과에서
    **가장 최근 발령**을 고른다.
    """
    search = (
        f"{API}/lawSearch.do?OC={OC}&type=XML&target=admrul"
        f"&query={urllib.parse.quote(query)}"
    )
    try:
        listing = _cached_get(search, f"admrul_search_{query}")
    except (urllib.error.URLError, OSError):
        return None

    best: tuple[str, str] = ("", "")
    for match in re.finditer(
        r"<행정규칙일련번호>(\d+)</행정규칙일련번호>.*?"
        r"<행정규칙명><!\[CDATA\[(.*?)\]\]></행정규칙명>.*?"
        r"<발령일자>(\d+)</발령일자>",
        listing,
        re.S,
    ):
        ident, title, issued = match.groups()
        if query.replace(" ", "") in title.replace(" ", "") and issued > best[1]:
            best = (ident, issued)

    if not best[0]:
        return None
    body = f"{API}/lawService.do?OC={OC}&type=XML&target=admrul&ID={best[0]}"
    try:
        return _plain(_cached_get(body, f"admrul_{best[0]}"))
    except (urllib.error.URLError, OSError):
        return None


# --------------------------------------------------------------------------- #
# 값 읽기·비교
# --------------------------------------------------------------------------- #


def dig(data: dict, dotted: str) -> Any:
    """`health_insurance.local.health_rate` 같은 경로로 값을 꺼낸다."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def extract(text: str, spec: dict) -> Optional[float]:
    """원문에서 숫자 하나를 뽑는다. 못 뽑으면 None — **추측하지 않는다.**

    뽑은 값이 `expect` 범위를 벗어나면 **못 찾은 것으로 본다.** 정규식은 본문이
    잘려 오거나 조문 구조가 바뀌면 엉뚱한 숫자를 문다. 장기요양보험료율 자리에
    "100" 이 잡히는 식이다. **틀린 값을 보고하는 검사기는 없느니만 못하다** —
    사람이 그것을 믿고 반영하기 때문이다. 확신이 없으면 "못 찾음"이 정답이다.
    """
    match = re.search(spec["pattern"], text)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None

    value *= spec.get("scale", 1)
    expect = spec.get("expect")
    if expect and not (expect["min"] <= value <= expect["max"]):
        return None
    return value


def is_stale(entry: dict, today: _dt.date) -> bool:
    until = entry.get("effective_until")
    if not until:
        return False
    try:
        return _dt.date.fromisoformat(str(until)) < today
    except ValueError:
        return False


def fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


# --------------------------------------------------------------------------- #
# 보고
# --------------------------------------------------------------------------- #


def report(entries: list[dict], today: _dt.date, *, fetch: bool) -> list[dict]:
    assumptions = yaml.safe_load(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    findings = []

    for entry in entries:
        current = dig(assumptions, entry["key"])
        row = {
            "key": entry["key"],
            "label": entry["label"],
            "current": current,
            "stale": is_stale(entry, today),
            "found": None,
            "period": None,
            "note": entry.get("note", ""),
        }

        if fetch:
            source = entry.get("source", {})
            text = (
                fetch_admrul(source["query"])
                if source.get("target") == "admrul"
                else fetch_law(source.get("name", ""))
            )
            if text:
                row["found"] = extract(text, entry["extract"])
                if entry.get("period_pattern"):
                    period = re.search(entry["period_pattern"], text)
                    row["period"] = period.group(1).strip() if period else None
        findings.append(row)

    return findings


def print_report(findings: list[dict], calendar: list[dict], today: _dt.date) -> None:
    changed = [f for f in findings if f["found"] is not None and f["found"] != f["current"]]
    same = [f for f in findings if f["found"] is not None and f["found"] == f["current"]]
    unknown = [f for f in findings if f["found"] is None]

    if changed:
        print(f"\n{BOLD}{RED}■ 원문과 다른 값 {len(changed)}건 — 확인이 필요합니다{RESET}")
        for row in changed:
            mark = f"{RED}(유효기간 지남){RESET}" if row["stale"] else ""
            print(f"\n  {BOLD}{row['label']}{RESET}  {mark}")
            print(f"    코드   {fmt(row['current'])}")
            print(f"    원문   {GREEN}{fmt(row['found'])}{RESET}")
            if row["period"]:
                print(f"    적용   {row['period']}")
            print(f"    {DIM}{row['key']}{RESET}")
            if row["note"]:
                print(f"    {DIM}{row['note']}{RESET}")

    if same:
        print(f"\n{GREEN}■ 원문과 일치 {len(same)}건{RESET}")
        for row in same:
            print(f"  {DIM}{row['label']:32} {fmt(row['current'])}{RESET}")

    if unknown:
        print(f"\n{YELLOW}■ 원문에서 찾지 못함 {len(unknown)}건 — 사람이 확인해야 합니다{RESET}")
        for row in unknown:
            mark = f" {RED}(유효기간 지남){RESET}" if row["stale"] else ""
            print(f"  {row['label']:32} 코드 {fmt(row['current'])}{mark}")
            if row["note"]:
                print(f"    {DIM}{row['note']}{RESET}")

    print(f"\n{DIM}확인 일정{RESET}")
    for item in calendar:
        soon = "  ← 이번 달" if item["month"] == today.month else ""
        print(f"  {DIM}{item['month']:>2}월  {item['what']}{soon}{RESET}")


def print_manual(manual: list[dict]) -> None:
    if not manual:
        return
    print(f"\n{DIM}■ 기계로 읽기 어려운 것 — 링크로 직접 확인{RESET}")
    for item in manual:
        print(f"  {item['label']}")
        print(f"    {DIM}{item['why']}{RESET}")
        print(f"    {DIM}{item['url']}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="가정값 유효기간·원문 확인")
    parser.add_argument("--all", action="store_true", help="유효기간과 무관하게 전부 확인")
    parser.add_argument("--no-fetch", action="store_true", help="원문 조회 없이 목록만")
    parser.add_argument("--manual", action="store_true", help="수동 확인 목록도 출력")
    args = parser.parse_args()

    sources = yaml.safe_load(SOURCES_PATH.read_text(encoding="utf-8"))
    today = _dt.date.today()

    entries = sources["values"]
    if not args.all:
        due = [e for e in entries if is_stale(e, today)]
        if due:
            entries = due

    print(f"{BOLD}가정값 확인{RESET}  {today}  ({len(entries)}개)")
    if OC == "test":
        print(
            f"{DIM}법제처 API 를 공용 데모 계정(OC=test)으로 조회합니다. 응답이 잘려 올 수 "
            f"있으니, open.law.go.kr 에서 발급받아 LAW_OC 에 넣으면 안정적입니다.{RESET}"
        )

    findings = report(entries, today, fetch=not args.no_fetch)
    print_report(findings, sources.get("calendar", []), today)
    if args.manual:
        print_manual(sources.get("manual", []))

    print(
        f"\n{DIM}이 스크립트는 값을 바꾸지 않습니다. 확인 후 data/assumptions.yaml 을 "
        f"직접 고치고, data/sources.yaml 의 checked_at·effective_until 도 함께 "
        f"갱신하세요.{RESET}"
    )


if __name__ == "__main__":
    main()
