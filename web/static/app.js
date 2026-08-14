/* 내 자산 AI 네비게이터 — 프런트엔드
 *
 * 빌드 도구 없이 순수 JS 로 짰다. HPC 환경에 node 를 얹고 빌드 단계를 두면
 * 발표 당일 실패 지점이 하나 늘어난다. 차트도 외부 라이브러리 없이 SVG 로 그린다.
 *
 * 금액 포맷은 서버가 이미 해서 보낸다 (core/formatting.py).
 * 여기서 다시 구현하면 화면마다 표기가 달라진다.
 */

const state = {
  sessionId: null,
  analysis: null,
  plainLanguage: false,
  user: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

/* 상대경로로 요청한다. code-server 의 /proxy/8000/ 처럼 base path 가 붙는 환경에서도
 * 그대로 동작하게 하기 위함이다. 절대경로(/api/...)를 쓰면 프록시 뒤에서 깨진다. */
const API_BASE = new URL(".", location.href).pathname;

async function api(path, options = {}) {
  const response = await fetch(API_BASE + path.replace(/^\//, ""), {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `요청에 실패했습니다 (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) {
      /* 본문이 JSON 이 아니면 기본 문구를 쓴다 */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* ------------------------------------------------------------------ */
/* 접근성 설정                                                          */
/* ------------------------------------------------------------------ */

function setupAccessibility() {
  $$("[data-font]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("[data-font]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      document.documentElement.style.setProperty("--font-scale", btn.dataset.font);
    });
  });

  $("#contrast-toggle").addEventListener("click", (e) => {
    const on = document.body.classList.toggle("high-contrast");
    e.currentTarget.setAttribute("aria-pressed", String(on));
  });

  $("#plain-toggle").addEventListener("click", (e) => {
    state.plainLanguage = !state.plainLanguage;
    e.currentTarget.setAttribute("aria-pressed", String(state.plainLanguage));
    if (state.analysis) renderAnalysis(state.analysis);
  });

  setupSpeech();
}

/* 브라우저 내장 음성합성. 외부 서비스도 API 키도 필요 없다. */
function setupSpeech() {
  const button = $("#tts-toggle");
  if (!("speechSynthesis" in window)) {
    button.classList.add("hidden");
    return;
  }

  button.addEventListener("click", () => {
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
      button.setAttribute("aria-pressed", "false");
      button.textContent = "🔊 읽어주기";
      return;
    }
    const view = $(".view:not(.hidden)");
    const text = (view?.innerText || "").slice(0, 4000);
    if (!text.trim()) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    utterance.rate = 0.95; // 시니어 청취를 고려해 조금 천천히
    utterance.onend = () => {
      button.setAttribute("aria-pressed", "false");
      button.textContent = "🔊 읽어주기";
    };
    speechSynthesis.speak(utterance);
    button.setAttribute("aria-pressed", "true");
    button.textContent = "⏹ 멈추기";
  });
}

/* ------------------------------------------------------------------ */
/* 화면 전환                                                            */
/* ------------------------------------------------------------------ */

function showView(name) {
  $$(".view").forEach((v) => v.classList.add("hidden"));
  $(`#view-${name}`).classList.remove("hidden");
  $$(".tab").forEach((t) =>
    t.setAttribute("aria-selected", String(t.dataset.view === name)),
  );
  if (speechSynthesis?.speaking) speechSynthesis.cancel();
}

function setupTabs() {
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => showView(tab.dataset.view)),
  );
  $$("[data-goto]").forEach((btn) =>
    btn.addEventListener("click", () => showView(btn.dataset.goto)),
  );
}

/* ------------------------------------------------------------------ */
/* 상담                                                                 */
/* ------------------------------------------------------------------ */

function renderConversation(data) {
  state.sessionId = data.session_id;

  const log = $("#chat-log");
  log.innerHTML = "";
  data.messages.forEach((m) => log.appendChild(el("div", `bubble ${m.role}`, m.content)));

  // 마지막 답변이 계산을 돌린 것이면 그 아래에 실행 기록을 붙인다 (A4).
  if (data.advice) log.appendChild(renderTrace(data.advice));
  if (data.fraud) log.appendChild(renderRouted(data));
  log.scrollTop = log.scrollHeight;

  const ratio = data.total ? data.answered / data.total : 0;
  $("#progress-fill").style.width = `${Math.round(ratio * 100)}%`;
  $("#progress-text").textContent = `${data.answered} / ${data.total}`;
  $(".progress-track").setAttribute("aria-valuenow", String(data.answered));
  $(".progress-track").setAttribute("aria-valuemax", String(data.total));

  $("#btn-see-result").classList.toggle("hidden", !(data.ready && !data.done));

  const finished = data.done;

  // 대화가 끝나도 입력창을 닫지 않는다. 결과를 두고 **이어서 묻는 것**이 이
  // 화면의 다음 역할이기 때문이다. 닫아 버리면 상담 에이전트에 닿을 길이 없다.
  $("#chat-input").disabled = false;
  $("#chat-send").disabled = false;
  $("#chat-input").placeholder = finished
    ? "결과를 보고 궁금한 점을 물어보세요"
    : "편하게 말씀해 주세요";
  $("#chat-hint").textContent =
    data.error ||
    (finished
      ? "예: 생활비를 50만원 줄이면 어떻게 되나요? / 국민연금을 더 내는 게 나을까요?"
      : "");

  // 결과 화면으로 넘기는 것은 **처음 한 번만.** 이후 질문마다 화면이 튀면
  // 대화를 이어갈 수 없다.
  if (finished && !state.analysis) loadAnalysis();
}

/* ------------------------------------------------------------------ */
/* 실행 기록 (A4) — 무엇을 어떤 값으로 계산했는지                            */
/* ------------------------------------------------------------------ */

function renderTrace(advice) {
  const box = el("details", "trace");
  const count = advice.trace.length;
  box.appendChild(
    el(
      "summary",
      null,
      count
        ? `이 답을 만들려고 계산 ${count}번을 돌렸습니다 — 눌러서 확인`
        : "계산 없이 답했습니다",
    ),
  );

  // 지어낸 숫자를 걸러냈으면 그 사실이 먼저다. 답변보다 중요한 정보다.
  if (advice.stop_reason === "blocked") {
    box.appendChild(
      el(
        "div",
        "alert alert-warn",
        `계산 결과에 없는 숫자(${advice.unverified_numbers.join(", ")})가 있어 ` +
          "원래 답변을 내보내지 않았습니다.",
      ),
    );
  }

  advice.trace.forEach((step, index) => {
    const item = el("div", "trace-step");
    item.appendChild(
      el("div", "trace-head", `${index + 1}. ${step.tool}${step.ok ? "" : " (실패)"}`),
    );

    const args = Object.entries(step.arguments || {});
    item.appendChild(
      el(
        "div",
        "trace-args",
        args.length
          ? args.map(([k, v]) => `${k} = ${v}`).join(" · ")
          : "인자 없이 — 현재 계획 그대로",
      ),
    );
    item.appendChild(el("div", "trace-result", step.summary));
    box.appendChild(item);
  });

  if (count) {
    box.appendChild(
      el(
        "p",
        "hint",
        "답변의 숫자는 전부 위 계산에서 나온 값입니다. " +
          "계산에 없는 숫자는 답변에서 걸러집니다.",
      ),
    );
  }
  return box;
}

function renderRouted(data) {
  const box = el("div", "trace");
  box.appendChild(
    el("div", "trace-head", `받은 연락 확인으로 보냈습니다 — ${data.routing_reason}`),
  );
  (data.fraud.signals || []).forEach((signal) =>
    box.appendChild(
      el("div", "trace-result", `${signal.label} — "${signal.evidence}"`),
    ),
  );
  return box;
}

/* 사용자 식별 — 인증이 아니다. 브라우저가 들고 있는 id 하나가 신원의 전부이고,
   그래서 이 기기에서만 지난 상담을 이어볼 수 있다. 화면에도 그렇게 적는다. */
const USER_KEY = "finagent-user-id";

async function ensureUser() {
  try {
    const info = await api("/api/users", {
      method: "POST",
      body: JSON.stringify({ user_id: localStorage.getItem(USER_KEY) }),
    });
    localStorage.setItem(USER_KEY, info.user_id);
    state.user = info;
    return info;
  } catch (_) {
    // 저장소가 없어도 상담은 되어야 한다. 이어보기만 못 할 뿐이다.
    return null;
  }
}

async function startSession() {
  try {
    const user = state.user || (await ensureUser());
    renderConversation(
      await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ user_id: user?.user_id ?? null }),
      }),
    );
  } catch (err) {
    $("#chat-hint").textContent = err.message;
  }
}

/** 지난 상담을 이어서 연다. 없으면 조용히 새로 시작한다. */
async function resumeLastSession() {
  const user = await ensureUser();
  const last = user?.sessions?.[0];
  if (!last) return false;

  try {
    renderConversation(await api(`/api/sessions/${last.id}`));
    return true;
  } catch (_) {
    return false;
  }
}

async function sendMessage(text) {
  const input = $("#chat-input");
  input.disabled = true;
  $("#chat-send").disabled = true;
  try {
    renderConversation(
      await api(`/api/sessions/${state.sessionId}/messages`, {
        method: "POST",
        body: JSON.stringify({ text }),
      }),
    );
    input.value = "";
  } catch (err) {
    $("#chat-hint").textContent = err.message;
  } finally {
    if (!$("#chat-input").disabled) input.focus();
  }
}

function setupChat() {
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("#chat-input").value.trim();
    if (text) sendMessage(text);
  });

  // 엔터로 보내되, 줄바꿈은 Shift+Enter 로
  $("#chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });

  $("#btn-restart").addEventListener("click", async () => {
    // 이전 대화를 저장소에서도 지운다. '처음부터 다시'를 눌렀는데 지난 상담이
    // 목록에 남아 있으면 지운 것이 아니다.
    const previous = state.sessionId;
    state.analysis = null;
    $("#result-body").classList.add("hidden");
    $("#result-empty").classList.remove("hidden");
    if (previous) {
      try {
        await api(`/api/sessions/${previous}`, { method: "DELETE" });
      } catch (_) {
        /* 지우기에 실패해도 새 상담은 시작한다 */
      }
    }
    startSession();
  });

  $("#btn-see-result").addEventListener("click", loadAnalysis);

  // 결과 화면의 예시 질문 버튼. 시니어 사용자에게는 "무엇이든 물어보세요" 보다
  // 눌러볼 수 있는 문장 세 개가 훨씬 낫다.
  $$("[data-ask]").forEach((btn) =>
    btn.addEventListener("click", () => {
      showView("chat");
      $("#chat-input").value = btn.dataset.ask;
      $("#chat-form").requestSubmit();
    }),
  );
}

/* ------------------------------------------------------------------ */
/* 결과                                                                 */
/* ------------------------------------------------------------------ */

async function loadAnalysis() {
  if (!state.sessionId) return;
  try {
    const data = await api(`/api/sessions/${state.sessionId}/analysis`);
    state.analysis = data;
    renderAnalysis(data);
    showView("result");
  } catch (err) {
    $("#chat-hint").textContent = err.message;
  }
}

async function loadSample(key) {
  try {
    const profile = await api(`/api/samples/${key}`);
    const data = await api("/api/analysis?briefing=true", {
      method: "POST",
      body: JSON.stringify(profile),
    });
    state.analysis = data;
    renderAnalysis(data);
  } catch (err) {
    alert(err.message);
  }
}

function renderAnalysis(data) {
  $("#result-empty").classList.add("hidden");
  $("#result-body").classList.remove("hidden");

  renderBriefing(data.briefing);
  renderStats(data.headline, data.assumed_fields);
  renderAssetMap(data.asset_map);
  renderBalanceChart(data.simulation);
  renderRiskScore(data.risk_score);
  renderScenarios(data.scenarios, data.monte_carlo);
  renderEvents(data.profile.life_events || []);
  loadSeverance();
  loadHealthInsurance();
  loadNationalPension();
  loadFamilyReport();
  loadPrescriptions();
  loadSensitivity();
}

function renderBriefing(briefing) {
  const box = $("#briefing");
  box.innerHTML = "";
  if (!briefing) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");

  box.appendChild(el("h2", null, briefing.headline));
  box.appendChild(el("p", null, briefing.situation));

  const priority = el("div", "alert alert-warn");
  priority.appendChild(el("strong", null, "지금 가장 중요한 것 "));
  priority.appendChild(document.createTextNode(briefing.priority));
  box.appendChild(priority);

  if (!state.plainLanguage) {
    box.appendChild(el("p", null, "이렇게 해보세요"));
  }
  const list = el("ul");
  briefing.next_steps.forEach((step) => list.appendChild(el("li", null, step)));
  box.appendChild(list);

  const verified = (briefing.unverified_numbers || []).length === 0;
  box.appendChild(
    el(
      "p",
      "hint",
      verified
        ? "✅ 위 설명의 숫자는 모두 계산 결과에서 나온 값입니다. AI가 임의로 만든 숫자가 없는지 자동 검사했습니다."
        : `⚠️ 다음 숫자는 계산 결과에서 확인되지 않았습니다: ${briefing.unverified_numbers.join(", ")}`,
    ),
  );
}

function renderStats(headline, assumed) {
  const grid = $("#stat-grid");
  grid.innerHTML = "";

  const items = [
    ["총자산", headline.total_assets, `금융자산 ${headline.financial_assets}`],
    ["소득공백기", headline.income_gap, "국민연금 받기 전까지"],
    ["생활비 충당 가능", headline.expense_coverage, "연금·기타소득 없이 계산"],
    ["금융자산 고갈", headline.depletion, ""],
    ["은퇴 재무 안정도", headline.risk_score, headline.risk_status],
  ];

  items.forEach(([label, value, note]) => {
    const card = el("div", "stat");
    card.appendChild(el("div", "stat-label", label));
    card.appendChild(el("div", "stat-value", value));
    if (note) card.appendChild(el("div", "stat-note", note));
    grid.appendChild(card);
  });

  if (assumed && assumed.length) {
    grid.appendChild(
      el(
        "p",
        "hint",
        `말씀하지 않으신 ${assumed.length}개 항목은 기본값으로 가정했습니다.`,
      ),
    );
  }
}

const BUCKET_COLORS = {
  survival: "#1f6f8b",
  income_gap: "#e08214",
  long_term: "#4d9078",
  growth: "#8b5fa8",
  real_asset: "#8c8c8c",
};

function renderAssetMap(map) {
  const box = $("#asset-map");
  box.innerHTML = "";

  const total = map.buckets.reduce((sum, b) => sum + b.amount, 0) || 1;
  const bar = el("div", "card");
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", "0 0 100 8");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    map.buckets.map((b) => `${b.label} ${b.amount}원`).join(", "),
  );

  let offset = 0;
  map.buckets.forEach((bucket) => {
    if (bucket.amount <= 0) return;
    const width = (bucket.amount / total) * 100;
    const rect = document.createElementNS(svgNs, "rect");
    rect.setAttribute("x", String(offset));
    rect.setAttribute("y", "0");
    rect.setAttribute("width", String(width));
    rect.setAttribute("height", "8");
    rect.setAttribute("fill", BUCKET_COLORS[bucket.key] || "#888");
    svg.appendChild(rect);
    offset += width;
  });
  bar.appendChild(svg);
  box.appendChild(bar);

  map.buckets.forEach((bucket) => {
    const row = el("div", "bucket-row");
    const name = el("span", "bucket-name", `${bucket.label} · ${bucket.horizon}`);
    row.appendChild(name);

    const amount = el("span", "bucket-amount");
    amount.textContent = formatWon(bucket.amount);
    if (bucket.shortfall > 0) {
      amount.appendChild(
        el("span", "bucket-short", ` (${formatWon(bucket.shortfall)} 부족)`),
      );
    }
    row.appendChild(amount);
    row.appendChild(el("span", "bucket-purpose", bucket.purpose));
    box.appendChild(row);
  });
}

/* 서버가 포맷한 값이 없는 자리에서만 쓰는 최소 포맷터 */
function formatWon(won) {
  if (!won) return "0원";
  const eok = Math.floor(won / 100000000);
  const man = Math.floor((won % 100000000) / 10000);
  const parts = [];
  if (eok) parts.push(`${eok}억`);
  if (man) parts.push(`${man.toLocaleString("ko-KR")}만`);
  return parts.length ? `${parts.join(" ")}원` : `${won.toLocaleString("ko-KR")}원`;
}

function renderBalanceChart(sim) {
  const box = $("#balance-chart");
  box.innerHTML = "";

  const rows = sim.rows;
  const maxBalance = Math.max(...rows.map((r) => r.start_balance), 1);
  const W = 100;
  const H = 40;

  const points = rows
    .map((row, i) => {
      const x = (i / Math.max(rows.length - 1, 1)) * W;
      const y = H - (row.start_balance / maxBalance) * H;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H + 6}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    sim.depletion_age
      ? `금융자산 잔액이 만 ${sim.depletion_age}세에 0이 됩니다`
      : "금융자산 잔액이 시뮬레이션 기간 내내 유지됩니다",
  );

  const line = document.createElementNS(svgNs, "polyline");
  line.setAttribute("points", points);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#1f6f8b");
  line.setAttribute("stroke-width", "1.2");
  svg.appendChild(line);

  if (sim.depletion_age) {
    const index = rows.findIndex((r) => r.age === sim.depletion_age);
    if (index >= 0) {
      const x = (index / Math.max(rows.length - 1, 1)) * W;
      const marker = document.createElementNS(svgNs, "line");
      marker.setAttribute("x1", String(x));
      marker.setAttribute("x2", String(x));
      marker.setAttribute("y1", "0");
      marker.setAttribute("y2", String(H));
      marker.setAttribute("stroke", "#b3261e");
      marker.setAttribute("stroke-width", "0.8");
      marker.setAttribute("stroke-dasharray", "2 1");
      svg.appendChild(marker);
    }
  }

  const card = el("div", "card");
  card.appendChild(svg);
  card.appendChild(
    el(
      "p",
      "hint",
      `만 ${rows[0].age}세부터 만 ${rows[rows.length - 1].age}세까지의 금융자산 잔액입니다.` +
        (sim.depletion_age ? ` 빨간 선은 자산이 바닥나는 만 ${sim.depletion_age}세입니다.` : ""),
    ),
  );
  box.appendChild(card);
}

function renderRiskScore(score) {
  const box = $("#risk-score");
  box.innerHTML = "";

  const head = el("div", "card");
  head.appendChild(el("div", "stat-value", `${score.total} / 100 · ${score.status}`));
  if (score.cap_reason) {
    head.appendChild(
      el(
        "p",
        "hint",
        `현재 상태 점수는 ${score.raw_total}점이지만, ${score.cap_reason} 계획을 유지할 수 없으므로 종합 점수를 ${score.cap}점으로 제한했습니다.`,
      ),
    );
  }
  box.appendChild(head);

  score.components.forEach((c) => {
    const row = el("div", "score-row");
    row.appendChild(el("span", "score-name", `${c.icon} ${c.label}`));
    row.appendChild(el("span", null, `${Math.round(c.score)}점 · ${c.status}`));
    row.appendChild(el("span", "score-detail", c.detail));
    box.appendChild(row);
  });
}

function renderScenarios(scenarios, _mc) {
  const box = $("#scenarios");
  box.innerHTML = "";

  const grid = el("div", "scenario-grid");
  scenarios.forEach((s) => {
    const card = el("div", "card");
    card.appendChild(el("h3", null, s.label));
    card.appendChild(el("p", "section-note", s.description));

    const list = el("ul");
    [
      ["현금성", s.amounts.cash],
      ["채권·중위험", s.amounts.bond],
      ["주식·투자", s.amounts.equity],
    ].forEach(([label, amount]) => {
      list.appendChild(el("li", null, `${label} ${formatWon(amount)}`));
    });
    card.appendChild(list);
    grid.appendChild(card);
  });
  box.appendChild(grid);

  const table = el("table");
  const head = el("thead");
  const headRow = el("tr");
  headRow.appendChild(el("th", null, "지표"));
  scenarios.forEach((s) => headRow.appendChild(el("th", null, s.label)));
  head.appendChild(headRow);
  table.appendChild(head);

  const body = el("tbody");
  const metrics = [
    ["소득공백기 안정성", (s) => `${Math.round(s.monte_carlo.prob_survive_income_gap * 100)}%`],
    ["변동성", (s) => volatilityLabel(s.volatility)],
  ];
  metrics.forEach(([label, fn]) => {
    const row = el("tr");
    row.appendChild(el("td", null, label));
    scenarios.forEach((s) => row.appendChild(el("td", null, fn(s))));
    body.appendChild(row);
  });
  table.appendChild(body);
  box.appendChild(table);
}

function volatilityLabel(volatility) {
  if (volatility < 0.05) return "낮음";
  if (volatility < 0.1) return "중간";
  return "높음";
}

/* ------------------------------------------------------------------ */
/* 받은 연락 확인                                                       */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* 퇴직금 수령 방식                                                      */
/* ------------------------------------------------------------------ */

async function loadSeverance() {
  const box = $("#severance");
  const profile = state.analysis?.profile;
  if (!profile) return;

  try {
    renderSeverance(
      await api("/api/severance", {
        method: "POST",
        body: JSON.stringify({
          profile,
          pension_years: Number($("#pension-years").value),
        }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderSeverance(result) {
  const box = $("#severance");
  box.innerHTML = "";
  box.appendChild(
    el("div", `alert ${result.computable ? "alert-ok" : "alert-warn"}`, result.headline),
  );

  // 근속연수를 모르면 세금이 전부 0으로 나온다. 그 표를 보여주면 '세금이 없다'는
  // 잘못된 인상을 준다.
  if (!result.computable) {
    box.appendChild(el("div", "advice", result.verdict));
    return;
  }

  const table = el("table");
  const thead = el("thead");
  const header = el("tr");
  ["항목", result.lump_sum.method, result.pension.method].forEach((text) =>
    header.appendChild(el("th", null, text)),
  );
  thead.appendChild(header);
  table.appendChild(thead);

  const tbody = el("tbody");
  [
    ["세금", "total_tax_label"],
    ["실효세율", "effective_rate_label"],
    ["언제 내나", "when_paid"],
    ["자산 고갈", "depletion_label"],
    ["마지막 잔액", "final_balance_label"],
  ].forEach(([label, key]) => {
    const row = el("tr");
    row.appendChild(el("td", null, label));
    row.appendChild(el("td", null, result.lump_sum[key]));
    row.appendChild(el("td", null, result.pension[key]));
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  box.appendChild(table);

  box.appendChild(el("div", "advice", `이렇게 하세요 — ${result.verdict}`));
  const notes = el("ul", "hint");
  result.notes.forEach((note) => notes.appendChild(el("li", null, note)));
  box.appendChild(notes);
}

/* ------------------------------------------------------------------ */
/* 건강보험료 절벽                                                       */
/* ------------------------------------------------------------------ */

async function loadHealthInsurance() {
  const box = $("#health-insurance");
  const profile = state.analysis?.profile;
  if (!profile) return;

  // 빈 칸은 0이 아니라 **모름**이다. 0으로 보내면 서버가 계산해 버린다.
  const won = (sel) => {
    const raw = $(sel).value.trim();
    return raw === "" ? null : Number(raw) * MAN;
  };
  const family = $("#hi-family").value;

  try {
    renderHealthInsurance(
      await api("/api/health-insurance", {
        method: "POST",
        body: JSON.stringify({
          profile,
          monthly_salary: won("#hi-salary"),
          property_tax_base: won("#hi-property"),
          has_employed_family: family === "" ? null : family === "yes",
        }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderHealthInsurance(report) {
  const box = $("#health-insurance");
  box.innerHTML = "";

  const safe = report.cliff_year === null;
  box.appendChild(el("div", `alert ${safe ? "alert-ok" : "alert-warn"}`, report.headline));

  // 절벽까지 남은 거리. 이 화면에서 가장 중요한 숫자다 —
  // "지금 얼마"가 아니라 "얼마 남았나"가 결정을 바꾼다.
  const grid = el("div", "stat-grid");
  [
    ["합산소득", report.counted_income_label, "피부양자 판정에 들어가는 소득"],
    ["소득 한도까지", report.income_headroom_label, "이만큼 늘면 탈락합니다"],
    [
      "이자·배당 계단까지",
      report.financial_headroom_label,
      report.headroom_rate_label
        ? `수익률 ${report.headroom_rate_label} — 넘으면 전액이 합산됩니다`
        : "넘으면 전액이 합산됩니다",
    ],
    [
      "탈락 시점",
      report.cliff_year === null ? "없음" : `${report.cliff_year}년`,
      report.cliff_age === null ? "지금 기준이 유지된다면" : `만 ${report.cliff_age}세`,
    ],
  ].forEach(([label, value, note]) => {
    const card = el("div", "stat");
    card.appendChild(el("div", "stat-label", label));
    card.appendChild(el("div", "stat-value", value));
    card.appendChild(el("div", "stat-note", note));
    grid.appendChild(card);
  });
  box.appendChild(grid);

  const table = el("table");
  const thead = el("thead");
  const header = el("tr");
  ["가입 방법", "매달", "1년", "근거"].forEach((text) =>
    header.appendChild(el("th", null, text)),
  );
  thead.appendChild(header);
  table.appendChild(thead);

  const tbody = el("tbody");
  [report.dependent, report.local, report.voluntary].forEach((path) => {
    const row = el("tr");
    row.appendChild(el("td", null, path.available ? path.method : `${path.method} (해당 없음)`));
    row.appendChild(el("td", null, path.monthly_label));
    row.appendChild(el("td", null, path.annual_label));
    row.appendChild(el("td", null, path.basis.replace(/\*\*/g, "")));
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  box.appendChild(table);

  if (report.depletion_advanced_years > 0) {
    box.appendChild(
      el(
        "div",
        "alert alert-warn",
        `보험료를 넣으면 금융자산이 바닥나는 시점이 ${report.depletion_advanced_years}년 ` +
          `앞당겨집니다 (평생 ${report.total_premiums_label}).`,
      ),
    );
  }

  box.appendChild(el("div", "advice", `이렇게 하세요 — ${report.verdict}`));
  const notes = el("ul", "hint");
  report.notes.forEach((note) =>
    notes.appendChild(el("li", null, note.replace(/\*\*/g, ""))),
  );
  box.appendChild(notes);
}

/* ------------------------------------------------------------------ */
/* 국민연금 임의계속가입 · 추납                                            */
/* ------------------------------------------------------------------ */

async function loadNationalPension() {
  const box = $("#national-pension");
  const profile = state.analysis?.profile;
  if (!profile) return;

  // 빈 칸은 0이 아니라 **모름**이다. 0으로 보내면 서버가 '가입기간 0개월'로 계산한다.
  const count = (sel) => {
    const raw = $(sel).value.trim();
    return raw === "" ? null : Number(raw);
  };

  try {
    renderNationalPension(
      await api("/api/national-pension", {
        method: "POST",
        body: JSON.stringify({
          profile,
          contributed_months: count("#np-months"),
          catchup_months: count("#np-catchup"),
        }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderNationalPension(report) {
  const box = $("#national-pension");
  box.innerHTML = "";
  const clean = (text) => (text || "").replace(/\*\*/g, "");

  if (!report.computable) {
    box.appendChild(el("div", "alert alert-warn", clean(report.headline)));
    box.appendChild(el("div", "advice", `이렇게 하세요 — ${report.verdict}`));
    const asked = el("ul", "hint");
    report.notes.forEach((note) => asked.appendChild(el("li", null, clean(note))));
    box.appendChild(asked);
    return;
  }

  // 수급자격이 없는 것은 '주의'가 아니라 '지금 연금이 0원'이라는 사실이다.
  const tone = !report.qualifies_now
    ? "alert-warn"
    : report.best_key === "none"
      ? "alert-ok"
      : "alert-warn";
  box.appendChild(el("div", `alert ${tone}`, clean(report.headline)));

  const grid = el("div", "stat-grid");
  [
    ["가입기간", report.contributed_label, "국민연금을 낸 총 기간"],
    [
      "수급자격",
      report.qualifies_now ? "있음" : `${report.months_to_qualify}개월 부족`,
      report.qualifies_now
        ? "최소 120개월을 넘겼습니다"
        : `채우는 데 ${report.cost_to_qualify_label}`,
    ],
    ["지금 월 연금", report.current.monthly_pension_label, "더 내지 않으실 경우"],
    [
      "가장 나은 선택",
      report.best_label,
      report.best_key === "none" ? "더 내실 이유가 없습니다" : "건보료까지 뺀 기준",
    ],
  ].forEach(([label, value, note]) => {
    const card = el("div", "stat");
    card.appendChild(el("div", "stat-label", label));
    card.appendChild(el("div", "stat-value", value));
    card.appendChild(el("div", "stat-note", note));
    grid.appendChild(card);
  });
  box.appendChild(grid);

  const options = [
    report.current,
    report.catchup,
    report.voluntary,
    report.both,
  ];

  const table = el("table");
  const thead = el("thead");
  const header = el("tr");
  [
    "수단",
    "더 내는 기간",
    "내는 돈",
    "월 연금 증가",
    "평생 더 받음",
    "추가 건보료",
    "빼고 남는 것",
    "본전",
  ].forEach((text) => header.appendChild(el("th", null, text)));
  thead.appendChild(header);
  table.appendChild(thead);

  const tbody = el("tbody");
  options.forEach((option) => {
    const row = el("tr");
    const dash = (text) => (option.available ? text : "—");
    row.appendChild(
      el("td", null, option.available ? option.label : `${option.label} (해당 없음)`),
    );
    row.appendChild(el("td", null, dash(`${option.months_added}개월`)));
    row.appendChild(el("td", null, dash(option.cost_label)));
    row.appendChild(el("td", null, dash(option.monthly_gain_label)));
    row.appendChild(el("td", null, dash(option.lifetime_gain_label)));
    row.appendChild(el("td", null, dash(option.extra_premium_label)));
    row.appendChild(el("td", null, dash(option.net_gain_label)));
    row.appendChild(el("td", null, option.breakeven_label || "—"));
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  box.appendChild(table);

  const reasons = options.filter((o) => !o.available && o.unavailable_reason);
  if (reasons.length) {
    const list = el("ul", "hint");
    reasons.forEach((o) =>
      list.appendChild(el("li", null, `${o.label} — ${o.unavailable_reason}`)),
    );
    box.appendChild(list);
  }

  // 표만 보면 '추가 건보료' 칸이 그냥 비용 한 줄로 읽힌다. 상충을 문장으로 한 번 더 짚는다.
  const shifted = options.filter((o) => o.available && o.cliff_shift_years > 0);
  if (shifted.length) {
    const worst = shifted.reduce((a, b) =>
      b.cliff_shift_years > a.cliff_shift_years ? b : a,
    );
    box.appendChild(
      el(
        "div",
        "alert alert-warn",
        `${worst.label}을 선택하시면 건강보험 피부양자 탈락이 ` +
          `${worst.cliff_shift_years}년 앞당겨집니다 ` +
          `(${report.current.cliff_year}년 → ${worst.cliff_year}년).`,
      ),
    );
  }

  box.appendChild(el("div", "advice", `이렇게 하세요 — ${report.verdict}`));
  const notes = el("ul", "hint");
  report.notes.forEach((note) => notes.appendChild(el("li", null, clean(note))));
  box.appendChild(notes);
}

/* ------------------------------------------------------------------ */
/* 가족에게 보여줄 한 장                                                  */
/* ------------------------------------------------------------------ */

async function loadFamilyReport() {
  const box = $("#family-report");
  const profile = state.analysis?.profile;
  if (!profile) return;

  try {
    renderFamilyReport(
      await api("/api/family-report", {
        method: "POST",
        body: JSON.stringify({ profile }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderFamilyReport(report) {
  const box = $("#family-report");
  box.innerHTML = "";

  box.appendChild(el("p", "hint", report.subject));
  box.appendChild(
    el("div", `alert ${report.now.length ? "alert-warn" : "alert-ok"}`, report.headline),
  );
  box.appendChild(el("p", "section-note", report.summary));

  const block = (title, items) => {
    if (!items.length) return;
    box.appendChild(el("h4", null, title));
    items.forEach((item) => {
      const card = el("div", "card");
      card.appendChild(
        el("div", "stat-label", item.value ? `${item.title} — ${item.value}` : item.title),
      );
      if (item.detail) card.appendChild(el("div", "stat-note", item.detail));
      // 기한은 눈에 띄어야 한다. 놓치면 되돌릴 수 없는 것들이다.
      if (item.deadline) card.appendChild(el("div", "alert alert-warn", `⏰ ${item.deadline}`));
      box.appendChild(card);
    });
  };

  block("기한이 있는 것 — 놓치면 되돌릴 수 없습니다", report.now);
  block("천천히 보셔도 되는 것", report.later);
  block("하지 않으셔도 되는 것", report.not_needed);

  if (report.watch_outs.length) {
    box.appendChild(el("h4", null, "가족이 함께 봐 주실 것"));
    const list = el("ul", "hint");
    report.watch_outs.forEach((note) => list.appendChild(el("li", null, note)));
    box.appendChild(list);
  }

  const notes = el("ul", "hint");
  [...report.assumptions, ...report.limits].forEach((note) =>
    notes.appendChild(el("li", null, note)),
  );
  box.appendChild(notes);

  // 실제 공유 경로는 카톡이다. 화면을 캡처해 보내면 숫자만 남고 단서가 사라진다.
  const share = el("details", "trace");
  share.appendChild(el("summary", null, "문자·카톡으로 보낼 수 있게 글로 보기"));
  const text = el("pre", "share-text", report.as_text);
  share.appendChild(text);
  const copy = el("button", "btn btn-secondary", "글 복사하기");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(report.as_text);
      copy.textContent = "복사했습니다";
    } catch (_) {
      // 클립보드 권한이 없으면 직접 선택하시게 둔다 — 실패를 숨기지 않는다.
      copy.textContent = "복사가 안 됩니다. 글을 직접 선택해 주세요";
    }
  });
  share.appendChild(copy);
  box.appendChild(share);
}

/* ------------------------------------------------------------------ */
/* 예정된 큰 지출                                                        */
/* ------------------------------------------------------------------ */

const MAN = 10000; // 입력은 만원 단위로 받는다 — 시니어에게 자연스럽고 오타가 줄어든다

function setupEvents() {
  $("#event-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const year = Number($("#event-year").value);
    const amount = Number($("#event-amount").value) * MAN;
    if (!year || !amount) return;

    applyEvents([
      ...(state.analysis?.profile?.life_events || []),
      { year, amount, label: $("#event-label").value.trim() || "큰 지출" },
    ]);
    $("#event-label").value = "";
    $("#event-year").value = "";
    $("#event-amount").value = "";
  });

  $("#btn-clear-events").addEventListener("click", () => applyEvents([]));
}

/* 프로파일에 지출을 반영해 전체 분석을 다시 돌린다. 지출이 들어가면 고갈 시점도
 * 처방도 민감도도 전부 달라지므로, 화면 일부만 고치지 않고 통째로 다시 그린다. */
async function applyEvents(events) {
  const profile = state.analysis?.profile;
  if (!profile) return;

  try {
    const data = await api("/api/analysis?briefing=false", {
      method: "POST",
      body: JSON.stringify({ ...profile, life_events: events }),
    });
    state.analysis = { ...data, briefing: state.analysis.briefing };
    renderAnalysis(state.analysis);
  } catch (err) {
    $("#event-list").textContent = err.message;
  }
}

function renderEvents(events) {
  const box = $("#event-list");
  box.innerHTML = "";
  if (!events.length) {
    box.appendChild(el("p", "hint", "아직 넣으신 큰 지출이 없습니다."));
    return;
  }
  events
    .slice()
    .sort((a, b) => a.year - b.year)
    .forEach((event) => {
      const row = el("div", "bucket-row");
      row.appendChild(el("div", "bucket-name", `${event.year}년 · ${event.label}`));
      row.appendChild(el("strong", null, formatWon(event.amount)));
      box.appendChild(row);
    });
}

/* ------------------------------------------------------------------ */
/* 처방 — "그래서 무엇을 하면 되나"                                      */
/* ------------------------------------------------------------------ */

async function loadPrescriptions() {
  const box = $("#prescriptions");
  const profile = state.analysis?.profile;
  if (!profile) return;

  box.textContent = "계산 중입니다...";
  try {
    renderPrescriptions(
      await api("/api/prescriptions", {
        method: "POST",
        body: JSON.stringify({
          profile,
          target_age: Number($("#target-age").value),
        }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderPrescriptions(plan) {
  const box = $("#prescriptions");
  box.innerHTML = "";

  box.appendChild(
    el("div", `alert ${plan.already_safe ? "alert-ok" : "alert-warn"}`, plan.summary),
  );

  plan.options.forEach((option) => {
    // 색이 아니라 아이콘과 글자로 구분한다 (시니어 접근성).
    const mark = option.feasible ? "✅" : option.improves ? "🟠" : "⬜";
    const card = el("div", `signal ${option.feasible ? "" : "high"}`);

    card.appendChild(el("div", "signal-title", `${mark} ${option.label}`));
    card.appendChild(el("p", null, option.headline));

    if (option.change_label) {
      card.appendChild(el("div", "advice", `해야 할 일 — ${option.change_label}`));
    }
    if (option.gained_years > 0) {
      card.appendChild(
        el("div", "evidence", `자산이 버티는 기간이 ${option.gained_years}년 늘어납니다.`),
      );
    }
    if (option.detail) card.appendChild(el("p", "hint", option.detail));
    if (option.caution) card.appendChild(el("div", "alert alert-warn", `⚠️ ${option.caution}`));

    box.appendChild(card);
  });
}

/* ------------------------------------------------------------------ */
/* 민감도 — 우리 숫자 자체를 흔들어 본다                                 */
/* ------------------------------------------------------------------ */

async function loadSensitivity() {
  const box = $("#sensitivity");
  const profile = state.analysis?.profile;
  if (!profile) return;

  try {
    renderSensitivity(
      await api("/api/sensitivity", {
        method: "POST",
        body: JSON.stringify({ profile }),
      }),
    );
  } catch (err) {
    box.textContent = err.message;
  }
}

function renderSensitivity(report) {
  const box = $("#sensitivity");
  box.innerHTML = "";
  box.appendChild(el("div", "alert alert-warn", report.summary.replace(/\*\*/g, "")));

  // 흔들림 폭을 막대 길이로. 가장 큰 항목을 100% 로 잡는다.
  const widest = Math.max(1, ...report.cases.map((c) => c.swing_years));

  report.cases.forEach((item) => {
    const row = el("div", "bucket-row");

    const name = el("div", "bucket-name");
    name.appendChild(el("strong", null, item.label));
    name.appendChild(el("div", "hint", item.detail));
    row.appendChild(name);

    const meter = el("div", "sensitivity-meter");
    const bar = el("div", "sensitivity-bar");
    bar.style.width = `${Math.round((item.swing_years / widest) * 100)}%`;
    // 색만으로 구분하지 않는다 — 숫자를 함께 적는다.
    meter.appendChild(bar);
    meter.appendChild(el("span", "sensitivity-value", `${item.swing_years}년`));
    row.appendChild(meter);

    box.appendChild(row);
  });

  box.appendChild(el("p", "hint", report.caveat));
}

/* ------------------------------------------------------------------ */
/* 제도 변화 (기능 ④·⑧)                                                */
/* ------------------------------------------------------------------ */

function setupPolicy() {
  $("#btn-policy-impact").addEventListener("click", () => {
    // 이미 분석한 프로파일이 있으면 그걸 쓰고, 없으면 대화 세션에서 꺼낸다.
    if (state.analysis?.profile) {
      requestPolicyImpact({ profile: state.analysis.profile });
    } else if (state.sessionId) {
      requestPolicyImpact({ session_id: state.sessionId });
    } else {
      $("#policy-impact").textContent = "먼저 상담을 진행해 주세요.";
    }
  });

  $$("[data-policy-sample]").forEach((btn) =>
    btn.addEventListener("click", () =>
      requestPolicyImpact({ sample: btn.dataset.policySample }),
    ),
  );
}

async function requestPolicyImpact(body) {
  try {
    renderPolicyImpact(
      await api("/api/policy-impact", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    );
  } catch (err) {
    $("#policy-impact").textContent = err.message;
  }
}

function policyBadge(fact) {
  return el("span", `badge ${fact.is_confirmed ? "badge-ok" : "badge-warn"}`, fact.badge);
}

async function loadPolicyList() {
  const box = $("#policy-list");
  try {
    const facts = await api("/api/policies");
    box.innerHTML = "";
    facts.forEach((fact) => {
      const card = el("div", "card");
      const title = el("div", "signal-title");
      title.appendChild(policyBadge(fact));
      title.appendChild(document.createTextNode(` ${fact.title}`));
      card.appendChild(title);
      card.appendChild(el("p", null, fact.summary));
      card.appendChild(el("div", "advice", fact.caution));
      card.appendChild(el("p", "hint", fact.trust_note));
      box.appendChild(card);
    });
  } catch (_) {
    box.textContent = "제도 정보를 불러오지 못했습니다.";
  }
}

function renderPolicyImpact(result) {
  const box = $("#policy-impact");
  box.innerHTML = "";
  const fact = result.policy;

  const head = el("div", "card");
  const title = el("div", "signal-title");
  title.appendChild(policyBadge(fact));
  title.appendChild(document.createTextNode(` ${fact.title}`));
  head.appendChild(title);
  head.appendChild(el("p", "hint", fact.trust_note));
  box.appendChild(head);

  // 확정되지 않은 제도라는 사실을 숫자보다 먼저 보여준다.
  if (!fact.is_confirmed) {
    box.appendChild(el("div", "alert alert-warn", `⚠️ ${fact.notice}`));
  }

  box.appendChild(el("div", `alert ${result.material ? "alert-warn" : "alert-ok"}`, result.headline));
  box.appendChild(el("p", null, result.reason));

  // 비교표는 금액 표기까지 서버가 완성해서 보낸다 (core/formatting.py 한 곳에서만 포맷).
  if (result.comparison.length) {
    const table = el("table");
    const thead = el("thead");
    const header = el("tr");
    ["항목", result.current.label, result.proposed.label, "차이"].forEach((text) =>
      header.appendChild(el("th", null, text)),
    );
    thead.appendChild(header);
    table.appendChild(thead);

    const tbody = el("tbody");
    result.comparison.forEach((row) => {
      const tr = el("tr");
      [row.label, row.current, row.proposed, row.delta].forEach((text) =>
        tr.appendChild(el("td", null, text)),
      );
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    box.appendChild(table);
  }

  box.appendChild(el("div", "advice", `이렇게 하세요 — ${result.verdict}`));

  const notes = el("ul", "hint");
  result.notes.forEach((note) => notes.appendChild(el("li", null, note)));
  box.appendChild(notes);
}

const SAMPLE_SCAM = `[특별안내] 고객님만 드리는 기회입니다.
정부 세법이 바뀌면서 ISA 비과세 혜택이 곧 폐지됩니다.
지금 갈아타지 않으시면 손해입니다.
원금 보장되면서 월 3% 확정 수익 나오는 상품이고요,
오늘까지만 선착순으로 받습니다.
자세한 내용은 텔레그램으로 연락 주세요. 가족한테는 비밀로 해주세요.`;

function setupFraud() {
  $("#btn-sample-scam").addEventListener("click", () => {
    $("#fraud-input").value = SAMPLE_SCAM;
    $("#fraud-form").requestSubmit();
  });

  $("#fraud-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#fraud-input").value.trim();
    if (!text) return;
    try {
      const result = await api("/api/fraud-check", {
        method: "POST",
        body: JSON.stringify({ text, session_id: state.sessionId }),
      });
      renderFraud(result);
    } catch (err) {
      $("#fraud-result").textContent = err.message;
    }
  });
}

function renderFraud(result) {
  const box = $("#fraud-result");
  box.innerHTML = "";

  const styles = { 위험: ["alert-danger", "🔴"], 주의: ["alert-warn", "🟠"], 낮음: ["alert-ok", "🟢"] };
  const [cls, icon] = styles[result.risk_level] || styles["주의"];

  const banner = el("div", `alert ${cls}`);
  banner.appendChild(el("strong", null, `${icon} ${result.risk_level} `));
  banner.appendChild(document.createTextNode(result.summary));
  box.appendChild(banner);

  if (result.signals.length) {
    box.appendChild(el("h3", null, "확인이 필요한 부분"));
    result.signals.forEach((signal) => {
      const card = el("div", `signal ${signal.severity}`);
      card.appendChild(
        el("div", "signal-title", `${signal.severity === "high" ? "🔴" : "🟠"} ${signal.label}`),
      );
      if (signal.evidence.length) {
        const ev = el("div", "evidence");
        ev.textContent = "걸린 표현: ";
        signal.evidence.forEach((text, i) => {
          if (i) ev.appendChild(document.createTextNode(", "));
          ev.appendChild(el("code", null, text));
        });
        card.appendChild(ev);
      }
      card.appendChild(el("p", null, signal.why));
      card.appendChild(el("div", "advice", `이렇게 하세요 — ${signal.advice}`));
      box.appendChild(card);
    });
  }

  if (result.policy_checks.length) {
    box.appendChild(el("h3", null, "언급된 제도의 실제 상태"));
    box.appendChild(
      el(
        "p",
        "section-note",
        "뉴스는 '발표'와 '시행'을 구분하지 않습니다. 발표된 개편안은 바뀌거나 무산될 수 있습니다.",
      ),
    );
    result.policy_checks.forEach((check) => {
      const card = el("div", "card");
      const badge = el(
        "span",
        `badge ${check.status === "시행" ? "badge-ok" : "badge-warn"}`,
        check.status === "시행" ? "✅ 시행 중" : `⚠️ ${check.status} (미확정)`,
      );
      const title = el("div", "signal-title");
      title.appendChild(badge);
      title.appendChild(document.createTextNode(` ${check.title}`));
      card.appendChild(title);
      card.appendChild(el("p", null, check.summary));
      card.appendChild(el("div", "advice", check.caution));
      card.appendChild(el("p", "hint", `출처: ${check.source}`));
      box.appendChild(card);
    });
  }

  if (result.profile_notes.length) {
    box.appendChild(el("h3", null, "고객님 자산과의 관계"));
    const list = el("ul");
    result.profile_notes.forEach((note) => list.appendChild(el("li", null, note)));
    box.appendChild(list);
  }

  box.appendChild(
    el(
      "p",
      "hint",
      "⚠️ 이 확인은 사기 여부를 확정하지 않습니다. 확인이 필요한 신호를 알려드릴 뿐입니다. " +
        "신호가 없어도 안전이 보장되지 않으며, 가입 전에는 반드시 거래하시는 금융회사에 직접 문의하세요. " +
        "금융감독원 파인(fine.fss.or.kr)에서 제도권 금융회사인지 조회하실 수 있습니다.",
    ),
  );
}

/* ------------------------------------------------------------------ */

async function init() {
  setupAccessibility();
  setupTabs();
  setupChat();
  setupPolicy();
  setupEvents();
  setupFraud();
  $("#target-age").addEventListener("change", loadPrescriptions);
  $("#pension-years").addEventListener("change", loadSeverance);
  $("#hi-apply").addEventListener("click", loadHealthInsurance);
  $("#hi-family").addEventListener("change", loadHealthInsurance);
  $("#np-apply").addEventListener("click", loadNationalPension);
  loadPolicyList();

  $$("[data-sample]").forEach((btn) =>
    btn.addEventListener("click", () => loadSample(btn.dataset.sample)),
  );

  try {
    const health = await api("/api/health");
    $("#llm-status").textContent = health.llm;
  } catch (_) {
    $("#llm-status").textContent = "서버 상태를 확인할 수 없습니다.";
  }

  // 지난 상담이 있으면 이어서 연다. 시니어 사용자에게 12개 질문을 다시 물어보는
  // 것이 가장 큰 이탈 요인이라, 이어보기가 저장소를 붙인 첫 번째 이유다.
  if (!(await resumeLastSession())) {
    await startSession();
  }
}

document.addEventListener("DOMContentLoaded", init);
