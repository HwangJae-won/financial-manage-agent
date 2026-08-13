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
  log.scrollTop = log.scrollHeight;

  const ratio = data.total ? data.answered / data.total : 0;
  $("#progress-fill").style.width = `${Math.round(ratio * 100)}%`;
  $("#progress-text").textContent = `${data.answered} / ${data.total}`;
  $(".progress-track").setAttribute("aria-valuenow", String(data.answered));
  $(".progress-track").setAttribute("aria-valuemax", String(data.total));

  $("#btn-see-result").classList.toggle("hidden", !(data.ready && !data.done));
  $("#chat-hint").textContent = data.error || "";

  const finished = data.done;
  $("#chat-input").disabled = finished;
  $("#chat-send").disabled = finished;

  if (finished) loadAnalysis();
}

async function startSession() {
  try {
    renderConversation(await api("/api/sessions", { method: "POST" }));
  } catch (err) {
    $("#chat-hint").textContent = err.message;
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

  $("#btn-restart").addEventListener("click", () => {
    state.analysis = null;
    $("#result-body").classList.add("hidden");
    $("#result-empty").classList.remove("hidden");
    startSession();
  });

  $("#btn-see-result").addEventListener("click", loadAnalysis);
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
  setupFraud();
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

  await startSession();
}

document.addEventListener("DOMContentLoaded", init);
