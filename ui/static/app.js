"use strict";

// ---------------------------------------------------------------- browser identity & state
// Chats live on the server (the recent-chats list and the agent's memory). This browser keeps
// only its random client id, which marks the chats as its own, and the id of the open chat.
const STORE = { client: "rag.clientId", thread: "rag.threadId" };

function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function save(key, value) {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* private mode */ }
}
try { localStorage.removeItem("rag.messages"); } catch { /* chats used to be kept here */ }

function newClientId() {
  if (crypto.randomUUID) return crypto.randomUUID();  // secure contexts (https, localhost)
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}
const clientId = load(STORE.client) || newClientId();
save(STORE.client, clientId);

let threadId = (load(STORE.thread) || "").replace(/"/g, "") || null;  // older versions stored it as JSON
let conversations = [];  // recent chats: [{id, title, created_at, updated_at}], newest first
let busy = false;
let messageSeq = 0;

// ---------------------------------------------------------------- elements
const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");   // scroll container
const threadEl = $("thread");       // centered message column
const emptyEl = $("empty-state");
const form = $("composer");
const input = $("question");
const sendBtn = $("send");

// ---------------------------------------------------------------- safe markdown
// Escapes HTML first, then applies a small, predictable subset of Markdown:
// fenced code, inline code, headings, lists, bold/italic, paragraphs, and [n] citations.
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderMarkdown(text, msgId, sourceNumbers) {
  // Native model citations (【1】, 【1†L3-L5】) as [1]; the server does the same for final answers.
  text = text.replace(/【\s*(\d{1,2})(?:\s*†[^】]*)?\s*】/g, "[$1]");
  const blocks = [];
  // Pull fenced code blocks out first so nothing inside them is formatted.
  let src = text.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  src = escapeHtml(src);

  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,2})\]/g, (m, n) => sourceNumbers.has(Number(n))
      ? `<a class="cite" href="#src-${msgId}-${n}" data-msg="${msgId}" data-n="${n}" title="Source ${n}">${n}</a>`
      : m);

  const out = [];
  let list = null;  // "ul" | "ol"
  let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join("<br>"))}</p>`); para = []; } };
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const line of src.split("\n")) {
    const code = line.match(/^\u0000(\d+)\u0000$/);
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (code) { flushPara(); closeList(); out.push(blocks[Number(code[1])]); }
    else if (heading) { flushPara(); closeList(); out.push(`<h${heading[1].length > 2 ? 4 : 3}>${inline(heading[2])}</h${heading[1].length > 2 ? 4 : 3}>`); }
    else if (bullet || numbered) {
      flushPara();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
    }
    else if (!line.trim()) { flushPara(); closeList(); }
    else { closeList(); para.push(line); }
  }
  flushPara(); closeList();
  // Code blocks embedded mid-line (rare) are restored here too.
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

// ---------------------------------------------------------------- rendering
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.append(c);
  return node;
}

function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
// While an answer streams in, follow it only if the reader hasn't scrolled up.
function isNearBottom() { return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120; }

function renderUser(msg) {
  return el("div", { class: "msg user" }, [
    el("div", { class: "bubble", text: msg.text }),
    msg.note ? el("div", { class: "note", text: msg.note }) : null,
  ]);
}

const ACTION_LABEL = { blocked: "blocked", redacted: "redacted", dropped: "dropped", fixed: "fixed", error: "error" };
const VALIDATOR_LABEL = {
  DetectPII: "PII", SecretsPresent: "secret", InjectionPatterns: "prompt injection", DetectJailbreak: "jailbreak",
  ToxicLanguage: "toxic language", CitationCheck: "invalid citation", RelevanceFloor: "low-relevance chunk", Grounding: "unsupported claim", Guard: "guardrail",
};

// One badge per (stage, validator, action), with a count: "retrieval · PII redacted ×2".
function guardrailBadges(events) {
  const groups = new Map();
  for (const g of events) {
    const key = `${g.stage}|${g.validator}|${g.action}`;
    const entry = groups.get(key) || { ...g, count: 0, details: [] };
    entry.count += 1;
    if (g.detail) entry.details.push(g.detail);
    groups.set(key, entry);
  }
  return [...groups.values()].map((g) => el("span", {
    class: `badge g-${g.action}`,
    title: g.details.join("\n") || `${g.stage} guardrail`,
    text: `${g.stage} · ${VALIDATOR_LABEL[g.validator] || g.validator} ${ACTION_LABEL[g.action] || g.action}${g.count > 1 ? ` ×${g.count}` : ""}`,
  }));
}

const messageData = new Map();  // rendered bot message id → message (for the source viewer)

function renderBot(msg) {
  const id = msg.id;
  messageData.set(id, msg);
  const sources = msg.sources || [];
  const numbers = new Set(sources.map((s) => s.number));
  const outer = el("div", { class: `msg bot${msg.route === "blocked" ? " blocked" : ""}${msg.error ? " error" : ""}` });
  const wrap = el("div", { class: "body" });
  outer.append(el("div", { class: "avatar", "aria-hidden": "true" }), wrap);

  const bubble = el("div", { class: "bubble" });
  if (msg.error) bubble.textContent = msg.text;
  else bubble.innerHTML = renderMarkdown(msg.text, id, numbers);  // escaped inside renderMarkdown
  wrap.append(bubble);

  if (!msg.error) {
    const meta = el("div", { class: "meta" });
    if (msg.route) meta.append(el("span", { class: `badge ${msg.route}`, text: msg.route }));
    if (msg.search_query) meta.append(el("span", { text: `search: “${msg.search_query}”` }));
    if (msg.duration_seconds != null) meta.append(el("span", { text: `${msg.duration_seconds.toFixed(1)}s` }));
    meta.append(...guardrailBadges(msg.guardrails || []));
    wrap.append(meta);
  }

  if (sources.length) {
    const list = el("ul", { class: "sources" }, sources.map((s) => {
      const where = [s.location, s.section, s.source_type].filter(Boolean).join(" · ");
      return el("li", { id: `src-${id}-${s.number}` }, el("button", {
        class: "source-link", type: "button", "data-msg": id, "data-n": String(s.number),
        title: s.chunk_id ? "Open this source" : "Source text unavailable",
        ...(s.chunk_id ? {} : { disabled: "" }),
      }, [
        el("span", { class: "num", text: `[${s.number}]` }),
        el("span", { class: "file", text: s.file }),
        where ? el("span", { class: "where", text: ` — ${where}` }) : null,
      ]));
    }));
    wrap.append(el("details", { class: "panel sources-panel" }, [
      el("summary", { text: `${sources.length} source${sources.length > 1 ? "s" : ""}` }), list,
    ]));
  }

  const events = msg.guardrails || [];
  if (events.length) {
    wrap.append(el("details", { class: "panel" }, [
      el("summary", { text: `Guardrails: ${events.length} event${events.length > 1 ? "s" : ""}` }),
      el("ul", { class: "events" }, events.map((g) =>
        el("li", { text: `${g.stage} · ${g.validator} · ${g.action}${g.detail ? ` — ${g.detail}` : ""}` }))),
    ]));
  }
  return outer;
}

function append(msg) {
  emptyEl.hidden = true;
  threadEl.append(msg.role === "user" ? renderUser(msg) : renderBot(msg));
  scrollToBottom();
}

function clearThread() {
  threadEl.querySelectorAll(".msg").forEach((node) => node.remove());
  messageData.clear();
  closeSource();
}

const REDACTED_NOTE = "Sensitive data was redacted before processing";
const inputRedacted = (data) => (data.guardrails || []).some((g) => g.stage === "input" && g.action === "redacted");

// A turn as the API returns it (ChatResponse) → the bot message to render.
function botMessage(data) {
  return {
    role: "bot", id: `m${++messageSeq}`, text: data.answer, route: data.route, search_query: data.search_query,
    sources: data.sources, guardrails: data.guardrails, duration_seconds: data.duration_seconds,
  };
}

// The in-progress answer: typing dots with a status ("Searching the knowledge base…") until
// the first token arrives, then the answer rendering live as tokens stream in. The status row
// comes back below the text for later steps ("Checking the answer…").
function streamingMessage() {
  const status = el("div", { class: "stream-status", text: "Thinking…" });
  const dots = el("div", { class: "typing", "aria-label": "Assistant is thinking" }, [el("span"), el("span"), el("span")]);
  const row = el("div", { class: "stream-row" }, [dots, status]);
  const bubble = el("div", { class: "bubble streaming" });
  bubble.hidden = true;
  const node = el("div", { class: "msg bot pending", id: "typing" }, [
    el("div", { class: "avatar", "aria-hidden": "true" }),
    el("div", { class: "body" }, [bubble, row]),
  ]);
  threadEl.append(node);
  scrollToBottom();

  let text = "";
  let frame = 0;
  const paint = () => {
    frame = 0;
    const follow = isNearBottom();
    bubble.innerHTML = renderMarkdown(text, "stream", new Set());  // escaped inside renderMarkdown
    if (follow) scrollToBottom();
  };
  return {
    node,
    status(value) {
      status.textContent = value;
      row.hidden = false;
      if (isNearBottom()) scrollToBottom();
    },
    token(piece) {
      if (!text) bubble.hidden = false;
      row.hidden = true;
      text += piece;
      if (!frame) frame = requestAnimationFrame(paint);  // at most one render per frame
    },
    remove() { if (frame) cancelAnimationFrame(frame); node.remove(); },
  };
}

// Citation [n] or source-list click: open that source in the right panel; a citation also
// opens and flashes the answer's source list entry.
messagesEl.addEventListener("click", (event) => {
  const target = event.target.closest("a.cite, button.source-link");
  if (!target) return;
  event.preventDefault();
  const { msg, n } = target.dataset;
  if (target.matches("a.cite")) {
    const item = document.getElementById(`src-${msg}-${n}`);
    if (item) {
      item.closest("details").open = true;
      item.scrollIntoView({ block: "nearest", behavior: "smooth" });
      item.classList.add("flash");
      setTimeout(() => item.classList.remove("flash"), 1600);
    }
  }
  openSource(msg, Number(n));
});

// ---------------------------------------------------------------- source viewer (right panel)
const sourcePanel = $("source-panel");
const sourceBody = $("source-body");
let sourceRequest = 0;  // ignore responses to superseded clicks

function markActiveSource(msgId, n) {
  document.querySelectorAll(".source-link.active").forEach((b) => b.classList.remove("active"));
  if (msgId) document.querySelector(`#src-${msgId}-${n} .source-link`)?.classList.add("active");
}

// Passage text with the highlighted ranges wrapped in <mark> (text nodes only: no HTML injection).
function highlightedText(text, ranges) {
  const nodes = [];
  let at = 0;
  for (const [start, end] of ranges) {
    if (start < at) continue;
    if (start > at) nodes.push(document.createTextNode(text.slice(at, start)));
    nodes.push(el("mark", { text: text.slice(start, end) }));
    at = end;
  }
  if (at < text.length) nodes.push(document.createTextNode(text.slice(at)));
  return nodes;
}

// Highlight ranges index the passage text as sent; follow the leading newlines trimmed for display.
function shiftRanges(ranges, text) {
  const shift = text.length - text.replace(/^\n+/, "").length;
  return ranges.map(([start, end]) => [Math.max(0, start - shift), end - shift]).filter(([, end]) => end > 0);
}

async function openSource(msgId, n) {
  const msg = messageData.get(msgId);
  const source = msg && (msg.sources || []).find((s) => s.number === n);
  if (!source || !source.chunk_id) return;
  const request = ++sourceRequest;

  $("source-kicker").textContent = `Source ${n}`;
  $("source-file").textContent = source.file;
  $("source-where").textContent = [source.location, source.section, source.source_type].filter(Boolean).join(" · ");
  sourceBody.replaceChildren(el("p", { class: "source-loading", text: "Loading source…" }));
  sourcePanel.hidden = false;
  document.documentElement.classList.add("source-open");
  markActiveSource(msgId, n);

  let data;
  try {
    data = await api("/sources/passage", { chunk_id: source.chunk_id, answer: msg.text, citation: n });
  } catch (error) {
    if (request === sourceRequest) {
      sourceBody.replaceChildren(el("p", { class: "source-error", text: `Couldn't load the source: ${error.message}` }));
    }
    return;
  }
  if (request !== sourceRequest) return;

  $("source-where").textContent = [data.location, data.section, data.source_type].filter(Boolean).join(" · ");
  const highlighted = data.passages.some((p) => p.highlights.length);
  sourceBody.replaceChildren(
    ...data.passages.map((p) => el("div", { class: `passage${p.cited ? " cited" : ""}` }, [
      p.cited ? el("span", { class: "passage-label", text: "Cited passage" }) : null,
      p.heading ? el("div", { class: "passage-heading", text: p.heading }) : null,
      ...highlightedText(p.content.replace(/^\n+/, ""), p.cited ? shiftRanges(p.highlights, p.content) : []),
    ])),
    highlighted ? null : el("p", { class: "source-note", text: "No specific sentence matched the answer's wording; the whole cited passage was used." }),
  );
  // Bring the first highlight (or the cited passage) into view.
  const focus = sourceBody.querySelector("mark") || sourceBody.querySelector(".passage.cited");
  if (focus) focus.scrollIntoView({ block: "center" });
}

function closeSource() {
  sourceRequest++;
  sourcePanel.hidden = true;
  document.documentElement.classList.remove("source-open");
  markActiveSource(null);
}
$("source-close").addEventListener("click", closeSource);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !sourcePanel.hidden && !document.documentElement.classList.contains("sidebar-open")) closeSource();
});

// ---------------------------------------------------------------- API
async function api(path, body, method) {
  const response = await fetch(path, {
    method: method || (body ? "POST" : "GET"),
    headers: { "X-Client-Id": clientId, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (response.status === 204) return null;
  let data = null;
  try { data = await response.json(); } catch { /* non-JSON error page */ }
  if (!response.ok) {
    const detail = data && data.detail;
    const message = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(message || `Request failed (HTTP ${response.status})`);
  }
  return data;
}

// POST /chat/stream answers with newline-delimited JSON events; yield them as they arrive.
async function* streamEvents(path, body) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json", "X-Client-Id": clientId }, body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    let data = null;
    try { data = await response.json(); } catch { /* non-JSON error page */ }
    const detail = data && data.detail;
    const message = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(message || `Request failed (HTTP ${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
    if (done) break;
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  const userMsg = { role: "user", text: question };
  append(userMsg);
  const userNode = threadEl.lastElementChild;
  const pending = streamingMessage();
  try {
    let data = null;
    for await (const event of streamEvents("/chat/stream", { question, thread_id: threadId })) {
      if (event.type === "status") pending.status(event.text);
      else if (event.type === "token") pending.token(event.text);
      else if (event.type === "done") data = event.response;
      else if (event.type === "error") throw new Error(event.detail);
    }
    if (!data) throw new Error("The connection closed before the answer was complete");
    pending.remove();
    threadId = data.thread_id;
    save(STORE.thread, threadId);
    // If the input guardrails redacted PII/secrets, show only the redacted text (that's also
    // what the server stored), so the original doesn't stay on screen either.
    if (inputRedacted(data) && data.question && data.question !== question) {
      userNode.replaceWith(renderUser({ role: "user", text: data.question, note: REDACTED_NOTE }));
    }
    // The final answer replaces the streamed text: the output guardrails may have changed it.
    append(botMessage(data));
    loadConversations();  // a new chat appears in the list; an old one moves to the top
  } catch (error) {
    pending.remove();
    append({ role: "bot", id: `m${++messageSeq}`, error: true, text: `Couldn't get an answer: ${error.message}` });
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

// ---------------------------------------------------------------- composer
function autosize() { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 200)}px`; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); form.requestSubmit(); }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  autosize();
  ask(question);
});
document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => ask(chip.dataset.q)));

function newChat() {
  if (busy) return;
  threadId = null;
  save(STORE.thread, null);
  clearThread();
  emptyEl.hidden = false;
  renderConversations();
  showView("chat-view");
}
$("new-chat").addEventListener("click", newChat);
$("new-chat-top").addEventListener("click", newChat);

// ---------------------------------------------------------------- recent chats
const chatListEl = $("chat-list");
const ICONS = {
  pencil: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
  trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6"/></svg>',
};

async function loadConversations() {
  try { conversations = await api("/conversations?limit=100"); } catch { /* keep the current list */ }
  renderConversations();
}

// ChatGPT-style groups by last activity.
function groupLabel(unixSeconds) {
  const day = 86400000;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const t = unixSeconds * 1000;
  if (t >= today.getTime()) return "Today";
  if (t >= today.getTime() - day) return "Yesterday";
  if (t >= today.getTime() - 7 * day) return "Previous 7 days";
  if (t >= today.getTime() - 30 * day) return "Previous 30 days";
  return new Date(t).toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function renderConversations() {
  const nodes = [];
  let group = null;
  for (const chat of conversations) {
    const label = groupLabel(chat.updated_at);
    if (label !== group) { group = label; nodes.push(el("li", { class: "chat-group", text: label })); }
    nodes.push(chatItem(chat));
  }
  chatListEl.replaceChildren(...nodes);
  $("chats-empty").hidden = conversations.length > 0;
}

function chatItem(chat) {
  const item = el("li", { class: `chat-item${chat.id === threadId ? " active" : ""}` });
  const open = el("button", { class: "chat-open", type: "button", title: chat.title },
    el("span", { class: "chat-title", text: chat.title }));
  if (chat.id === threadId) open.setAttribute("aria-current", "true");
  const rename = el("button", { class: "chat-action", type: "button", title: "Rename", "aria-label": `Rename “${chat.title}”` });
  const remove = el("button", { class: "chat-action danger", type: "button", title: "Delete", "aria-label": `Delete “${chat.title}”` });
  rename.innerHTML = ICONS.pencil;  // static markup
  remove.innerHTML = ICONS.trash;
  open.addEventListener("click", () => openConversation(chat.id));
  rename.addEventListener("click", () => startRename(item, chat));
  remove.addEventListener("click", () => deleteConversation(chat));
  item.append(open, el("div", { class: "chat-actions" }, [rename, remove]));
  return item;
}

async function openConversation(id) {
  if (busy) return;
  if (id === threadId && threadEl.querySelector(".msg")) { showView("chat-view"); return; }
  let chat;
  try {
    chat = await api(`/conversations/${encodeURIComponent(id)}`);
  } catch {
    // Deleted elsewhere, or a chat from before chats were saved: start fresh.
    if (id === threadId) newChat();
    loadConversations();
    return;
  }
  threadId = id;
  save(STORE.thread, id);
  clearThread();
  emptyEl.hidden = chat.turns.length > 0;
  for (const turn of chat.turns) {
    append({ role: "user", text: turn.question, note: inputRedacted(turn) ? REDACTED_NOTE : "" });
    append(botMessage(turn));
  }
  renderConversations();
  showView("chat-view");
  scrollToBottom();
}

function startRename(item, chat) {
  const field = el("input", { class: "chat-rename", value: chat.title, maxlength: "200", "aria-label": "Chat title" });
  item.classList.add("editing");
  item.replaceChildren(field);
  field.focus();
  field.select();
  let finished = false;
  const finish = async (keep) => {
    if (finished) return;
    finished = true;
    const title = field.value.trim();
    if (keep && title && title !== chat.title) {
      try { await api(`/conversations/${encodeURIComponent(chat.id)}`, { title }, "PATCH"); } catch { /* list reload shows the old title */ }
    }
    loadConversations();
  };
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    else if (event.key === "Escape") { event.preventDefault(); finish(false); }
  });
  field.addEventListener("blur", () => finish(true));
}

async function deleteConversation(chat) {
  if (!confirm(`Delete “${chat.title}”?\n\nThe conversation and the assistant's memory of it are removed for good.`)) return;
  try {
    await api(`/conversations/${encodeURIComponent(chat.id)}`, null, "DELETE");
  } catch (error) {
    alert(`Couldn't delete the chat: ${error.message}`);
    return;
  }
  if (chat.id === threadId) newChat();
  conversations = conversations.filter((c) => c.id !== chat.id);
  renderConversations();
}

// ---------------------------------------------------------------- modes (sidebar)
function showView(viewId) {
  document.querySelectorAll(".mode").forEach((mode) => {
    const active = mode.dataset.view === viewId;
    mode.classList.toggle("active", active);
    mode.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".view").forEach((view) => { view.hidden = view.id !== viewId; });
  $("view-title").textContent = viewId === "chat-view" ? "Chat" : "Search";
  if (isMobile()) setSidebar(false);
  (viewId === "chat-view" ? input : $("search-query")).focus();
}
document.querySelectorAll(".mode").forEach((mode) => mode.addEventListener("click", () => showView(mode.dataset.view)));

// ---------------------------------------------------------------- sidebar
// Desktop: the sidebar is collapsible and the choice is remembered. Mobile: it slides over the chat.
const mobileQuery = matchMedia("(max-width: 820px)");
const isMobile = () => mobileQuery.matches;
function setSidebar(open) {
  const root = document.documentElement;
  if (isMobile()) {
    root.classList.toggle("sidebar-open", open);
    $("scrim").hidden = !open;
  } else {
    root.classList.toggle("sidebar-closed", !open);
    try { localStorage.setItem("rag.sidebar", open ? "open" : "closed"); } catch { /* ignore */ }
  }
}
$("sidebar-open").addEventListener("click", () => setSidebar(true));
$("sidebar-close").addEventListener("click", () => setSidebar(false));
$("scrim").addEventListener("click", () => setSidebar(false));
mobileQuery.addEventListener("change", () => {
  document.documentElement.classList.remove("sidebar-open");
  $("scrim").hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.documentElement.classList.contains("sidebar-open")) setSidebar(false);
});

// ---------------------------------------------------------------- search
$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("search-query").value.trim();
  if (!query) return;
  const results = $("search-results");
  results.replaceChildren(el("p", { class: "hint", text: "Searching…" }));
  const source = $("search-source").value;
  const type = $("search-type").value;
  try {
    const data = await api("/search", {
      query, k: Number($("search-k").value),
      source_types: source ? [source] : [], file_types: type ? [type] : [],
    });
    if (!data.results.length) {
      results.replaceChildren(el("p", { class: "hint", text: "No results." }));
      return;
    }
    results.replaceChildren(
      el("p", { class: "hint", text: `${data.results.length} results in ${data.duration_seconds.toFixed(2)}s` }),
      ...data.results.map((r) => {
        const scores = [
          r.search_score != null ? `${r.search_mode || "search"} ${r.search_score.toFixed(3)}` : null,
          r.rerank_score != null ? `rerank ${r.rerank_score.toFixed(2)}` : null,
        ].filter(Boolean).join(" · ");
        return el("article", { class: "result" }, [
          el("header", {}, [
            el("span", { class: "rank", text: `#${r.rank}` }),
            el("span", { class: "file", text: r.file }),
            el("span", { class: "badge", text: [r.location, r.file_type, r.source_type].filter(Boolean).join(" · ") }),
            el("span", { class: "scores", text: scores }),
          ]),
          el("pre", { text: r.content }),
        ]);
      }),
    );
  } catch (error) {
    results.replaceChildren(el("p", { class: "hint", text: `Search failed: ${error.message}` }));
  }
});

// ---------------------------------------------------------------- status
async function loadStatus() {
  const status = $("status");
  try {
    const h = await api("/health");
    status.className = "status ok";
    status.innerHTML = "";
    status.append(el("span", { class: "dot" }), "Connected");
    const rows = [
      ["Model", h.llm_model], ...(h.llm_fallbacks || []).map((m) => ["Fallback", m]),
      ["Provider", h.llm_provider], ["Planner", h.planner],
      ["Retrieval", `${h.retrieval_mode} search`], ["Embeddings", h.embedding_model],
      ["Collection", h.collection], ["Qdrant", h.qdrant], ["Guardrails", h.guardrails ? "on" : "off"],
    ];
    $("sysinfo").replaceChildren(...rows.map(([k, v]) =>
      el("div", {}, [el("dt", { text: k }), el("dd", { text: String(v ?? "—"), title: String(v ?? "") })])));
  } catch {
    status.className = "status down";
    status.innerHTML = "";
    status.append(el("span", { class: "dot" }), "API unavailable");
    $("sysinfo").replaceChildren(el("div", {}, [el("dt", { text: "Status" }), el("dd", { text: "Unavailable" })]));
  }
}

// ---------------------------------------------------------------- theme
// Auto follows the OS; Light/Dark pin a theme.
const themeButtons = document.querySelectorAll("#theme-picker [data-theme-value]");
function markTheme() {
  const current = document.documentElement.dataset.theme || "";
  themeButtons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.themeValue === current)));
}
themeButtons.forEach((button) => button.addEventListener("click", () => {
  const root = document.documentElement;
  const value = button.dataset.themeValue;
  if (value) root.dataset.theme = value; else delete root.dataset.theme;
  try { value ? localStorage.setItem("rag.theme", value) : localStorage.removeItem("rag.theme"); } catch { /* ignore */ }
  markTheme();
}));
markTheme();

// ---------------------------------------------------------------- start
loadConversations();
if (threadId) openConversation(threadId);
loadStatus();
input.focus();
