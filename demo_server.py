from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from demo_analyzer import AnalyzerError, ReasoningAnalyzer


ANALYZER = ReasoningAnalyzer()


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "HalluciNOTDemo/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_html(INDEX_HTML)
            return
        if self.path == "/health":
            self.send_json({"status": "ok", "analyzer_configured": ANALYZER.is_configured()})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self.read_json()
            result = ANALYZER.analyze(
                problem=str(payload.get("problem", "")),
                draft_reasoning=str(payload.get("draft_reasoning", "")),
            )
            self.send_json(result)
        except (ValueError, AnalyzerError) as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception:
            self.send_json({"error": "Reasoning audit failed. Check the server logs and input."}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed

    def send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HalluciNOT.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"HalluciNOT running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HalluciNOT</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef2f5;
      --panel: #ffffff;
      --soft: #f8fafc;
      --ink: #111827;
      --muted: #667085;
      --line: #d8e1ec;
      --accent: #0b6f63;
      --accent-dark: #075b52;
      --danger: #b42318;
      --danger-soft: #fff1ef;
      --ok: #157f3b;
      --ok-soft: #edf9f0;
      --warn: #a15c07;
      --warn-soft: #fff7e8;
      --shadow: 0 18px 38px rgba(17, 24, 39, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }
    header {
      min-height: 70px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(14px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background:
        linear-gradient(90deg, transparent 14px, rgba(255,255,255,.34) 14px, rgba(255,255,255,.34) 17px, transparent 17px),
        linear-gradient(0deg, transparent 14px, rgba(255,255,255,.28) 14px, rgba(255,255,255,.28) 17px, transparent 17px),
        #0b6f63;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.35), 0 8px 18px rgba(11,111,99,.20);
      flex: 0 0 auto;
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 20px; line-height: 1.1; }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      background: #f2f6fa;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 13px;
      white-space: nowrap;
    }
    .status-pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--ok);
      box-shadow: 0 0 0 3px rgba(21,127,59,.12);
    }
    main {
      width: min(1320px, calc(100vw - 32px));
      margin: 22px auto 48px;
      display: grid;
      grid-template-columns: minmax(320px, 440px) minmax(0, 1fr);
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .input-panel {
      align-self: start;
      position: sticky;
      top: 92px;
      overflow: hidden;
    }
    .panel-title {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-title h2 { font-size: 15px; }
    .panel-title span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    form { padding: 18px; }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 7px;
      text-transform: uppercase;
    }
    textarea {
      width: 100%;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 12px;
      font: inherit;
      line-height: 1.45;
      color: var(--ink);
      background: #fff;
      min-height: 138px;
      margin-bottom: 14px;
    }
    textarea#draft { min-height: 182px; }
    textarea:focus-visible,
    button:focus-visible {
      outline: 3px solid rgba(11,111,99,.22);
      outline-offset: 2px;
    }
    .examples {
      display: grid;
      gap: 8px;
      margin: -2px 0 14px;
    }
    .example-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .example-btn {
      border: 1px solid var(--line);
      background: var(--soft);
      color: var(--muted);
      border-radius: 999px;
      padding: 7px 10px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    .example-btn:hover {
      border-color: var(--accent);
      color: var(--accent);
      background: #f2fbf9;
    }
    button.primary {
      width: 100%;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      min-height: 44px;
      cursor: pointer;
      transition: background 140ms ease, transform 140ms ease;
    }
    button.primary:hover { background: var(--accent-dark); }
    button.primary:active { transform: translateY(1px); }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
      transform: none;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      min-height: 24px;
      margin-top: 14px;
      font-size: 13px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--ok);
      box-shadow: 0 0 0 4px rgba(21,127,59,.12);
    }
    .dot.busy { background: var(--warn); box-shadow: 0 0 0 4px rgba(161,92,7,.12); }
    .dot.error { background: var(--danger); box-shadow: 0 0 0 4px rgba(180,35,24,.12); }
    .results {
      display: grid;
      gap: 20px;
      min-width: 0;
    }
    .empty {
      min-height: 520px;
      padding: 72px 24px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
    }
    .empty-inner { max-width: 440px; }
    .empty-icon {
      width: 64px;
      height: 64px;
      margin: 0 auto 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background:
        linear-gradient(90deg, transparent 27px, #d9e3ed 27px, #d9e3ed 31px, transparent 31px),
        linear-gradient(0deg, transparent 27px, #d9e3ed 27px, #d9e3ed 31px, transparent 31px),
        var(--soft);
    }
    .empty h2 { color: var(--ink); font-size: 20px; margin-bottom: 8px; }
    .summary {
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(170px, 210px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .verdict {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 136px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      background: var(--soft);
    }
    .verdict strong {
      display: block;
      font-size: 24px;
      line-height: 1.05;
      margin-bottom: 7px;
    }
    .verdict span { color: var(--muted); font-size: 13px; }
    .verdict.supported { background: var(--ok-soft); border-color: #c6e8cd; }
    .verdict.supported strong { color: var(--ok); }
    .verdict.drift { background: var(--danger-soft); border-color: #f2c8c3; }
    .verdict.drift strong { color: var(--danger); }
    .verdict.corrected,
    .verdict.uncertain { background: var(--warn-soft); border-color: #f3d8a8; }
    .verdict.corrected strong,
    .verdict.uncertain strong { color: var(--warn); }
    .summary-main p {
      color: var(--muted);
      line-height: 1.5;
      margin-top: 8px;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(112px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 10px 11px;
      min-height: 74px;
    }
    .stat span { color: var(--muted); font-size: 12px; }
    .stat strong {
      display: block;
      font-size: 21px;
      line-height: 1;
      margin-top: 7px;
    }
    .section {
      padding: 18px;
    }
    .section-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .section-head h2 { font-size: 16px; }
    .section-head span { color: var(--muted); font-size: 12px; }
    .answer-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .answer-card {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 8px;
      padding: 13px;
      min-height: 108px;
    }
    .answer-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      font-weight: 700;
    }
    .answer-card p {
      white-space: pre-wrap;
      line-height: 1.45;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: block;
    }
    thead, tbody, tr { display: table; width: 100%; table-layout: fixed; }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px;
      vertical-align: top;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      background: var(--soft);
      text-transform: uppercase;
      font-size: 11px;
    }
    tr:last-child td { border-bottom: 0; }
    .drift {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      margin-top: 10px;
      background: white;
    }
    .drift h3 { font-size: 15px; margin-bottom: 7px; }
    .drift p { color: var(--muted); line-height: 1.45; }
    .notes {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .note {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      color: var(--muted);
      background: var(--soft);
      font-size: 12px;
    }
    .loading {
      display: grid;
      gap: 12px;
      padding: 18px;
    }
    .skeleton {
      height: 106px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, #f4f7fa 0%, #e8eef5 48%, #f4f7fa 100%);
      background-size: 200% 100%;
      animation: shimmer 1.4s infinite linear;
    }
    .skeleton.tall { height: 260px; }
    @keyframes shimmer {
      from { background-position: 200% 0; }
      to { background-position: -200% 0; }
    }
    .error { color: var(--danger); font-weight: 700; }
    @media (max-width: 980px) {
      header { flex-direction: column; align-items: flex-start; padding: 16px; }
      main { grid-template-columns: 1fr; width: min(100vw - 24px, 760px); }
      .input-panel { position: static; }
      .summary { grid-template-columns: 1fr; }
      .stat-grid, .answer-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .status-pill { align-self: stretch; justify-content: center; }
    }
    @media (max-width: 560px) {
      main { width: min(100vw - 18px, 760px); margin-top: 12px; gap: 12px; }
      .stat-grid, .answer-grid { grid-template-columns: 1fr; }
      .section-head { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="mark" aria-hidden="true"></div>
      <div>
        <h1>HalluciNOT</h1>
        <div class="subtitle">Logic-grounded symbolic drift audit</div>
      </div>
    </div>
    <div class="status-pill" id="healthPill">System ready</div>
  </header>
  <main>
    <section class="panel input-panel">
      <div class="panel-title">
        <h2>Reasoning Input</h2>
        <span>Single query</span>
      </div>
      <form id="auditForm" novalidate>
        <label for="problem">Problem Or Claim</label>
        <textarea id="problem" placeholder="Example: A store sold 35 tickets at $12 each and 18 tickets at $15 each. What was total revenue?"></textarea>
        <div class="examples">
          <div class="example-row">
            <button class="example-btn" type="button" data-example="revenue">Revenue</button>
            <button class="example-btn" type="button" data-example="drift">Drift Case</button>
            <button class="example-btn" type="button" data-example="rate">Rate Math</button>
          </div>
        </div>
        <label for="draft">Draft Reasoning Or Answer</label>
        <textarea id="draft" placeholder="Optional: paste a draft answer or reasoning chain to verify. Leave blank to generate and audit a fresh answer."></textarea>
        <button class="primary" id="runButton" type="submit">Run Reasoning Audit</button>
        <div class="status" id="status"><span class="dot"></span><span>Ready.</span></div>
      </form>
    </section>
    <section id="results" class="results">
      <div class="panel empty">
        <div class="empty-inner">
          <div class="empty-icon" aria-hidden="true"></div>
          <h2>Awaiting Query</h2>
          <p>Enter a reasoning problem and optional draft answer to generate a symbolic audit.</p>
        </div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById('auditForm');
    const problemEl = document.getElementById('problem');
    const draftEl = document.getElementById('draft');
    const results = document.getElementById('results');
    const button = document.getElementById('runButton');
    const statusEl = document.getElementById('status');
    const healthPill = document.getElementById('healthPill');

    const examples = {
      revenue: {
        problem: 'A store sold 35 student tickets at $12 each and 18 adult tickets at $15 each. What was the total revenue?',
        draft: 'student = 35 * 12 = 420\nadult = 18 * 15 = 280\ntotal = 420 + 280 = 700\nFinal answer: 700'
      },
      drift: {
        problem: 'A train travels 120 km in 2 hours, then 90 km in 1.5 hours. What is its average speed over the whole trip?',
        draft: 'total_distance = 120 + 90 = 210\ntotal_time = 2 + 1.5 = 2.5\naverage_speed = 210 / 2.5 = 84\nFinal answer: 84 km/h'
      },
      rate: {
        problem: 'Maya has 48 marbles. She gives 1/4 to Liam, then buys 15 more. How many marbles does she have now?',
        draft: ''
      }
    };

    document.querySelectorAll('.example-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const example = examples[btn.dataset.example];
        problemEl.value = example.problem;
        draftEl.value = example.draft;
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const problem = problemEl.value.trim();
      const draft = draftEl.value.trim();
      if (!problem) {
        setStatus('Enter a problem or claim first.', 'error');
        return;
      }
      button.disabled = true;
      setStatus('Running symbolic audit...', 'busy');
      renderLoading();
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem, draft_reasoning: draft })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Audit failed.');
        renderResults(data);
        setStatus('Complete.', 'ok');
      } catch (error) {
        setStatus(error.message, 'error');
      } finally {
        button.disabled = false;
      }
    });

    async function loadHealth() {
      try {
        const response = await fetch('/health');
        const data = await response.json();
        healthPill.textContent = data.analyzer_configured ? 'System ready' : 'Server key missing';
      } catch {
        healthPill.textContent = 'Health unavailable';
      }
    }

    function renderLoading() {
      results.innerHTML = `
        <section class="panel loading">
          <div class="skeleton"></div>
          <div class="skeleton tall"></div>
          <div class="skeleton"></div>
        </section>
      `;
    }

    function renderResults(data) {
      const confidence = Math.round((Number(data.confidence) || 0) * 100);
      const verdictClass = verdictClassName(data.verdict);
      const steps = data.symbolic_steps || [];
      const drifts = data.drift_reports || [];
      const aggregate = data.aggregate || {};
      results.innerHTML = `
        <section class="panel summary">
          <div class="verdict ${verdictClass}">
            <strong>${escapeHtml(data.verdict || 'Uncertain')}</strong>
            <span>${confidence}% confidence</span>
          </div>
          <div class="summary-main">
            <h2>Audit Summary</h2>
            <p>${escapeHtml(data.summary || 'Reasoning audit completed.')}</p>
            <div class="stat-grid">
              ${renderStat('Depth', String(aggregate.reasoning_depth ?? steps.length))}
              ${renderStat('Drifts', String(aggregate.drift_frequency ?? drifts.length))}
              ${renderStat('Latency', `${Math.round(Number(data.latency_ms) || 0)} ms`)}
              ${renderStat('Mode', 'Audit')}
            </div>
          </div>
        </section>
        <section class="panel section">
          <div class="section-head">
            <h2>Answer Review</h2>
            <span>baseline vs verified</span>
          </div>
          <div class="answer-grid">
            <div class="answer-card"><span>Baseline</span><p>${escapeHtml(data.baseline_answer || 'Not supplied')}</p></div>
            <div class="answer-card"><span>Verified</span><p>${escapeHtml(data.verified_answer || 'Not available')}</p></div>
          </div>
        </section>
        <section class="panel section">
          <div class="section-head">
            <h2>Symbolic Trace</h2>
            <span>${steps.length} steps</span>
          </div>
          ${renderSteps(steps)}
        </section>
        <section class="panel section">
          <div class="section-head">
            <h2>Drift Reports</h2>
            <span>${drifts.length} findings</span>
          </div>
          ${renderDrifts(drifts)}
        </section>
        <section class="panel section">
          <div class="section-head">
            <h2>Correction</h2>
            <span>review-ready</span>
          </div>
          <div class="answer-card"><p>${escapeHtml(data.corrected_reasoning || 'No correction needed.')}</p></div>
          ${renderNotes(data.audit_notes || [])}
        </section>
      `;
    }

    function renderSteps(steps) {
      if (!steps.length) return '<p class="subtitle">No symbolic steps returned.</p>';
      return `
        <table>
          <thead><tr><th>Step</th><th>Claim</th><th>Operation</th><th>Value</th><th>Status</th></tr></thead>
          <tbody>
            ${steps.map((step) => `
              <tr>
                <td>${escapeHtml(step.step ?? '')}</td>
                <td>${escapeHtml(step.claim || '')}</td>
                <td>${escapeHtml(step.operation || '')}</td>
                <td>${escapeHtml(step.computed_value || '')}</td>
                <td>${escapeHtml(step.status || '')}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    function renderDrifts(drifts) {
      if (!drifts.length) return '<p class="subtitle">No symbolic drift was detected.</p>';
      return drifts.map((drift) => `
        <article class="drift">
          <h3>${escapeHtml(drift.claim || 'Drift finding')}</h3>
          <p><strong>Claimed:</strong> ${escapeHtml(drift.claimed_value || '-')} &nbsp; <strong>Verified:</strong> ${escapeHtml(drift.verified_value || '-')}</p>
          <p>${escapeHtml(drift.explanation || '')}</p>
        </article>
      `).join('');
    }

    function renderNotes(notes) {
      if (!notes.length) return '';
      return `<div class="notes" style="margin-top:12px">${notes.slice(0, 5).map((note) => `<span class="note">${escapeHtml(note)}</span>`).join('')}</div>`;
    }

    function renderStat(label, value) {
      return `<div class="stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
    }

    function verdictClassName(verdict) {
      const value = String(verdict || '').toLowerCase();
      if (value.includes('drift')) return 'drift';
      if (value.includes('correct')) return 'corrected';
      if (value.includes('support')) return 'supported';
      return 'uncertain';
    }

    function setStatus(message, tone) {
      statusEl.innerHTML = `<span class="dot ${tone}"></span><span class="${tone === 'error' ? 'error' : ''}">${escapeHtml(message)}</span>`;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    loadHealth();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
