"""
build_agent_report.py — turn agent-results.json into a self-contained
interactive HTML report for GitHub Pages.

Usage:
    python scripts/build_agent_report.py \
        --input  reports/agent-results.json \
        --output reports/agent-validation.html
"""

import argparse
import json
import html
import pathlib
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Badge helpers
# ---------------------------------------------------------------------------

STATUS_BADGE = {
    "pass":    '<span class="badge pass">PASS</span>',
    "warn":    '<span class="badge warn">WARN</span>',
    "timeout": '<span class="badge timeout">TIMEOUT</span>',
    "error":   '<span class="badge error">ERROR</span>',
}

CATEGORY_ICON = {
    "docker":     "🐳",
    "kubernetes": "☸️",
    "reasoning":  "🧠",
}


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(data: dict) -> str:
    model = html.escape(data.get("model", "unknown"))
    ollama_url = html.escape(data.get("ollama_url", ""))
    total = data["total"]
    passed = data["pass"]
    warned = data["warn"]
    errors = data["error"]
    results = data["results"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Summary bar ──────────────────────────────────────────────────────
    pct = round(passed / total * 100) if total else 0
    bar_color = "#1a7f37" if errors == 0 else "#cf222e"

    # ── Result cards ─────────────────────────────────────────────────────
    cards = []
    for r in results:
        icon = CATEGORY_ICON.get(r["category"], "❓")
        badge = STATUS_BADGE.get(r["status"], r["status"])
        question = html.escape(r["question"])
        answer = html.escape(r["answer"]) if r["answer"] else ""
        error_msg = html.escape(r["error"]) if r["error"] else ""
        tools = ", ".join(r.get("tool_calls", [])) or "none"
        mode = html.escape(r.get("mode", "react"))
        elapsed = r.get("elapsed_secs", "?")
        idx = r["index"]

        answer_block = ""
        if answer:
            answer_block = f"""
            <div class="answer">
              <pre>{answer}</pre>
            </div>"""

        error_block = ""
        if error_msg:
            error_block = f"""
            <div class="error-msg">
              <strong>Error:</strong> {error_msg}
            </div>"""

        cards.append(f"""
        <div class="card {r['status']}" id="q{idx}">
          <div class="card-header">
            <span class="q-num">#{idx}</span>
            <span class="category">{icon} {r['category'].upper()}</span>
            {badge}
            <span class="meta">🔧 tools: {tools} &nbsp;|&nbsp; ⏱ {elapsed}s</span>
          </div>
          <div class="question">{question}</div>
          {answer_block}
          {error_block}
        </div>""")

    cards_html = "\n".join(cards)

    # ── Full page ─────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agent Validation Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f6f8fa;
      color: #24292f;
      padding: 32px 16px;
    }}

    .container {{ max-width: 900px; margin: 0 auto; }}

    h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    .subtitle {{ color: #656d76; font-size: 0.9rem; margin-bottom: 24px; }}

    /* ── Summary ───────────────────────────────────────────── */
    .summary {{
      background: #fff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 24px;
      display: flex;
      gap: 32px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .stat {{ text-align: center; }}
    .stat .value {{ font-size: 2rem; font-weight: 700; }}
    .stat .label {{ font-size: 0.8rem; color: #656d76; text-transform: uppercase; }}
    .stat.pass .value  {{ color: #1a7f37; }}
    .stat.warn .value  {{ color: #9a6700; }}
    .stat.error .value {{ color: #cf222e; }}
    .stat.total .value {{ color: #0969da; }}

    .progress-wrap {{
      flex: 1;
      min-width: 200px;
    }}
    .progress-bar-bg {{
      height: 10px;
      background: #d0d7de;
      border-radius: 6px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .progress-bar-fg {{
      height: 100%;
      background: {bar_color};
      width: {pct}%;
      border-radius: 6px;
      transition: width 0.4s ease;
    }}
    .progress-label {{ font-size: 0.85rem; color: #656d76; }}

    /* ── Filter bar ────────────────────────────────────────── */
    .filters {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .filter-btn {{
      border: 1px solid #d0d7de;
      background: #fff;
      border-radius: 20px;
      padding: 4px 14px;
      cursor: pointer;
      font-size: 0.85rem;
      transition: background 0.15s, border-color 0.15s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: #0969da;
      border-color: #0969da;
      color: #fff;
    }}

    /* ── Cards ─────────────────────────────────────────────── */
    .card {{
      background: #fff;
      border: 1px solid #d0d7de;
      border-left: 4px solid #d0d7de;
      border-radius: 8px;
      margin-bottom: 16px;
      overflow: hidden;
      transition: box-shadow 0.15s;
    }}
    .card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .card.pass  {{ border-left-color: #1a7f37; }}
    .card.warn  {{ border-left-color: #9a6700; }}
    .card.error {{ border-left-color: #cf222e; }}

    .card-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      background: #f6f8fa;
      border-bottom: 1px solid #d0d7de;
      flex-wrap: wrap;
    }}
    .q-num    {{ font-weight: 700; color: #656d76; font-size: 0.85rem; }}
    .category {{ font-size: 0.8rem; font-weight: 600; color: #0969da; }}
    .meta     {{ margin-left: auto; font-size: 0.78rem; color: #656d76; }}

    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 700;
    }}
    .badge.pass    {{ background: #dafbe1; color: #1a7f37; }}
    .badge.warn    {{ background: #fff8c5; color: #9a6700; }}
    .badge.timeout {{ background: #fff0b3; color: #7d4e00; }}
    .badge.error   {{ background: #ffebe9; color: #cf222e; }}
    .card.timeout  {{ border-left-color: #7d4e00; }}

    .question {{
      padding: 14px 16px;
      font-weight: 600;
      font-size: 0.95rem;
      border-bottom: 1px solid #f0f0f0;
    }}

    .answer {{
      padding: 14px 16px;
    }}
    .answer pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.82rem;
      line-height: 1.5;
      background: #f6f8fa;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #d0d7de;
      max-height: 400px;
      overflow-y: auto;
    }}

    .error-msg {{
      padding: 14px 16px;
      color: #cf222e;
      font-size: 0.85rem;
      background: #ffebe9;
    }}

    .meta-info {{
      margin-top: 32px;
      font-size: 0.8rem;
      color: #656d76;
      text-align: center;
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>🤖 Agent Validation Report</h1>
  <p class="subtitle">
    Model: <strong>{model}</strong> &nbsp;|&nbsp;
    Ollama: <code>{ollama_url}</code> &nbsp;|&nbsp;
    Generated: {generated}
  </p>

  <div class="summary">
    <div class="stat total">
      <div class="value">{total}</div>
      <div class="label">Total</div>
    </div>
    <div class="stat pass">
      <div class="value">{passed}</div>
      <div class="label">Pass</div>
    </div>
    <div class="stat warn">
      <div class="value">{warned}</div>
      <div class="label">Warn</div>
    </div>
    <div class="stat error">
      <div class="value">{errors}</div>
      <div class="label">Error</div>
    </div>
    <div class="progress-wrap">
      <div class="progress-label">{pct}% passed</div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fg"></div>
      </div>
    </div>
  </div>

  <div class="filters">
    <button class="filter-btn active" onclick="filter('all')">All ({total})</button>
    <button class="filter-btn" onclick="filter('pass')">✅ Pass ({passed})</button>
    <button class="filter-btn" onclick="filter('warn')">⚠️ Warn ({warned})</button>
    <button class="filter-btn" onclick="filter('error')">❌ Error ({errors})</button>
    <button class="filter-btn" onclick="filter('docker')">🐳 Docker</button>
    <button class="filter-btn" onclick="filter('kubernetes')">☸️ Kubernetes</button>
    <button class="filter-btn" onclick="filter('reasoning')">🧠 Reasoning</button>
  </div>

  <div id="cards">
{cards_html}
  </div>

  <p class="meta-info">
    Generated by GitHub Actions &bull; {generated}
  </p>
</div>

<script>
  function filter(key) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    document.querySelectorAll('.card').forEach(card => {{
      const matchStatus   = card.classList.contains(key);
      const matchCategory = card.querySelector('.category') &&
                            card.querySelector('.category')
                                .textContent.toLowerCase().includes(key);
      if (key === 'all' || matchStatus || matchCategory) {{
        card.style.display = '';
      }} else {{
        card.style.display = 'none';
      }}
    }});
  }}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build agent validation HTML report")
    parser.add_argument("--input",  default="reports/agent-results.json")
    parser.add_argument("--output", default="reports/agent-validation.html")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        raise SystemExit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    html_content = build_html(data)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
