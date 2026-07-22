"""Operator dashboard (M10): one self-contained HTML file, the whole desk at a glance.

render_dashboard() is PURE - it takes already-loaded records and returns HTML;
all file IO lives in cli_dashboard. No external assets, no JavaScript: the
file must open identically on any machine, forever (it is an audit artifact
as much as a UI).
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from quant_platform.execution.session import AuditRecord
from quant_platform.execution.state import PaperState
from quant_platform.journal import DecisionJournal
from quant_platform.strategies.candidates import LoadedCandidate

_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1a1d21}
header{background:#1a1d21;color:#fff;padding:14px 24px}
header h1{margin:0;font-size:18px}header p{margin:4px 0 0;font-size:12px;color:#9aa3ad}
main{padding:16px 24px;max-width:1100px;margin:0 auto}
section{background:#fff;border:1px solid #e1e4e8;border-radius:6px;margin:14px 0;padding:14px 18px}
h2{font-size:14px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em;color:#57606a}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;color:#57606a;font-weight:600;border-bottom:2px solid #e1e4e8;padding:4px 8px}
td{border-bottom:1px solid #eef0f2;padding:4px 8px;vertical-align:top}
.kpi{display:inline-block;margin-right:28px}.kpi b{display:block;font-size:20px}
.kpi span{font-size:11px;color:#57606a;text-transform:uppercase}
.pos{color:#116329}.neg{color:#a40e26}.muted{color:#8b949e}
.ok{color:#116329;font-weight:600}.bad{color:#a40e26;font-weight:600}
.tag{display:inline-block;background:#eef1f4;border-radius:3px;padding:1px 6px;font-size:11px}
.pred{font-size:12px;color:#57606a;max-width:520px}
footer{padding:10px 24px;font-size:11px;color:#8b949e;text-align:center}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _signed_pct(value: float) -> str:
    cls = "pos" if value >= 0 else "neg"
    return f'<span class="{cls}">{value:+.2f}%</span>'


def _equity_svg(points: list[float], width: int = 1040, height: int = 120) -> str:
    if len(points) < 2:
        return '<p class="muted">Equity curve appears after the first two audited fills.</p>'
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    step = width / (len(points) - 1)
    coords = " ".join(
        f"{i * step:.1f},{height - (p - lo) / span * (height - 10) - 5:.1f}"
        for i, p in enumerate(points)
    )
    color = "#116329" if points[-1] >= points[0] else "#a40e26"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" role="img" aria-label="paper equity curve">'
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'
        f'<p class="muted" style="font-size:11px">equity after each audited decision: '
        f'{points[0]:.2f} &#8594; {points[-1]:.2f} (min {lo:.2f}, max {hi:.2f})</p>'
    )


def _positions_section(state: PaperState) -> str:
    if not state.open_positions:
        return '<p class="muted">No open paper positions.</p>'
    rows = "".join(
        f"<tr><td>{_esc(p.candidate_id)}</td><td>{_esc(p.symbol)}</td>"
        f"<td><span class='tag'>{_esc(getattr(p, 'direction', 'long'))}</span></td>"
        f"<td>{p.quantity:.8f}</td><td>{p.entry_price:.4f}</td><td>{p.stop_price:.4f}</td>"
        f"<td>{_esc(p.entry_ts.strftime('%Y-%m-%d %H:%M UTC'))}</td></tr>"
        for p in state.open_positions
    )
    return (
        "<table><tr><th>candidate</th><th>symbol</th><th>dir</th><th>quantity</th>"
        f"<th>entry</th><th>stop</th><th>entered</th></tr>{rows}</table>"
    )


def _audit_section(records: list[AuditRecord], limit: int = 20) -> str:
    if not records:
        return '<p class="muted">No execution decisions yet - run m9-cycle.bat.</p>'
    rows = []
    for r in reversed(records[-limit:]):
        failed = ", ".join(c["name"] for c in r.checks if not c["passed"])
        verdict = '<span class="ok">approved</span>' if r.approved else \
            f'<span class="bad">rejected</span> <span class="muted">({_esc(failed)})</span>'
        fill = f"{r.fill['fill_price']:.4f}" if r.fill else "&#8212;"
        equity = f"{r.equity_after:.2f}" if r.equity_after is not None else "&#8212;"
        rows.append(
            f"<tr><td>{_esc(r.ts.strftime('%m-%d %H:%M'))}</td>"
            f'<td><span class="tag">{_esc(r.tier)}</span></td>'
            f"<td>{_esc(r.strategy_id)}</td><td>{_esc(r.side.value)}</td>"
            f"<td>{verdict}</td><td>{fill}</td><td>{equity}</td></tr>"
        )
    return (
        "<table><tr><th>time (UTC)</th><th>tier</th><th>strategy</th><th>side</th>"
        f"<th>decision</th><th>fill</th><th>equity after</th></tr>{''.join(rows)}</table>"
    )


def _candidates_section(candidates: list[LoadedCandidate], records: list[AuditRecord]) -> str:
    if not candidates:
        return '<p class="muted">No candidates registered (config/candidates/).</p>'
    fills = {}
    for r in records:
        if r.fill:
            fills[r.strategy_id] = fills.get(r.strategy_id, 0) + 1
    rows = "".join(
        f"<tr><td>{_esc(c.id)} <span class='muted'>v{_esc(c.version)}</span></td>"
        f"<td>{_esc(c.definition['tracking'].get('hypothesis', '&#8212;'))}</td>"
        f"<td>{fills.get(c.id, 0)}</td>"
        f"<td class='pred'>{_esc(c.prediction)}</td></tr>"
        for c in candidates
    )
    return (
        "<table><tr><th>candidate</th><th>hyp</th><th>fills</th>"
        f"<th>pre-registered prediction (ADR-0006)</th></tr>{rows}</table>"
    )


def _journal_section(journal: DecisionJournal, limit: int = 10) -> str:
    memos = journal.memos()
    if not memos:
        return '<p class="muted">No desk memos yet - run m5-desk.bat.</p>'
    rows = []
    for memo in reversed(memos[-limit:]):
        outcomes = journal.outcomes_for(memo.record_id)
        if outcomes:
            o = outcomes[-1]
            outcome = f"{_signed_pct(o.realized_return_pct)} <span class='muted'>@{o.horizon_days}d</span>"
        else:
            outcome = '<span class="muted">pending</span>'
        rows.append(
            f"<tr><td>{_esc(memo.created_at.strftime('%Y-%m-%d'))}</td>"
            f"<td>{_esc(memo.symbol)}</td>"
            f"<td>{_esc(memo.confidence or '&#8212;')}</td><td>{outcome}</td>"
            f"<td class='muted'>{_esc(memo.record_id)}</td></tr>"
        )
    return (
        "<table><tr><th>date</th><th>symbol</th><th>confidence</th>"
        f"<th>realized outcome</th><th>memo id</th></tr>{''.join(rows)}</table>"
    )


def _ledger_section(ledger_rows: list[dict]) -> str:
    if not ledger_rows:
        return '<p class="muted">No signed validation reports.</p>'
    rows = "".join(
        f"<tr><td>{_esc(r['report'])}</td><td>{_esc(r['result'])}</td>"
        f"<td>{_esc(r['date'])}</td><td class='muted'>{_esc(r['sha256'][:12])}&#8230;</td></tr>"
        for r in ledger_rows
    )
    return (
        "<table><tr><th>validation report</th><th>result</th><th>signed</th>"
        f"<th>sha256</th></tr>{rows}</table>"
    )


def render_dashboard(
    state: PaperState | None,
    audit_records: list[AuditRecord],
    candidates: list[LoadedCandidate],
    journal: DecisionJournal,
    ledger_rows: list[dict],
    generated_at: datetime | None = None,
    equity_history: list[dict] | None = None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    banner = ""
    if state is not None:
        age_h = (generated_at - state.updated_at).total_seconds() / 3600
        if age_h > 3.0:
            banner = (
                '<div style="background:#a40e26;color:#fff;padding:10px 24px;'
                'font-size:13px;font-weight:600">&#9888; CYCLE STALE - last paper '
                f'cycle ran {age_h:.1f}h ago (&gt; 3h). The forward-evidence clock '
                '(protocol v2, F1) is NOT ticking. Check the m9 schedule / machine '
                'uptime, then run quant-status.</div>'
            )
    if state is not None:
        equity_now = state.last_equity if state.last_equity is not None else state.cash
        ret = (equity_now / state.starting_cash - 1.0) * 100.0
        kpis = (
            f'<div class="kpi"><b>{equity_now:.2f}</b><span>equity (last mark)</span></div>'
            f'<div class="kpi"><b>{state.cash:.2f}</b><span>cash</span></div>'
            f'<div class="kpi"><b>{_signed_pct(ret)}</b><span>total return</span></div>'
            f'<div class="kpi"><b>{state.cycle_count}</b><span>cycles run</span></div>'
            f'<div class="kpi"><b>{len(state.open_positions)}</b><span>open positions</span></div>'
        )
    else:
        kpis = '<p class="muted">No paper state yet - run m9-cycle.bat for the first time.</p>'

    # Per-cycle history (one mark per cycle) is the real curve; the audit
    # trail's equity_after (one point per FILL) is only a fallback for
    # workspaces predating the equity-history sidecar.
    if equity_history:
        equity_points = [
            float(row["equity"]) for row in equity_history
            if isinstance(row.get("equity"), (int, float))
        ]
        equity_label = f"Equity (per cycle, {len(equity_points)} marks)"
    else:
        equity_points = [r.equity_after for r in audit_records if r.equity_after is not None]
        equity_label = "Equity"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PROJECT GENESIS - paper desk dashboard</title><style>{_CSS}</style></head>
<body>
{banner}
<header><h1>PROJECT GENESIS &#8212; paper trading desk</h1>
<p>generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} &#183; paper tier only &#183;
zero validated strategies &#183; not financial advice</p></header>
<main>
<section><h2>Account</h2>{kpis}</section>
<section><h2>{equity_label}</h2>{_equity_svg(equity_points)}</section>
<section><h2>Open positions</h2>{_positions_section(state) if state else '<p class="muted">&#8212;</p>'}</section>
<section><h2>Recent execution decisions</h2>{_audit_section(audit_records)}</section>
<section><h2>Registered candidates</h2>{_candidates_section(candidates, audit_records)}</section>
<section><h2>Desk memos &amp; outcomes</h2>{_journal_section(journal)}</section>
<section><h2>Research ledger (signed validation reports)</h2>{_ledger_section(ledger_rows)}</section>
</main>
<footer>All records above are paper-tier research artifacts (ADR-0006). Promotion beyond paper
requires a signed validation report (ADR-0005). Live trading does not exist in this codebase.</footer>
</body></html>
"""
