"""End-of-day Markdown report for the paper-trading session.

Writes to data/reports/<YYYY-MM-DD>.md. Includes trade log, cost breakdown,
sentiment accuracy, and diversification summary.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backtest.report import _diagnostic_observations, _summary_stats, _trades_to_frame
from src.features.technical import IST_OFFSET_SECONDS
from src.paper.persistence import (
    get_halt_state,
    ist_date_str,
    list_closed_positions_today,
    list_open_positions,
    todays_realised_pnl,
)

REPORTS_DIR = Path("data/reports")


def _ist_str(ts: int, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.utcfromtimestamp(ts + IST_OFFSET_SECONDS).strftime(fmt)


def _costs_for_trade(t) -> float:
    """Gross - net realised = costs + slippage paid."""
    if t.entry_price and t.exit_price and t.realised_pnl_inr is not None:
        gross = t.qty * (
            (t.exit_price - t.entry_price) if t.side == "long"
            else (t.entry_price - t.exit_price)
        )
        return gross - t.realised_pnl_inr
    return 0.0


async def generate_report(now_ts: Optional[float] = None) -> Path:
    """Build today's report. Overwrites on repeated calls."""
    now = now_ts if now_ts is not None else time.time()
    closed = await list_closed_positions_today(now)
    open_now = await list_open_positions()
    pnl = await todays_realised_pnl(now)
    halt = await get_halt_state(now)
    date_str = ist_date_str(now)

    trades_df = _trades_to_frame(closed)
    summary = _summary_stats(trades_df)

    md = []
    md.append(f"# Paper-trading report — {date_str}")
    md.append("")
    md.append(f"Generated {datetime.utcnow().isoformat()}Z.")
    md.append("")

    # ---------- Headline ----------
    md.append("## Headline")
    md.append("")
    md.append(f"- **Trades closed today**: {summary.get('trades', 0)}")
    md.append(f"- **Open at EOD**: {len(open_now)}")
    md.append(f"- **Net realised P&L**: ₹{pnl:,.2f}")
    md.append(f"- **Win rate**: {summary.get('win_rate_pct', 0)}%")
    if halt["halted"]:
        md.append(f"- **HALTED**: {halt['halt_reason']} at "
                  f"{_ist_str(halt['halted_at']) if halt['halted_at'] else 'unknown'}")
    md.append("")

    # ---------- Trade stats ----------
    md.append("## Trade stats")
    md.append("")
    if summary.get("trades", 0) == 0:
        md.append("_No trades closed today._")
    else:
        for k in (
            "trades", "wins", "losses", "win_rate_pct",
            "gross_pnl_inr", "mean_win_inr", "mean_loss_inr",
            "best_trade_inr", "worst_trade_inr",
            "long_count", "short_count",
            "eod_exits", "sl_exits", "target_exits",
        ):
            md.append(f"- {k}: {summary[k]}")
    md.append("")

    # ---------- Costs breakdown ----------
    md.append("## Costs & charges")
    md.append("")
    if closed:
        total_costs = sum(_costs_for_trade(t) for t in closed)
        gross_pnl = pnl + total_costs
        md.append(f"- Gross P&L (before costs): ₹{gross_pnl:,.2f}")
        md.append(f"- Costs + slippage: -₹{abs(total_costs):,.2f}")
        md.append(f"- Net realised P&L: ₹{pnl:,.2f}")
        avg_cost = total_costs / len(closed)
        md.append(f"- Average cost per trade: ₹{avg_cost:,.2f}")
        md.append("")
        md.append("_Costs include 5 bps/leg slippage, min(₹20, 0.05%) brokerage per leg, STT 0.025% sell-side, exchange charges._")
    else:
        md.append("_No trades._")
    md.append("")

    # ---------- Sentiment accuracy ----------
    md.append("## Sentiment accuracy")
    md.append("")
    trades_with_sentiment = [t for t in closed if t.entry_sentiment_score is not None]
    if trades_with_sentiment:
        correct = 0
        for t in trades_with_sentiment:
            pnl_sign = (t.realised_pnl_inr or 0) >= 0
            # Bullish sentiment (>0) on a long → correct if profitable; bearish on short → correct if profitable
            sent_aligned = (t.entry_sentiment_score > 0 and t.side == "long") or \
                           (t.entry_sentiment_score < 0 and t.side == "short")
            if sent_aligned == pnl_sign:
                correct += 1
        acc = correct / len(trades_with_sentiment) * 100
        md.append(f"- Trades with sentiment at entry: {len(trades_with_sentiment)}")
        md.append(f"- Sentiment aligned with outcome: {correct} / {len(trades_with_sentiment)} ({acc:.1f}%)")
    else:
        md.append("_No trades had sentiment scores at entry._")
    md.append("")

    # ---------- Diversification ----------
    md.append("## Diversification")
    md.append("")
    if closed:
        symbols = {t.instrument_key for t in closed}
        sides = {t.side for t in closed}
        md.append(f"- Distinct symbols traded: {len(symbols)}")
        md.append(f"- Direction mix: {', '.join(sorted(sides))}")
        symbol_counts = {}
        for t in closed:
            symbol_counts[t.instrument_key] = symbol_counts.get(t.instrument_key, 0) + 1
        churned = {k: v for k, v in symbol_counts.items() if v > 1}
        if churned:
            md.append(f"- Symbols traded >1× (potential churn): {len(churned)}")
            for sym, cnt in sorted(churned.items(), key=lambda x: -x[1])[:5]:
                md.append(f"  - `{sym}` × {cnt}")
    else:
        md.append("_No trades._")
    md.append("")

    # ---------- Open at EOD ----------
    md.append("## Open positions at EOD")
    md.append("")
    if not open_now:
        md.append("_None._")
    else:
        md.append("| symbol | side | qty | entry | sl | tp | entry_ts |")
        md.append("|--------|------|----:|------:|----:|---:|----------|")
        for p in open_now:
            md.append(
                f"| `{p.instrument_key}` | {p.side} | {p.qty} | "
                f"{p.entry_price:.2f} | {p.stop_loss_price:.2f} | {p.target_price:.2f} | "
                f"{_ist_str(p.entry_ts)} |"
            )
        md.append("")
        md.append("**Note**: open positions at EOD — investigate if 14:55 IST hard-exit didn't fire.")
    md.append("")

    # ---------- Diagnostic observations ----------
    md.append("## Things worth looking at")
    md.append("")
    md.extend(_diagnostic_observations(summary, daily=pd.DataFrame(), report=_empty_report_stub()))
    md.append("")

    # ---------- Full trade log ----------
    md.append("## Trade log")
    md.append("")
    if trades_df.empty:
        md.append("_None._")
    else:
        md.append("| entry | exit | symbol | side | qty | entry₹ | exit₹ | reason | pnl₹ | ret% | pred% | sent |")
        md.append("|-------|------|--------|------|----:|-------:|------:|--------|-----:|-----:|------:|------|")
        for t in sorted(closed, key=lambda x: x.entry_ts):
            exit_price = t.exit_price or 0.0
            ret_pct = 0.0
            if t.entry_price and t.exit_price:
                ret_pct = ((t.exit_price - t.entry_price) / t.entry_price * 100
                           if t.side == "long"
                           else (t.entry_price - t.exit_price) / t.entry_price * 100)
            pred_str = f"{t.predicted_return * 100:+.2f}" if t.predicted_return is not None else "—"
            sent_str = f"{t.entry_sentiment_score:+.2f}" if t.entry_sentiment_score is not None else "—"
            md.append(
                f"| {_ist_str(t.entry_ts, '%H:%M')} | {_ist_str(t.exit_ts, '%H:%M') if t.exit_ts else '—'} "
                f"| `{t.instrument_key}` | {t.side} | {t.qty} | {t.entry_price:.2f} | {exit_price:.2f} | "
                f"{t.exit_reason} | {(t.realised_pnl_inr or 0):.2f} | {ret_pct:.3f} | {pred_str} | {sent_str} |"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{date_str}.md"
    path.write_text("\n".join(md))
    return path


def _empty_report_stub():
    class _Stub:
        equity_curve: list = []
    return _Stub()
