"""Markdown report generation from a BacktestReport.

This format is what the daily paper-run report will mirror in Sprint 3 — keep
the schema stable so the user reads the same shape across both worlds.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.backtest.runner import BacktestReport, EquityPoint
from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS


def _ist_str(ts: int, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.utcfromtimestamp(ts + IST_OFFSET_SECONDS).strftime(fmt)


def _trades_to_frame(positions: List[Position]) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()
    rows = []
    for p in positions:
        rows.append(
            {
                "entry_ts_ist": _ist_str(p.entry_ts, "%Y-%m-%d %H:%M"),
                "exit_ts_ist": _ist_str(p.exit_ts, "%Y-%m-%d %H:%M") if p.exit_ts else "",
                "symbol_key": p.instrument_key,
                "side": p.side,
                "qty": p.qty,
                "entry": round(p.entry_price, 2),
                "exit": round(p.exit_price, 2) if p.exit_price else None,
                "exit_reason": p.exit_reason,
                "pnl_inr": round(p.realised_pnl_inr or 0.0, 2),
                "return_pct": round(
                    (p.realised_pnl_inr or 0.0) / (p.qty * p.entry_price) * 100, 3
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    wins = trades[trades["pnl_inr"] > 0]
    losses = trades[trades["pnl_inr"] < 0]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if len(trades) else 0.0,
        "gross_pnl_inr": round(trades["pnl_inr"].sum(), 2),
        "mean_win_inr": round(wins["pnl_inr"].mean(), 2) if not wins.empty else 0.0,
        "mean_loss_inr": round(losses["pnl_inr"].mean(), 2) if not losses.empty else 0.0,
        "best_trade_inr": round(trades["pnl_inr"].max(), 2),
        "worst_trade_inr": round(trades["pnl_inr"].min(), 2),
        "long_count": int((trades["side"] == "long").sum()),
        "short_count": int((trades["side"] == "short").sum()),
        "eod_exits": int((trades["exit_reason"] == "eod").sum()),
        "sl_exits": int((trades["exit_reason"] == "stop_loss").sum()),
        "target_exits": int((trades["exit_reason"] == "target").sum()),
    }


def _daily_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["date"] = trades["exit_ts_ist"].str.slice(0, 10)
    return (
        trades.groupby("date")["pnl_inr"]
        .agg(["count", "sum"])
        .rename(columns={"count": "trades", "sum": "pnl_inr"})
        .reset_index()
    )


def _max_drawdown(equity: List[EquityPoint]) -> tuple[float, str]:
    """Return (max DD in INR, IST timestamp of trough)."""
    if not equity:
        return 0.0, ""
    series = pd.Series([e.cash_pnl + e.open_unrealised for e in equity])
    running_max = series.cummax()
    drawdown = series - running_max
    idx = int(drawdown.idxmin())
    return float(drawdown.iloc[idx]), _ist_str(equity[idx].minute_ts)


def render_markdown(report: BacktestReport) -> str:
    trades = _trades_to_frame(report.completed_positions)
    summary = _summary_stats(trades)
    daily = _daily_pnl(trades)
    dd, dd_at = _max_drawdown(report.equity_curve)
    final_pnl = (
        report.equity_curve[-1].cash_pnl + report.equity_curve[-1].open_unrealised
        if report.equity_curve
        else 0.0
    )

    cfg = report.config
    md: List[str] = []
    md.append(f"# Backtest report — {_ist_str(report.start_ts, '%Y-%m-%d')} to {_ist_str(report.end_ts, '%Y-%m-%d')}")
    md.append("")
    md.append(f"Generated at {datetime.utcnow().isoformat()}Z. Runtime {report.runtime_seconds:.1f}s.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- **Trades**: {summary.get('trades', 0)}")
    md.append(f"- **Net P&L (incl. costs + slippage)**: ₹{final_pnl:,.2f}")
    md.append(f"- **Win rate**: {summary.get('win_rate_pct', 0)}%")
    md.append(f"- **Max drawdown**: ₹{dd:,.2f} (at {dd_at})")
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append("```")
    for k, v in asdict(cfg).items():
        md.append(f"  {k}: {v}")
    md.append("```")
    md.append("")
    md.append("## Trade stats")
    md.append("")
    if summary.get("trades", 0) == 0:
        md.append("_No trades were taken._ Check the **Filters and skips** section below — most likely")
        md.append("the model's predictions never cleared `min_predicted_edge`, or the entry window")
        md.append("never overlapped with bars that have all features defined (warmup).")
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
    md.append("## Daily P&L")
    md.append("")
    if daily.empty:
        md.append("_No closed trades._")
    else:
        md.append("| date | trades | pnl_inr |")
        md.append("|------|--------|---------|")
        for _, row in daily.iterrows():
            md.append(f"| {row['date']} | {int(row['trades'])} | ₹{row['pnl_inr']:,.2f} |")
    md.append("")
    md.append("## Filters and skips")
    md.append("")
    md.append("Counts of decisions the engine skipped (and why). Heavy skips here ≠ bug;")
    md.append("they usually mean the model's predictions weren't strong enough or the")
    md.append("entry window was inactive. Use these to debug strategy logic.")
    md.append("")
    if not report.skipped:
        md.append("_No skips recorded._")
    else:
        md.append("| reason | count |")
        md.append("|--------|------:|")
        for reason, count in sorted(report.skipped.items(), key=lambda kv: -kv[1]):
            md.append(f"| `{reason}` | {count:,} |")
    md.append("")
    md.append("## Things worth looking at")
    md.append("")
    md.extend(_diagnostic_observations(summary, daily, report))
    md.append("")
    md.append("## Trade log (latest 50)")
    md.append("")
    if trades.empty:
        md.append("_None._")
    else:
        latest = trades.tail(50)
        md.append("| entry | exit | symbol | side | qty | entry₹ | exit₹ | reason | pnl₹ | ret% |")
        md.append("|-------|------|--------|------|----:|-------:|------:|--------|-----:|-----:|")
        for _, row in latest.iterrows():
            exit_val = row["exit"] if row["exit"] is not None else 0.0
            md.append(
                f"| {row['entry_ts_ist']} | {row['exit_ts_ist']} | `{row['symbol_key']}` | "
                f"{row['side']} | {row['qty']} | {row['entry']:.2f} | {exit_val:.2f} | "
                f"{row['exit_reason']} | {row['pnl_inr']:.2f} | {row['return_pct']:.3f} |"
            )
    return "\n".join(md)


def _diagnostic_observations(summary: dict, daily: pd.DataFrame, report: BacktestReport) -> List[str]:
    """Auto-generated 'things to improve' bullets. Each is a heuristic, not a verdict."""
    out: List[str] = []
    trades = summary.get("trades", 0)
    if trades == 0:
        out.append("- **No trades taken.** The most likely cause is `min_predicted_edge` "
                   "being above what the model produces. Inspect `data/model_v1_metrics.json` "
                   "and compare to `EngineConfig.min_predicted_edge`.")
        return out

    if summary.get("long_count", 0) == 0 or summary.get("short_count", 0) == 0:
        out.append("- **One-sided book**: every trade was the same direction. Either the "
                   "model has a directional bias, or the data window was strongly trending.")

    win_rate = summary.get("win_rate_pct", 0)
    if 45 <= win_rate <= 55 and trades > 20:
        out.append(f"- **Win rate ~{win_rate}%** is statistically indistinguishable from a "
                   "coin flip. The model isn't ranking well — try feature engineering or a "
                   "different label horizon before tuning the engine.")

    if summary.get("sl_exits", 0) > summary.get("target_exits", 0) * 2 and trades > 20:
        out.append("- **SL exits dominate target exits 2:1+**: either SL is too tight, "
                   "target is too far, or signals are systematically wrong-direction.")

    if not daily.empty:
        losing_days = (daily["pnl_inr"] < 0).sum()
        total_days = len(daily)
        if losing_days >= total_days * 0.6:
            out.append(f"- **{losing_days}/{total_days} days are losing.** Costs may be "
                       "eating the edge, or the engine's entry-window timing is off.")

    final_pnl = (
        report.equity_curve[-1].cash_pnl + report.equity_curve[-1].open_unrealised
        if report.equity_curve else 0
    )
    if trades > 0 and abs(final_pnl) < trades * 5:
        out.append("- **Net P&L is tiny vs trade count**: the model is generating noise. "
                   "Costs are dominating. Either raise `min_predicted_edge` or stop trading "
                   "this signal.")

    if not out:
        out.append("- No obvious red flags from the heuristic checks. Read the trade log "
                   "for anything that looks wrong.")
    return out


def save_report(report: BacktestReport, out_dir: Path = Path("data/backtests")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"backtest_{stamp}.md"
    path.write_text(render_markdown(report))
    return path
