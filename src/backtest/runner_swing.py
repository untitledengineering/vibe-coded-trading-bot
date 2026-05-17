"""Multi-day swing backtester.

Validates `swing_momentum.evaluate` against historical daily bars. Mirrors the
ad-hoc test in /tmp/swing_test.py but lives in the production tree with proper
output + integration into the existing backtest report directory.

Differences from the intraday runners:
    - Iterates days, not minutes
    - Fills at next day's open (NOT next bar's open)
    - SL/TP check against each day's high/low
    - Time exit after N trading days (config.hold_max_days)
    - No EOD forced exit — positions ride across days
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.bot.positions import Position
from src.strategy.bars_daily import aggregate_to_daily
from src.strategy.swing_momentum import SwingConfig, _add_features, evaluate
from src.utils.config import DB_PATH
from src.utils.logger import logger

REPORTS_DIR = Path("data/backtests")


@dataclass
class SwingTrade:
    instrument_key: str
    entry_day: int  # ist_day integer
    exit_day: int
    entry_price: float
    exit_price: float
    qty: int
    exit_reason: str
    net_pnl_inr: float


@dataclass
class SwingBacktestReport:
    config: dict
    trades: List[SwingTrade] = field(default_factory=list)
    skipped_total: Dict[str, int] = field(default_factory=dict)
    days_evaluated: int = 0
    runtime_seconds: float = 0.0


def _ist_day_to_date(d: int) -> datetime:
    return datetime(1970, 1, 1) + timedelta(days=int(d))


def _load_daily_by_symbol(db_path: str, instrument_keys: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        q = "SELECT instrument_key, minute_ts, open, high, low, close, volume FROM bars_1m"
        params: tuple = ()
        if instrument_keys:
            q += " WHERE instrument_key IN (" + ",".join("?" * len(instrument_keys)) + ")"
            params = tuple(instrument_keys)
        bars = pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()

    if bars.empty:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for key, g in bars.groupby("instrument_key"):
        daily = aggregate_to_daily(g)
        if not daily.empty:
            out[key] = daily.reset_index(drop=True)
    return out


def run_swing_backtest(
    config: Optional[SwingConfig] = None,
    db_path: Optional[str] = None,
    instrument_keys: Optional[List[str]] = None,
) -> SwingBacktestReport:
    config = config or SwingConfig()
    db_path = db_path or DB_PATH

    t0 = time.monotonic()
    logger.info("Loading bars and aggregating to daily...")
    daily_by_symbol = _load_daily_by_symbol(db_path, instrument_keys)
    if not daily_by_symbol:
        raise RuntimeError("No bars in DB. Run the historical backfill first.")
    # Restrict universe: equities only (NSE_EQ|...). Indices and other
    # segments would distort the cost model and the S1 signal.
    daily_by_symbol = {
        k: v for k, v in daily_by_symbol.items() if k.startswith("NSE_EQ|")
    }
    logger.info(f"Loaded daily bars for {len(daily_by_symbol)} equities")

    # Pre-compute features for every symbol so the inner loop is just lookups.
    indexed: Dict[str, pd.DataFrame] = {}
    for key, daily in daily_by_symbol.items():
        feat = _add_features(daily, config).set_index("ist_day")
        indexed[key] = feat

    all_days = sorted({d for df in indexed.values() for d in df.index})
    report = SwingBacktestReport(config=asdict(config))

    open_positions: List[Position] = []
    pending_intents: List[tuple] = []  # (instrument_key, intended_day) — fill next day's open

    for day in all_days:
        report.days_evaluated += 1

        # 1. Fill yesterday's pending intents at today's open.
        if pending_intents:
            still_pending: List[tuple] = []
            for key, _ in pending_intents:
                df = indexed[key]
                if day not in df.index:
                    report.skipped_total["no_fill_bar"] = report.skipped_total.get("no_fill_bar", 0) + 1
                    continue
                row = df.loc[day]
                entry_price = float(row["open"])
                qty = max(0, int(config.notional_per_slot_inr // entry_price))
                if qty == 0:
                    report.skipped_total["qty_zero_at_fill"] = report.skipped_total.get("qty_zero_at_fill", 0) + 1
                    continue
                pos = Position(
                    instrument_key=key, side="long", qty=qty,
                    entry_ts=day,  # using ist_day as a stand-in epoch unit
                    entry_price=entry_price,
                    stop_loss_price=entry_price * (1 - config.stop_loss_pct),
                    target_price=entry_price * (1 + config.target_pct),
                )
                open_positions.append(pos)
            pending_intents = still_pending

        # 2. Exit checks for open positions.
        survivors: List[Position] = []
        for pos in open_positions:
            df = indexed[pos.instrument_key]
            if day not in df.index:
                survivors.append(pos)
                continue
            row = df.loc[day]
            high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            reason = pos.should_exit_at(high=high, low=low)
            if reason == "stop_loss":
                exit_price = pos.stop_loss_price
            elif reason == "target":
                exit_price = pos.target_price
            elif day - pos.entry_ts >= config.hold_max_days:
                reason = "time_exit"
                exit_price = close
            else:
                survivors.append(pos)
                continue
            cost = config.notional_per_slot_inr * config.round_trip_cost_pct
            net = pos.qty * (exit_price - pos.entry_price) - cost
            report.trades.append(SwingTrade(
                instrument_key=pos.instrument_key,
                entry_day=pos.entry_ts, exit_day=day,
                entry_price=pos.entry_price, exit_price=exit_price,
                qty=pos.qty, exit_reason=reason, net_pnl_inr=net,
            ))
        open_positions = survivors

        # 3. Decision step: gather candidates from indexed for `day`.
        decision_universe = {}
        for key, df in indexed.items():
            if day not in df.index:
                continue
            # Slice up to and including `day` for the strategy.
            decision_universe[key] = df.loc[:day].reset_index()
        decision = evaluate(
            daily_by_symbol=decision_universe,
            open_positions=open_positions,
            closed_positions=[],
            now_ts=day,
            config=config,
        )
        for k, v in decision.skipped.items():
            report.skipped_total[k] = report.skipped_total.get(k, 0) + v
        for intent in decision.intents:
            pending_intents.append((intent.instrument_key, day))

    # Close any survivors at the last bar's close.
    if open_positions and all_days:
        last_day = all_days[-1]
        for pos in open_positions:
            df = indexed[pos.instrument_key]
            if last_day in df.index:
                close = float(df.loc[last_day]["close"])
                cost = config.notional_per_slot_inr * config.round_trip_cost_pct
                net = pos.qty * (close - pos.entry_price) - cost
                report.trades.append(SwingTrade(
                    instrument_key=pos.instrument_key,
                    entry_day=pos.entry_ts, exit_day=last_day,
                    entry_price=pos.entry_price, exit_price=close,
                    qty=pos.qty, exit_reason="end_of_backtest", net_pnl_inr=net,
                ))

    report.runtime_seconds = time.monotonic() - t0
    return report


def render_report(report: SwingBacktestReport) -> str:
    trades = report.trades
    n = len(trades)
    if n == 0:
        return "# Swing backtest\n\nNo trades."

    net = sum(t.net_pnl_inr for t in trades)
    wins = [t for t in trades if t.net_pnl_inr > 0]
    config = report.config

    # Monthly breakdown
    by_month: Dict[str, List[SwingTrade]] = {}
    for t in trades:
        m = _ist_day_to_date(t.exit_day).strftime("%Y-%m")
        by_month.setdefault(m, []).append(t)

    md: List[str] = []
    md.append(f"# Swing backtest — S1 momentum continuation")
    md.append("")
    md.append(f"Capital ₹{config['total_capital_inr']:,.0f} · slots {config['max_positions']} · "
              f"target {config['target_pct']*100:.1f}% · SL {config['stop_loss_pct']*100:.1f}% · "
              f"hold ≤{config['hold_max_days']}d")
    md.append("")
    md.append("## Headline")
    md.append(f"- **Trades**: {n}")
    md.append(f"- **Net P&L**: ₹{net:,.0f}  ({net/config['total_capital_inr']*100:+.2f}% on capital)")
    md.append(f"- **Win rate**: {len(wins)/n*100:.1f}%  ({len(wins)}/{n})")
    md.append(f"- **Days evaluated**: {report.days_evaluated}")
    md.append(f"- **Runtime**: {report.runtime_seconds:.1f}s")
    md.append("")
    md.append("## Monthly breakdown")
    md.append("| month | trades | wins | net |")
    md.append("|-------|-------:|-----:|----:|")
    for month in sorted(by_month):
        ts = by_month[month]
        m_net = sum(t.net_pnl_inr for t in ts)
        m_wins = sum(1 for t in ts if t.net_pnl_inr > 0)
        md.append(f"| {month} | {len(ts)} | {m_wins} | ₹{m_net:+,.0f} |")
    md.append("")
    md.append("## Exit reasons")
    md.append("| reason | count |")
    md.append("|--------|------:|")
    reason_counts: Dict[str, int] = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| {reason} | {count} |")
    md.append("")
    md.append("## Skip reasons (decision-level)")
    md.append("| reason | count |")
    md.append("|--------|------:|")
    for reason, count in sorted(report.skipped_total.items(), key=lambda kv: -kv[1])[:15]:
        md.append(f"| {reason} | {count:,} |")
    md.append("")
    md.append("## Trade log (latest 30)")
    md.append("| entry_date | exit_date | symbol | qty | entry₹ | exit₹ | reason | pnl₹ |")
    md.append("|------------|-----------|--------|----:|-------:|------:|--------|-----:|")
    latest = sorted(trades, key=lambda t: t.exit_day)[-30:]
    for t in latest:
        md.append(
            f"| {_ist_day_to_date(t.entry_day).date()} | {_ist_day_to_date(t.exit_day).date()} | "
            f"`{t.instrument_key}` | {t.qty} | {t.entry_price:.2f} | {t.exit_price:.2f} | "
            f"{t.exit_reason} | {t.net_pnl_inr:+.2f} |"
        )
    return "\n".join(md)


def save_report(report: SwingBacktestReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = REPORTS_DIR / f"swing_{stamp}.md"
    path.write_text(render_report(report))
    return path


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest the S1 multi-day swing strategy.")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--slots", type=int, default=5)
    parser.add_argument("--target-pct", type=float, default=0.06)
    parser.add_argument("--sl-pct", type=float, default=0.03)
    parser.add_argument("--hold-days", type=int, default=10)
    args = parser.parse_args()

    config = SwingConfig(
        total_capital_inr=args.capital,
        max_positions=args.slots,
        target_pct=args.target_pct,
        stop_loss_pct=args.sl_pct,
        hold_max_days=args.hold_days,
    )
    report = run_swing_backtest(config=config)
    path = save_report(report)
    net = sum(t.net_pnl_inr for t in report.trades)
    n = len(report.trades)
    wins = sum(1 for t in report.trades if t.net_pnl_inr > 0)
    print(f"Report: {path}")
    print(f"Trades: {n}  Wins: {wins} ({wins/n*100 if n else 0:.1f}%)")
    print(f"Net:    ₹{net:+,.0f}  ({net/args.capital*100:+.2f}% of capital)")


if __name__ == "__main__":
    _cli()
