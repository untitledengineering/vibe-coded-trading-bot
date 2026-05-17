"""Rule-based strategy backtester (VWAP + Supertrend on 5-min bars).

Walks 1-minute bars in chronological order. On every 5-minute boundary, builds
the just-closed 5-min candles for each symbol and asks the strategy module for
intents. Fills at the next 1-min bar's open with slippage. SL/TP checks against
each 1-min bar's high/low. Trailing-to-cost moves SL to entry once price has
moved 1R in our favour.

Output: data/backtests/rules_<timestamp>.md and a returned BacktestReport-like
dict. Re-uses the existing cost model + slippage handling via BacktestExecutor.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.backtest.executor import BacktestExecutor
from src.bot.engine import EngineConfig, OrderIntent
from src.bot.positions import Position
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.strategy.bars5m import FIVE_MIN_SECONDS, aggregate_to_5m
from src.strategy.rule_based import (
    RULE_BASED_INSTRUMENT_KEYS,
    RuleBasedConfig,
    StrategyDecision,
    consecutive_losses,
    evaluate as evaluate_rules,
    is_forced_exit,
)
from src.utils.config import DB_PATH
from src.utils.logger import logger

REPORTS_DIR = Path("data/backtests")


@dataclass
class RulesBacktestReport:
    start_ts: int
    end_ts: int
    config: dict
    completed_positions: List[Position] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    halt_events: List[dict] = field(default_factory=list)
    minutes_seen: int = 0
    runtime_seconds: float = 0.0


def _ist_session_date(ts: int) -> int:
    return (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY


def _parse_iso_date_to_ts(iso_date: str, end_of_day: bool = False) -> int:
    dt = datetime.fromisoformat(iso_date)
    day_index = (dt - datetime(1970, 1, 1)).days
    minute = (15 * 60 + 30) if end_of_day else (9 * 60 + 15)
    return day_index * 86400 + minute * 60 - IST_OFFSET_SECONDS


def _read_universe_bars(
    db_path: str,
    instrument_keys: List[str],
    start_ts: int,
    end_ts: int,
) -> Dict[str, pd.DataFrame]:
    """Bulk-load 1-min bars in the window for the rule-based universe."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    out: Dict[str, pd.DataFrame] = {}
    try:
        for key in instrument_keys:
            df = pd.read_sql_query(
                """
                SELECT minute_ts, open, high, low, close, volume
                FROM bars_1m
                WHERE instrument_key = ? AND minute_ts >= ? AND minute_ts <= ?
                ORDER BY minute_ts ASC
                """,
                conn,
                params=(key, start_ts, end_ts),
            )
            if not df.empty:
                out[key] = df.reset_index(drop=True)
    finally:
        conn.close()
    return out


def _build_minute_index(bars_by_symbol: Dict[str, pd.DataFrame]) -> List[int]:
    """Sorted distinct minute_ts across all symbols."""
    all_mins: set = set()
    for df in bars_by_symbol.values():
        all_mins.update(df["minute_ts"].tolist())
    return sorted(all_mins)


def _bars_dict_at_minute(
    bars_by_symbol: Dict[str, pd.DataFrame],
    minute_ts: int,
) -> Dict[str, pd.Series]:
    """For each symbol, return the bar at minute_ts as a Series (or skip)."""
    out: Dict[str, pd.Series] = {}
    for key, df in bars_by_symbol.items():
        row = df.loc[df["minute_ts"] == minute_ts]
        if not row.empty:
            out[key] = row.iloc[0]
    return out


def _aggregate_through(
    bars_by_symbol: Dict[str, pd.DataFrame],
    upto_ts_exclusive: int,
) -> Dict[str, pd.DataFrame]:
    """5-min aggregation of every symbol's bars whose minute_ts < upto_ts_exclusive.
    upto_ts_exclusive is typically the current 5-min boundary; we exclude it so
    we only act on the *just-closed* bucket."""
    out: Dict[str, pd.DataFrame] = {}
    for key, df in bars_by_symbol.items():
        sub = df[df["minute_ts"] < upto_ts_exclusive]
        if sub.empty:
            continue
        out[key] = aggregate_to_5m(sub)
    return out


EvaluateFn = Callable[..., StrategyDecision]


def run_rules_backtest(
    start_date: str,
    end_date: str,
    config: Optional[RuleBasedConfig] = None,
    db_path: Optional[str] = None,
    slippage_bps: float = 5.0,
    evaluate_fn: Optional[EvaluateFn] = None,
    label: str = "rules",
    instrument_keys: Optional[List[str]] = None,
) -> RulesBacktestReport:
    """Run a backtest using any strategy that conforms to the
    `evaluate(bars_5m_by_symbol, open_positions, closed_today, now_ts, config)`
    contract from strategy.rule_based."""
    config = config or RuleBasedConfig()
    db_path = db_path or DB_PATH
    evaluate_fn = evaluate_fn or evaluate_rules
    keys = instrument_keys or RULE_BASED_INSTRUMENT_KEYS

    start_ts = _parse_iso_date_to_ts(start_date)
    end_ts = _parse_iso_date_to_ts(end_date, end_of_day=True)
    logger.info(f"Backtest [{label}] range: {start_date} .. {end_date}")

    t0 = time.monotonic()
    logger.info("Loading bars...")
    bars_by_symbol = _read_universe_bars(db_path, keys, start_ts, end_ts)
    if not bars_by_symbol:
        raise RuntimeError(
            "No bars in window for rule-based universe. Run the historical "
            f"backfill, or check that {RULE_BASED_INSTRUMENT_KEYS} are present in bars_1m."
        )
    minute_index = _build_minute_index(bars_by_symbol)
    logger.info(f"Loaded {sum(len(df) for df in bars_by_symbol.values()):,} bars across "
                f"{len(bars_by_symbol)} symbols in {time.monotonic() - t0:.1f}s")

    # Use the existing EngineConfig only for executor cost wiring; the rule-based
    # strategy uses its own RuleBasedConfig everywhere else.
    executor = BacktestExecutor(config=EngineConfig(), slippage_bps=slippage_bps)
    report = RulesBacktestReport(
        start_ts=start_ts, end_ts=end_ts, config=asdict(config),
    )
    open_positions: List[Position] = []
    pending_intents: List[OrderIntent] = []
    halt_until_ts: int = 0  # consec-loss cooldown
    halted_for_day_at: Dict[int, str] = {}  # session_date -> reason

    t1 = time.monotonic()
    for minute_ts in minute_index:
        report.minutes_seen += 1
        session = _ist_session_date(minute_ts)
        bars_at_now = _bars_dict_at_minute(bars_by_symbol, minute_ts)

        # 1. Fill pending intents at this minute's open.
        if pending_intents:
            still_pending: List[OrderIntent] = []
            for intent in pending_intents:
                bar = bars_at_now.get(intent.instrument_key)
                if bar is None:
                    report.skipped["no_fill_bar"] = report.skipped.get("no_fill_bar", 0) + 1
                    continue
                pos = executor.open_position(intent, fill_ts=int(minute_ts), bar_open_price=float(bar["open"]))
                open_positions.append(pos)
            pending_intents = still_pending

        # 2. SL/TP triggers + trailing-to-cost against this minute's high/low.
        survivors: List[Position] = []
        for pos in open_positions:
            bar = bars_at_now.get(pos.instrument_key)
            if bar is None:
                survivors.append(pos)
                continue
            high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            trigger = pos.should_exit_at(high=high, low=low)
            if trigger == "stop_loss":
                exit_price = pos.stop_loss_price
            elif trigger == "target":
                exit_price = pos.target_price
            else:
                # Trail-to-breakeven check on this bar's close.
                pos.maybe_trail_to_breakeven(current_price=close)
                survivors.append(pos)
                continue
            executor.close_position(pos, exit_ts=int(minute_ts), exit_price=exit_price, reason=trigger)
            report.completed_positions.append(pos)
        open_positions = survivors

        # 3. Forced exit (15:15 IST) — close everything at this bar's close.
        if is_forced_exit(int(minute_ts), config):
            for pos in open_positions:
                bar = bars_at_now.get(pos.instrument_key)
                if bar is None:
                    continue
                executor.close_position(
                    pos, exit_ts=int(minute_ts),
                    exit_price=float(bar["close"]), reason="eod",
                )
                report.completed_positions.append(pos)
            open_positions = []

        # 4. Kill-switch checks (run BEFORE we decide so we don't queue new entries).
        if session not in halted_for_day_at:
            todays = [p for p in report.completed_positions
                      if p.exit_ts is not None and _ist_session_date(p.exit_ts) == session]
            todays_pnl = sum(p.realised_pnl_inr or 0.0 for p in todays)
            if todays_pnl <= -config.daily_loss_cap_inr:
                halted_for_day_at[session] = f"daily_loss_cap (₹{todays_pnl:.2f})"
                report.halt_events.append({
                    "ts": int(minute_ts), "session": session,
                    "reason": halted_for_day_at[session],
                })
            else:
                consec = consecutive_losses(todays)
                if consec >= config.consecutive_loss_halt and minute_ts >= halt_until_ts:
                    halt_until_ts = int(minute_ts) + config.consec_loss_pause_minutes * 60
                    report.halt_events.append({
                        "ts": int(minute_ts), "session": session,
                        "reason": f"consec_loss_pause ({consec} in a row)",
                    })

        if session in halted_for_day_at:
            report.skipped["halt_daily_loss"] = report.skipped.get("halt_daily_loss", 0) + 1
            continue
        if minute_ts < halt_until_ts:
            report.skipped["halt_consec_loss"] = report.skipped.get("halt_consec_loss", 0) + 1
            continue

        # 5. Decision happens only at 5-min boundaries (close of the just-finished candle).
        if minute_ts % FIVE_MIN_SECONDS != 0:
            continue
        bars_5m = _aggregate_through(bars_by_symbol, upto_ts_exclusive=int(minute_ts))
        if not bars_5m:
            continue

        decision = evaluate_fn(
            bars_5m_by_symbol=bars_5m,
            open_positions=open_positions,
            closed_today=[p for p in report.completed_positions
                          if p.exit_ts is not None and _ist_session_date(p.exit_ts) == session],
            now_ts=int(minute_ts),
            config=config,
        )
        for reason, n in decision.skipped.items():
            report.skipped[reason] = report.skipped.get(reason, 0) + n
        held_keys = {p.instrument_key for p in open_positions}
        for intent in decision.intents:
            if intent.instrument_key in held_keys:
                continue
            pending_intents.append(intent)

    # Close any survivors at the last bar.
    if open_positions and minute_index:
        last_ts = minute_index[-1]
        last_bars = _bars_dict_at_minute(bars_by_symbol, last_ts)
        for pos in open_positions:
            bar = last_bars.get(pos.instrument_key)
            if bar is None:
                continue
            executor.close_position(
                pos, exit_ts=int(last_ts),
                exit_price=float(bar["close"]), reason="end_of_backtest",
            )
            report.completed_positions.append(pos)

    report.runtime_seconds = time.monotonic() - t1
    logger.info(
        f"Rules backtest finished in {report.runtime_seconds:.1f}s. "
        f"Trades: {len(report.completed_positions)}, minutes: {report.minutes_seen}"
    )
    return report


# ---------- Report rendering ----------

def _ist_str(ts: int) -> str:
    return datetime.utcfromtimestamp(ts + IST_OFFSET_SECONDS).strftime("%Y-%m-%d %H:%M")


def render_report_markdown(report: RulesBacktestReport) -> str:
    positions = report.completed_positions
    n = len(positions)
    wins = [p for p in positions if (p.realised_pnl_inr or 0) > 0]
    losses = [p for p in positions if (p.realised_pnl_inr or 0) < 0]
    net = sum(p.realised_pnl_inr or 0.0 for p in positions)
    longs = sum(1 for p in positions if p.side == "long")
    shorts = sum(1 for p in positions if p.side == "short")
    sl_exits = sum(1 for p in positions if p.exit_reason == "stop_loss")
    target_exits = sum(1 for p in positions if p.exit_reason == "target")
    eod_exits = sum(1 for p in positions if p.exit_reason == "eod")
    trail_locked = sum(1 for p in positions if p.breakeven_locked)

    md: List[str] = []
    md.append(f"# Rule-based backtest — {_ist_str(report.start_ts)[:10]} to {_ist_str(report.end_ts)[:10]}")
    md.append("")
    md.append(f"Strategy: VWAP + Supertrend(7,3) · runtime {report.runtime_seconds:.1f}s")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Trades: **{n}**")
    md.append(f"- Net P&L (incl. costs + slippage): **₹{net:,.2f}**")
    md.append(f"- Win rate: **{(len(wins)/n*100 if n else 0):.2f}%**" )
    md.append(f"- Wins / Losses: {len(wins)} / {len(losses)}")
    md.append(f"- Long / Short: {longs} / {shorts}")
    md.append(f"- Exits: SL {sl_exits} · Target {target_exits} · EOD {eod_exits}")
    md.append(f"- Breakeven-trail engaged: {trail_locked}")
    md.append("")
    md.append("## Halt events")
    if not report.halt_events:
        md.append("_None._")
    else:
        md.append("| ts | session | reason |")
        md.append("|----|---------|--------|")
        for h in report.halt_events:
            md.append(f"| {_ist_str(h['ts'])} | {h['session']} | {h['reason']} |")
    md.append("")
    md.append("## Skips")
    if not report.skipped:
        md.append("_None._")
    else:
        md.append("| reason | count |")
        md.append("|--------|------:|")
        for reason, count in sorted(report.skipped.items(), key=lambda kv: -kv[1]):
            md.append(f"| `{reason}` | {count:,} |")
    md.append("")
    md.append("## Trade log (latest 50)")
    if not positions:
        md.append("_None._")
    else:
        latest = sorted(positions, key=lambda p: p.exit_ts or 0)[-50:]
        md.append("| entry | exit | symbol | side | qty | entry₹ | exit₹ | reason | pnl₹ |")
        md.append("|-------|------|--------|------|----:|-------:|------:|--------|-----:|")
        for p in latest:
            md.append(
                f"| {_ist_str(p.entry_ts)} | {_ist_str(p.exit_ts) if p.exit_ts else ''} | "
                f"`{p.instrument_key}` | {p.side} | {p.qty} | "
                f"{p.entry_price:.2f} | {(p.exit_price or 0):.2f} | "
                f"{p.exit_reason} | {(p.realised_pnl_inr or 0):.2f} |"
            )
    return "\n".join(md)


def save_report(report: RulesBacktestReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = REPORTS_DIR / f"rules_{stamp}.md"
    path.write_text(render_report_markdown(report))
    return path


# ---------- CLI ----------

def _combined_evaluate(*args, **kwargs):
    """ORB first (higher-conviction breakout), Gap Fade for symbols ORB didn't claim."""
    from src.strategy import gap_fade, orb
    open_positions = kwargs["open_positions"]
    d1 = orb.evaluate(*args, **kwargs)
    d2 = gap_fade.evaluate(*args, **kwargs)
    combined = StrategyDecision()
    seen = {p.instrument_key for p in open_positions}
    for d in (d1, d2):
        for intent in d.intents:
            if intent.instrument_key in seen:
                continue
            combined.intents.append(intent)
            seen.add(intent.instrument_key)
        for k, v in d.skipped.items():
            combined.skipped[k] = combined.skipped.get(k, 0) + v
    return combined


_STRATEGY_REGISTRY = {
    "rules": (evaluate_rules, "rules"),
    "orb": ("src.strategy.orb:evaluate", "orb"),
    "orb_fade": ("src.strategy.orb_fade:evaluate", "orb_fade"),
    "gap_fade": ("src.strategy.gap_fade:evaluate", "gap_fade"),
    "combined": (_combined_evaluate, "combined"),
}


def _resolve_strategy(name: str):
    if name not in _STRATEGY_REGISTRY:
        raise SystemExit(f"Unknown strategy {name!r}. Choices: {list(_STRATEGY_REGISTRY)}")
    entry, label = _STRATEGY_REGISTRY[name]
    if isinstance(entry, str):
        mod_name, attr = entry.split(":")
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr), label
    return entry, label


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest a rule-based strategy.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--capital", type=float, default=25_000.0)
    parser.add_argument(
        "--strategy", default="rules",
        choices=list(_STRATEGY_REGISTRY),
        help="rules=VWAP+Supertrend, orb=Opening Range Breakout, "
             "gap_fade=Gap Fade, combined=ORB + Gap Fade",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Override the default universe with one or more instrument_keys. "
             "Useful for running on indices: --symbols 'NSE_INDEX|Nifty 50'",
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Override max_positions in config (e.g., 1 for single-instrument index runs)",
    )
    args = parser.parse_args()

    cfg_kwargs = {"total_capital_inr": args.capital}
    if args.max_positions is not None:
        cfg_kwargs["max_positions"] = args.max_positions
    config = RuleBasedConfig(**cfg_kwargs)
    evaluate_fn, label = _resolve_strategy(args.strategy)
    report = run_rules_backtest(
        args.start, args.end, config=config,
        evaluate_fn=evaluate_fn, label=label,
        instrument_keys=args.symbols,
    )
    path = save_report(report)
    print(f"Strategy: {label}")
    print(f"Report:   {path}")
    net = sum(p.realised_pnl_inr or 0.0 for p in report.completed_positions)
    print(f"Trades:   {len(report.completed_positions)}")
    print(f"Net P&L:  ₹{net:,.2f}")


if __name__ == "__main__":
    _cli()
