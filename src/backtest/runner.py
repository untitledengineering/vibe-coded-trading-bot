"""Backtest runner — replays bars_1m, feeds the SAME decision engine the live
paper engine will use, and produces a structured report.

Replay protocol:
    For each minute m in [start, end] sorted ascending:
        1. For every open Position, check whether bar m's high/low hit SL/TP.
           If so, close at the trigger price. SL is checked before TP (worst case).
        2. If m is at-or-past the forced-exit minute (14:55 IST), close ALL
           remaining open positions at bar m's close. No new entries today.
        3. Otherwise, if m is inside the entry window, run the decision engine
           on the cross-section of features at m, producing OrderIntents.
        4. Intents fill at bar (m+1)'s open price for their symbol. Symbols
           without a bar at m+1 (data gaps) cause the intent to be dropped
           with a 'no_fill_bar' skip count.

The runner does NOT simulate concurrent positions correctly if two trades on
the same symbol are queued (we drop redundant intents). It does NOT simulate
intra-bar order book — fills are conservative (worse-of-open vs avg). These
are deliberate v1 simplifications, called out in the report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.executor import BacktestExecutor
from src.bot.engine import (
    EngineConfig,
    OrderIntent,
    decide,
    is_forced_exit,
)
from src.bot.positions import Position
from src.features.technical import FEATURE_COLUMNS, IST_OFFSET_SECONDS
from src.model.dataset import DatasetSpec, assemble_full_dataset
from src.model.infer import ModelArtifact, load_model
from src.utils.logger import logger


@dataclass
class EquityPoint:
    minute_ts: int
    cash_pnl: float           # realised P&L since start (after costs)
    open_unrealised: float    # mark-to-market on currently open positions


@dataclass
class BacktestReport:
    start_ts: int
    end_ts: int
    config: EngineConfig
    model_path: str
    completed_positions: List[Position] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    minutes_seen: int = 0
    runtime_seconds: float = 0.0


def _ist_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts + IST_OFFSET_SECONDS).strftime("%Y-%m-%d")


def _filter_by_date(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    mask = (df["minute_ts"] >= start_ts) & (df["minute_ts"] <= end_ts)
    return df[mask]


def _parse_iso_date_to_ts(iso_date: str, end_of_day: bool = False) -> int:
    """ISO date (YYYY-MM-DD) -> epoch seconds at IST 09:15 (start) or 15:30 (end)."""
    dt = datetime.fromisoformat(iso_date)
    day_index = (dt - datetime(1970, 1, 1)).days
    if end_of_day:
        # 15:30 IST = 555 + 15*60 + 30 - 555 = 15h30m IST minute-of-day = 930
        ist_minute = 15 * 60 + 30
    else:
        ist_minute = 9 * 60 + 15
    return day_index * 86400 + ist_minute * 60 - IST_OFFSET_SECONDS


def _bars_lookup(minute_slice: pd.DataFrame) -> Dict[str, pd.Series]:
    """instrument_key -> row, for fast SL/TP and fill lookups within a minute."""
    return {row["instrument_key"]: row for _, row in minute_slice.iterrows()}


def run_backtest(
    start_date: str,
    end_date: str,
    model: Optional[ModelArtifact] = None,
    config: Optional[EngineConfig] = None,
    symbols: Optional[List[str]] = None,
) -> BacktestReport:
    config = config or EngineConfig()
    model = model or load_model()

    start_ts = _parse_iso_date_to_ts(start_date)
    end_ts = _parse_iso_date_to_ts(end_date, end_of_day=True)

    logger.info(f"Backtest range: {start_date} ({start_ts}) .. {end_date} ({end_ts})")
    logger.info("Loading dataset...")
    t0 = time.monotonic()
    spec = DatasetSpec(feature_columns=tuple(model.feature_columns))
    df = assemble_full_dataset(spec=spec, symbols=symbols)
    df = _filter_by_date(df, start_ts, end_ts)
    if df.empty:
        raise RuntimeError(
            f"No rows in the requested window. Adjust --start/--end or run a wider backfill."
        )
    df = df.sort_values("minute_ts").reset_index(drop=True)
    logger.info(f"Dataset loaded in {time.monotonic() - t0:.1f}s. Rows: {len(df):,}")

    # Pre-bucket by minute_ts for fast iteration. This is O(N) and avoids repeated groupby.
    minute_groups: List[tuple[int, pd.DataFrame]] = list(df.groupby("minute_ts", sort=True))

    executor = BacktestExecutor(config=config)
    report = BacktestReport(
        start_ts=start_ts,
        end_ts=end_ts,
        config=config,
        model_path=str(load_model.__module__),  # we'll let the CLI overwrite this if a path is passed
    )
    open_positions: List[Position] = []
    pending_intents: List[OrderIntent] = []

    # Loop start
    t1 = time.monotonic()
    for i, (minute_ts, minute_slice) in enumerate(minute_groups):
        report.minutes_seen += 1
        bars_by_key = _bars_lookup(minute_slice)

        # 1. Fill any pending intents at this minute's OPEN.
        if pending_intents:
            still_pending: List[OrderIntent] = []
            for intent in pending_intents:
                row = bars_by_key.get(intent.instrument_key)
                if row is None:
                    report.skipped["no_fill_bar"] = report.skipped.get("no_fill_bar", 0) + 1
                    continue
                # If the intent was an exit (eod or kill-switch) it targets a position
                # already in open_positions — handle below in step 4, not here.
                if intent.reason in ("exit_eod", "exit_kill_switch"):
                    still_pending.append(intent)
                    continue
                pos = executor.open_position(intent, fill_ts=int(minute_ts), bar_open_price=float(row["open"]))
                open_positions.append(pos)
            pending_intents = still_pending

        # 2. SL/TP triggers against this minute's high/low.
        survivors: List[Position] = []
        for pos in open_positions:
            row = bars_by_key.get(pos.instrument_key)
            if row is None:
                # No bar at this minute — keep position open, will check next minute.
                survivors.append(pos)
                continue
            trigger = pos.should_exit_at(high=float(row["high"]), low=float(row["low"]))
            if trigger == "stop_loss":
                exit_price = pos.stop_loss_price
            elif trigger == "target":
                exit_price = pos.target_price
            else:
                survivors.append(pos)
                continue
            executor.close_position(pos, exit_ts=int(minute_ts), exit_price=exit_price, reason=trigger)
            report.completed_positions.append(pos)
        open_positions = survivors

        # 3. Forced exit at 14:55 IST — close all at this minute's close.
        if is_forced_exit(int(minute_ts), config):
            for pos in open_positions:
                row = bars_by_key.get(pos.instrument_key)
                if row is None:
                    continue
                executor.close_position(
                    pos, exit_ts=int(minute_ts), exit_price=float(row["close"]), reason="eod"
                )
                report.completed_positions.append(pos)
            open_positions = []

        # 4. Decide for the NEXT minute. We pass this minute's features; fills happen next.
        feature_frame = minute_slice[["instrument_key", "close", *FEATURE_COLUMNS]].copy()
        result = decide(
            features_at_minute=feature_frame,
            model=model,
            open_positions=open_positions,
            config=config,
            now_ts=int(minute_ts),
            closed_positions=report.completed_positions,
        )
        for reason, n in result.skipped_reasons.items():
            report.skipped[reason] = report.skipped.get(reason, 0) + n
        # Queue intents to fill next minute (we drop self-redundant intents).
        existing_keys = {p.instrument_key for p in open_positions}
        for intent in result.intents:
            if intent.instrument_key in existing_keys:
                continue
            pending_intents.append(intent)

        # 5. Mark-to-market for the equity curve.
        unrealised = 0.0
        for pos in open_positions:
            row = bars_by_key.get(pos.instrument_key)
            if row is not None:
                unrealised += pos.unrealised_pnl_inr(float(row["close"]))
        cash = sum(p.realised_pnl_inr or 0.0 for p in report.completed_positions)
        report.equity_curve.append(
            EquityPoint(minute_ts=int(minute_ts), cash_pnl=cash, open_unrealised=unrealised)
        )

    # Any positions still open at end of range get closed at the last seen price.
    if open_positions and minute_groups:
        last_ts, last_slice = minute_groups[-1]
        last_bars = _bars_lookup(last_slice)
        for pos in open_positions:
            row = last_bars.get(pos.instrument_key)
            if row is None:
                continue
            executor.close_position(
                pos, exit_ts=int(last_ts), exit_price=float(row["close"]), reason="end_of_backtest"
            )
            report.completed_positions.append(pos)

    report.runtime_seconds = time.monotonic() - t1
    logger.info(
        f"Backtest replay finished in {report.runtime_seconds:.1f}s. "
        f"Trades: {len(report.completed_positions)}, minutes: {report.minutes_seen}"
    )
    return report
