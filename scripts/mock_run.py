#!/usr/bin/env python3
"""Mock dry-run of the full trading pipeline.

Uses the real bars_1m DB (17M rows, real history) for feature warmup.
Injects synthetic "tomorrow morning" bars into bars_live, fake LTPs into
the streamer cache, and fresh news headlines, then runs one complete
engine cycle. All injected data is cleaned up at the end.

Run from repo root:
    python3.11 scripts/mock_run.py
"""

from __future__ import annotations

import asyncio
import calendar
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── repo root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Provide a dummy encryption key so database.py doesn't crash on import
import base64
os.environ.setdefault(
    "TOKEN_ENCRYPTION_KEY",
    base64.urlsafe_b64encode(b"mock_key_for_dry_run_only_32byte").decode(),
)

# ── imports after path setup ───────────────────────────────────────────────
import aiosqlite

from src.api import streamer_manager
from src.bot.engine import EngineConfig, decide
from src.bot.positions import Position
from src.features.sentiment_cache import live_sentiment_by_symbol
from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY
from src.model.infer import load_model, score as model_score
from src.model.train import paths_for_name
from src.paper.executor import PaperExecutor
from src.paper.feature_cache import build_live_feature_frame
from src.paper.persistence import (
    list_closed_positions_today,
    list_open_positions,
    todays_realised_pnl,
)
from src.utils.config import DB_PATH
from src.utils.logger import logger

# ── config ─────────────────────────────────────────────────────────────────
MOCK_SYMBOLS_N = 209         # full universe
MOCK_BARS_PER_SYMBOL = 45    # 45 minutes of live bars (9:15-10:00 IST)
MOCK_MODEL_NAME = "v1"

# "tomorrow" 09:30 IST as the mock decision timestamp (market is open)
MOCK_NOW_TS = calendar.timegm((2026, 5, 19, 4, 0, 0, 0, 0, 0))   # 09:30 IST

ENGINE_CONFIG = EngineConfig(
    max_concurrent_positions=10,
    top_k_long=4,
    top_k_short=4,
)

# ── helpers ────────────────────────────────────────────────────────────────
SECTION = "=" * 60

def section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def inr(v) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}₹{v:,.2f}"


def pct(v) -> str:
    if v is None:
        return "—"
    return f"{v*100:+.3f}%"


# ── synthetic bar generation ───────────────────────────────────────────────
def last_close_from_db(conn, instrument_key: str) -> float:
    """Get the most recent close price from bars_1m for this symbol."""
    row = conn.execute(
        "SELECT close FROM bars_1m WHERE instrument_key = ? ORDER BY minute_ts DESC LIMIT 1",
        (instrument_key,),
    ).fetchone()
    return float(row[0]) if row else 1000.0


def generate_live_bars(instrument_key: str, start_price: float, n_bars: int, session_start_ts: int):
    """Synthetic 1-min bars starting at session_start_ts for today's mock session."""
    rng = np.random.default_rng(abs(hash(instrument_key)) % 2**31)
    bars = []
    price = start_price
    for i in range(n_bars):
        ts = session_start_ts + i * 60
        ret = rng.normal(0.0002, 0.0015)   # slight upward drift, realistic vol
        open_ = price
        close = price * (1 + ret)
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.0004)))
        low  = min(open_, close) * (1 - abs(rng.normal(0, 0.0004)))
        vol  = int(rng.integers(50_000, 800_000))
        bars.append((instrument_key, int(ts), round(open_, 2), round(high, 2), round(low, 2), round(close, 2), vol))
        price = close
    return bars, round(price, 2)


# ── main mock logic ────────────────────────────────────────────────────────
async def main() -> None:
    t0 = time.time()

    section("SETUP")

    # Load universe and pick N symbols
    uni = json.loads((ROOT / "data" / "universe.json").read_text())
    symbols = [u["instrument_key"] for u in uni[:MOCK_SYMBOLS_N]]
    sym_names = {u["instrument_key"]: u["trading_symbol"] for u in uni[:MOCK_SYMBOLS_N]}
    print(f"Mock symbols ({len(symbols)}): {', '.join(sym_names.values())}")

    # Tomorrow 9:15 IST in epoch seconds
    session_start = MOCK_NOW_TS - 15 * 60   # 09:15 IST

    # Collect last close prices and generate synthetic live bars
    import sqlite3
    conn_ro = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    last_closes = {sym: last_close_from_db(conn_ro, sym) for sym in symbols}
    conn_ro.close()

    print("\nLast known close prices (from bars_1m):")
    for sym in symbols:
        print(f"  {sym_names[sym]:<20} ₹{last_closes[sym]:>10,.2f}")

    # Generate synthetic live bars
    live_bars_by_sym: dict[str, tuple[list, float]] = {}
    for sym in symbols:
        bars, last_price = generate_live_bars(sym, last_closes[sym], MOCK_BARS_PER_SYMBOL, session_start)
        live_bars_by_sym[sym] = (bars, last_price)

    # ── Inject live bars into bars_live ──────────────────────────────────
    section("INJECT LIVE BARS → bars_live")
    injected_keys_ts: list[tuple] = []

    async with aiosqlite.connect(DB_PATH) as db:
        for sym, (bars, _) in live_bars_by_sym.items():
            await db.executemany(
                "INSERT OR REPLACE INTO bars_live (instrument_key, minute_ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
                bars,
            )
            injected_keys_ts.extend((sym, b[1]) for b in bars)
        await db.commit()

    total_bars = sum(len(b[0]) for b in live_bars_by_sym.values())
    print(f"Injected {total_bars} synthetic bars ({MOCK_BARS_PER_SYMBOL} bars × {len(symbols)} symbols)")
    print(f"Session window: 09:15 IST → 09:15+{MOCK_BARS_PER_SYMBOL}min IST (tomorrow 2026-05-19)")

    # ── Inject fake LTPs into streamer cache ──────────────────────────────
    section("INJECT LIVE QUOTES → streamer_manager")
    for sym, (_, last_price) in live_bars_by_sym.items():
        streamer_manager.last_quote_by_symbol[sym] = last_price
        print(f"  {sym_names[sym]:<20}  LTP = ₹{last_price:,.2f}")

    # ── Inject mock news headlines ────────────────────────────────────────
    section("INJECT NEWS HEADLINES")
    mock_headlines = [
        (MOCK_NOW_TS - 1800, "ET Markets", "Nifty opens flat; RBI policy in focus", "https://mock/1", None, 0.15, 0.85, 120, "claude-haiku-mock"),
        (MOCK_NOW_TS - 3600, "Livemint", "ADANI stocks rally on infrastructure push", "https://mock/2", "NSE_EQ|INE423A01024", 0.62, 0.78, 90, "claude-haiku-mock"),
        (MOCK_NOW_TS - 5400, "ET Markets", "FII outflows continue; caution advised", "https://mock/3", None, -0.41, 0.71, 180, "claude-haiku-mock"),
    ]
    injected_news_ids = []
    async with aiosqlite.connect(DB_PATH) as db:
        for pub_at, source, headline, url, ikey, score, conf, decay, model in mock_headlines:
            cur = await db.execute(
                """INSERT OR IGNORE INTO news
                   (published_at, fetched_at, source, headline, url, instrument_key,
                    sentiment_score, sentiment_confidence, sentiment_decay_minutes, sentiment_model, sentiment_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (pub_at, MOCK_NOW_TS, source, headline, url, ikey, score, conf, decay, model, MOCK_NOW_TS),
            )
            if cur.lastrowid:
                injected_news_ids.append(cur.lastrowid)
            await db.commit()
        for row in mock_headlines:
            print(f"  [{row[1]}] {row[2][:55]:<55}  score={row[6]:+.2f}")

    # ── Build feature frame ───────────────────────────────────────────────
    section("FEATURE CACHE  (build_live_feature_frame)")
    t_feat = time.time()
    features = await asyncio.get_event_loop().run_in_executor(
        None, build_live_feature_frame, symbols
    )
    elapsed_feat = time.time() - t_feat
    print(f"Computed in {elapsed_feat:.1f}s  →  {len(features)} symbols with valid features")

    if features.empty:
        print("\n⚠  Feature frame is empty — bars_1m may not have enough history for these symbols.")
        print("   Check that the selected symbols appear in data/bars_1m and have 20+ trading days.")
        await _cleanup(injected_keys_ts, injected_news_ids)
        return

    print("\nFeature sample (first 3 rows):")
    cols = ["instrument_key", "close", "ret_5m", "ret_15m", "rsi_14", "atr_14", "vwap_dev", "vol_z_20d", "gap_pct"]
    print(f"{'Symbol':<25} {'close':>9} {'ret5m':>8} {'ret15m':>8} {'rsi14':>7} {'atr14':>8} {'vwap_dev':>9}")
    for _, row in features.head(3).iterrows():
        name = sym_names.get(row["instrument_key"], row["instrument_key"])
        print(f"  {name:<23} {row['close']:>9.2f} {row['ret_5m']*100:>7.3f}% {row['ret_15m']*100:>7.3f}% {row['rsi_14']:>7.1f} {row['atr_14']:>8.3f} {row['vwap_dev']*100:>8.3f}%")

    # ── Load model + score all symbols ────────────────────────────────────
    section("MODEL INFERENCE  (v1, horizon=15m)")
    model_path, metrics_path = paths_for_name(MOCK_MODEL_NAME)
    artifact = load_model(model_path=model_path, metrics_path=metrics_path)
    print(f"Model: {MOCK_MODEL_NAME}  |  horizon: {artifact.label_horizon_minutes}m  |  features: {len(artifact.feature_columns)}")

    preds = model_score(artifact, features)
    features = features.copy()
    features["pred"] = preds

    features_sorted = features.sort_values("pred", ascending=False)
    print("\nTop 5 long candidates (highest predicted return):")
    for _, row in features_sorted.head(5).iterrows():
        name = sym_names.get(row["instrument_key"], row["instrument_key"])
        print(f"  {name:<20}  pred={pct(row['pred'])}  close=₹{row['close']:.2f}")

    print("\nTop 5 short candidates (lowest predicted return):")
    for _, row in features_sorted.tail(5).iterrows():
        name = sym_names.get(row["instrument_key"], row["instrument_key"])
        print(f"  {name:<20}  pred={pct(row['pred'])}  close=₹{row['close']:.2f}")

    # ── Sentiment scores ──────────────────────────────────────────────────
    section("SENTIMENT CACHE")
    sentiment = live_sentiment_by_symbol(symbols, now_ts=MOCK_NOW_TS)
    if sentiment:
        for sym, score_val in sorted(sentiment.items(), key=lambda x: -abs(x[1])):
            name = sym_names.get(sym, sym)
            print(f"  {name:<20}  sentiment={score_val:+.3f}")
    else:
        print("  (no sentiment scores — all symbols neutral / no recent news)")

    # ── Decision engine ───────────────────────────────────────────────────
    section("DECISION ENGINE  (decide())")
    result = decide(
        features_at_minute=features.drop(columns=["pred"]),
        model=artifact,
        open_positions=[],
        config=ENGINE_CONFIG,
        now_ts=MOCK_NOW_TS,
        closed_positions=[],
        sentiment_scores=sentiment,
    )

    print(f"Intents generated: {len(result.intents)}")
    print(f"Skip reasons:      {dict(result.skipped_reasons) or 'none'}")

    if not result.intents:
        print("\n  No entries generated this cycle. Possible reasons:")
        print("  • All symbol predictions below min_predicted_edge (0.15%)")
        print("  • Sentiment veto fired")
        print("  • This is expected — the model's edge on 10 symbols is thin.")
    else:
        print()
        for intent in result.intents:
            name = sym_names.get(intent.instrument_key, intent.instrument_key)
            print(f"  {intent.side.upper():<6}  {name:<20}  qty={intent.qty}  pred={pct(intent.predicted_return)}")

    # ── Fill intents (paper executor) ─────────────────────────────────────
    section("PAPER EXECUTOR  (fill_intent)")
    executor = PaperExecutor(config=ENGINE_CONFIG, slippage_bps=5.0, model_name=MOCK_MODEL_NAME)
    injected_positions: list[tuple[str, int]] = []   # (instrument_key, entry_ts) for cleanup

    filled: list[Position] = []
    for intent in result.intents:
        if intent.reason in ("exit_eod", "exit_kill_switch"):
            continue
        quote = streamer_manager.last_quote_by_symbol.get(intent.instrument_key)
        if quote is None:
            print(f"  ⚠  No quote for {intent.instrument_key}, skipping fill")
            continue
        pos = await executor.fill_intent(
            intent,
            fill_ts=MOCK_NOW_TS + 60,    # filled on next bar (T+1 convention)
            last_quote_price=quote,
            entry_sentiment_score=sentiment.get(intent.instrument_key),
        )
        filled.append(pos)
        injected_positions.append((pos.instrument_key, pos.entry_ts))

    if not filled:
        print("  No fills (no intents or no live quotes for intent symbols).")
    else:
        print(f"  Filled {len(filled)} position(s):\n")
        for pos in filled:
            name = sym_names.get(pos.instrument_key, pos.instrument_key)
            print(f"  {'LONG' if pos.side=='long' else 'SHORT':<6}  {name:<20}  qty={pos.qty}")
            print(f"           entry=₹{pos.entry_price:.2f}  SL=₹{pos.stop_loss_price:.2f}  TP=₹{pos.target_price:.2f}")
            print(f"           risk/share=₹{abs(pos.entry_price-pos.stop_loss_price):.2f}  "
                  f"reward/share=₹{abs(pos.target_price-pos.entry_price):.2f}")
            if pos.entry_sentiment_score is not None:
                print(f"           sentiment={pos.entry_sentiment_score:+.3f}")

    # ── Verify DB state ───────────────────────────────────────────────────
    section("DB VERIFICATION")
    open_pos = await list_open_positions()
    pnl = await todays_realised_pnl(MOCK_NOW_TS)
    print(f"Open positions in DB:   {len(open_pos)}")
    print(f"Realised P&L today:     {inr(pnl)}")

    # ── Simulate SL/TP hit on first position ──────────────────────────────
    if filled:
        section("SIMULATE EXIT  (stop-loss hit on first position)")
        pos = filled[0]
        name = sym_names.get(pos.instrument_key, pos.instrument_key)
        sl_price = pos.stop_loss_price
        print(f"  Simulating SL hit for {name} at ₹{sl_price:.2f}")
        await executor.close_position(pos, exit_ts=MOCK_NOW_TS + 300, exit_price=sl_price, reason="stop_loss")
        open_pos_after = await list_open_positions()
        pnl_after = await todays_realised_pnl(MOCK_NOW_TS)
        print(f"  Realised P&L on this trade: {inr(pos.realised_pnl_inr)}")
        print(f"  Open positions remaining:   {len(open_pos_after)}")
        print(f"  Session realised P&L:       {inr(pnl_after)}")

    # ── Pipeline summary ──────────────────────────────────────────────────
    section("PIPELINE SUMMARY")
    elapsed = time.time() - t0
    print(f"  Symbols tested:       {len(symbols)}")
    print(f"  Feature frame rows:   {len(features)}")
    print(f"  Feature compute time: {elapsed_feat:.1f}s")
    print(f"  Model scores:         min={preds.min():+.5f}  max={preds.max():+.5f}  mean={preds.mean():+.5f}")
    print(f"  Intents generated:    {len(result.intents)}")
    print(f"  Fills executed:       {len(filled)}")
    print(f"  Total wall time:      {elapsed:.1f}s")

    status_lines = []
    if not features.empty:      status_lines.append("✓ feature cache")
    if preds is not None:       status_lines.append("✓ model inference")
    if result is not None:      status_lines.append("✓ decision engine")
    if executor is not None:    status_lines.append("✓ paper executor")
    if open_pos is not None:    status_lines.append("✓ DB persistence")
    print("\n  " + "  ".join(status_lines))

    # ── Cleanup injected data ─────────────────────────────────────────────
    section("CLEANUP  (removing injected mock data)")
    async with aiosqlite.connect(DB_PATH) as db:
        # Remove injected live bars
        if injected_keys_ts:
            for sym, ts in injected_keys_ts:
                await db.execute(
                    "DELETE FROM bars_live WHERE instrument_key = ? AND minute_ts = ?",
                    (sym, ts),
                )
            print(f"  Deleted {len(injected_keys_ts)} rows from bars_live")

        # Remove injected news
        if injected_news_ids:
            await db.execute(
                f"DELETE FROM news WHERE id IN ({','.join('?' * len(injected_news_ids))})",
                injected_news_ids,
            )
            print(f"  Deleted {len(injected_news_ids)} rows from news")

        # Remove injected positions (paper_positions)
        if injected_positions:
            for ikey, entry_ts in injected_positions:
                await db.execute(
                    "DELETE FROM paper_positions WHERE instrument_key = ? AND entry_ts = ?",
                    (ikey, entry_ts),
                )
            print(f"  Deleted {len(injected_positions)} rows from paper_positions")

        await db.commit()

    # Clear injected LTPs from streamer cache
    for sym in symbols:
        streamer_manager.last_quote_by_symbol.pop(sym, None)
    print(f"  Cleared {len(symbols)} LTPs from streamer cache")
    print("\n  DB is clean. Mock run complete.\n")


async def _cleanup(injected_keys_ts, injected_news_ids):
    async with aiosqlite.connect(DB_PATH) as db:
        for sym, ts in injected_keys_ts:
            await db.execute("DELETE FROM bars_live WHERE instrument_key = ? AND minute_ts = ?", (sym, ts))
        for nid in injected_news_ids:
            await db.execute("DELETE FROM news WHERE id = ?", (nid,))
        await db.commit()


if __name__ == "__main__":
    asyncio.run(main())
