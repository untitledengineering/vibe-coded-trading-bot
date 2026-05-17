"""Label generation for the intraday return prediction model.

The model predicts the realised return from time t to t + HORIZON minutes, where
HORIZON defaults to 15. Two things make this non-trivial:

1. Session boundaries — we MUST NOT compute a label that crosses 15:30 IST into
   the next trading day. Predicting "close to next open" would teach the model
   overnight gaps, which are unrelated to intraday momentum.
2. Missing bars — the historical backfill has gaps (chunks that failed). We
   require the t+HORIZON bar to exist in the SAME session, not just any later
   bar HORIZON minutes ahead, because we look up t+HORIZON by minute_ts, not
   by row offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import IST_OFFSET_SECONDS, SECONDS_PER_DAY

DEFAULT_HORIZON_MINUTES = 15


def _session_date(minute_ts: pd.Series) -> pd.Series:
    """Same definition as in features.technical — kept in sync via the shared constants."""
    return ((minute_ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY).astype("int64")


def forward_return(
    bars: pd.DataFrame,
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
) -> pd.Series:
    """Return a Series aligned with `bars` whose value at row i is the close-to-close
    return from bar i to the bar HORIZON_MINUTES later WITHIN THE SAME SESSION.

    NaN where:
      - The t+HORIZON bar is missing (gap in history)
      - The t+HORIZON bar falls in a different session

    The caller is expected to dropna() before training. We do not drop here so the
    output stays aligned with the input row-for-row.
    """
    if not bars["minute_ts"].is_monotonic_increasing:
        raise ValueError("forward_return requires bars sorted ascending by minute_ts")

    horizon_seconds = horizon_minutes * 60
    target_ts = bars["minute_ts"] + horizon_seconds

    # Map minute_ts -> close, so we can look up the close of the t+HORIZON bar.
    close_by_ts = pd.Series(bars["close"].values, index=bars["minute_ts"].values)
    future_close = target_ts.map(close_by_ts)

    # Session boundary check: the t+HORIZON bar must be in the same session.
    session = _session_date(bars["minute_ts"])
    target_session = _session_date(target_ts)
    cross_session = session.values != target_session.values

    ret = (future_close.values - bars["close"].values) / bars["close"].values
    ret = pd.Series(ret, index=bars.index, name="fwd_ret_15m")
    ret[cross_session] = np.nan
    return ret


def attach_labels(bars: pd.DataFrame, horizon_minutes: int = DEFAULT_HORIZON_MINUTES) -> pd.DataFrame:
    """Convenience wrapper: returns the input DataFrame with two label columns.

        fwd_ret_<H>m       — float, what the regressor learns
        fwd_up_<H>m        — int {0,1}, what a classifier would learn (NaN propagates)
    """
    out = bars.copy()
    fwd = forward_return(out, horizon_minutes=horizon_minutes)
    out[f"fwd_ret_{horizon_minutes}m"] = fwd
    # Binary up label, with NaN preserved where fwd is NaN. Using float dtype so
    # NaN survives the column (int columns can't hold NaN).
    binary = pd.Series(np.where(fwd > 0, 1.0, 0.0), index=out.index)
    binary[fwd.isna()] = np.nan
    out[f"fwd_up_{horizon_minutes}m"] = binary
    return out
