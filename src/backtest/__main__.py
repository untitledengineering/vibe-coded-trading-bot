"""CLI: python -m src.backtest --start YYYY-MM-DD --end YYYY-MM-DD

Writes a Markdown report to data/backtests/. The report is the gating artifact
before any live paper trading.
"""

import argparse

from src.backtest.report import save_report
from src.backtest.runner import run_backtest
from src.bot.engine import EngineConfig
from src.model.infer import load_model
from src.model.train import paths_for_name
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Run a backtest of the trained model.")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (IST)")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD (IST)")
    parser.add_argument("--symbols", nargs="*", help="Restrict universe (smoke testing)")
    parser.add_argument("--model", default="v1",
                        help="Model name to load (data/model_<name>.json). Default v1.")
    parser.add_argument("--min-edge", type=float, default=None,
                        help="Override EngineConfig.min_predicted_edge")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top_k_long/top_k_short (sets both to same value)")
    parser.add_argument("--cooldown", type=int, default=None,
                        help="Override cooldown_minutes")
    parser.add_argument("--max-per-symbol", type=int, default=None,
                        help="Override max_trades_per_symbol_per_day")
    parser.add_argument("--total-capital", type=float, default=None,
                        help="Total account capital. ML config sizes positions as "
                             "total/max_concurrent × 5x leverage. Defaults match the "
                             "original ₹25k account.")
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.min_edge is not None:
        cfg_kwargs["min_predicted_edge"] = args.min_edge
    if args.top_k is not None:
        cfg_kwargs["top_k_long"] = args.top_k
        cfg_kwargs["top_k_short"] = args.top_k
    if args.cooldown is not None:
        cfg_kwargs["cooldown_minutes"] = args.cooldown
    if args.max_per_symbol is not None:
        cfg_kwargs["max_trades_per_symbol_per_day"] = args.max_per_symbol
    if args.total_capital is not None:
        # ML config uses per-position notional; convert from total account capital.
        # Default 4 positions × 5x leverage → per-position notional = total_capital × 5 / 4.
        cfg_kwargs["capital_inr"] = args.total_capital * 5.0 / 4.0
    config = EngineConfig(**cfg_kwargs)

    model_path, metrics_path = paths_for_name(args.model)
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path=model_path, metrics_path=metrics_path)
    report = run_backtest(
        start_date=args.start,
        end_date=args.end,
        model=model,
        config=config,
        symbols=args.symbols,
    )
    path = save_report(report)
    logger.info(f"Report written to {path}")
    print(f"Report: {path}")
    if report.completed_positions:
        pnls = [p.realised_pnl_inr or 0.0 for p in report.completed_positions]
        net = sum(pnls)
        print(f"Trades: {len(pnls)}  Net P&L (incl. costs): ₹{net:,.2f}")
    else:
        print("No trades. See the report's 'Filters and skips' section.")


if __name__ == "__main__":
    main()
