# scripts/run_backtest.py

import argparse
import sys
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------
# Make sure Python can import from src/
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.config import StrategyConfig
from src.data_loader import prepare_lob_data
from src.features import compute_features_v1
from src.calibration import (
    calibrate_strategy_parameters,
    update_config_from_calibration,
)
from src.strategy import AvellanedaMarketMaker
from src.backtester import Backtester
from src.pnl import compute_pnl_decomposition
from src.metrics import (
    compute_performance_summary,
    print_performance_summary,
)


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run Avellaneda-Stoikov market-making backtest."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to raw LOB data file, csv or parquet.",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol to filter.",
    )

    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.1,
        help="Minimum price tick size.",
    )

    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0002,
        help="Fee rate per trade.",
    )

    parser.add_argument(
        "--vol-window",
        type=int,
        default=100,
        help="Rolling window for volatility estimation.",
    )

    parser.add_argument(
        "--fill-horizon",
        type=int,
        default=5,
        help="Horizon used for fill probability calibration.",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.3,
        help="Fraction of data used for calibration before backtesting.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to use.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where results are saved.",
    )
    parser.add_argument(
        "--min-spread-bps",
        type=float,
        default=None,
        help="Override calibrated minimum spread in bps.",
    )
    return parser.parse_args()


def split_calibration_backtest_data(
    df: pd.DataFrame,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into calibration sample and backtest sample.

    We calibrate on the beginning of the dataset and backtest on the rest.
    This avoids using future data to calibrate the strategy.
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    split_index = int(len(df) * calibration_fraction)

    calibration_data = df.iloc[:split_index].copy()
    backtest_data = df.iloc[split_index:].copy()

    return calibration_data, backtest_data


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("Loading and cleaning data")
    print("======================================")

    df = prepare_lob_data(path=args.data_path)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print()
    print("======================================")
    print("Computing features")
    print("======================================")

    df_features = compute_features_v1(
        df=df,
        vol_window=args.vol_window,
    )

    print("Feature columns added:")
    print(
        [
            "spread_bps",
            "imbalance",
            "total_depth",
            "log_mid_return",
            "rolling_volatility",
        ]
    )

    print()
    print("======================================")
    print("Creating initial config")
    print("======================================")

    config = StrategyConfig(
        symbol=args.symbol,
        tick_size=args.tick_size,
        fee_rate=args.fee_rate,
        fill_horizon=args.fill_horizon,
    )

    print(config)

    print()
    print("======================================")
    print("Splitting calibration/backtest data")
    print("======================================")

    calibration_data, backtest_data = split_calibration_backtest_data(
        df=df_features,
        calibration_fraction=args.calibration_fraction,
    )

    print(f"Calibration rows: {len(calibration_data):,}")
    print(f"Backtest rows:    {len(backtest_data):,}")

    print()
    print("======================================")
    print("Calibrating strategy parameters")
    print("======================================")

    calibration = calibrate_strategy_parameters(
        df=calibration_data,
        config=config,
        tick_distances=range(1, 11),
        horizons=[1, 3, 5, 10],
    )
    if args.min_spread_bps is not None:
        calibration["min_spread_bps"] = args.min_spread_bps
    fill_table = calibration["fill_table"]

    fill_table_path = output_dir / "fill_table.csv"
    fill_table.to_csv(fill_table_path, index=False)

    print("Calibration results:")
    print(f"p0:              {calibration['p0']:.6f}")
    print(f"k_fill:          {calibration['k_fill']:.6f}")
    print(f"kappa:           {calibration['kappa']:.6f}")
    print(f"max_volatility:  {calibration['max_volatility']:.8f}")
    print(f"min_depth:       {calibration['min_depth']:.6f}")
    print(f"min_spread_bps:  {calibration['min_spread_bps']:.6f}")
    print(f"fit_ok:          {calibration['fit_ok']}")
    print(f"Saved fill table to: {fill_table_path}")

    config = update_config_from_calibration(
        config=config,
        calibration=calibration,
    )

    print()
    print("======================================")
    print("Updated config")
    print("======================================")
    print(config)

    print()
    print("======================================")
    print("Running backtest")
    print("======================================")

    strategy = AvellanedaMarketMaker(config=config)

    backtester = Backtester(
        data=backtest_data,
        strategy=strategy,
    )

    results = backtester.run(show_progress=True)
    trades = backtester.get_trades()

    print()
    print(f"Backtest finished.")
    print(f"Number of states: {len(results):,}")
    print(f"Number of trades: {len(trades):,}")

    print()
    print("======================================")
    print("Computing PnL decomposition")
    print("======================================")

    pnl_decomp = compute_pnl_decomposition(
        results=results,
        trades=trades,
        market_data=backtest_data,
        adverse_selection_horizon=args.fill_horizon,
    )

    print(pnl_decomp)

    print()
    print("======================================")
    print("Computing performance summary")
    print("======================================")

    summary = compute_performance_summary(
        results=results,
        trades=trades,
        pnl_decomposition=pnl_decomp,
    )

    print_performance_summary(summary)

    print()
    print("======================================")
    print("Saving outputs")
    print("======================================")

    results_path = output_dir / "backtest_results.csv"
    trades_path = output_dir / "trades.csv"
    summary_path = output_dir / "performance_summary.csv"
    config_path = output_dir / "calibrated_config.txt"

    results.to_csv(results_path, index=False)
    trades.to_csv(trades_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    with open(config_path, "w") as f:
        f.write(str(config))

    print(f"Saved results to: {results_path}")
    print(f"Saved trades to:  {trades_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved config to:  {config_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
