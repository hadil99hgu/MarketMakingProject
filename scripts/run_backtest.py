# scripts/run_backtest.py

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester import Backtester
from src.calibration import (
    calibrate_strategy_parameters,
    future_rolling_max,
    future_rolling_min,
    update_config_from_calibration,
)
from src.config import StrategyConfig
from src.data_loader import prepare_lob_data, sampling_diagnostics
from src.features import compute_features_v1
from src.metrics import (
    compute_performance_summary,
    print_performance_summary,
)
from src.pnl import compute_pnl_decomposition
from src.strategy import AvellanedaMarketMaker


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the corrected Avellaneda-Stoikov-inspired "
            "BTCUSDT market-making backtest."
        )
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the existing raw BTCUSDT top-of-book CSV.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol to retain if the CSV has a symbol column.",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=0.01,
        help="BTCUSDT minimum price increment in USDT.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0002,
        help="Fee rate per executed trade.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Avellaneda-Stoikov CARA risk aversion in 1/USDT.",
    )
    parser.add_argument(
        "--as-horizon-seconds",
        type=float,
        default=1.0,
        help="Receding A-S inventory-risk horizon in seconds.",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=100,
        help="Rolling volatility estimation window in observations.",
    )
    parser.add_argument(
        "--fill-horizon",
        type=int,
        default=5,
        help=(
            "Forward horizon in observations used to estimate empirical "
            "touch probabilities before hazard conversion."
        ),
    )
    parser.add_argument(
        "--markout-horizon-seconds",
        type=float,
        default=5.0,
        help="Physical-time horizon for post-fill signed markouts.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.30,
        help="Initial fraction of observations used for calibration.",
    )
    parser.add_argument(
        "--calibration-distance-points",
        type=int,
        default=30,
        help="Approximate number of quote distances used in fill calibration.",
    )
    parser.add_argument(
        "--calibration-distance-quantile",
        type=float,
        default=0.99,
        help=(
            "Upper quantile of empirical forward price excursion used "
            "to set the largest calibration distance."
        ),
    )
    parser.add_argument(
        "--max-calibration-distance-ticks",
        type=int,
        default=5000,
        help="Safety cap on the largest calibrated quote distance.",
    )
    parser.add_argument(
        "--min-spread-bps",
        type=float,
        default=None,
        help=(
            "Optional observed-BBO regime filter. Default None leaves "
            "the calibrated/default value (normally 0 bps)."
        ),
    )
    parser.add_argument(
        "--enable-imbalance-filter",
        action="store_true",
        help=(
            "Enable the optional hard imbalance-side blocker. "
            "Microprice skew remains active regardless."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of existing rows to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reduced-form execution simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where backtest outputs are saved.",
    )

    return parser.parse_args()


def split_calibration_backtest_data(
    df: pd.DataFrame,
    calibration_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: calibration uses only the initial sample.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "calibration_fraction must lie strictly between 0 and 1."
        )

    if len(df) < 4:
        raise ValueError("Dataset is too short for calibration/backtesting.")

    split_index = int(len(df) * calibration_fraction)

    if split_index < 2 or len(df) - split_index < 2:
        raise ValueError(
            "Calibration and backtest samples must each contain "
            "at least two observations."
        )

    calibration_data = df.iloc[:split_index].copy()
    backtest_data = df.iloc[split_index:].copy()

    return calibration_data, backtest_data


def choose_calibration_tick_distances(
    calibration_data: pd.DataFrame,
    tick_size: float,
    horizon: int,
    n_points: int = 30,
    upper_quantile: float = 0.99,
    max_distance_ticks: int = 5000,
):
    """
    Choose a distance grid from the empirical forward mid-price excursion.

    For each calibration row, compute the largest one-sided excursion over
    the chosen fill horizon:

        max(mid_t - min_future_mid,
            max_future_mid - mid_t).

    The upper quantile of this excursion determines the largest distance
    represented in the fit. Distances are placed on a geometric grid so
    both near-touch and farther quotes are represented.

    This avoids fitting kappa only on an arbitrary 1--10 tick range and
    then extrapolating hundreds of ticks beyond the calibration support.
    """
    if tick_size <= 0:
        raise ValueError("tick_size must be strictly positive.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if not 0.0 < upper_quantile < 1.0:
        raise ValueError("upper_quantile must lie in (0, 1).")
    if max_distance_ticks < 1:
        raise ValueError("max_distance_ticks must be at least 1.")

    mid = calibration_data["mid_price"].reset_index(drop=True)

    future_min = future_rolling_min(mid, horizon)
    future_max = future_rolling_max(mid, horizon)

    valid = future_min.notna() & future_max.notna()

    if not valid.any():
        raise ValueError(
            "No complete forward windows available for distance selection."
        )

    downside = (mid.loc[valid] - future_min.loc[valid]).clip(lower=0.0)
    upside = (future_max.loc[valid] - mid.loc[valid]).clip(lower=0.0)
    excursion_ticks = np.maximum(
        downside.to_numpy(dtype=float),
        upside.to_numpy(dtype=float),
    ) / tick_size

    excursion_ticks = excursion_ticks[
        np.isfinite(excursion_ticks)
    ]

    if len(excursion_ticks) == 0:
        raise ValueError("Could not estimate empirical price excursions.")

    empirical_max = int(
        np.ceil(np.quantile(excursion_ticks, upper_quantile))
    )
    empirical_max = max(empirical_max, 10)
    empirical_max = min(empirical_max, max_distance_ticks)

    if empirical_max == 1:
        return [1]

    distances = np.unique(
        np.rint(
            np.geomspace(
                1,
                empirical_max,
                num=min(n_points, empirical_max),
            )
        ).astype(int)
    )

    distances = distances[
        (distances >= 1) & (distances <= empirical_max)
    ]

    # Guarantee the empirical upper endpoint is included.
    distances = np.unique(
        np.append(distances, empirical_max)
    )

    return distances.tolist()


def resolve_data_path(path_text: str) -> Path:
    """Resolve a relative data path against the project root."""
    path = Path(path_text).expanduser()

    if not path.is_absolute():
        project_relative = PROJECT_ROOT / path
        if project_relative.exists():
            path = project_relative

    return path.resolve()


def filter_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Filter the dataset when a symbol column is present."""
    if "symbol" not in df.columns:
        return df

    filtered = df[df["symbol"].astype(str) == symbol].copy()

    if filtered.empty:
        raise ValueError(
            f"No observations found for symbol={symbol!r}."
        )

    return filtered.reset_index(drop=True)


def main():
    args = parse_args()

    if args.markout_horizon_seconds <= 0:
        raise ValueError(
            "markout-horizon-seconds must be strictly positive."
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("Loading and cleaning existing data")
    print("======================================")

    data_path = resolve_data_path(args.data_path)
    df = prepare_lob_data(path=data_path)
    df = filter_symbol(df, args.symbol)

    if args.max_rows is not None:
        if args.max_rows < 2:
            raise ValueError("max-rows must be at least 2.")
        df = df.iloc[: args.max_rows].copy()

    diagnostics = sampling_diagnostics(df)
    median_dt = diagnostics["median_dt_seconds"]

    print(f"Data path:                {data_path}")
    print(f"Loaded rows:              {len(df):,}")
    print(f"Start:                    {diagnostics['start_time']}")
    print(f"End:                      {diagnostics['end_time']}")
    print(
        f"Duration:                 "
        f"{diagnostics['duration_seconds'] / 3600.0:.3f} h"
    )
    print(
        f"Median timestamp spacing: "
        f"{median_dt:.6f} s"
    )
    print(
        f"Median effective rate:    "
        f"{diagnostics['median_effective_hz']:.4f} Hz"
    )
    print(
        f"99% timestamp spacing:    "
        f"{diagnostics['p99_dt_seconds']:.6f} s"
    )

    print()
    print("======================================")
    print("Computing causal features")
    print("======================================")

    df_features = compute_features_v1(
        df=df,
        vol_window=args.vol_window,
    )

    print(
        "Features: spread_bps, imbalance, total_depth, "
        "log_mid_return, rolling_log_volatility, "
        "rolling_price_volatility"
    )

    print()
    print("======================================")
    print("Chronological calibration/test split")
    print("======================================")

    calibration_data, backtest_data = split_calibration_backtest_data(
        df=df_features,
        calibration_fraction=args.calibration_fraction,
    )

    if len(calibration_data) <= args.fill_horizon:
        raise ValueError(
            "Calibration sample is shorter than the fill horizon."
        )

    print(f"Calibration rows: {len(calibration_data):,}")
    print(f"Backtest rows:    {len(backtest_data):,}")

    print()
    print("======================================")
    print("Creating initial configuration")
    print("======================================")

    config = StrategyConfig(
        symbol=args.symbol,
        tick_size=args.tick_size,
        fee_rate=args.fee_rate,
        gamma=args.gamma,
        horizon_seconds=args.as_horizon_seconds,
        fill_horizon=args.fill_horizon,
        seconds_per_observation=median_dt,
        execution_step_seconds=median_dt,
        enable_imbalance_filter=args.enable_imbalance_filter,
        seed=args.seed,
    )

    print(config)

    print()
    print("======================================")
    print("Choosing empirical calibration distances")
    print("======================================")

    tick_distances = choose_calibration_tick_distances(
        calibration_data=calibration_data,
        tick_size=config.tick_size,
        horizon=config.fill_horizon,
        n_points=args.calibration_distance_points,
        upper_quantile=args.calibration_distance_quantile,
        max_distance_ticks=args.max_calibration_distance_ticks,
    )

    print(f"Number of distances: {len(tick_distances)}")
    print(
        f"Distance support:   "
        f"{min(tick_distances)}--{max(tick_distances)} ticks "
        f"({min(tick_distances) * config.tick_size:.4f}--"
        f"{max(tick_distances) * config.tick_size:.4f} USDT)"
    )
    print(f"Distances:          {tick_distances}")

    print()
    print("======================================")
    print("Calibrating execution and risk parameters")
    print("======================================")

    horizons = sorted(
        set([1, 3, 5, 10, config.fill_horizon])
    )

    calibration = calibrate_strategy_parameters(
        df=calibration_data,
        config=config,
        tick_distances=tick_distances,
        horizons=horizons,
        seconds_per_observation=median_dt,
        execution_step_seconds=median_dt,
    )

    if args.min_spread_bps is not None:
        if args.min_spread_bps < 0:
            raise ValueError(
                "min-spread-bps must be non-negative."
            )
        calibration["min_spread_bps"] = args.min_spread_bps

    config = update_config_from_calibration(
        config=config,
        calibration=calibration,
    )

    fill_table = calibration["fill_table"]
    fill_table_path = output_dir / "fill_table.csv"
    fill_table.to_csv(fill_table_path, index=False)

    print(
        f"A (per second):             "
        f"{calibration['arrival_rate_A']:.8f}"
    )
    print(
        f"kappa_ticks (per tick):     "
        f"{calibration['kappa_ticks']:.8f}"
    )
    print(
        f"kappa (per USDT):           "
        f"{calibration['kappa']:.8f}"
    )
    print(
        f"Intensity log-fit R^2:       "
        f"{calibration['intensity_fit_r_squared']:.6f}"
    )
    print(
        f"1-step p0 approximation:     "
        f"{calibration['p0']:.8f}"
    )
    print(
        f"1-step k_fill approximation: "
        f"{calibration['k_fill']:.8f}"
    )
    print(
        f"Direct h-step p0:            "
        f"{calibration['p0_horizon']:.8f}"
    )
    print(
        f"Direct h-step k_fill:        "
        f"{calibration['k_fill_horizon']:.8f}"
    )
    print(
        f"max_volatility:              "
        f"{calibration['max_volatility']:.10f}"
    )
    print(
        f"min_depth:                   "
        f"{calibration['min_depth']:.6f}"
    )
    print(
        f"min_spread_bps:              "
        f"{calibration['min_spread_bps']:.6f}"
    )
    print(f"Saved fill table to:         {fill_table_path}")

    print()
    print("======================================")
    print("Running out-of-sample backtest")
    print("======================================")

    strategy = AvellanedaMarketMaker(config=config)
    backtester = Backtester(
        data=backtest_data,
        strategy=strategy,
    )

    results = backtester.run(show_progress=True)
    trades = backtester.get_trades()

    print(f"Number of states: {len(results):,}")
    print(f"All trade records: {len(trades):,}")

    print()
    print("======================================")
    print("PnL decomposition and reconciliation")
    print("======================================")

    pnl_decomp = compute_pnl_decomposition(
        results=results,
        trades=trades,
        market_data=backtest_data,
        adverse_selection_horizon=args.markout_horizon_seconds,
    )

    print(
        f"Final PnL:                  "
        f"{pnl_decomp['final_pnl']:.8f} USDT"
    )
    print(
        f"Strategy spread capture:    "
        f"{pnl_decomp['strategy_spread_capture']:.8f} USDT"
    )
    print(
        f"Terminal liquidation cost:  "
        f"{pnl_decomp['terminal_liquidation_spread_cost']:.8f} USDT"
    )
    print(
        f"Inventory PnL:              "
        f"{pnl_decomp['inventory_pnl']:.8f} USDT"
    )
    print(
        f"Fees:                       "
        f"{pnl_decomp['total_fees']:.8f} USDT"
    )
    print(
        f"Reconciliation error:       "
        f"{pnl_decomp['reconciliation_error']:.3e}"
    )
    print(
        f"{args.markout_horizon_seconds:g}s signed markout total: "
        f"{pnl_decomp['signed_markout_total']:.8f} USDT"
    )

    print()
    print("======================================")
    print("Performance summary")
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
    config_path = output_dir / "calibrated_config.json"
    diagnostics_path = output_dir / "sampling_diagnostics.json"

    results.to_csv(results_path, index=False)
    trades.to_csv(trades_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            asdict(config),
            file,
            indent=2,
            default=str,
        )

    serializable_diagnostics = {
        key: (
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else value
        )
        for key, value in diagnostics.items()
    }
    with open(
        diagnostics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_diagnostics,
            file,
            indent=2,
        )

    print(f"Results:              {results_path}")
    print(f"Trades:               {trades_path}")
    print(f"Summary:              {summary_path}")
    print(f"Calibrated config:    {config_path}")
    print(f"Sampling diagnostics: {diagnostics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
