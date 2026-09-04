# src/metrics.py

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.pnl import compute_pnl_increments


def compute_sharpe_ratio(
    pnl_increments: pd.Series,
    annualization_factor: Optional[float] = None,
) -> float:
    """
    Mean PnL increment divided by its sample standard deviation.

    With annualization_factor=None this is a per-observation signal-to-noise
    ratio, not an annualized Sharpe ratio.

    Annualization is optional because a few hours of approximately 1 Hz
    market-making PnL should not be presented as a reliable annual Sharpe.
    """
    increments = pd.Series(pnl_increments, dtype=float).dropna()

    if increments.empty:
        return 0.0

    mean_pnl = float(increments.mean())
    std_pnl = float(increments.std(ddof=1))

    if not np.isfinite(std_pnl) or std_pnl <= 0:
        return 0.0

    sharpe = mean_pnl / std_pnl

    if annualization_factor is not None:
        if (
            not np.isfinite(annualization_factor)
            or annualization_factor <= 0
        ):
            raise ValueError(
                "annualization_factor must be strictly positive."
            )
        sharpe *= np.sqrt(annualization_factor)

    return float(sharpe)


def compute_max_drawdown(results: pd.DataFrame) -> float:
    """
    Maximum absolute drawdown in USDT, including the initial zero wealth.

        DD_t = V_t - max(0, max_{u <= t} V_u).

    The returned number is non-positive.
    """
    if results.empty:
        return 0.0

    if "pnl" not in results.columns:
        raise ValueError("results is missing the 'pnl' column.")

    pnl = results["pnl"].astype(float)

    # Initial portfolio wealth is zero. Without this floor, an immediately
    # negative first PnL observation would incorrectly have zero drawdown.
    running_max = pnl.cummax().clip(lower=0.0)
    drawdown = pnl - running_max

    return float(drawdown.min())


def _time_intervals(results: pd.DataFrame) -> pd.Series:
    """
    Duration for which each non-terminal state is carried until the next
    observed state.
    """
    if "timestamp" not in results.columns:
        raise ValueError("results is missing the 'timestamp' column.")

    timestamps = pd.to_datetime(
        results["timestamp"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError("results contains invalid timestamps.")

    dt = (
        timestamps.shift(-1) - timestamps
    ).dt.total_seconds()

    # The final terminal state has no subsequent holding interval.
    return dt


def compute_inventory_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Inventory statistics.

    Both row-weighted and time-weighted averages are reported. The latter
    is preferable when the nominal 1 Hz snapshots are not perfectly regular.
    """
    if results.empty:
        return {
            "avg_inventory": 0.0,
            "max_inventory": 0.0,
            "min_inventory": 0.0,
            "avg_abs_inventory": 0.0,
            "max_abs_inventory": 0.0,
            "time_weighted_avg_inventory": 0.0,
            "time_weighted_avg_abs_inventory": 0.0,
        }

    if "inventory" not in results.columns:
        raise ValueError("results is missing the 'inventory' column.")

    inventory = results["inventory"].astype(float)

    metrics = {
        "avg_inventory": float(inventory.mean()),
        "max_inventory": float(inventory.max()),
        "min_inventory": float(inventory.min()),
        "avg_abs_inventory": float(inventory.abs().mean()),
        "max_abs_inventory": float(inventory.abs().max()),
    }

    dt = _time_intervals(results)
    valid = dt.notna() & (dt > 0)

    if valid.any():
        weights = dt.loc[valid].astype(float)
        held_inventory = inventory.loc[valid]

        metrics["time_weighted_avg_inventory"] = float(
            np.average(held_inventory, weights=weights)
        )
        metrics["time_weighted_avg_abs_inventory"] = float(
            np.average(held_inventory.abs(), weights=weights)
        )
    else:
        metrics["time_weighted_avg_inventory"] = 0.0
        metrics["time_weighted_avg_abs_inventory"] = 0.0

    return metrics


def compute_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """
    Separate ordinary strategy fills from the forced terminal liquidation.
    """
    if trades.empty:
        return {
            "number_of_trades": 0,
            "number_of_strategy_fills": 0,
            "number_of_terminal_trades": 0,
            "number_of_buys": 0,
            "number_of_sells": 0,
            "buy_ratio": 0.0,
            "sell_ratio": 0.0,
            "total_volume": 0.0,
            "strategy_volume": 0.0,
            "total_traded_notional": 0.0,
            "strategy_traded_notional": 0.0,
            "avg_trade_size": 0.0,
            "avg_fill_probability": 0.0,
            "avg_quote_distance_ticks": 0.0,
        }

    required = ["side", "quantity", "price"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        raise ValueError(f"trades is missing columns: {missing}")

    if "is_terminal_unwind" in trades.columns:
        terminal_mask = trades["is_terminal_unwind"].astype(bool)
    else:
        terminal_mask = pd.Series(
            False,
            index=trades.index,
            dtype=bool,
        )

    strategy_trades = trades.loc[~terminal_mask].copy()

    number_of_trades = int(len(trades))
    number_of_strategy_fills = int(len(strategy_trades))
    number_of_terminal_trades = int(terminal_mask.sum())

    number_of_buys = int((strategy_trades["side"] == "buy").sum())
    number_of_sells = int((strategy_trades["side"] == "sell").sum())

    if number_of_strategy_fills > 0:
        buy_ratio = number_of_buys / number_of_strategy_fills
        sell_ratio = number_of_sells / number_of_strategy_fills
        strategy_volume = float(
            strategy_trades["quantity"].astype(float).sum()
        )
        avg_trade_size = float(
            strategy_trades["quantity"].astype(float).mean()
        )

        if "p_fill" in strategy_trades.columns:
            avg_fill_probability = float(
                strategy_trades["p_fill"].astype(float).mean()
            )
        else:
            avg_fill_probability = np.nan

        if "distance_ticks" in strategy_trades.columns:
            distances = pd.to_numeric(
                strategy_trades["distance_ticks"],
                errors="coerce",
            ).dropna()
            avg_quote_distance_ticks = (
                float(distances.mean())
                if not distances.empty
                else np.nan
            )
        else:
            avg_quote_distance_ticks = np.nan
    else:
        buy_ratio = 0.0
        sell_ratio = 0.0
        strategy_volume = 0.0
        avg_trade_size = 0.0
        avg_fill_probability = 0.0
        avg_quote_distance_ticks = 0.0

    total_volume = float(
        trades["quantity"].astype(float).sum()
    )
    total_traded_notional = float(
        (
            trades["quantity"].astype(float)
            * trades["price"].astype(float)
        ).sum()
    )
    strategy_traded_notional = float(
        (
            strategy_trades["quantity"].astype(float)
            * strategy_trades["price"].astype(float)
        ).sum()
    )

    return {
        "number_of_trades": number_of_trades,
        "number_of_strategy_fills": number_of_strategy_fills,
        "number_of_terminal_trades": number_of_terminal_trades,
        "number_of_buys": number_of_buys,
        "number_of_sells": number_of_sells,
        "buy_ratio": float(buy_ratio),
        "sell_ratio": float(sell_ratio),
        "total_volume": total_volume,
        "strategy_volume": strategy_volume,
        "total_traded_notional": total_traded_notional,
        "strategy_traded_notional": strategy_traded_notional,
        "avg_trade_size": avg_trade_size,
        "avg_fill_probability": avg_fill_probability,
        "avg_quote_distance_ticks": avg_quote_distance_ticks,
    }


def compute_risk_filter_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze quoting permissions, excluding the terminal liquidation state.
    """
    if results.empty:
        return {}

    required = ["risk_reason", "quote_bid", "quote_ask"]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError(f"results is missing columns: {missing}")

    quoting_results = results[
        results["risk_reason"] != "terminal_unwind"
    ].copy()

    if quoting_results.empty:
        return {
            "total_quoting_steps": 0,
            "quote_bid_ratio": 0.0,
            "quote_ask_ratio": 0.0,
            "quote_both_ratio": 0.0,
            "quote_neither_ratio": 0.0,
        }

    total_steps = len(quoting_results)

    reason_counts = (
        quoting_results["risk_reason"]
        .value_counts()
        .to_dict()
    )

    quote_bid = quoting_results["quote_bid"].astype(bool)
    quote_ask = quoting_results["quote_ask"].astype(bool)

    metrics = {
        "total_quoting_steps": int(total_steps),
        # Backward-compatible key.
        "total_steps": int(total_steps),
        "quote_bid_ratio": float(quote_bid.mean()),
        "quote_ask_ratio": float(quote_ask.mean()),
        "quote_both_ratio": float((quote_bid & quote_ask).mean()),
        "quote_neither_ratio": float((~quote_bid & ~quote_ask).mean()),
    }

    for reason, count in reason_counts.items():
        metrics[f"risk_reason_count_{reason}"] = int(count)
        metrics[f"risk_reason_ratio_{reason}"] = float(
            count / total_steps
        )

    return metrics


def compute_sample_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Physical-time description of the backtest sample."""
    if results.empty:
        return {
            "sample_duration_seconds": 0.0,
            "sample_duration_hours": 0.0,
            "median_state_interval_seconds": np.nan,
        }

    timestamps = pd.to_datetime(
        results["timestamp"],
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        raise ValueError("results contains invalid timestamps.")

    duration_seconds = float(
        (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    )

    dt = timestamps.diff().dt.total_seconds()
    positive_dt = dt[dt > 0]

    return {
        "sample_duration_seconds": duration_seconds,
        "sample_duration_hours": duration_seconds / 3600.0,
        "median_state_interval_seconds": (
            float(positive_dt.median())
            if not positive_dt.empty
            else np.nan
        ),
    }


def _add_economic_bps_metrics(
    summary: Dict[str, float],
    pnl_decomposition: Dict[str, float],
) -> None:
    """
    Add economically interpretable PnL-per-notional metrics.

    The denominator for strategy spread capture excludes terminal unwind
    notional; total gross PnL uses total traded notional.
    """
    total_notional = float(
        pnl_decomposition.get("total_traded_notional", 0.0)
    )
    strategy_notional = float(
        pnl_decomposition.get("strategy_traded_notional", 0.0)
    )

    if total_notional > 0:
        summary["gross_pnl_bps_per_total_notional"] = float(
            10_000.0
            * pnl_decomposition["gross_pnl_before_fees"]
            / total_notional
        )
        summary["net_pnl_bps_per_total_notional"] = float(
            10_000.0
            * pnl_decomposition["final_pnl"]
            / total_notional
        )
        summary["fee_bps_per_total_notional"] = float(
            10_000.0
            * pnl_decomposition["total_fees"]
            / total_notional
        )

    if strategy_notional > 0:
        summary["strategy_spread_capture_bps"] = float(
            10_000.0
            * pnl_decomposition["strategy_spread_capture"]
            / strategy_notional
        )


def compute_performance_summary(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    pnl_decomposition: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the complete performance summary."""
    if results.empty:
        raise ValueError("Results dataframe is empty.")

    pnl_increments = compute_pnl_increments(results)

    final_pnl = float(results["pnl"].iloc[-1])
    per_step_sharpe = compute_sharpe_ratio(
        pnl_increments,
        annualization_factor=None,
    )

    summary: Dict[str, float] = {
        "final_pnl": final_pnl,
        "mean_pnl_increment": float(pnl_increments.mean()),
        "std_pnl_increment": float(
            pnl_increments.std(ddof=1)
        ),
        "pnl_increment_mean_over_std": per_step_sharpe,
        # Backward-compatible name, but this is explicitly not annualized.
        "sharpe_non_annualized": per_step_sharpe,
        "max_drawdown": compute_max_drawdown(results),
    }

    summary.update(compute_sample_metrics(results))
    summary.update(compute_inventory_metrics(results))
    summary.update(compute_trade_metrics(trades))
    summary.update(compute_risk_filter_metrics(results))

    if pnl_decomposition is not None:
        summary.update(
            {
                f"decomp_{key}": value
                for key, value in pnl_decomposition.items()
            }
        )
        _add_economic_bps_metrics(
            summary=summary,
            pnl_decomposition=pnl_decomposition,
        )

    return summary


def print_performance_summary(summary: Dict[str, float]) -> None:
    """Pretty-print the key interview-relevant metrics."""
    print("========== Performance Summary ==========")
    print(
        f"Final PnL:                    "
        f"{summary.get('final_pnl', 0.0):.6f} USDT"
    )
    print(
        f"PnL mean/std per step:        "
        f"{summary.get('pnl_increment_mean_over_std', 0.0):.4f}"
    )
    print(
        f"Max drawdown:                 "
        f"{summary.get('max_drawdown', 0.0):.6f} USDT"
    )
    print(
        f"Sample duration:              "
        f"{summary.get('sample_duration_hours', 0.0):.3f} h"
    )

    if "gross_pnl_bps_per_total_notional" in summary:
        print(
            f"Gross PnL / notional:         "
            f"{summary['gross_pnl_bps_per_total_notional']:.4f} bps"
        )
    if "strategy_spread_capture_bps" in summary:
        print(
            f"Strategy spread capture:      "
            f"{summary['strategy_spread_capture_bps']:.4f} bps"
        )

    print()
    print("========== Executions ==========")
    print(
        f"Strategy fills:                "
        f"{summary.get('number_of_strategy_fills', 0)}"
    )
    print(
        f"Terminal trades:               "
        f"{summary.get('number_of_terminal_trades', 0)}"
    )
    print(
        f"Buys / sells:                  "
        f"{summary.get('number_of_buys', 0)} / "
        f"{summary.get('number_of_sells', 0)}"
    )
    print(
        f"Strategy volume:               "
        f"{summary.get('strategy_volume', 0.0):.6f} BTC"
    )
    print(
        f"Avg quote distance:            "
        f"{summary.get('avg_quote_distance_ticks', 0.0):.3f} ticks"
    )

    print()
    print("========== Inventory ==========")
    print(
        f"Time-weighted avg inventory:   "
        f"{summary.get('time_weighted_avg_inventory', 0.0):.6f} BTC"
    )
    print(
        f"Time-weighted avg |inventory|: "
        f"{summary.get('time_weighted_avg_abs_inventory', 0.0):.6f} BTC"
    )
    print(
        f"Max |inventory|:               "
        f"{summary.get('max_abs_inventory', 0.0):.6f} BTC"
    )

    print()
    print("========== Quoting ==========")
    print(
        f"Quote bid ratio:               "
        f"{summary.get('quote_bid_ratio', 0.0):.2%}"
    )
    print(
        f"Quote ask ratio:               "
        f"{summary.get('quote_ask_ratio', 0.0):.2%}"
    )
    print(
        f"Both sides ratio:              "
        f"{summary.get('quote_both_ratio', 0.0):.2%}"
    )
