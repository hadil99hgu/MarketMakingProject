# src/metrics.py

import numpy as np
import pandas as pd


def compute_pnl_increments(results: pd.DataFrame) -> pd.Series:
    """
    Compute PnL changes between timestamps.
    """

    if results.empty:
        return pd.Series(dtype=float)

    return results["pnl"].diff().fillna(0.0)


def compute_sharpe_ratio(
    pnl_increments: pd.Series,
    annualization_factor: float | None = None,
) -> float:
    """
    Compute Sharpe ratio from PnL increments.

    For high-frequency data, we first keep it non-annualized.
    Annualization can be misleading if the data frequency is irregular.
    """

    if pnl_increments.empty:
        return 0.0

    mean_pnl = pnl_increments.mean()
    std_pnl = pnl_increments.std()

    if std_pnl == 0 or np.isnan(std_pnl):
        return 0.0

    sharpe = mean_pnl / std_pnl

    if annualization_factor is not None:
        sharpe *= np.sqrt(annualization_factor)

    return float(sharpe)


def compute_max_drawdown(results: pd.DataFrame) -> float:
    """
    Compute maximum drawdown from the PnL curve.
    """

    if results.empty:
        return 0.0

    pnl = results["pnl"]
    running_max = pnl.cummax()
    drawdown = pnl - running_max

    return float(drawdown.min())


def compute_inventory_metrics(results: pd.DataFrame) -> dict:
    """
    Compute inventory statistics.
    """

    if results.empty:
        return {
            "avg_inventory": 0.0,
            "max_inventory": 0.0,
            "min_inventory": 0.0,
            "avg_abs_inventory": 0.0,
            "max_abs_inventory": 0.0,
        }

    inventory = results["inventory"]

    return {
        "avg_inventory": float(inventory.mean()),
        "max_inventory": float(inventory.max()),
        "min_inventory": float(inventory.min()),
        "avg_abs_inventory": float(inventory.abs().mean()),
        "max_abs_inventory": float(inventory.abs().max()),
    }


def compute_trade_metrics(trades: pd.DataFrame) -> dict:
    """
    Compute trade and fill statistics.
    """

    if trades.empty:
        return {
            "number_of_trades": 0,
            "number_of_buys": 0,
            "number_of_sells": 0,
            "buy_ratio": 0.0,
            "sell_ratio": 0.0,
            "total_volume": 0.0,
            "avg_trade_size": 0.0,
            "avg_fill_probability": 0.0,
        }

    number_of_trades = len(trades)
    number_of_buys = (trades["side"] == "buy").sum()
    number_of_sells = (trades["side"] == "sell").sum()

    return {
        "number_of_trades": int(number_of_trades),
        "number_of_buys": int(number_of_buys),
        "number_of_sells": int(number_of_sells),
        "buy_ratio": float(number_of_buys / number_of_trades),
        "sell_ratio": float(number_of_sells / number_of_trades),
        "total_volume": float(trades["quantity"].sum()),
        "avg_trade_size": float(trades["quantity"].mean()),
        "avg_fill_probability": float(trades["p_fill"].mean()),
    }


def compute_risk_filter_metrics(results: pd.DataFrame) -> dict:
    """
    Analyze how often the strategy was allowed or blocked from quoting.
    """

    if results.empty:
        return {}

    total_steps = len(results)

    reason_counts = results["risk_reason"].value_counts().to_dict()
    reason_ratios = {
        f"risk_reason_ratio_{reason}": count / total_steps
        for reason, count in reason_counts.items()
    }

    quote_bid_ratio = results["quote_bid"].mean()
    quote_ask_ratio = results["quote_ask"].mean()

    metrics = {
        "total_steps": int(total_steps),
        "quote_bid_ratio": float(quote_bid_ratio),
        "quote_ask_ratio": float(quote_ask_ratio),
    }

    metrics.update({
        f"risk_reason_count_{reason}": int(count)
        for reason, count in reason_counts.items()
    })

    metrics.update(reason_ratios)

    return metrics


def compute_performance_summary(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    pnl_decomposition: dict | None = None,
) -> dict:
    """
    Compute full performance summary.
    """

    if results.empty:
        raise ValueError("Results dataframe is empty.")

    pnl_increments = compute_pnl_increments(results)

    final_pnl = float(results["pnl"].iloc[-1])
    sharpe = compute_sharpe_ratio(pnl_increments)
    max_drawdown = compute_max_drawdown(results)

    inventory_metrics = compute_inventory_metrics(results)
    trade_metrics = compute_trade_metrics(trades)
    risk_metrics = compute_risk_filter_metrics(results)

    summary = {
        "final_pnl": final_pnl,
        "mean_pnl_increment": float(pnl_increments.mean()),
        "std_pnl_increment": float(pnl_increments.std()),
        "sharpe_non_annualized": sharpe,
        "max_drawdown": max_drawdown,
    }

    summary.update(inventory_metrics)
    summary.update(trade_metrics)
    summary.update(risk_metrics)

    if pnl_decomposition is not None:
        summary.update({
            f"decomp_{key}": value
            for key, value in pnl_decomposition.items()
        })

    return summary


def print_performance_summary(summary: dict) -> None:
    """
    Pretty-print the most important metrics.
    """

    print("========== Performance Summary ==========")
    print(f"Final PnL:              {summary.get('final_pnl', 0.0):.6f}")
    print(f"Sharpe non-annualized:  {summary.get('sharpe_non_annualized', 0.0):.4f}")
    print(f"Max drawdown:           {summary.get('max_drawdown', 0.0):.6f}")
    print()
    print("========== Trades ==========")
    print(f"Number of trades:       {summary.get('number_of_trades', 0)}")
    print(f"Number of buys:         {summary.get('number_of_buys', 0)}")
    print(f"Number of sells:        {summary.get('number_of_sells', 0)}")
    print(f"Total volume:           {summary.get('total_volume', 0.0):.6f}")
    print()
    print("========== Inventory ==========")
    print(f"Average inventory:      {summary.get('avg_inventory', 0.0):.6f}")
    print(f"Average abs inventory:  {summary.get('avg_abs_inventory', 0.0):.6f}")
    print(f"Max abs inventory:      {summary.get('max_abs_inventory', 0.0):.6f}")
    print()
    print("========== Quoting ==========")
    print(f"Quote bid ratio:        {summary.get('quote_bid_ratio', 0.0):.2%}")
    print(f"Quote ask ratio:        {summary.get('quote_ask_ratio', 0.0):.2%}")