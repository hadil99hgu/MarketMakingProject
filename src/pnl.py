# src/pnl.py

import numpy as np
import pandas as pd


def compute_total_pnl(results: pd.DataFrame) -> pd.Series:
    """
    Total mark-to-market PnL already computed by the strategy.

    pnl_t = cash_t + inventory_t * mid_price_t
    """

    return results["pnl"]


def compute_pnl_returns(results: pd.DataFrame) -> pd.Series:
    """
    Compute PnL increments.
    """

    pnl_returns = results["pnl"].diff().fillna(0.0)

    return pnl_returns


def compute_total_fees(trades: pd.DataFrame) -> float:
    """
    Total transaction fees paid.
    """

    if trades.empty:
        return 0.0

    return trades["fee"].sum()


def compute_gross_trading_pnl(trades: pd.DataFrame) -> float:
    """
    Gross cash PnL from executions, before terminal inventory value.

    Convention:
        signed_quantity > 0 means buy
        signed_quantity < 0 means sell

    Cash change from a trade:
        - signed_quantity * price
    """

    if trades.empty:
        return 0.0

    gross_cash = -(trades["signed_quantity"] * trades["price"]).sum()

    return gross_cash


def compute_terminal_inventory_value(results: pd.DataFrame) -> float:
    """
    Terminal inventory value marked at final mid-price.
    """

    if results.empty:
        return 0.0

    final_inventory = results["inventory"].iloc[-1]
    final_mid = results["mid_price"].iloc[-1]

    return final_inventory * final_mid


def compute_spread_capture(trades: pd.DataFrame) -> float:
    """
    Approximate spread capture relative to the mid-price at execution time.

    For a buy:
        spread capture = mid - execution_price

    For a sell:
        spread capture = execution_price - mid

    This measures how much better than mid we traded.
    """

    if trades.empty:
        return 0.0

    signed_qty = trades["signed_quantity"]
    price = trades["price"]
    mid = trades["mid_price"]

    spread_capture_per_trade = np.where(
        signed_qty > 0,
        mid - price,      # buy below mid
        price - mid,      # sell above mid
    )

    spread_capture = (
        trades["quantity"] * spread_capture_per_trade
    ).sum()

    return spread_capture


def compute_adverse_selection(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon: int = 5,
) -> float:
    """
    Estimate adverse selection after fills.

    For a buy:
        good if future_mid > execution_mid
        bad if future_mid < execution_mid

    For a sell:
        good if future_mid < execution_mid
        bad if future_mid > execution_mid

    We compute:
        signed_quantity * (future_mid - mid_at_trade)

    If this is negative on average, fills are toxic.
    """

    if trades.empty:
        return 0.0

    market_data = market_data.reset_index(drop=True).copy()
    trades = trades.copy()

    future_mid_col = f"future_mid_{horizon}"

    market_data[future_mid_col] = market_data["mid_price"].shift(-horizon)

    merged = trades.merge(
        market_data[["timestamp", future_mid_col]],
        on="timestamp",
        how="left",
    )

    merged = merged.dropna(subset=[future_mid_col])

    if merged.empty:
        return 0.0

    adverse_selection = (
        merged["signed_quantity"]
        * (merged[future_mid_col] - merged["mid_price"])
    ).sum()

    return adverse_selection


def compute_inventory_pnl(results: pd.DataFrame) -> float:
    """
    Approximate inventory mark-to-market PnL.

    inventory_pnl_t = inventory_{t-1} * (mid_t - mid_{t-1})

    This measures how much PnL came from holding inventory while the mid moved.
    """

    if results.empty:
        return 0.0

    inventory_lag = results["inventory"].shift(1).fillna(0.0)
    mid_change = results["mid_price"].diff().fillna(0.0)

    inventory_pnl = (inventory_lag * mid_change).sum()

    return inventory_pnl


def compute_pnl_decomposition(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    adverse_selection_horizon: int = 5,
) -> dict:
    """
    Compute a first PnL decomposition.
    """

    if results.empty:
        raise ValueError("Results dataframe is empty.")

    final_pnl = results["pnl"].iloc[-1]

    total_fees = compute_total_fees(trades)
    gross_trading_pnl = compute_gross_trading_pnl(trades)
    terminal_inventory_value = compute_terminal_inventory_value(results)
    spread_capture = compute_spread_capture(trades)
    inventory_pnl = compute_inventory_pnl(results)

    adverse_selection = compute_adverse_selection(
        trades=trades,
        market_data=market_data,
        horizon=adverse_selection_horizon,
    )

    decomposition = {
        "final_pnl": final_pnl,
        "gross_trading_pnl": gross_trading_pnl,
        "terminal_inventory_value": terminal_inventory_value,
        "total_fees": total_fees,
        "spread_capture": spread_capture,
        "inventory_pnl": inventory_pnl,
        "adverse_selection": adverse_selection,
        "number_of_trades": len(trades),
        "final_inventory": results["inventory"].iloc[-1],
    }

    return decomposition