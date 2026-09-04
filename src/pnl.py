# src/pnl.py

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _require_columns(
    df: pd.DataFrame,
    columns,
    dataframe_name: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def compute_total_pnl(results: pd.DataFrame) -> pd.Series:
    """
    Mark-to-market PnL path already computed by the strategy:

        V_t = cash_t + inventory_t * mid_price_t.
    """
    _require_columns(results, ["pnl"], "results")
    return results["pnl"].astype(float).copy()


def compute_pnl_increments(results: pd.DataFrame) -> pd.Series:
    """
    PnL increments relative to zero initial wealth.

    The first increment is V_0 - 0, not zero. This preserves any execution
    PnL generated at the first backtest timestamp.
    """
    _require_columns(results, ["pnl"], "results")

    pnl = results["pnl"].astype(float)

    if pnl.empty:
        return pd.Series(dtype=float, index=results.index)

    increments = pnl.diff()
    increments.iloc[0] = pnl.iloc[0]

    return increments


def compute_pnl_returns(results: pd.DataFrame) -> pd.Series:
    """
    Backward-compatible alias.

    These are PnL increments in USDT, not percentage returns.
    """
    return compute_pnl_increments(results)


def compute_total_fees(
    trades: pd.DataFrame,
    include_terminal_unwind: bool = True,
) -> float:
    """Total transaction fees paid."""
    if trades.empty:
        return 0.0

    _require_columns(trades, ["fee"], "trades")

    data = trades
    if (
        not include_terminal_unwind
        and "is_terminal_unwind" in trades.columns
    ):
        data = trades[~trades["is_terminal_unwind"].astype(bool)]

    return float(data["fee"].astype(float).sum())


def compute_total_traded_notional(
    trades: pd.DataFrame,
    include_terminal_unwind: bool = True,
) -> float:
    """Sum of absolute executed notional in USDT."""
    if trades.empty:
        return 0.0

    _require_columns(trades, ["quantity", "price"], "trades")

    data = trades
    if (
        not include_terminal_unwind
        and "is_terminal_unwind" in trades.columns
    ):
        data = trades[~trades["is_terminal_unwind"].astype(bool)]

    return float(
        (
            data["quantity"].astype(float)
            * data["price"].astype(float)
        ).sum()
    )


def compute_gross_trading_pnl(trades: pd.DataFrame) -> float:
    """
    Gross realized cash PnL from all executions, before fees:

        - sum_i q_i p_i,

    where signed quantity q_i is positive for buys and negative for sells.

    With zero initial and final inventory this equals economic gross PnL
    before fees. It is retained mainly as an accounting diagnostic.
    """
    if trades.empty:
        return 0.0

    _require_columns(
        trades,
        ["signed_quantity", "price"],
        "trades",
    )

    return float(
        -(
            trades["signed_quantity"].astype(float)
            * trades["price"].astype(float)
        ).sum()
    )


def compute_terminal_inventory_value(results: pd.DataFrame) -> float:
    """
    Final inventory marked at the final mid.

    Under the corrected backtester this should be zero because residual
    inventory is explicitly liquidated at the terminal BBO.
    """
    if results.empty:
        return 0.0

    _require_columns(
        results,
        ["inventory", "mid_price"],
        "results",
    )

    return float(
        results["inventory"].iloc[-1]
        * results["mid_price"].iloc[-1]
    )


def trade_spread_capture(trades: pd.DataFrame) -> pd.Series:
    """
    Immediate execution edge relative to the contemporaneous mid:

        SC_i = q_i * (m_i - p_i),

    with signed quantity q_i > 0 for buys and q_i < 0 for sells.

    This single formula works for both sides. A terminal unwind at the bid
    or ask typically has negative spread capture because it crosses half of
    the terminal market spread.
    """
    if trades.empty:
        return pd.Series(dtype=float, index=trades.index)

    _require_columns(
        trades,
        ["signed_quantity", "price", "mid_price"],
        "trades",
    )

    return (
        trades["signed_quantity"].astype(float)
        * (
            trades["mid_price"].astype(float)
            - trades["price"].astype(float)
        )
    )


def compute_spread_capture(
    trades: pd.DataFrame,
    include_terminal_unwind: bool = True,
) -> float:
    """Total immediate execution edge relative to mid."""
    if trades.empty:
        return 0.0

    capture = trade_spread_capture(trades)

    if (
        not include_terminal_unwind
        and "is_terminal_unwind" in trades.columns
    ):
        mask = ~trades["is_terminal_unwind"].astype(bool)
        capture = capture.loc[mask]

    return float(capture.sum())


def compute_terminal_liquidation_spread_cost(
    trades: pd.DataFrame,
) -> float:
    """
    Cost from crossing the terminal BBO, excluding fees.

    Returned as a non-negative cost. It is the negative of terminal-unwind
    spread capture.
    """
    if trades.empty or "is_terminal_unwind" not in trades.columns:
        return 0.0

    mask = trades["is_terminal_unwind"].astype(bool)
    if not mask.any():
        return 0.0

    terminal_capture = trade_spread_capture(trades.loc[mask]).sum()
    return float(-terminal_capture)


def compute_inventory_pnl(results: pd.DataFrame) -> float:
    """
    PnL from carrying inventory between successive observed mids:

        InventoryPnL
            = sum_i Q_{i-1} * (m_i - m_{i-1}).

    In the corrected backtester, the final results row is the terminal BBO
    after liquidation. The lagged inventory on that row is precisely the
    position carried from the penultimate snapshot to the terminal snapshot.
    """
    if results.empty:
        return 0.0

    _require_columns(
        results,
        ["inventory", "mid_price"],
        "results",
    )

    inventory_lag = (
        results["inventory"]
        .astype(float)
        .shift(1)
        .fillna(0.0)
    )
    mid_change = (
        results["mid_price"]
        .astype(float)
        .diff()
        .fillna(0.0)
    )

    return float((inventory_lag * mid_change).sum())


def compute_signed_markouts(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon_seconds: float = 5.0,
    tolerance_seconds: Optional[float] = 2.0,
) -> pd.DataFrame:
    """
    Compute post-fill signed mid-price markouts in physical time.

    For each non-terminal strategy fill at time t, locate the first market
    observation at or after t + horizon_seconds and compute

        markout_i = q_i * (m_{t+h} - m_t).

    Positive markout is favorable; negative markout indicates adverse
    selection.

    Forced terminal liquidation is excluded because it is not a passive
    strategy fill.
    """
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be strictly positive.")

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "target_timestamp",
                "future_timestamp",
                "signed_quantity",
                "mid_price",
                "future_mid_price",
                "markout",
            ]
        )

    _require_columns(
        trades,
        ["timestamp", "signed_quantity", "mid_price"],
        "trades",
    )
    _require_columns(
        market_data,
        ["timestamp", "mid_price"],
        "market_data",
    )

    fill_data = trades.copy()

    if "is_terminal_unwind" in fill_data.columns:
        fill_data = fill_data[
            ~fill_data["is_terminal_unwind"].astype(bool)
        ].copy()

    if fill_data.empty:
        return pd.DataFrame()

    fill_data["timestamp"] = pd.to_datetime(
        fill_data["timestamp"],
        errors="coerce",
        utc=True,
    )
    fill_data = fill_data.dropna(subset=["timestamp"]).copy()

    fill_data["target_timestamp"] = (
        fill_data["timestamp"]
        + pd.to_timedelta(horizon_seconds, unit="s")
    )

    market = market_data[["timestamp", "mid_price"]].copy()
    market["timestamp"] = pd.to_datetime(
        market["timestamp"],
        errors="coerce",
        utc=True,
    )
    market = (
        market.dropna(subset=["timestamp", "mid_price"])
        .sort_values("timestamp")
        .rename(
            columns={
                "timestamp": "future_timestamp",
                "mid_price": "future_mid_price",
            }
        )
    )

    left = fill_data.sort_values("target_timestamp")

    tolerance = None
    if tolerance_seconds is not None:
        if tolerance_seconds < 0:
            raise ValueError(
                "tolerance_seconds must be non-negative or None."
            )
        tolerance = pd.Timedelta(seconds=tolerance_seconds)

    markouts = pd.merge_asof(
        left,
        market,
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="forward",
        tolerance=tolerance,
    )

    markouts = markouts.dropna(
        subset=["future_mid_price"]
    ).copy()

    markouts["markout"] = (
        markouts["signed_quantity"].astype(float)
        * (
            markouts["future_mid_price"].astype(float)
            - markouts["mid_price"].astype(float)
        )
    )

    return markouts


def compute_adverse_selection(
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon: int = 5,
) -> float:
    """
    Backward-compatible aggregate signed markout.

    `horizon` is interpreted in seconds for the corrected 1 Hz project.

    Positive = favorable post-fill movement.
    Negative = adverse selection.
    """
    markouts = compute_signed_markouts(
        trades=trades,
        market_data=market_data,
        horizon_seconds=float(horizon),
    )

    if markouts.empty:
        return 0.0

    return float(markouts["markout"].sum())


def compute_pnl_decomposition(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    market_data: pd.DataFrame,
    adverse_selection_horizon: int = 5,
    reconciliation_tolerance: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute and verify the economic PnL decomposition.

    Starting from zero wealth and ending with zero inventory,

        Final PnL
            = Spread capture
            + Inventory PnL
            - Fees.

    Terminal liquidation is a real trade:
    - its bid/ask crossing cost is included in spread capture;
    - its fee is included in total fees.

    Adverse selection / signed markout is a diagnostic and is NOT an
    additional additive PnL component.
    """
    if results.empty:
        raise ValueError("Results dataframe is empty.")

    _require_columns(
        results,
        ["pnl", "inventory"],
        "results",
    )

    final_pnl = float(results["pnl"].iloc[-1])
    final_inventory = float(results["inventory"].iloc[-1])

    total_fees = compute_total_fees(trades)
    spread_capture = compute_spread_capture(trades)
    strategy_spread_capture = compute_spread_capture(
        trades,
        include_terminal_unwind=False,
    )
    inventory_pnl = compute_inventory_pnl(results)

    reconstructed_pnl = (
        spread_capture
        + inventory_pnl
        - total_fees
    )
    reconciliation_error = final_pnl - reconstructed_pnl

    scale = max(
        1.0,
        abs(final_pnl),
        abs(reconstructed_pnl),
    )
    if abs(reconciliation_error) > reconciliation_tolerance * scale:
        raise RuntimeError(
            "PnL decomposition does not reconcile: "
            f"final_pnl={final_pnl:.12f}, "
            f"reconstructed={reconstructed_pnl:.12f}, "
            f"error={reconciliation_error:.12e}."
        )

    markouts = compute_signed_markouts(
        trades=trades,
        market_data=market_data,
        horizon_seconds=float(adverse_selection_horizon),
    )

    signed_markout_total = (
        float(markouts["markout"].sum())
        if not markouts.empty
        else 0.0
    )
    signed_markout_mean_per_fill = (
        float(markouts["markout"].mean())
        if not markouts.empty
        else np.nan
    )

    ordinary_trade_mask = pd.Series(
        True,
        index=trades.index,
        dtype=bool,
    )
    terminal_trade_count = 0

    if not trades.empty and "is_terminal_unwind" in trades.columns:
        terminal_mask = trades["is_terminal_unwind"].astype(bool)
        ordinary_trade_mask = ~terminal_mask
        terminal_trade_count = int(terminal_mask.sum())

    number_of_strategy_fills = int(
        ordinary_trade_mask.sum()
    ) if not trades.empty else 0

    total_traded_notional = compute_total_traded_notional(
        trades,
        include_terminal_unwind=True,
    )
    strategy_traded_notional = compute_total_traded_notional(
        trades,
        include_terminal_unwind=False,
    )

    gross_pnl_before_fees = (
        spread_capture + inventory_pnl
    )

    return {
        "final_pnl": final_pnl,
        "gross_pnl_before_fees": float(gross_pnl_before_fees),
        "gross_trading_pnl": compute_gross_trading_pnl(trades),
        "terminal_inventory_value": compute_terminal_inventory_value(
            results
        ),
        "total_fees": float(total_fees),
        "spread_capture": float(spread_capture),
        "strategy_spread_capture": float(strategy_spread_capture),
        "terminal_liquidation_spread_cost": (
            compute_terminal_liquidation_spread_cost(trades)
        ),
        "inventory_pnl": float(inventory_pnl),
        "signed_markout_total": signed_markout_total,
        "signed_markout_mean_per_fill": signed_markout_mean_per_fill,
        # Backward-compatible name. Sign convention:
        # positive = favorable, negative = adverse.
        "adverse_selection": signed_markout_total,
        "markout_horizon_seconds": float(
            adverse_selection_horizon
        ),
        "number_of_strategy_fills": number_of_strategy_fills,
        "number_of_terminal_trades": terminal_trade_count,
        "number_of_trades": int(len(trades)),
        "total_traded_notional": float(total_traded_notional),
        "strategy_traded_notional": float(strategy_traded_notional),
        "final_inventory": final_inventory,
        "reconstructed_pnl": float(reconstructed_pnl),
        "reconciliation_error": float(reconciliation_error),
    }
