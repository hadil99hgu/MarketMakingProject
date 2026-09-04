# src/features.py

import numpy as np
import pandas as pd
from typing import Optional


def compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Top-of-book displayed quantity on both sides."""
    df = df.copy()
    df["volume"] = df["best_bid_qty"] + df["best_ask_qty"]
    return df


def compute_relative_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the bid-ask spread relative to the mid-price."""
    df = df.copy()

    if (df["mid_price"] <= 0).any():
        raise ValueError("mid_price must be strictly positive.")

    df["relative_spread"] = df["spread"] / df["mid_price"]
    df["spread_bps"] = 10_000.0 * df["relative_spread"]
    return df


def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Level-1 order-book imbalance:

        (bid_qty - ask_qty) / (bid_qty + ask_qty).
    """
    df = df.copy()

    denominator = df["best_bid_qty"] + df["best_ask_qty"]

    df["imbalance"] = np.where(
        denominator > 0,
        (df["best_bid_qty"] - df["best_ask_qty"]) / denominator,
        np.nan,
    )

    return df


def compute_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Store bid, ask and total displayed top-of-book depth."""
    df = df.copy()
    df["bid_depth"] = df["best_bid_qty"]
    df["ask_depth"] = df["best_ask_qty"]
    df["total_depth"] = df["bid_depth"] + df["ask_depth"]
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price increments and log returns.

    Timestamp differences are retained so that volatility can be expressed
    per square-root second even if the nominal 1 Hz snapshots are not
    perfectly equally spaced.
    """
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column.")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    dt_seconds = timestamps.diff().dt.total_seconds()
    dt_seconds = dt_seconds.where(dt_seconds > 0)

    df["dt_seconds"] = dt_seconds
    df["log_mid_return"] = np.log(df["mid_price"]).diff()
    df["mid_price_change"] = df["mid_price"].diff()

    # Under a diffusion, increments scale like sqrt(dt). These normalized
    # increments therefore have volatility units per sqrt(second).
    sqrt_dt = np.sqrt(df["dt_seconds"])
    df["log_return_per_sqrt_second"] = df["log_mid_return"] / sqrt_dt
    df["price_change_per_sqrt_second"] = df["mid_price_change"] / sqrt_dt

    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    vol_window: int = 100,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimate short-horizon volatility over the latest observations.

    rolling_log_volatility:
        dimensionless volatility per sqrt(second), used as a relative
        volatility/risk-regime feature.

    rolling_price_volatility:
        absolute mid-price volatility in USDT per sqrt(second), suitable
        for the arithmetic-price Avellaneda-Stoikov equations.

    rolling_volatility is retained as a backward-compatible alias for
    rolling_log_volatility while the rest of the project is migrated.
    """
    df = df.copy()

    if vol_window <= 1:
        raise ValueError("vol_window must be greater than 1.")

    if min_periods is None:
        min_periods = vol_window

    if not 2 <= min_periods <= vol_window:
        raise ValueError("min_periods must lie between 2 and vol_window.")

    df["rolling_log_volatility"] = (
        df["log_return_per_sqrt_second"]
        .rolling(window=vol_window, min_periods=min_periods)
        .std(ddof=1)
    )

    df["rolling_price_volatility"] = (
        df["price_change_per_sqrt_second"]
        .rolling(window=vol_window, min_periods=min_periods)
        .std(ddof=1)
    )

    # Temporary compatibility with risk.py/calibration.py.
    df["rolling_volatility"] = df["rolling_log_volatility"]

    return df


def compute_features_v1(
    df: pd.DataFrame,
    vol_window: int = 100,
) -> pd.DataFrame:
    """Compute all features used by the current strategy."""
    df = df.copy()
    df = compute_volume(df)
    df = compute_relative_spread(df)
    df = compute_imbalance(df)
    df = compute_depth(df)
    df = compute_returns(df)
    df = compute_rolling_volatility(df, vol_window=vol_window)

    return df
