# src/features.py
import numpy as np
import pandas as pd


def compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume"] = df["best_bid_qty"] + df["best_ask_qty"]
    return df




def compute_relative_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relative_spread"] = df["spread"] / df["mid_price"]
    df["spread_bps"] = 10_000 * df["relative_spread"]
    return df


def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    denominator = df["best_bid_qty"] + df["best_ask_qty"]

    df["imbalance"] = np.where(
        denominator > 0,
        (df["best_bid_qty"] - df["best_ask_qty"]) / denominator,
        0.0,
    )

    return df


def compute_depth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bid_depth"] = df["best_bid_qty"]
    df["ask_depth"] = df["best_ask_qty"]
    df["total_depth"] = df["bid_depth"] + df["ask_depth"]
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_mid_return"] = np.log(df["mid_price"]).diff()
    df["log_mid_return"] = df["log_mid_return"].fillna(0.0)
    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    vol_window: int = 100,
) -> pd.DataFrame:
    df = df.copy()

    df["rolling_volatility"] = (
        df["log_mid_return"]
        .rolling(window=vol_window, min_periods=10)
        .std()
        .fillna(0.0)
    )

    return df


def compute_features_v1(
    df: pd.DataFrame,
    vol_window: int = 100,
) -> pd.DataFrame:
    df = df.copy()
    df = compute_volume(df)
    df = compute_relative_spread(df)
    df = compute_imbalance(df)
    df = compute_depth(df)
    df = compute_returns(df)
    df = compute_rolling_volatility(df, vol_window=vol_window)

    return df