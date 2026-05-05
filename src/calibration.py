import numpy as np
import pandas as pd
from dataclasses import replace


def future_rolling_min(series: pd.Series, horizon: int) -> pd.Series:
    """
    For each time t, compute min value over the next `horizon` observations.
    """

    return (
        series.shift(-1)
        .iloc[::-1]
        .rolling(window=horizon, min_periods=1)
        .min()
        .iloc[::-1]
        .reset_index(drop=True)
    )


def future_rolling_max(series: pd.Series, horizon: int) -> pd.Series:
    """
    For each time t, compute max value over the next `horizon` observations.
    """

    return (
        series.shift(-1)
        .iloc[::-1]
        .rolling(window=horizon, min_periods=1)
        .max()
        .iloc[::-1]
        .reset_index(drop=True)
    )


def estimate_fill_probabilities(
    df: pd.DataFrame,
    tick_size: float,
    tick_distances: list[int] | range,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Estimate fill probabilities for virtual bid/ask quotes.

    For each row:
        virtual bid = mid_price - distance * tick_size
        virtual ask = mid_price + distance * tick_size

    Bid fill proxy:
        future mid goes below our bid

    Ask fill proxy:
        future mid goes above our ask

    This is not a perfect queue model.
    It is a first simple approximation.
    """

    df = df.reset_index(drop=True).copy()

    results = []

    for horizon in horizons:
        future_min_mid = future_rolling_min(df["mid_price"], horizon)
        future_max_mid = future_rolling_max(df["mid_price"], horizon)

        for distance in tick_distances:
            bid_quote = df["mid_price"] - distance * tick_size
            ask_quote = df["mid_price"] + distance * tick_size

            bid_filled = future_min_mid <= bid_quote
            ask_filled = future_max_mid >= ask_quote

            p_bid = bid_filled.mean()
            p_ask = ask_filled.mean()
            p_mean = 0.5 * (p_bid + p_ask)

            results.append(
                {
                    "horizon": horizon,
                    "distance_ticks": distance,
                    "p_fill_bid": p_bid,
                    "p_fill_ask": p_ask,
                    "p_fill_mean": p_mean,
                }
            )

    return pd.DataFrame(results)


def fit_exponential_fill_model(
    fill_table: pd.DataFrame,
    horizon: int,
    min_probability: float = 1e-6,
) -> dict:
    """
    Fit:

        p_fill(distance) ≈ p0 * exp(-k_fill * distance)

    where distance is measured in ticks.

    Taking logs:

        log(p_fill) ≈ log(p0) - k_fill * distance
    """

    data = fill_table[fill_table["horizon"] == horizon].copy()

    # p_fill_mean is strictly positive enough
    # and
    # p_fill_mean is not exactly 1
    # to avoid issues with log and fitting.
    data = data[(data["p_fill_mean"] > min_probability) & (data["p_fill_mean"] < 1.0)]

    if len(data) < 2:
        return {
            "p0": 0.5,
            "k_fill": 1.0,
            "fit_ok": False,
        }

    x = data["distance_ticks"].values.astype(float)
    y = np.log(data["p_fill_mean"].values.astype(float))

    slope, intercept = np.polyfit(x, y, deg=1)

    k_fill = -slope
    p0 = np.exp(intercept)

    p0 = float(np.clip(p0, 0.0, 1.0))
    k_fill = float(max(k_fill, 0.0))

    return {
        "p0": p0,
        "k_fill": k_fill,
        "fit_ok": True,
    }


def estimate_risk_thresholds(
    df: pd.DataFrame,
    fee_rate: float,
    volatility_quantile: float = 0.95,
    depth_quantile: float = 0.10,
    spread_buffer_bps: float = 0.5,
) -> dict:
    """
    Estimate simple risk filter thresholds from the data.
    """

    if "rolling_volatility" not in df.columns:
        raise ValueError("Missing rolling_volatility column.")

    if "total_depth" not in df.columns:
        raise ValueError("Missing total_depth column.")

    if "spread_bps" not in df.columns:
        raise ValueError("Missing spread_bps column.")

    max_volatility = df["rolling_volatility"].quantile(volatility_quantile)

    min_depth = df["total_depth"].quantile(depth_quantile)

    fee_bps_per_side = fee_rate * 10_000
    round_trip_fee_bps = 2 * fee_bps_per_side
    # I only quote if the observed market spread is bigger than fees by some margin.
    # I add a small buffer to avoid adverse selection
    # latency
    # queue cost
    # slippage
    # rounding to tick
    # bad fills
    min_spread_bps = round_trip_fee_bps + spread_buffer_bps

    return {
        "max_volatility": float(max_volatility),
        "min_depth": float(min_depth),
        "min_spread_bps": float(min_spread_bps),
    }


def calibrate_strategy_parameters(
    df: pd.DataFrame,
    config,
    tick_distances: list[int] | range = range(1, 11),
    horizons: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Full first-pass calibration.

    Outputs:
        fill_table
        p0
        k_fill
        kappa
        max_volatility
        min_depth
        min_spread_bps
    """

    fill_table = estimate_fill_probabilities(
        df=df,
        tick_size=config.tick_size,
        tick_distances=tick_distances,
        horizons=horizons,
    )

    fill_fit = fit_exponential_fill_model(
        fill_table=fill_table,
        horizon=config.fill_horizon,
    )

    risk_thresholds = estimate_risk_thresholds(
        df=df,
        fee_rate=config.fee_rate,
    )

    # k_fill is calibrated per tick.
    # kappa in the Avellaneda formula is usually per price unit.
    # Since distance_price = distance_ticks * tick_size:
    #
    # exp(-k_fill * distance_ticks)
    # =
    # exp(-(k_fill / tick_size) * distance_price)
    #
    # So we convert:
    kappa = fill_fit["k_fill"] / config.tick_size

    calibration = {
        "fill_table": fill_table,
        "p0": fill_fit["p0"],
        "k_fill": fill_fit["k_fill"],
        "kappa": kappa,
        "fit_ok": fill_fit["fit_ok"],
        **risk_thresholds,
    }

    return calibration


def update_config_from_calibration(config, calibration: dict):
    """
    Return a new config object updated with calibrated parameters.
    """

    new_config = replace(
        config,
        p0=calibration["p0"],
        k_fill=calibration["k_fill"],
        kappa=calibration["kappa"],
        max_volatility=calibration["max_volatility"],
        min_depth=calibration["min_depth"],
        min_spread_bps=calibration["min_spread_bps"],
    )

    return new_config
