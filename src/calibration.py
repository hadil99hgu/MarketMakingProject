# src/calibration.py

from dataclasses import replace
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def future_rolling_min(series: pd.Series, horizon: int) -> pd.Series:
    """
    Minimum over the next `horizon` observations, excluding the current row.

    Rows without a complete forward window are returned as NaN so that they
    are not given a shorter effective calibration horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    return (
        series.shift(-1)
        .iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .min()
        .iloc[::-1]
        .reset_index(drop=True)
    )


def future_rolling_max(series: pd.Series, horizon: int) -> pd.Series:
    """
    Maximum over the next `horizon` observations, excluding the current row.

    Rows without a complete forward window are returned as NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    return (
        series.shift(-1)
        .iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .max()
        .iloc[::-1]
        .reset_index(drop=True)
    )


def estimate_fill_probabilities(
    df: pd.DataFrame,
    tick_size: float,
    tick_distances: Iterable[int],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """
    Estimate empirical virtual-quote touch probabilities.

    For distance d ticks from the current mid:

        bid_t(d) = mid_t - d * tick_size
        ask_t(d) = mid_t + d * tick_size

    A bid is labelled as touched if the future mid reaches or crosses the
    virtual bid within the complete forward horizon. The ask side is treated
    symmetrically.

    These are touch probabilities, not queue-aware fill probabilities.
    """
    if tick_size <= 0:
        raise ValueError("tick_size must be strictly positive.")

    if "mid_price" not in df.columns:
        raise ValueError("Missing mid_price column.")

    distances = list(tick_distances)
    if not distances or any(distance < 0 for distance in distances):
        raise ValueError("tick_distances must contain non-negative values.")

    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")

    data = df.reset_index(drop=True).copy()
    results = []

    for horizon in horizons:
        future_min_mid = future_rolling_min(data["mid_price"], horizon)
        future_max_mid = future_rolling_max(data["mid_price"], horizon)

        valid = future_min_mid.notna() & future_max_mid.notna()
        n_valid = int(valid.sum())

        if n_valid == 0:
            continue

        current_mid = data.loc[valid, "mid_price"]
        future_min = future_min_mid.loc[valid]
        future_max = future_max_mid.loc[valid]

        for distance in distances:
            bid_quote = current_mid - distance * tick_size
            ask_quote = current_mid + distance * tick_size

            bid_touched = future_min <= bid_quote
            ask_touched = future_max >= ask_quote

            p_bid = float(bid_touched.mean())
            p_ask = float(ask_touched.mean())

            results.append(
                {
                    "horizon": int(horizon),
                    "distance_ticks": int(distance),
                    "n_observations": n_valid,
                    "p_fill_bid": p_bid,
                    "p_fill_ask": p_ask,
                    "p_fill_mean": 0.5 * (p_bid + p_ask),
                }
            )

    if not results:
        raise ValueError("No valid fill-probability observations were produced.")

    return pd.DataFrame(results)


def _linear_log_fit(x: np.ndarray, positive_y: np.ndarray) -> Dict[str, float]:
    """Fit log(y) = intercept + slope * x and return diagnostics."""
    log_y = np.log(positive_y)

    slope, intercept = np.polyfit(x, log_y, deg=1)
    fitted = intercept + slope * x

    ss_res = float(np.sum((log_y - fitted) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def fit_exponential_fill_model(
    fill_table: pd.DataFrame,
    horizon: int,
    min_probability: float = 1e-6,
) -> Dict[str, float]:
    """
    Diagnostic direct fit of the horizon-h touch probability:

        p_h(d) ~= p0_h * exp(-k_h * d).

    This fit is retained for diagnostics. It is not used directly as the
    per-step execution model when horizon > 1.
    """
    data = fill_table[fill_table["horizon"] == horizon].copy()
    data = data[
        (data["p_fill_mean"] > min_probability)
        & (data["p_fill_mean"] < 1.0)
    ]

    if len(data) < 2:
        raise ValueError(
            f"Not enough non-degenerate fill probabilities for horizon={horizon}."
        )

    x = data["distance_ticks"].to_numpy(dtype=float)
    p = data["p_fill_mean"].to_numpy(dtype=float)

    fit = _linear_log_fit(x, p)

    return {
        "p0_horizon": float(np.clip(np.exp(fit["intercept"]), 0.0, 1.0)),
        "k_fill_horizon": float(max(-fit["slope"], 0.0)),
        "probability_fit_r_squared": fit["r_squared"],
    }


def fit_exponential_intensity_model(
    fill_table: pd.DataFrame,
    horizon: int,
    tick_size: float,
    seconds_per_observation: float = 1.0,
    execution_step_seconds: float = 1.0,
    min_probability: float = 1e-6,
) -> Dict[str, float]:
    """
    Convert horizon-h touch probabilities into a per-second intensity and fit

        lambda(d) = A * exp(-kappa_ticks * d).

    If p_h(d) is the probability of at least one event over h observations,
    the constant-hazard conversion is

        lambda(d) = -log(1 - p_h(d)) / (h * seconds_per_observation).

    The fitted intensity is then converted back to a one-step probability
    curve for the existing Bernoulli execution simulator:

        p_step(d) = 1 - exp(-lambda(d) * execution_step_seconds).

    A second exponential fit,

        p_step(d) ~= p0 * exp(-k_fill * d),

    provides backward-compatible p0 and k_fill parameters for execution.py.

    The Avellaneda-Stoikov kappa is obtained from the intensity decay, not
    from the finite-horizon probability decay.
    """
    if tick_size <= 0:
        raise ValueError("tick_size must be strictly positive.")
    if seconds_per_observation <= 0 or execution_step_seconds <= 0:
        raise ValueError("Time scales must be strictly positive.")

    data = fill_table[fill_table["horizon"] == horizon].copy()
    data = data[
        (data["p_fill_mean"] > min_probability)
        & (data["p_fill_mean"] < 1.0 - min_probability)
    ]

    if len(data) < 2:
        raise ValueError(
            f"Not enough non-degenerate fill probabilities for horizon={horizon}."
        )

    x = data["distance_ticks"].to_numpy(dtype=float)
    p_h = data["p_fill_mean"].to_numpy(dtype=float)

    horizon_seconds = float(horizon) * seconds_per_observation
    intensity = -np.log1p(-p_h) / horizon_seconds

    intensity_fit = _linear_log_fit(x, intensity)

    arrival_rate_A = float(np.exp(intensity_fit["intercept"]))
    kappa_ticks = float(max(-intensity_fit["slope"], 0.0))

    fitted_intensity = arrival_rate_A * np.exp(-kappa_ticks * x)
    one_step_probability = 1.0 - np.exp(
        -fitted_intensity * execution_step_seconds
    )

    probability_fit = _linear_log_fit(x, one_step_probability)

    p0 = float(np.clip(np.exp(probability_fit["intercept"]), 0.0, 1.0))
    k_fill = float(max(-probability_fit["slope"], 0.0))

    return {
        "arrival_rate_A": arrival_rate_A,
        "kappa_ticks": kappa_ticks,
        "kappa": kappa_ticks / tick_size,
        "intensity_fit_r_squared": intensity_fit["r_squared"],
        "p0": p0,
        "k_fill": k_fill,
        "execution_step_seconds": float(execution_step_seconds),
        "calibration_horizon_seconds": horizon_seconds,
    }


def estimate_risk_thresholds(
    df: pd.DataFrame,
    volatility_quantile: float = 0.95,
    depth_quantile: float = 0.10,
    min_spread_bps: float = 0.0,
) -> Dict[str, float]:
    """
    Estimate simple volatility and depth regime filters.

    The minimum observed market spread is not inferred from transaction fees:
    fees apply to our executed quotes, whereas the observed BBO spread is a
    different object. Fees remain accounted for trade by trade in the PnL.
    """
    volatility_column = (
        "rolling_log_volatility"
        if "rolling_log_volatility" in df.columns
        else "rolling_volatility"
    )

    required = [volatility_column, "total_depth", "spread_bps"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing calibration columns: {missing}")

    if not 0.0 < volatility_quantile < 1.0:
        raise ValueError("volatility_quantile must lie in (0, 1).")
    if not 0.0 < depth_quantile < 1.0:
        raise ValueError("depth_quantile must lie in (0, 1).")
    if min_spread_bps < 0:
        raise ValueError("min_spread_bps must be non-negative.")

    max_volatility = df[volatility_column].dropna().quantile(
        volatility_quantile
    )
    min_depth = df["total_depth"].dropna().quantile(depth_quantile)

    if pd.isna(max_volatility) or pd.isna(min_depth):
        raise ValueError("Risk thresholds could not be estimated.")

    return {
        "max_volatility": float(max_volatility),
        "min_depth": float(min_depth),
        "min_spread_bps": float(min_spread_bps),
    }


def calibrate_strategy_parameters(
    df: pd.DataFrame,
    config,
    tick_distances: Iterable[int] = range(1, 11),
    horizons: Sequence[int] = (1, 3, 5, 10),
    seconds_per_observation: float = 1.0,
    execution_step_seconds: float = 1.0,
) -> Dict[str, object]:
    """
    Calibrate the reduced-form execution model and risk filters.

    `config.fill_horizon` selects the forward touch horizon. The resulting
    p0 and k_fill are always converted to the execution-step time scale, while
    kappa is calibrated from the per-second intensity decay.
    """
    horizons = tuple(horizons)

    if config.fill_horizon not in horizons:
        horizons = tuple(sorted(set(horizons + (config.fill_horizon,))))

    fill_table = estimate_fill_probabilities(
        df=df,
        tick_size=config.tick_size,
        tick_distances=tick_distances,
        horizons=horizons,
    )

    direct_probability_fit = fit_exponential_fill_model(
        fill_table=fill_table,
        horizon=config.fill_horizon,
    )

    intensity_fit = fit_exponential_intensity_model(
        fill_table=fill_table,
        horizon=config.fill_horizon,
        tick_size=config.tick_size,
        seconds_per_observation=seconds_per_observation,
        execution_step_seconds=execution_step_seconds,
    )

    risk_thresholds = estimate_risk_thresholds(df=df)

    return {
        "fill_table": fill_table,
        **intensity_fit,
        **direct_probability_fit,
        **risk_thresholds,
        "fit_ok": True,
    }


def update_config_from_calibration(config, calibration: Dict[str, object]):
    """Return a new strategy configuration with calibrated parameters."""
    return replace(
        config,
        p0=float(calibration["p0"]),
        k_fill=float(calibration["k_fill"]),
        kappa=float(calibration["kappa"]),
        max_volatility=float(calibration["max_volatility"]),
        min_depth=float(calibration["min_depth"]),
        min_spread_bps=float(calibration["min_spread_bps"]),
    )
