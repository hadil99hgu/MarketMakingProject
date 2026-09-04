# src/data_loader.py

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "Largebookticker.csv"
)

RAW_COLUMNS = [
    "timestamp",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
]

NUMERIC_COLUMNS = [
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
]


def load_lob_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw top-of-book dataset."""
    data_path = DEFAULT_DATA_PATH if path is None else Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"LOB data file not found: {data_path}")

    return pd.read_csv(data_path)


def validate_lob_columns(df: pd.DataFrame) -> None:
    """Check that the raw columns required by the backtest are present."""
    missing_cols = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def clean_lob_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Binance top-of-book snapshots.

    The raw dataset is not modified in place. Mid-price and spread are
    recomputed from best bid/ask rather than trusted from the CSV.
    """
    df = df.copy()
    validate_lob_columns(df)

    # Invalid timestamps become NaT and are removed below.
    try:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
            format="mixed",
        )
    except (TypeError, ValueError):
        # Compatibility with older pandas versions that do not support
        # format="mixed".
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", *NUMERIC_COLUMNS])

    # A valid top-of-book quote has positive prices/quantities and is not
    # locked or crossed.
    df = df[
        (df["best_bid_price"] > 0)
        & (df["best_ask_price"] > 0)
        & (df["best_bid_qty"] > 0)
        & (df["best_ask_qty"] > 0)
        & (df["best_bid_price"] < df["best_ask_price"])
    ].copy()

    # Derived quantities are recomputed from the cleaned raw BBO.
    df["mid_price"] = 0.5 * (
        df["best_bid_price"] + df["best_ask_price"]
    )
    df["spread"] = (
        df["best_ask_price"] - df["best_bid_price"]
    )

    # Stable sorting preserves acquisition order if two timestamps coincide.
    df = df.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid LOB observations remain after cleaning.")

    return df


def sampling_diagnostics(df: pd.DataFrame) -> dict:
    """
    Summarize the empirical sampling frequency without altering the data.

    Useful because the collector targets approximately one observation
    per second, while the actual timestamp spacing also contains request
    and processing latency.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' column.")

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    ).dropna().sort_values()

    if len(timestamps) < 2:
        raise ValueError("At least two valid timestamps are required.")

    dt = timestamps.diff().dt.total_seconds().dropna()
    positive_dt = dt[dt > 0]

    if positive_dt.empty:
        raise ValueError("No positive timestamp increments were found.")

    duration_seconds = (
        timestamps.iloc[-1] - timestamps.iloc[0]
    ).total_seconds()

    median_dt = float(positive_dt.median())

    return {
        "n_observations": int(len(timestamps)),
        "start_time": timestamps.iloc[0],
        "end_time": timestamps.iloc[-1],
        "duration_seconds": float(duration_seconds),
        "median_dt_seconds": median_dt,
        "mean_dt_seconds": float(positive_dt.mean()),
        "p95_dt_seconds": float(positive_dt.quantile(0.95)),
        "p99_dt_seconds": float(positive_dt.quantile(0.99)),
        "max_dt_seconds": float(positive_dt.max()),
        "median_effective_hz": float(1.0 / median_dt),
        "nonpositive_dt_count": int((dt <= 0).sum()),
    }


def prepare_lob_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the top-of-book dataset."""
    return clean_lob_data(load_lob_data(path))
