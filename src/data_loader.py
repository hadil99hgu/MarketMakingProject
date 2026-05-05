# src/data_loader.py
import pandas as pd
path_large = "../data/raw/Largebookticker.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid_price",
    "spread",
   
]


NUMERIC_COLUMNS = [
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid_price",
    "spread",
   
]

def load_lob_data(path=path_large):
    return pd.read_csv(path)
    



def validate_lob_columns(df: pd.DataFrame) -> None:
    """
    Check that all required LOB columns are present.
    """

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")



def clean_lob_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean LOB data:
    - parse timestamp
    - convert numeric columns
    - remove invalid bid/ask rows
    - recompute mid_price and spread
    - sort by timestamp
    """

    df = df.copy()

    validate_lob_columns(df)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing values in important columns
    df = df.dropna(subset=[
        "timestamp",
        "best_bid_price",
        "best_ask_price",
        "best_bid_qty",
        "best_ask_qty",
    ])

    # Keep only valid prices and quantities
    df = df[
        (df["best_bid_price"] > 0)
        & (df["best_ask_price"] > 0)
        & (df["best_bid_qty"] >= 0)
        & (df["best_ask_qty"] >= 0)
    ]

    # Remove crossed or invalid markets
    df = df[df["best_bid_price"] < df["best_ask_price"]]

    # Recompute mid and spread from bid/ask
    df["mid_price"] = (
        df["best_bid_price"] + df["best_ask_price"]
    ) / 2

    df["spread"] = (
        df["best_ask_price"] - df["best_bid_price"]
    )

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def prepare_lob_data(path: str | None = None) -> pd.DataFrame:
    """
    Full data preparation pipeline.
    """

    df = load_lob_data(path)
    df = clean_lob_data(df)

    return df
   