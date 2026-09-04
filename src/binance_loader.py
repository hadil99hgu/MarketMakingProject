# src/binance_loader.py

import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    SYMBOL,
    N_OBSERVATIONS,
    SLEEP_SECONDS,
    BOOK_TICKER_URL,
    OUTPUT_CSV,
)

FIELDNAMES = [
    "timestamp",
    "symbol",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid_price",
    "spread",
    "request_latency_ms",
]


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_book_ticker(session: requests.Session) -> dict:
    request_start = time.perf_counter()
    response = session.get(BOOK_TICKER_URL, timeout=10)
    response.raise_for_status()

    latency_ms = 1000.0 * (time.perf_counter() - request_start)
    received_at = pd.Timestamp.now(tz="UTC")
    data = response.json()

    bid_price = float(data["bidPrice"])
    ask_price = float(data["askPrice"])
    bid_qty = float(data["bidQty"])
    ask_qty = float(data["askQty"])

    if bid_price <= 0 or ask_price <= 0:
        raise ValueError("Bid and ask prices must be positive.")
    if bid_qty < 0 or ask_qty < 0:
        raise ValueError("Bid and ask quantities must be non-negative.")
    if bid_price >= ask_price:
        raise ValueError(
            f"Invalid BBO: bid={bid_price} must be below ask={ask_price}."
        )

    return {
        "timestamp": received_at.isoformat(),
        "symbol": SYMBOL,
        "best_bid_price": bid_price,
        "best_bid_qty": bid_qty,
        "best_ask_price": ask_price,
        "best_ask_qty": ask_qty,
        "mid_price": 0.5 * (bid_price + ask_price),
        "spread": ask_price - bid_price,
        "request_latency_ms": latency_ms,
    }


def prepare_output_file():
    path = Path(OUTPUT_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()
        return path, 0

    try:
        existing = pd.read_csv(path, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame(columns=FIELDNAMES)

    missing = [c for c in FIELDNAMES if c not in existing.columns]
    if missing:
        raise ValueError(
            "Existing output file has an incompatible schema. "
            f"Missing columns: {missing}"
        )

    existing = existing.dropna(
        subset=["timestamp", "best_bid_price", "best_ask_price"]
    )

    # Clean a possible truncated last line before resuming.
    existing[FIELDNAMES].to_csv(path, index=False)
    return path, int(len(existing))


def append_row(path: Path, row: dict) -> None:
    # Every successful observation is saved immediately.
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def run() -> None:
    path, collected = prepare_output_file()

    if collected >= N_OBSERVATIONS:
        print(
            f"{path} already contains {collected} observations. "
            "Nothing to collect."
        )
        return

    if collected:
        print(
            f"Resuming collection from {collected}/{N_OBSERVATIONS} observations."
        )
    else:
        print(f"Starting a new collection: 0/{N_OBSERVATIONS}.")

    session = build_session()

    try:
        while collected < N_OBSERVATIONS:
            loop_start = time.perf_counter()

            try:
                row = fetch_book_ticker(session)
            except (
                requests.RequestException,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                print(f"Collection error: {exc}. Will keep retrying.")
            else:
                append_row(path, row)
                collected += 1
                print(
                    f"{collected}/{N_OBSERVATIONS} collected",
                    flush=True,
                )

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, SLEEP_SECONDS - elapsed))

    except KeyboardInterrupt:
        print(
            "\nCollection interrupted. "
            f"{collected} observations are already saved."
        )
    finally:
        session.close()

    print(
        f"Current file: {path} "
        f"({collected}/{N_OBSERVATIONS} observations)."
    )


if __name__ == "__main__":
    run()
