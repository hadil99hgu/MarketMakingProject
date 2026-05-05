import os
import time
import requests
import pandas as pd

from config import SYMBOL, N_OBSERVATIONS, SLEEP_SECONDS, BOOK_TICKER_URL, OUTPUT_CSV


def fetch_book_ticker():
    response = requests.get(BOOK_TICKER_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    bid_price = float(data["bidPrice"])
    ask_price = float(data["askPrice"])
    bid_qty = float(data["bidQty"])
    ask_qty = float(data["askQty"])

    mid_price = (bid_price + ask_price) / 2
    spread = ask_price - bid_price

    return {
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "symbol": SYMBOL,
        "best_bid_price": bid_price,
        "best_bid_qty": bid_qty,
        "best_ask_price": ask_price,
        "best_ask_qty": ask_qty,
        "mid_price": mid_price,
        "spread": spread,
    }


def run():
    rows = []

    for i in range(N_OBSERVATIONS):
        row = fetch_book_ticker()
        rows.append(row)
        print(f"{i+1}/{N_OBSERVATIONS} collected")
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    run()