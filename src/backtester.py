# src/backtester.py

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class Backtester:
    """
    Discrete-time backtester for the market-making strategy.

    Observation i is used to form quotes for the interval [t_i, t_{i+1}).
    The final observation is reserved for terminal marking/liquidation,
    because there is no observed interval after it.
    """

    def __init__(self, data: pd.DataFrame, strategy):
        self.data = self._validate_and_prepare_data(data)
        self.strategy = strategy
        self.results: Optional[pd.DataFrame] = None
        self.trades: Optional[pd.DataFrame] = None

    @staticmethod
    def _validate_and_prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        required = {
            "timestamp",
            "mid_price",
            "best_bid_price",
            "best_ask_price",
        }
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing required backtest columns: {missing}")

        if len(data) < 2:
            raise ValueError(
                "At least two observations are required: one quoting "
                "observation and one terminal observation."
            )

        prepared = data.copy().reset_index(drop=True)
        prepared["timestamp"] = pd.to_datetime(
            prepared["timestamp"],
            errors="coerce",
            utc=True,
        )

        if prepared["timestamp"].isna().any():
            raise ValueError("Backtest data contain invalid timestamps.")

        if not prepared["timestamp"].is_monotonic_increasing:
            raise ValueError(
                "Backtest timestamps must be sorted in increasing order."
            )

        forward_dt = (
            prepared["timestamp"].shift(-1) - prepared["timestamp"]
        ).dt.total_seconds()

        if (forward_dt.iloc[:-1] <= 0).any():
            raise ValueError(
                "Backtest timestamps must be strictly increasing."
            )

        prepared["execution_interval_seconds"] = forward_dt

        return prepared

    def run(self, show_progress: bool = True) -> pd.DataFrame:
        """Run the complete backtest and terminal liquidation."""
        self.strategy.reset()

        # Every quoting decision needs an observed future interval. Therefore
        # the last market snapshot is not used to initiate a new quote.
        indices = range(len(self.data) - 1)

        iterator = indices
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(self.data) - 1,
                desc="Running backtest",
            )

        for i in iterator:
            row = self.data.iloc[i]
            self.strategy.step(row)

        last_row = self.data.iloc[-1]

        terminal_inventory_before = float(self.strategy.inventory)
        terminal_cash_before = float(self.strategy.cash)

        unwind_trade = self.strategy.liquidate_inventory(last_row)

        terminal_net_quantity = (
            float(unwind_trade["signed_quantity"])
            if unwind_trade is not None
            else 0.0
        )

        # One terminal state at the final observed BBO. There is no new quote
        # at this timestamp.
        self.strategy.state_history.append(
            {
                "timestamp": last_row["timestamp"],
                "mid_price": float(last_row["mid_price"]),
                "inventory_before": terminal_inventory_before,
                "inventory": float(self.strategy.inventory),
                "cash_before": terminal_cash_before,
                "cash": float(self.strategy.cash),
                "pnl": float(self.strategy.pnl),
                "quote_bid": False,
                "quote_ask": False,
                "risk_reason": "terminal_unwind",
                "n_fills": 1 if unwind_trade is not None else 0,
                "net_fill_quantity": terminal_net_quantity,
                "reservation_price": np.nan,
                "optimal_spread": np.nan,
                "inventory_risk_spread": np.nan,
                "execution_spread": np.nan,
                "half_spread": np.nan,
                "imbalance_skew": np.nan,
                "bid_price": np.nan,
                "ask_price": np.nan,
                "quote_spread": np.nan,
            }
        )

        self.results = pd.DataFrame(
            self.strategy.get_state_history()
        )
        self.trades = pd.DataFrame(
            self.strategy.get_trades()
        )

        return self.results

    def get_results(self) -> pd.DataFrame:
        """Return strategy state history."""
        if self.results is None:
            raise ValueError("Backtest has not been run yet.")

        return self.results

    def get_trades(self) -> pd.DataFrame:
        """Return executed trades."""
        if self.trades is None:
            raise ValueError("Backtest has not been run yet.")

        return self.trades

    def save_results(
        self,
        results_path: str,
        trades_path: str,
    ) -> None:
        """Save state and trade outputs to CSV."""
        if self.results is None or self.trades is None:
            raise ValueError("Backtest has not been run yet.")

        results_file = Path(results_path)
        trades_file = Path(trades_path)

        results_file.parent.mkdir(parents=True, exist_ok=True)
        trades_file.parent.mkdir(parents=True, exist_ok=True)

        self.results.to_csv(results_file, index=False)
        self.trades.to_csv(trades_file, index=False)
