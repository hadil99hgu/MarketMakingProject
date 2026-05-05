# src/backtester.py

import pandas as pd
from tqdm import tqdm


class Backtester:
    """
    Backtester for the Avellaneda market-making strategy.

    Responsibilities:
        - loop over market data
        - call strategy.step(row)
        - collect state history
        - collect trades
    """

    def __init__(self, data: pd.DataFrame, strategy):
        self.data = data
        self.strategy = strategy
        self.results = None
        self.trades = None

    def run(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Run the backtest.
        """

        self.strategy.reset()

        iterator = self.data.iterrows()

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(self.data),
                desc="Running backtest",
            )

        for _, row in iterator:
            self.strategy.step(row)
        # Terminal inventory liquidation
        last_row = self.data.iloc[-1]
        
        unwind_trade = self.strategy.liquidate_inventory(last_row)
        # Store final state after liquidation
        self.strategy.state_history.append(
            {
                "timestamp": last_row["timestamp"] if "timestamp" in last_row else None,
                "mid_price": last_row["mid_price"],
                "inventory": self.strategy.inventory,
                "cash": self.strategy.cash,
                "pnl": self.strategy.pnl,
                "quote_bid": False,
                "quote_ask": False,
                "risk_reason": "terminal_unwind",
                "n_fills": 1 if unwind_trade is not None else 0,
                "reservation_price": float("nan"),
                "optimal_spread": float("nan"),
                "half_spread": float("nan"),
                "bid_price": float("nan"),
                "ask_price": float("nan"),
                "quote_spread": float("nan"),
            }
        )

        self.results = pd.DataFrame(self.strategy.get_state_history())

        self.trades = pd.DataFrame(self.strategy.get_trades())

        return self.results

    def get_results(self) -> pd.DataFrame:
        """
        Return strategy state history.
        """

        if self.results is None:
            raise ValueError("Backtest has not been run yet.")

        return self.results

    def get_trades(self) -> pd.DataFrame:
        """
        Return executed trades.
        """

        if self.trades is None:
            raise ValueError("Backtest has not been run yet.")

        return self.trades

    def save_results(
        self,
        results_path: str,
        trades_path: str,
    ) -> None:
        """
        Save results and trades to csv files.
        """

        if self.results is None or self.trades is None:
            raise ValueError("Backtest has not been run yet.")

        self.results.to_csv(results_path, index=False)
        self.trades.to_csv(trades_path, index=False)
