# src/strategy.py
import numpy as np
import numpy as np

from src.quote_engine import compute_final_quotes

from src.execution import simulate_fills
from src.risk import should_quote

class AvellanedaMarketMaker:
    """
    Avellaneda-Stoikov market-making strategy with:
    - inventory tracking
    - cash tracking
    - risk filters
    - imbalance skew
    - probabilistic execution
    """

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self):
        """
        Reset strategy state.
        """

        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0

        self.trades = []
        self.state_history = []

    def get_trades(self):
        """
        Return all executed trades.
        """

        return self.trades

    def get_state_history(self):
        """
        Return full strategy state history.
        """

        return self.state_history

    def update_after_fills(self, fills: list[dict], row) -> None:
        """
        Update cash and inventory after executions.

        Convention:
            signed_quantity > 0 means buy
            signed_quantity < 0 means sell

        Cash update:
            cash -= signed_quantity * price
            cash -= fee
        """

        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = row["mid_price"]

        for fill in fills:
            signed_quantity = fill["signed_quantity"]
            price = fill["price"]
            fee = fill["fee"]

            # Update inventory
            self.inventory += signed_quantity

            # Update cash
            self.cash -= signed_quantity * price
            self.cash -= fee

            trade_record = {
                "timestamp": timestamp,
                "side": fill["side"],
                "price": price,
                "quantity": fill["quantity"],
                "signed_quantity": signed_quantity,
                "fee": fee,
                "p_fill": fill["p_fill"],
                "mid_price": mid_price,
                "inventory_after": self.inventory,
                "cash_after": self.cash,
                "is_terminal_unwind": False,
            }

            self.trades.append(trade_record)

    def mark_to_market(self, mid_price: float) -> float:
        """
        Mark-to-market portfolio value.

        total_pnl = cash + inventory * mid_price
        """
        self.pnl = self.cash + self.inventory * mid_price
        return self.pnl

    def step(self, row):
        """
        Run one strategy step at one timestamp.

        Steps:
            1. Check whether we are allowed to quote.
            2. Compute quotes if allowed.
            3. Simulate fills.
            4. Update cash and inventory.
            5. Compute mark-to-market PnL.
            6. Store state.
        """
        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = row["mid_price"]

        # 1. Check whether we are allowed to quote
        permissions = should_quote(
            row=row,
            inventory=self.inventory,
            config=self.config,
        )
        # 2. Compute quotes if allowed
        quotes = None
        fills = []

        if permissions["quote_bid"] or permissions["quote_ask"]:
            quotes = compute_final_quotes(
                row=row,
                inventory=self.inventory,
                config=self.config,
            )
            fills = simulate_fills(
                row=row,
                quotes=quotes,
                permissions=permissions,
                config=self.config,
                rng=self.rng,
            )

            self.update_after_fills(fills=fills, row=row)
        pnl = self.mark_to_market(mid_price=mid_price)

        state = {
            "timestamp": timestamp,
            "mid_price": mid_price,
            "inventory": self.inventory,
            "cash": self.cash,
            "pnl": pnl,
            "quote_bid": permissions["quote_bid"],
            "quote_ask": permissions["quote_ask"],
            "risk_reason": permissions["reason"],
            "n_fills": len(fills),
        }

        if quotes is not None:
            state.update(
                {
                    "reservation_price": quotes["reservation_price"],
                    "optimal_spread": quotes["optimal_spread"],
                    "half_spread": quotes["half_spread"],
                    "bid_price": quotes["bid_price"],
                    "ask_price": quotes["ask_price"],
                    "quote_spread": quotes["quote_spread"],
                }
            )
        else:
            state.update(
                {
                    "reservation_price": np.nan,
                    "optimal_spread": np.nan,
                    "half_spread": np.nan,
                    "bid_price": np.nan,
                    "ask_price": np.nan,
                    "quote_spread": np.nan,
                }
            )

        self.state_history.append(state)

        return state

    def liquidate_inventory(self, row) -> dict | None:
        """
        Liquidate remaining inventory at the end of the backtest.

        If inventory > 0:
            sell at best bid

        If inventory < 0:
            buy at best ask
        """

        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = row["mid_price"]

        if self.inventory == 0:
            return None

        if self.inventory > 0:
            side = "sell"
            price = row["best_bid_price"]
            quantity = abs(self.inventory)
            signed_quantity = -quantity

        else:
            side = "buy"
            price = row["best_ask_price"]
            quantity = abs(self.inventory)
            signed_quantity = quantity

        fee = self.config.fee_rate * quantity * price

        self.inventory += signed_quantity
        self.cash -= signed_quantity * price
        self.cash -= fee

        trade_record = {
            "timestamp": timestamp,
            "side": side,
            "price": price,
            "quantity": quantity,
            "signed_quantity": signed_quantity,
            "fee": fee,
            "p_fill": 1.0,
            "mid_price": mid_price,
            "inventory_after": self.inventory,
            "cash_after": self.cash,
            "is_terminal_unwind": True,
        }

        self.trades.append(trade_record)

        self.mark_to_market(mid_price)

        return trade_record
