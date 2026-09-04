# src/strategy.py

from typing import Dict, List, Optional

import numpy as np

from src.execution import simulate_fills
from src.quote_engine import compute_final_quotes
from src.risk import should_quote


class AvellanedaMarketMaker:
    """
    Avellaneda-Stoikov-inspired market-making strategy with:
    - inventory and cash accounting,
    - market-regime and inventory risk controls,
    - microprice-based quote skew,
    - reduced-form probabilistic maker execution.
    """

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self) -> None:
        """Reset all strategy state."""
        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        self.trades: List[Dict[str, object]] = []
        self.state_history: List[Dict[str, object]] = []

    def get_trades(self) -> List[Dict[str, object]]:
        """Return all executed trades."""
        return self.trades

    def get_state_history(self) -> List[Dict[str, object]]:
        """Return the full strategy state history."""
        return self.state_history

    def _check_inventory_limit(self) -> None:
        """Assert that realized inventory respects the hard limit."""
        limit = float(self.config.inventory_limit)
        tolerance = 1e-12

        if abs(self.inventory) > limit + tolerance:
            raise RuntimeError(
                "Inventory limit violated after execution: "
                f"|{self.inventory}| > {limit}."
            )

    def update_after_fills(self, fills: List[Dict[str, float]], row) -> None:
        """
        Update cash and inventory after all executions in one time bucket.

        signed_quantity > 0: buy
        signed_quantity < 0: sell

        Cash update for each fill:
            cash -= signed_quantity * execution_price + fee

        Bid and ask fills may both occur in the same one-second bucket. Their
        intrasecond ordering is not observable in the Level-1 data, so trade
        records store the common inventory/cash state before and after the
        whole bucket rather than attributing an arbitrary path to that order.
        """
        if not fills:
            return

        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = float(row["mid_price"])

        inventory_before = float(self.inventory)
        cash_before = float(self.cash)

        net_quantity = 0.0
        cash_change = 0.0

        for fill in fills:
            signed_quantity = float(fill["signed_quantity"])
            price = float(fill["price"])
            fee = float(fill["fee"])

            if not np.isfinite(signed_quantity):
                raise ValueError("signed_quantity must be finite.")
            if not np.isfinite(price) or price <= 0:
                raise ValueError("Execution price must be positive and finite.")
            if not np.isfinite(fee) or fee < 0:
                raise ValueError("Fee must be finite and non-negative.")

            net_quantity += signed_quantity
            cash_change -= signed_quantity * price + fee

        self.inventory += net_quantity
        self.cash += cash_change
        self._check_inventory_limit()

        inventory_after = float(self.inventory)
        cash_after = float(self.cash)

        for fill in fills:
            trade_record = {
                "timestamp": timestamp,
                "side": fill["side"],
                "price": float(fill["price"]),
                "quantity": float(fill["quantity"]),
                "signed_quantity": float(fill["signed_quantity"]),
                "fee": float(fill["fee"]),
                "p_fill": float(fill["p_fill"]),
                "distance_ticks": float(fill["distance_ticks"]),
                "mid_price": mid_price,
                "inventory_before_bucket": inventory_before,
                "inventory_after_bucket": inventory_after,
                "cash_before_bucket": cash_before,
                "cash_after_bucket": cash_after,
                "is_terminal_unwind": False,
            }
            self.trades.append(trade_record)

    def mark_to_market(self, mid_price: float) -> float:
        """
        Mark the portfolio to the current mid-price:

            equity = cash + inventory * mid_price.
        """
        mid_price = float(mid_price)

        if not np.isfinite(mid_price) or mid_price <= 0:
            raise ValueError("mid_price must be positive and finite.")

        self.pnl = self.cash + self.inventory * mid_price
        return float(self.pnl)

    def step(self, row) -> Dict[str, object]:
        """
        Run one strategy step:
        1. apply risk permissions,
        2. compute passive quotes,
        3. simulate fills,
        4. update cash/inventory,
        5. mark to market,
        6. store diagnostics.
        """
        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = float(row["mid_price"])

        inventory_before = float(self.inventory)

        permissions = should_quote(
            row=row,
            inventory=self.inventory,
            config=self.config,
        )

        quotes = None
        fills: List[Dict[str, float]] = []

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

        state: Dict[str, object] = {
            "timestamp": timestamp,
            "mid_price": mid_price,
            "inventory_before": inventory_before,
            "inventory": float(self.inventory),
            "cash": float(self.cash),
            "pnl": pnl,
            "quote_bid": bool(permissions["quote_bid"]),
            "quote_ask": bool(permissions["quote_ask"]),
            "risk_reason": permissions["reason"],
            "n_fills": len(fills),
            "net_fill_quantity": float(
                sum(fill["signed_quantity"] for fill in fills)
            ),
        }

        if quotes is not None:
            state.update(
                {
                    "reservation_price": quotes["reservation_price"],
                    "optimal_spread": quotes["optimal_spread"],
                    "inventory_risk_spread": quotes[
                        "inventory_risk_spread"
                    ],
                    "execution_spread": quotes["execution_spread"],
                    "half_spread": quotes["half_spread"],
                    "imbalance_skew": quotes["imbalance_skew"],
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
                    "inventory_risk_spread": np.nan,
                    "execution_spread": np.nan,
                    "half_spread": np.nan,
                    "imbalance_skew": np.nan,
                    "bid_price": np.nan,
                    "ask_price": np.nan,
                    "quote_spread": np.nan,
                }
            )

        self.state_history.append(state)
        return state

    def liquidate_inventory(self, row) -> Optional[Dict[str, object]]:
        """
        Liquidate residual inventory at the observed terminal BBO.

        Long inventory is sold at the best bid.
        Short inventory is bought back at the best ask.
        """
        timestamp = row["timestamp"] if "timestamp" in row else None
        mid_price = float(row["mid_price"])

        inventory_before = float(self.inventory)

        if abs(inventory_before) <= 1e-12:
            self.inventory = 0.0
            self.mark_to_market(mid_price)
            return None

        if inventory_before > 0:
            side = "sell"
            price = float(row["best_bid_price"])
            quantity = inventory_before
            signed_quantity = -quantity
        else:
            side = "buy"
            price = float(row["best_ask_price"])
            quantity = -inventory_before
            signed_quantity = quantity

        if price <= 0 or not np.isfinite(price):
            raise ValueError("Invalid terminal liquidation price.")

        fee = float(self.config.fee_rate) * quantity * price

        cash_before = float(self.cash)
        self.cash -= signed_quantity * price + fee

        # The liquidation quantity is exactly the outstanding inventory.
        # Set to zero explicitly to avoid floating-point residuals.
        self.inventory = 0.0

        trade_record: Dict[str, object] = {
            "timestamp": timestamp,
            "side": side,
            "price": price,
            "quantity": float(quantity),
            "signed_quantity": float(signed_quantity),
            "fee": float(fee),
            "p_fill": 1.0,
            "distance_ticks": np.nan,
            "mid_price": mid_price,
            "inventory_before_bucket": inventory_before,
            "inventory_after_bucket": 0.0,
            "cash_before_bucket": cash_before,
            "cash_after_bucket": float(self.cash),
            "is_terminal_unwind": True,
        }

        self.trades.append(trade_record)
        self.mark_to_market(mid_price)

        return trade_record
