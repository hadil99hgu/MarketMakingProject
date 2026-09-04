# src/quote_engine.py

from typing import Dict, Tuple

import numpy as np


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")


def round_down_to_tick(price: float, tick_size: float) -> float:
    """Round a bid price down to the exchange tick grid."""
    _validate_positive("tick_size", tick_size)
    if not np.isfinite(price):
        raise ValueError("price must be finite.")

    return float(np.floor(price / tick_size + 1e-12) * tick_size)


def round_up_to_tick(price: float, tick_size: float) -> float:
    """Round an ask price up to the exchange tick grid."""
    _validate_positive("tick_size", tick_size)
    if not np.isfinite(price):
        raise ValueError("price must be finite.")

    return float(np.ceil(price / tick_size - 1e-12) * tick_size)


def compute_reservation_price(
    mid_price: float,
    inventory: float,
    sigma_price: float,
    gamma: float,
    horizon_seconds: float,
) -> float:
    """
    Avellaneda-Stoikov reservation price in USDT/BTC.

    inventory:
        BTC inventory.

    sigma_price:
        Mid-price volatility in USDT/BTC/sqrt(second).

    gamma:
        CARA risk-aversion parameter in inverse wealth units.

    horizon_seconds:
        Receding quoting/risk horizon in seconds.

    With inventory measured in BTC, the reservation-price shift is

        inventory * gamma * sigma_price^2 * horizon_seconds.
    """
    _validate_positive("mid_price", mid_price)
    _validate_positive("gamma", gamma)
    _validate_positive("horizon_seconds", horizon_seconds)

    if not np.isfinite(inventory):
        raise ValueError("inventory must be finite.")
    if not np.isfinite(sigma_price) or sigma_price < 0:
        raise ValueError("sigma_price must be finite and non-negative.")

    return float(
        mid_price
        - inventory * gamma * sigma_price**2 * horizon_seconds
    )


def compute_optimal_spread(
    sigma_price: float,
    gamma: float,
    kappa: float,
    horizon_seconds: float,
    order_size: float,
) -> Dict[str, float]:
    """
    Avellaneda-Stoikov total spread in USDT/BTC for a finite order size.

    The canonical model is written for one inventory unit. If one inventory
    unit is an order of size z BTC, then, expressed back in BTCUSDT price
    units,

        spread =
            gamma * z * sigma_price^2 * horizon
            + 2 / (gamma * z)
              * log(1 + gamma * z / kappa),

    where kappa is the exponential decay coefficient with respect to
    BTCUSDT price distance.

    In the small-risk-aversion limit, the execution term tends to 2/kappa.
    """
    _validate_positive("gamma", gamma)
    _validate_positive("kappa", kappa)
    _validate_positive("horizon_seconds", horizon_seconds)
    _validate_positive("order_size", order_size)

    if not np.isfinite(sigma_price) or sigma_price < 0:
        raise ValueError("sigma_price must be finite and non-negative.")

    gamma_lot = gamma * order_size

    inventory_risk_term = (
        gamma_lot * sigma_price**2 * horizon_seconds
    )
    execution_term = (
        2.0 / gamma_lot
        * np.log1p(gamma_lot / kappa)
    )

    total_spread = inventory_risk_term + execution_term

    return {
        "optimal_spread": float(total_spread),
        "inventory_risk_spread": float(inventory_risk_term),
        "execution_spread": float(execution_term),
    }


def compute_avellaneda_quotes(
    mid_price: float,
    inventory: float,
    sigma_price: float,
    config,
) -> Dict[str, float]:
    """Compute unrounded Avellaneda-Stoikov bid and ask quotes."""
    gamma = config.lambda_as
    kappa = config.kappa
    horizon_seconds = config.horizon
    order_size = config.base_order_size

    reservation_price = compute_reservation_price(
        mid_price=mid_price,
        inventory=inventory,
        sigma_price=sigma_price,
        gamma=gamma,
        horizon_seconds=horizon_seconds,
    )

    spread_components = compute_optimal_spread(
        sigma_price=sigma_price,
        gamma=gamma,
        kappa=kappa,
        horizon_seconds=horizon_seconds,
        order_size=order_size,
    )

    optimal_spread = spread_components["optimal_spread"]
    half_spread = 0.5 * optimal_spread

    return {
        "reservation_price": reservation_price,
        "optimal_spread": optimal_spread,
        "inventory_risk_spread": spread_components[
            "inventory_risk_spread"
        ],
        "execution_spread": spread_components["execution_spread"],
        "half_spread": half_spread,
        "bid_price": reservation_price - half_spread,
        "ask_price": reservation_price + half_spread,
    }


def compute_microprice(
    best_bid_price: float,
    best_bid_qty: float,
    best_ask_price: float,
    best_ask_qty: float,
) -> float:
    """
    Level-1 microprice.

    More bid depth moves the microprice toward the ask; more ask depth
    moves it toward the bid.
    """
    total_qty = best_bid_qty + best_ask_qty

    if (
        not np.isfinite(total_qty)
        or total_qty <= 0
        or best_bid_price >= best_ask_price
    ):
        raise ValueError("Invalid best bid/ask state for microprice.")

    return float(
        (
            best_ask_price * best_bid_qty
            + best_bid_price * best_ask_qty
        )
        / total_qty
    )


def apply_imbalance_skew(
    bid_price: float,
    ask_price: float,
    mid_price: float,
    best_bid_price: float,
    best_bid_qty: float,
    best_ask_price: float,
    best_ask_qty: float,
    config,
) -> Tuple[float, float, float]:
    """
    Shift both quotes toward the Level-1 microprice.

    For the standard imbalance

        I = (Q_bid - Q_ask) / (Q_bid + Q_ask),

    the exact Level-1 identity is

        microprice - mid = 0.5 * I * market_spread.

    Hence alpha_imbalance = 1 means using the full microprice displacement,
    rather than an arbitrary full-spread imbalance shift.
    """
    microprice = compute_microprice(
        best_bid_price=best_bid_price,
        best_bid_qty=best_bid_qty,
        best_ask_price=best_ask_price,
        best_ask_qty=best_ask_qty,
    )

    skew = config.alpha_imbalance * (microprice - mid_price)

    return (
        float(bid_price + skew),
        float(ask_price + skew),
        float(skew),
    )


def round_and_validate_quotes(
    bid_price: float,
    ask_price: float,
    mid_price: float,
    tick_size: float,
    best_bid_price: float,
    best_ask_price: float,
) -> Tuple[float, float]:
    """
    Round quotes to the tick grid and keep them passive.

    A passive bid must stay at least one tick below the current best ask.
    A passive ask must stay at least one tick above the current best bid.
    Quotes may improve the current BBO when the market spread exceeds one
    tick, but they cannot cross the book.
    """
    _validate_positive("tick_size", tick_size)

    if best_bid_price >= best_ask_price:
        raise ValueError("Current BBO is locked or crossed.")

    bid_price = round_down_to_tick(bid_price, tick_size)
    ask_price = round_up_to_tick(ask_price, tick_size)

    max_passive_bid = round_down_to_tick(
        best_ask_price - tick_size,
        tick_size,
    )
    min_passive_ask = round_up_to_tick(
        best_bid_price + tick_size,
        tick_size,
    )

    bid_price = min(bid_price, max_passive_bid)
    ask_price = max(ask_price, min_passive_ask)

    if bid_price >= ask_price:
        bid_price = round_down_to_tick(
            mid_price - 0.5 * tick_size,
            tick_size,
        )
        ask_price = round_up_to_tick(
            mid_price + 0.5 * tick_size,
            tick_size,
        )

        bid_price = min(bid_price, max_passive_bid)
        ask_price = max(ask_price, min_passive_ask)

    if bid_price >= ask_price:
        raise ValueError("Could not construct valid passive quotes.")

    return float(bid_price), float(ask_price)


def compute_final_quotes(
    row,
    inventory: float,
    config,
) -> Dict[str, float]:
    """
    Full quote engine.

    Required row fields:
        mid_price
        rolling_price_volatility
        best_bid_price
        best_bid_qty
        best_ask_price
        best_ask_qty
    """
    mid_price = float(row["mid_price"])
    sigma_price = float(row["rolling_price_volatility"])

    if not np.isfinite(sigma_price):
        raise ValueError(
            "rolling_price_volatility is unavailable; "
            "the volatility warm-up period should not be quoted."
        )

    quotes = compute_avellaneda_quotes(
        mid_price=mid_price,
        inventory=inventory,
        sigma_price=sigma_price,
        config=config,
    )

    bid_price, ask_price, imbalance_skew = apply_imbalance_skew(
        bid_price=quotes["bid_price"],
        ask_price=quotes["ask_price"],
        mid_price=mid_price,
        best_bid_price=float(row["best_bid_price"]),
        best_bid_qty=float(row["best_bid_qty"]),
        best_ask_price=float(row["best_ask_price"]),
        best_ask_qty=float(row["best_ask_qty"]),
        config=config,
    )

    bid_price, ask_price = round_and_validate_quotes(
        bid_price=bid_price,
        ask_price=ask_price,
        mid_price=mid_price,
        tick_size=config.tick_size,
        best_bid_price=float(row["best_bid_price"]),
        best_ask_price=float(row["best_ask_price"]),
    )

    quotes["bid_price"] = bid_price
    quotes["ask_price"] = ask_price
    quotes["quote_spread"] = ask_price - bid_price
    quotes["imbalance_skew"] = imbalance_skew

    return quotes
