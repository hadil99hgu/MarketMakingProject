# src/quote_engine.py

import numpy as np


def round_down_to_tick(price: float, tick_size: float) -> float:
    """
    Round price down to the nearest tick.
    Used for bid prices.
    """
    return np.floor(price / tick_size) * tick_size


def round_up_to_tick(price: float, tick_size: float) -> float:
    """
    Round price up to the nearest tick.
    Used for ask prices.
    """
    return np.ceil(price / tick_size) * tick_size


def compute_reservation_price(
    mid_price: float,
    inventory: float,
    sigma: float,
    gamma: float,
    horizon: float,
) -> float:
    """
    Avellaneda-Stoikov reservation price.

    If inventory > 0, we are long, so reservation price decreases.
    If inventory < 0, we are short, so reservation price increases.
    """

    reservation_price = mid_price - inventory * gamma * sigma**2 * horizon

    return reservation_price


def compute_optimal_spread(
    sigma: float,
    gamma: float,
    kappa: float,
    horizon: float,
) -> float:
    """
    Avellaneda-Stoikov optimal total spread.
    """

    inventory_risk_term = gamma * sigma**2 * horizon

    execution_term = (2.0 / gamma) * np.log(1.0 + gamma / kappa)

    optimal_spread = inventory_risk_term + execution_term

    return optimal_spread


def compute_avellaneda_quotes(
    mid_price: float,
    inventory: float,
    sigma: float,
    config,
) -> dict:
    """
    Compute raw Avellaneda bid and ask quotes.
    """

    gamma = config.lambda_as
    kappa = config.kappa
    horizon = config.horizon

    reservation_price = compute_reservation_price(
        mid_price=mid_price,
        inventory=inventory,
        sigma=sigma,
        gamma=gamma,
        horizon=horizon,
    )

    optimal_spread = compute_optimal_spread(
        sigma=sigma,
        gamma=gamma,
        kappa=kappa,
        horizon=horizon,
    )

    half_spread = optimal_spread / 2.0

    bid_price = reservation_price - half_spread
    ask_price = reservation_price + half_spread

    return {
        "reservation_price": reservation_price,
        "optimal_spread": optimal_spread,
        "half_spread": half_spread,
        "bid_price": bid_price,
        "ask_price": ask_price,
    }


# imbalance tells direction
# spread gives price scale
# alpha controls intensity

def apply_imbalance_skew(
    bid_price: float,
    ask_price: float,
    imbalance: float,
    market_spread: float,
    config,
) -> tuple[float, float]:
    """
    Adjust quotes using order book imbalance.

    Positive imbalance:
        bid side is stronger,
        price may go up,
        we shift quotes upward.

    Negative imbalance:
        ask side is stronger,
        price may go down,
        we shift quotes downward.
    """

    alpha = config.alpha_imbalance

    skew = alpha * imbalance * market_spread

    bid_price = bid_price + skew
    ask_price = ask_price + skew

    return bid_price, ask_price


def round_and_validate_quotes(
    bid_price: float,
    ask_price: float,
    mid_price: float,
    tick_size: float,
) -> tuple[float, float]:
    """
    Round bid and ask prices to the tick size and ensure bid < ask.
    """

    bid_price = round_down_to_tick(bid_price, tick_size)
    ask_price = round_up_to_tick(ask_price, tick_size)

    if bid_price >= ask_price:
        bid_price = round_down_to_tick(mid_price - tick_size, tick_size)
        ask_price = round_up_to_tick(mid_price + tick_size, tick_size)

    return bid_price, ask_price


def compute_final_quotes(
    row,
    inventory: float,
    config,
) -> dict:
    """
    Full quote engine.

    Input:
        row contains mid_price, rolling_volatility, spread, imbalance
        inventory is current inventory
        config contains strategy parameters

    Output:
        final bid and ask quotes
    """

    mid_price = row["mid_price"]
    sigma = row["rolling_volatility"]
    market_spread = row["spread"]
    imbalance = row["imbalance"]

    # Avoid zero volatility in the formula
    sigma = max(sigma, 1e-8)

    quotes = compute_avellaneda_quotes(
        mid_price=mid_price,
        inventory=inventory,
        sigma=sigma,
        config=config,
    )

    bid_price = quotes["bid_price"]
    ask_price = quotes["ask_price"]

    bid_price, ask_price = apply_imbalance_skew(
        bid_price=bid_price,
        ask_price=ask_price,
        imbalance=imbalance,
        market_spread=market_spread,
        config=config,
    )

    bid_price, ask_price = round_and_validate_quotes(
        bid_price=bid_price,
        ask_price=ask_price,
        mid_price=mid_price,
        tick_size=config.tick_size,
    )

    quotes["bid_price"] = bid_price
    quotes["ask_price"] = ask_price
    quotes["quote_spread"] = ask_price - bid_price

    return quotes