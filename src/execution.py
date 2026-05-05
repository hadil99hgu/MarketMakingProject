# src/execution.py

import numpy as np


def compute_distance_in_ticks(
    quote_price: float,
    mid_price: float,
    tick_size: float,
) -> float:
    """
    Compute the distance between a quote and the mid-price in ticks.
    """

    distance = abs(quote_price - mid_price)
    distance_ticks = distance / tick_size

    return distance_ticks


def compute_fill_probability(
    quote_price: float,
    mid_price: float,
    config,
) -> float:
    """
    Simple exponential fill probability.

    The closer the quote is to the mid-price, the higher the probability
    of being filled.
    """

    distance_ticks = compute_distance_in_ticks(
        quote_price=quote_price,
        mid_price=mid_price,
        tick_size=config.tick_size,
    )

    p_fill = config.p0 * np.exp(-config.k_fill * distance_ticks)

    p_fill = float(np.clip(p_fill, 0.0, 1.0))

    return p_fill


def simulate_bid_fill(
    row,
    bid_price: float,
    config,
    rng: np.random.Generator,
) -> dict | None:
    """
    Simulate a bid fill.

    If our bid is filled, we buy base_order_size.
    """

    mid_price = row["mid_price"]

    p_fill = compute_fill_probability(
        quote_price=bid_price,
        mid_price=mid_price,
        config=config,
    )

    is_filled = rng.random() < p_fill

    if not is_filled:
        return None

    quantity = config.base_order_size
    fee = config.fee_rate * quantity * bid_price

    return {
        "side": "buy",
        "price": bid_price,
        "quantity": quantity,
        "signed_quantity": quantity,
        "fee": fee,
        "p_fill": p_fill,
    }


def simulate_ask_fill(
    row,
    ask_price: float,
    config,
    rng: np.random.Generator,
) -> dict | None:
    """
    Simulate an ask fill.

    If our ask is filled, we sell base_order_size.
    """

    mid_price = row["mid_price"]

    p_fill = compute_fill_probability(
        quote_price=ask_price,
        mid_price=mid_price,
        config=config,
    )

    is_filled = rng.random() < p_fill

    if not is_filled:
        return None

    quantity = config.base_order_size
    fee = config.fee_rate * quantity * ask_price

    return {
        "side": "sell",
        "price": ask_price,
        "quantity": quantity,
        "signed_quantity": -quantity,
        "fee": fee,
        "p_fill": p_fill,
    }


def simulate_fills(
    row,
    quotes: dict,
    permissions: dict,
    config,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Simulate bid and ask fills.

    permissions tells us whether we are allowed to quote bid and/or ask.
    """

    fills = []

    if permissions["quote_bid"]:
        bid_fill = simulate_bid_fill(
            row=row,
            bid_price=quotes["bid_price"],
            config=config,
            rng=rng,
        )

        if bid_fill is not None:
            fills.append(bid_fill)

    if permissions["quote_ask"]:
        ask_fill = simulate_ask_fill(
            row=row,
            ask_price=quotes["ask_price"],
            config=config,
            rng=rng,
        )

        if ask_fill is not None:
            fills.append(ask_fill)

    return fills