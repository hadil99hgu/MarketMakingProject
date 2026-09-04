# src/execution.py

from typing import Dict, List, Optional

import numpy as np


def compute_distance_in_ticks(
    quote_price: float,
    mid_price: float,
    tick_size: float,
    side: str,
) -> float:
    """
    Outward quote distance from the mid-price, in ticks.

    In the Avellaneda-Stoikov intensity model,

        delta_bid = mid - bid,
        delta_ask = ask - mid.

    If a passive quote improves through the mid in a wide-spread market,
    the reduced-form model treats it as distance zero rather than making
    the fill probability decrease again through an absolute value.
    """
    if not np.isfinite(quote_price) or not np.isfinite(mid_price):
        raise ValueError("quote_price and mid_price must be finite.")
    if not np.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("tick_size must be finite and strictly positive.")

    if side == "bid":
        distance = max(mid_price - quote_price, 0.0)
    elif side == "ask":
        distance = max(quote_price - mid_price, 0.0)
    else:
        raise ValueError("side must be 'bid' or 'ask'.")

    return float(distance / tick_size)


def compute_fill_probability(
    quote_price: float,
    mid_price: float,
    config,
    side: str,
) -> float:
    """
    Compute the execution probability over one backtest step.

    Preferred parameterization, when available:

        lambda(d) = A * exp(-kappa_ticks * d),
        p_step(d) = 1 - exp(-lambda(d) * dt).

    This is consistent with the Avellaneda-Stoikov arrival-intensity model.

    For backward compatibility, if A and kappa_ticks are not yet present
    in the configuration, use the calibrated one-step approximation

        p_step(d) = p0 * exp(-k_fill * d).
    """
    distance_ticks = compute_distance_in_ticks(
        quote_price=quote_price,
        mid_price=mid_price,
        tick_size=config.tick_size,
        side=side,
    )

    arrival_rate_A = getattr(config, "arrival_rate_A", None)
    kappa_ticks = getattr(config, "kappa_ticks", None)

    if (
        arrival_rate_A is not None
        and kappa_ticks is not None
        and np.isfinite(arrival_rate_A)
        and np.isfinite(kappa_ticks)
    ):
        arrival_rate_A = float(arrival_rate_A)
        kappa_ticks = float(kappa_ticks)

        if arrival_rate_A < 0 or kappa_ticks < 0:
            raise ValueError(
                "arrival_rate_A and kappa_ticks must be non-negative."
            )

        step_seconds = float(
            getattr(config, "execution_step_seconds", 1.0)
        )
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError(
                "execution_step_seconds must be strictly positive."
            )

        intensity = arrival_rate_A * np.exp(
            -kappa_ticks * distance_ticks
        )
        p_fill = -np.expm1(-intensity * step_seconds)

    else:
        p0 = float(config.p0)
        k_fill = float(config.k_fill)

        if not np.isfinite(p0) or not 0.0 <= p0 <= 1.0:
            raise ValueError("p0 must lie in [0, 1].")
        if not np.isfinite(k_fill) or k_fill < 0:
            raise ValueError("k_fill must be non-negative.")

        p_fill = p0 * np.exp(-k_fill * distance_ticks)

    return float(np.clip(p_fill, 0.0, 1.0))


def _validate_passive_quote(
    row,
    quote_price: float,
    side: str,
) -> None:
    """
    Ensure the order is passive at the observed BBO.

    The current execution model is a maker-fill model. A marketable order
    must not be sent through it.
    """
    best_bid = float(row["best_bid_price"])
    best_ask = float(row["best_ask_price"])

    if best_bid >= best_ask:
        raise ValueError("Observed BBO is locked or crossed.")

    if side == "bid" and quote_price >= best_ask:
        raise ValueError(
            "Bid quote is marketable and cannot use maker execution."
        )

    if side == "ask" and quote_price <= best_bid:
        raise ValueError(
            "Ask quote is marketable and cannot use maker execution."
        )


def _simulate_fill(
    row,
    quote_price: float,
    side: str,
    config,
    rng: np.random.Generator,
) -> Optional[Dict[str, float]]:
    """Simulate one passive bid or ask fill."""
    _validate_passive_quote(
        row=row,
        quote_price=quote_price,
        side=side,
    )

    mid_price = float(row["mid_price"])

    p_fill = compute_fill_probability(
        quote_price=quote_price,
        mid_price=mid_price,
        config=config,
        side=side,
    )

    distance_ticks = compute_distance_in_ticks(
        quote_price=quote_price,
        mid_price=mid_price,
        tick_size=config.tick_size,
        side=side,
    )

    if rng.random() >= p_fill:
        return None

    quantity = float(config.base_order_size)
    fee_rate = float(config.fee_rate)

    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("base_order_size must be strictly positive.")
    if not np.isfinite(fee_rate) or fee_rate < 0:
        raise ValueError("fee_rate must be non-negative.")

    signed_quantity = quantity if side == "bid" else -quantity
    fee = fee_rate * quantity * quote_price

    return {
        "side": "buy" if side == "bid" else "sell",
        "price": float(quote_price),
        "quantity": quantity,
        "signed_quantity": signed_quantity,
        "fee": float(fee),
        "p_fill": p_fill,
        "distance_ticks": distance_ticks,
        "mid_at_quote": mid_price,
    }


def simulate_bid_fill(
    row,
    bid_price: float,
    config,
    rng: np.random.Generator,
) -> Optional[Dict[str, float]]:
    """Simulate a passive bid fill."""
    return _simulate_fill(
        row=row,
        quote_price=bid_price,
        side="bid",
        config=config,
        rng=rng,
    )


def simulate_ask_fill(
    row,
    ask_price: float,
    config,
    rng: np.random.Generator,
) -> Optional[Dict[str, float]]:
    """Simulate a passive ask fill."""
    return _simulate_fill(
        row=row,
        quote_price=ask_price,
        side="ask",
        config=config,
        rng=rng,
    )


def simulate_fills(
    row,
    quotes: Dict[str, float],
    permissions: Dict[str, object],
    config,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """
    Simulate passive bid and ask fills for one one-second backtest step.

    Conditional on the observed state, the two Bernoulli events are treated
    as independent. Both sides can therefore fill within the same one-second
    interval. This is a reduced-form assumption necessitated by 1 Hz Level-1
    data; an event-level queue model would require richer data.
    """
    fills = []

    if bool(permissions["quote_bid"]):
        bid_fill = simulate_bid_fill(
            row=row,
            bid_price=float(quotes["bid_price"]),
            config=config,
            rng=rng,
        )
        if bid_fill is not None:
            fills.append(bid_fill)

    if bool(permissions["quote_ask"]):
        ask_fill = simulate_ask_fill(
            row=row,
            ask_price=float(quotes["ask_price"]),
            config=config,
            rng=rng,
        )
        if ask_fill is not None:
            fills.append(ask_fill)

    return fills
