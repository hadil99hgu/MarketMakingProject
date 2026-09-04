# src/risk.py

from typing import Dict

import numpy as np


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def check_spread_condition(row, config) -> bool:
    """
    Optional market-regime filter on the observed BBO spread.

    This is not a profitability test: transaction fees are accounted for
    separately on executed trades.
    """
    spread_bps = row["spread_bps"]

    return (
        _finite(spread_bps)
        and _finite(config.min_spread_bps)
        and float(spread_bps) >= float(config.min_spread_bps)
    )


def check_volatility_condition(row, config) -> bool:
    """
    Avoid quoting when relative short-horizon volatility is unavailable
    (warm-up) or above the calibrated regime threshold.
    """
    volatility_column = (
        "rolling_log_volatility"
        if "rolling_log_volatility" in row.index
        else "rolling_volatility"
    )

    volatility = row[volatility_column]

    return (
        _finite(volatility)
        and _finite(config.max_volatility)
        and float(volatility) <= float(config.max_volatility)
    )


def check_depth_condition(row, config) -> bool:
    """Avoid quoting when displayed Level-1 depth is unavailable or too low."""
    total_depth = row["total_depth"]

    return (
        _finite(total_depth)
        and _finite(config.min_depth)
        and float(total_depth) >= float(config.min_depth)
    )


def get_inventory_quote_permissions(
    inventory: float,
    config,
) -> Dict[str, bool]:
    """
    Decide which sides can be quoted without violating the hard inventory
    limit after one full fill.

    If a bid fills:
        Q_new = Q + order_size.

    If an ask fills:
        Q_new = Q - order_size.
    """
    if not _finite(inventory):
        return {"quote_bid": False, "quote_ask": False}

    inventory = float(inventory)
    limit = float(config.inventory_limit)
    order_size = float(config.base_order_size)

    if limit <= 0:
        raise ValueError("inventory_limit must be strictly positive.")
    if order_size <= 0:
        raise ValueError("base_order_size must be strictly positive.")

    tolerance = 1e-12

    quote_bid = (
        inventory + order_size
        <= limit + tolerance
    )
    quote_ask = (
        inventory - order_size
        >= -limit - tolerance
    )

    return {
        "quote_bid": bool(quote_bid),
        "quote_ask": bool(quote_ask),
    }


def get_imbalance_quote_permissions(row, config) -> Dict[str, bool]:
    """
    Optional hard toxicity gate based on Level-1 imbalance.

    Imbalance already enters the quote engine continuously through the
    microprice skew. A second hard gate is therefore disabled by default
    unless config.enable_imbalance_filter is explicitly set to True.
    """
    enabled = bool(
        getattr(config, "enable_imbalance_filter", False)
    )

    if not enabled:
        return {
            "quote_bid": True,
            "quote_ask": True,
        }

    imbalance = row["imbalance"]
    threshold = float(config.imbalance_threshold)

    if not _finite(imbalance):
        return {
            "quote_bid": False,
            "quote_ask": False,
        }

    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "imbalance_threshold must lie in (0, 1]."
        )

    imbalance = float(imbalance)

    if imbalance > threshold:
        # Upward pressure: avoid passive selling.
        return {
            "quote_bid": True,
            "quote_ask": False,
        }

    if imbalance < -threshold:
        # Downward pressure: avoid passive buying.
        return {
            "quote_bid": False,
            "quote_ask": True,
        }

    return {
        "quote_bid": True,
        "quote_ask": True,
    }


def should_quote(row, inventory: float, config) -> Dict[str, object]:
    """
    Apply the market-regime and side-specific risk controls.

    Output:
        quote_bid: whether a passive bid may be placed
        quote_ask: whether a passive ask may be placed
        reason: compact diagnostic label
    """
    if not check_spread_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "spread_filter",
        }

    if not check_volatility_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "volatility_filter_or_warmup",
        }

    if not check_depth_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "depth_filter",
        }

    inventory_permissions = get_inventory_quote_permissions(
        inventory=inventory,
        config=config,
    )

    imbalance_permissions = get_imbalance_quote_permissions(
        row=row,
        config=config,
    )

    quote_bid = (
        inventory_permissions["quote_bid"]
        and imbalance_permissions["quote_bid"]
    )
    quote_ask = (
        inventory_permissions["quote_ask"]
        and imbalance_permissions["quote_ask"]
    )

    if quote_bid and quote_ask:
        reason = "ok"
    elif not quote_bid and not quote_ask:
        reason = "both_sides_blocked"
    elif not quote_bid:
        reason = "bid_blocked"
    else:
        reason = "ask_blocked"

    return {
        "quote_bid": bool(quote_bid),
        "quote_ask": bool(quote_ask),
        "reason": reason,
    }
