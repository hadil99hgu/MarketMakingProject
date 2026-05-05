# src/risk.py


def check_spread_condition(row, config) -> bool:
    """
    Check whether the market spread is large enough to quote.

    If the spread is too small, fees and adverse selection may destroy the edge.
    """

    return row["spread_bps"] >= config.min_spread_bps


def check_volatility_condition(row, config) -> bool:
    """
    Avoid quoting when short-term volatility is too high.
    """

    return row["rolling_volatility"] <= config.max_volatility


def check_depth_condition(row, config) -> bool:
    """
    Avoid quoting when top-of-book liquidity is too thin.
    """

    return row["total_depth"] >= config.min_depth


def check_inventory_condition(inventory: float, config) -> bool:
    """
    Check whether inventory is within the hard inventory limit.
    """

    return abs(inventory) <= config.inventory_limit


def get_inventory_quote_permissions(inventory: float, config) -> dict:
    """
    Decide which side we are allowed to quote based on inventory.

    If we are too long:
        we should stop buying
        we should only sell

    If we are too short:
        we should stop selling
        we should only buy
    """

    quote_bid = True
    quote_ask = True

    if inventory >= config.inventory_limit:
        quote_bid = False
        quote_ask = True

    elif inventory <= -config.inventory_limit:
        quote_bid = True
        quote_ask = False

    return {
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
    }


def get_imbalance_quote_permissions(row, config) -> dict:
    """
    Decide whether imbalance is too toxic on one side.

    Positive imbalance:
        price may go up
        selling at the ask is risky

    Negative imbalance:
        price may go down
        buying at the bid is risky
    """

    imbalance = row["imbalance"]

    quote_bid = True
    quote_ask = True

    if imbalance > config.imbalance_threshold:
        # Upward pressure: avoid selling too aggressively
        quote_ask = False

    elif imbalance < -config.imbalance_threshold:
        # Downward pressure: avoid buying too aggressively
        quote_bid = False

    return {
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
    }


def should_quote(row, inventory: float, config) -> dict:
    """
    Main risk filter.

    Output:
        quote_bid: whether we are allowed to place a bid
        quote_ask: whether we are allowed to place an ask
        reason: explanation for debugging
    """

    if not check_spread_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "spread_too_small",
        }

    if not check_volatility_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "volatility_too_high",
        }

    if not check_depth_condition(row, config):
        return {
            "quote_bid": False,
            "quote_ask": False,
            "reason": "depth_too_low",
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
        inventory_permissions["quote_bid"] and imbalance_permissions["quote_bid"]
    )

    quote_ask = (
        inventory_permissions["quote_ask"] and imbalance_permissions["quote_ask"]
    )

    if not quote_bid and not quote_ask:
        reason = "both_sides_blocked"
    elif not quote_bid:
        reason = "bid_blocked"
    elif not quote_ask:
        reason = "ask_blocked"
    else:
        reason = "ok"

    return {
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
        "reason": reason,
    }



