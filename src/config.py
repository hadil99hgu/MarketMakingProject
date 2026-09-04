# src/config.py

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =========================
# Market data collection
# =========================

SYMBOL = "BTCUSDT"
N_OBSERVATIONS = 20_000

# Target start-to-start collection interval.
SLEEP_SECONDS = 1.0

BASE_URL = "https://api.binance.com"
BOOK_TICKER_ENDPOINT = "/api/v3/ticker/bookTicker"
BOOK_TICKER_URL = (
    f"{BASE_URL}{BOOK_TICKER_ENDPOINT}?symbol={SYMBOL}"
)

OUTPUT_CSV = "data/raw/Largebookticker.csv"


@dataclass
class StrategyConfig:
    """
    Configuration for the corrected Avellaneda-Stoikov-inspired strategy.

    Units used throughout the project:
        price: USDT/BTC
        inventory/order size: BTC
        time: seconds
        sigma_price: USDT/BTC/sqrt(second)
        gamma: 1/USDT
        kappa: BTC/USDT
    """

    # -------------------------
    # Market
    # -------------------------
    symbol: str = "BTCUSDT"
    tick_size: float = 0.01

    # -------------------------
    # Trading costs
    # -------------------------
    fee_rate: float = 0.0002

    # -------------------------
    # Avellaneda-Stoikov
    # -------------------------
    # CARA risk-aversion coefficient, in 1/USDT.
    gamma: float = 0.1

    # Exponential intensity decay per unit of BTCUSDT price distance.
    # Placeholder before empirical calibration.
    kappa: float = 1.5

    # Receding A-S inventory-risk horizon, in seconds.
    horizon_seconds: float = 1.0

    # -------------------------
    # Reduced-form execution
    # -------------------------
    # Empirical touch-probability calibration horizon, measured in
    # observations. With the existing data this is approximately seconds.
    fill_horizon: int = 5

    # Nominal duration represented by one observation.
    seconds_per_observation: float = 1.0

    # Baseline Bernoulli execution step used by the calibrated fallback
    # probability curve.
    execution_step_seconds: float = 1.0

    # Backward-compatible one-step probability approximation:
    #     p(d) = p0 * exp(-k_fill * d_ticks).
    p0: float = 0.5
    k_fill: float = 1.0

    # Preferred intensity parameterization after calibration:
    #     lambda(d) = A * exp(-kappa_ticks * d_ticks).
    arrival_rate_A: Optional[float] = None
    kappa_ticks: Optional[float] = None

    # -------------------------
    # Inventory
    # -------------------------
    base_order_size: float = 0.001
    inventory_limit: float = 0.01

    # -------------------------
    # Market-regime filters
    # -------------------------
    # Disabled by default. Fees are handled on executed trades and should
    # not be conflated with the observed BBO spread.
    min_spread_bps: float = 0.0

    # These are overwritten by calibration.
    max_volatility: float = np.inf
    min_depth: float = 0.0

    # -------------------------
    # Imbalance / microprice
    # -------------------------
    # alpha_imbalance = 1 uses the full Level-1 microprice displacement.
    alpha_imbalance: float = 1.0

    # Hard imbalance blocking is optional because imbalance is already used
    # continuously through the microprice skew.
    enable_imbalance_filter: bool = False
    imbalance_threshold: float = 0.4

    # -------------------------
    # Reproducibility
    # -------------------------
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be non-empty.")

        if not np.isfinite(self.tick_size) or self.tick_size <= 0:
            raise ValueError("tick_size must be strictly positive.")

        if not np.isfinite(self.fee_rate) or self.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative.")

        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be strictly positive.")

        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be strictly positive.")

        if (
            not np.isfinite(self.horizon_seconds)
            or self.horizon_seconds <= 0
        ):
            raise ValueError(
                "horizon_seconds must be strictly positive."
            )

        if self.fill_horizon < 1:
            raise ValueError("fill_horizon must be at least 1.")

        if (
            not np.isfinite(self.seconds_per_observation)
            or self.seconds_per_observation <= 0
        ):
            raise ValueError(
                "seconds_per_observation must be strictly positive."
            )

        if (
            not np.isfinite(self.execution_step_seconds)
            or self.execution_step_seconds <= 0
        ):
            raise ValueError(
                "execution_step_seconds must be strictly positive."
            )

        if not np.isfinite(self.p0) or not 0.0 <= self.p0 <= 1.0:
            raise ValueError("p0 must lie in [0, 1].")

        if not np.isfinite(self.k_fill) or self.k_fill < 0:
            raise ValueError("k_fill must be non-negative.")

        if self.arrival_rate_A is not None:
            if (
                not np.isfinite(self.arrival_rate_A)
                or self.arrival_rate_A < 0
            ):
                raise ValueError(
                    "arrival_rate_A must be non-negative or None."
                )

        if self.kappa_ticks is not None:
            if (
                not np.isfinite(self.kappa_ticks)
                or self.kappa_ticks < 0
            ):
                raise ValueError(
                    "kappa_ticks must be non-negative or None."
                )

        if (
            not np.isfinite(self.base_order_size)
            or self.base_order_size <= 0
        ):
            raise ValueError(
                "base_order_size must be strictly positive."
            )

        if (
            not np.isfinite(self.inventory_limit)
            or self.inventory_limit <= 0
        ):
            raise ValueError(
                "inventory_limit must be strictly positive."
            )

        if self.base_order_size > self.inventory_limit:
            raise ValueError(
                "base_order_size cannot exceed inventory_limit."
            )

        if (
            not np.isfinite(self.min_spread_bps)
            or self.min_spread_bps < 0
        ):
            raise ValueError(
                "min_spread_bps must be non-negative."
            )

        if (
            not np.isfinite(self.min_depth)
            or self.min_depth < 0
        ):
            raise ValueError("min_depth must be non-negative.")

        if (
            not np.isfinite(self.alpha_imbalance)
            or self.alpha_imbalance < 0
        ):
            raise ValueError(
                "alpha_imbalance must be non-negative."
            )

        if not 0.0 < self.imbalance_threshold <= 1.0:
            raise ValueError(
                "imbalance_threshold must lie in (0, 1]."
            )

    @property
    def lambda_as(self) -> float:
        """
        Backward-compatible alias for older project code/report terminology.

        New code should use `gamma`.
        """
        return self.gamma

    @property
    def horizon(self) -> float:
        """
        Backward-compatible alias.

        New code should use `horizon_seconds`.
        """
        return self.horizon_seconds
