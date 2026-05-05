
from dataclasses import dataclass

# =========================
# Market data configuration
# =========================

SYMBOL = "BTCUSDT"

# =========================
# Collection parameters
# =========================

N_OBSERVATIONS = 20000
SLEEP_SECONDS = 1.0

# =========================
# Binance REST endpoints
# =========================

BASE_URL = "https://api.binance.com"
BOOK_TICKER_ENDPOINT = "/api/v3/ticker/bookTicker"

BOOK_TICKER_URL = f"{BASE_URL}{BOOK_TICKER_ENDPOINT}?symbol={SYMBOL}"

# =========================
# Output
# =========================

OUTPUT_CSV = "data/raw/Largebookticker.csv"

# strategy parameters 


@dataclass
class StrategyConfig:
    """
    Configuration for the Avellaneda-Stoikov market-making strategy.
    """

    # -------------------------
    # Market / data parameters
    # -------------------------
    symbol: str = "BTCUSDT"
    tick_size: float = 0.01

    # -------------------------
    # Trading costs
    # -------------------------
    fee_rate: float = 0.0002

    # -------------------------
    # Avellaneda-Stoikov parameters
    # -------------------------
    lambda_as: float = 0.1
    kappa: float = 1.5
    horizon: float = 1.0

    # -------------------------
    # Execution model parameters
    # -------------------------
    p0: float = 0.5
    k_fill: float = 1.0
    fill_horizon: int = 5

    # -------------------------
    # Inventory parameters
    # -------------------------
    base_order_size: float = 0.001
    Qmax: float = 0.01
    inventory_limit: float = 0.01
    alpha_inventory: float = 2.0

    # -------------------------
    # Microstructure filters
    # -------------------------
    min_spread_bps: float = 1.0
    max_volatility: float = 0.01
    min_depth: float = 0.0

    # -------------------------
    # Imbalance / adverse selection
    # -------------------------
    imbalance_threshold: float = 0.4
    alpha_imbalance: float = 1.0

    # -------------------------
    # Risk management
    # -------------------------
    stop_loss: float = -100.0
    max_drawdown: float = -200.0

    # -------------------------
    # Backtest parameters
    # -------------------------
    seed: int = 42
    # Quote edge filter
    edge_buffer_ticks: float = 1.0