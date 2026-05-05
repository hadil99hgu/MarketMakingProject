# BTCUSDT Market-Making Backtester

This project implements a Python backtesting framework for a BTCUSDT market-making strategy using Binance Spot limit order book data.

The strategy is based on the Avellaneda–Stoikov market-making framework and includes inventory-aware quoting, imbalance-based quote skewing, empirical fill-probability calibration, transaction costs, terminal inventory liquidation, and PnL decomposition.

---

## Project Goal

The goal of this project is to study whether a calibrated market-making strategy can generate positive gross edge on BTCUSDT, and how sensitive this edge is to fees, inventory risk, and execution assumptions.

This is a research backtest, not a production trading system.

---

## Strategy Overview

The strategy follows the following pipeline:

```text
Raw Binance LOB data
        ↓
Data cleaning
        ↓
Microstructure feature engineering
        ↓
Fill probability calibration
        ↓
Avellaneda–Stoikov quote generation
        ↓
Execution simulation
        ↓
PnL decomposition and performance analysis
```

---

## Project Structure

```text
MarketMakingProject/
│
├── data/
│   └── raw/
│       └── Largebookticker.csv
|__ Notebooks/
│
├── outputs/  
|   ├── ShowcasingResults.py
│   └── FirstExploration.py
├── reports/  
|   ├── rapport_market_making_avellaneda_stoikov_btcusdt.pdf
│   └── report_market_making_avellaneda_stoikov_btcusdt_EN.pdf
│
├── scripts/
│   └── run_backtest.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── calibration.py
│   ├── quote_engine.py
│   ├── risk.py
│   ├── execution.py
│   ├── strategy.py
│   ├── backtester.py
│   ├── pnl.py
│   └── metrics.py
│
└── README.md
```

---

## How to Run

```bash
source activate_project.sh
python scripts/run_backtest.py \
  --data-path "data/raw/Largebookticker.csv" \
  --symbol BTCUSDT \
  --tick-size 0.01 \
  --fee-rate 0.0001 \
  --min-spread-bps 0.0
```