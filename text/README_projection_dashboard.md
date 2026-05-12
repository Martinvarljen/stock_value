# AI Stock Projection Dashboard

## Goal

Build a minimalist AI-assisted stock projection dashboard that combines:

- valuation analysis
- momentum/technical analysis
- risk analysis
- optional news sentiment
- projection lines on a live chart

The system is NOT intended to predict exact prices.

The main goal is to estimate:

- probability that price will be higher in 20 / 60 / 120 trading days
- expected return over those horizons
- bullish/base/bearish future scenarios

---

# Existing Architecture

Current engines already implemented:

- valuation_engine.py
- momentum_engine.py
- quality_engine.py
- growth_engine.py
- risk_engine.py
- scenario_engine.py
- peer_engine.py
- classification_engine.py
- backtest_engine.py
- explanation_engine.py

The system already calculates:
- fair value
- upside/downside
- quality/risk metrics
- momentum metrics
- classification (BUY / HOLD / AVOID)

---

# New Goal

Add a new module:

```text
projection_engine.py
```

This module should:
- estimate probabilities of future upside
- estimate expected returns
- generate future projection paths
- output minimalist prediction summaries
- later support live chart overlays

---

# Core Prediction Outputs

The model should estimate:

```python
{
    "p_up_20d": 0.64,
    "p_up_60d": 0.71,
    "p_up_120d": 0.78,

    "expected_return_20d": 0.042,
    "expected_return_60d": 0.115,
    "expected_return_120d": 0.198,

    "signal": "BULLISH",
    "confidence": "MEDIUM_HIGH"
}
```

---

# Phase 1 (No AI Yet)

First version should NOT use neural networks.

Use a weighted scoring approach based on:
- valuation upside
- momentum
- RSI
- trend strength
- volatility
- quality score
- risk score
- macro trend

Goal:
Generate:
- probabilities
- expected returns
- bullish/base/bearish paths

without ML training initially.

---

# Projection Line Logic

Projection lines should NOT pretend to predict exact candles.

Instead:
- build smooth scenario paths
- gradually move price toward estimated fair value
- adjust curve speed using momentum/trend

Example logic:

```python
expected_price_t =
    current_price +
    (fair_value - current_price)
    * (1 - exp(-speed * t))
```

Where:
- speed increases with bullish momentum
- speed decreases with bearish momentum
- volatility widens projection channels

---

# Projection Paths

Generate 3 future paths:

## Bull Path
Optimistic scenario.

## Base Path
Most likely scenario.

## Bear Path
Risk scenario.

---

# Example Dashboard Output

```text
AAPL
--------------------------------

P(up 20d):   64%
P(up 60d):   71%
P(up 120d):  78%

Expected Return 20d:   +4.2%
Expected Return 60d:   +11.5%
Expected Return 120d:  +19.8%

Signal: BULLISH
Confidence: MEDIUM-HIGH
```

---

# Live Chart Goal

Create a live dashboard showing:
- candlestick chart
- projection lines
- volatility channels
- fair value target
- probabilities
- signal strength

---

# Recommended Stack

## Backend
- Python

## Dashboard
- Streamlit

## Charting
- Plotly

## Data APIs
Choose one:
- yfinance (prototype)
- Polygon
- Alpaca
- Finnhub
- TwelveData

---

# Chart Features

The dashboard should support:
- live updates
- ticker search
- zooming
- multiple projection lines
- volatility bands
- technical indicators
- auto refresh

---

# Streamlit Architecture

## Main Flow

```text
ticker input
    ->
download market data
    ->
run valuation engine
    ->
run projection engine
    ->
draw live chart
```

---

# Future ML Upgrade

Later phases should train ML models on historical data.

Recommended models:
- XGBoost
- LightGBM
- RandomForest

Avoid neural networks initially.

---

# Training Dataset

Need:
- 5–10 years daily OHLCV data
- technical indicators
- valuation metrics
- quality/risk metrics
- macro context

Targets:

```python
future_return_20d
future_return_60d
future_return_120d
```

and:

```python
target_up_20d
target_up_60d
target_up_120d
```

---

# Future Improvements

Potential future additions:
- news sentiment analysis
- earnings analysis
- macro indicators
- VIX
- sector rotation
- insider trading data
- options flow
- AI explanation engine

---

# Immediate Next Steps

## Step 1
Create:
```text
projection_engine.py
```

## Step 2
Generate:
- probabilities
- expected returns
- projection paths

## Step 3
Create Streamlit dashboard.

## Step 4
Add live chart overlays.

## Step 5
Train ML model later.

---

# Philosophy

The system is NOT intended to:
- predict exact candles
- scalp intraday noise
- behave like a magical AI oracle

The system IS intended to:
- estimate probabilities
- visualize scenarios
- support investment decisions
- combine valuation with technicals
- provide clean visual guidance
