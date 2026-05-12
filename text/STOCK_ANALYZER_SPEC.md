# STOCK ANALYZER — Full Project Specification

## Overview

Build a single-file Python stock evaluation tool (`stock_analyzer.py`) that:
- Takes a list of EU-exchange tickers (e.g., `["OR.PA", "SAP.DE", "ASML.AS", "NESN.SW", "NOVO-B.CO"]`)
- Pulls all data from `yfinance` (free, no API key)
- Scores each stock on business quality, financial strength, valuation, growth, and risk
- Runs a 3-scenario DCF (bear / base / bull)
- Classifies each stock as **Ideal / Medium / Risky**
- Calculates a buy-below price using margin of safety
- Adds a momentum/technical timing overlay
- Compares stocks to sector peers
- Generates a plain-English explanation for every classification
- Outputs everything to a single `.xlsx` Excel workbook with multiple sheets
- Supports batch mode: analyze 10-20 tickers at once, ranked by composite score

The entire codebase lives in ONE Python file for simplicity.

---

## Tech Stack

- Python 3.10+
- `yfinance` — all financial data
- `openpyxl` — Excel output with formatting
- `numpy` / `pandas` — calculations
- No paid APIs. No API keys.

---

## EU Ticker Format

European tickers in yfinance use the format `SYMBOL.EXCHANGE`:
- `.PA` — Euronext Paris (e.g., `OR.PA` for L'Oréal)
- `.DE` — Frankfurt / XETRA (e.g., `SAP.DE`)
- `.AS` — Euronext Amsterdam (e.g., `ASML.AS`)
- `.SW` — SIX Swiss Exchange (e.g., `NESN.SW`)
- `.CO` — Copenhagen (e.g., `NOVO-B.CO`)
- `.MI` — Milan (e.g., `ENI.MI`)
- `.MC` — Madrid (e.g., `SAN.MC`)
- `.L`  — London (e.g., `SHEL.L`, prices in GBp — divide by 100 for GBP)
- `.HE` — Helsinki (e.g., `NOKIA.HE`)
- `.ST` — Stockholm (e.g., `VOLV-B.ST`)
- `.OL` — Oslo (e.g., `EQNR.OL`)
- `.BR` — Brussels (e.g., `ABI.BR`)
- `.LS` — Lisbon
- `.VI` — Vienna
- `.IR` — Dublin

Currency: all values stay in the stock's native currency. Label the currency in the Excel output.

---

## Architecture — 6 Engines + Output

### 1. DATA LAYER (`collect_data`)

For each ticker, pull from `yfinance`:

**Price & Identity:**
- current price, currency, market cap, shares outstanding
- company name, sector, industry, exchange
- 52-week high / low
- beta

**Income Statement (annual, last 5 years where available):**
- revenue
- gross profit
- operating income (EBIT)
- net income
- EPS (basic)
- interest expense

**Balance Sheet (annual, last 5 years):**
- total assets
- total liabilities
- total debt (long-term + short-term)
- cash and equivalents
- total stockholders equity
- book value per share
- shares outstanding history (for dilution check)

**Cash Flow (annual, last 5 years):**
- operating cash flow
- capital expenditure
- free cash flow (OCF - capex)
- dividends paid

**Derived Ratios (calculate from raw data):**
- gross margin, operating margin, net margin
- ROE, ROA, ROIC
- debt/equity
- net debt / EBITDA
- interest coverage (EBIT / interest expense)
- current ratio
- FCF yield
- dividend yield
- P/E, forward P/E
- EV/EBIT, EV/EBITDA, EV/FCF
- P/B (price to book)
- PEG ratio

**Price History:**
- 1 year of daily prices (for 200-day MA, 50-day MA, RSI)
- 5 years of monthly prices (for historical P/E range)

**Data Quality:**
- Track which fields are missing or stale
- Add a `data_quality_score` (0-100) per stock
- Flag stocks with >30% missing critical fields

---

### 2. QUALITY ENGINE (`score_quality`)

**Business Quality Score (0-100), weight: 25% of total**

Sub-scores:
- Revenue growth consistency (5yr): are revenues growing steadily or erratic?
  - 5yr CAGR > 8% = high, 3-8% = medium, <3% or negative = low
  - Consistency: count years of positive growth out of last 5
- Earnings consistency: how many of last 5 years had positive net income?
- FCF consistency: how many of last 5 years had positive free cash flow?
- Gross margin level: >50% = strong, 30-50% = decent, <30% = weak
- Operating margin level: >20% = strong, 10-20% = decent, <10% = weak
- ROIC: >15% = excellent, 10-15% = good, <10% = mediocre
- Share dilution: compare shares outstanding now vs 5 years ago
  - Decrease = buybacks (positive)
  - Increase >5% = dilution (negative)

---

### 3. FINANCIAL STRENGTH ENGINE (`score_financials`)

**Financial Strength Score (0-100), weight: 20% of total**

Sub-scores:
- Debt/Equity: <0.5 = strong, 0.5-1.5 = ok, >1.5 = weak
- Net Debt/EBITDA: <1 = strong, 1-3 = ok, >3 = concerning, >5 = dangerous
- Interest Coverage (EBIT/interest): >8 = strong, 4-8 = ok, <4 = weak
- Current Ratio: >1.5 = strong, 1-1.5 = ok, <1 = weak
- FCF consistency: positive in 4+ of last 5 years = strong
- Cash as % of total debt: higher = safer

---

### 4. VALUATION ENGINE (`score_valuation`)

**Valuation Score (0-100), weight: 25% of total**

#### A. DCF Model (60% of valuation score)

**Inputs:**
- Last full year revenue
- 5-year revenue CAGR (as base growth rate)
- Operating margin (use trailing average)
- Tax rate: use effective tax rate from financials, default 22% if unavailable
- Capex as % of revenue (trailing average)
- Working capital changes (simplified: use trailing average as % of revenue)

**WACC Calculation:**
- Risk-free rate: hardcode 3.5% (approximate EU 10Y bund, can be updated)
- Equity risk premium: 5.5%
- Beta: from yfinance
- Cost of equity = risk-free + beta × ERP
- Cost of debt = interest expense / total debt (if available), else 4.5%
- Tax shield on debt
- WACC = weighted average based on debt/equity mix

**3-Scenario DCF:**

| Parameter | Bear | Base | Bull |
|---|---|---|---|
| Revenue growth | CAGR × 0.5 | CAGR × 0.85 | CAGR × 1.2 |
| Operating margin | margin × 0.85 | margin × 1.0 | margin × 1.1 |
| Terminal growth | 1.5% | 2.5% | 3.0% |

- Project 10 years of free cash flow
- Terminal value using perpetuity growth method
- Discount all to present value
- Divide by shares outstanding = intrinsic value per share

**Terminal Value Sanity Check:**
- If terminal value > 75% of total enterprise value, flag as "valuation heavily dependent on terminal assumptions"

**Output:**
- Bear fair value per share
- Base fair value per share
- Bull fair value per share
- Probability-weighted fair value = 25% bear + 50% base + 25% bull

#### B. Relative Valuation (25% of valuation score)

Compare current multiples vs:
- Own 5-year average (if available from historical price / historical EPS)
- Sector average (from peer group — see Peer Engine)

Multiples to check:
- P/E vs own history
- P/E vs sector median
- EV/EBIT vs sector median
- EV/FCF vs sector median
- P/B (only for financials sector)

Score: if trading below historical and sector averages = cheap, above = expensive

#### C. Historical Range (15% of valuation score)

Where is the current P/E relative to its own 5-year range?
- Bottom quartile = cheap
- Middle = fair
- Top quartile = expensive

**Final Valuation Output:**
- Blended fair value = 60% DCF weighted + 25% relative + 15% historical
- Buy-below price = blended fair value × (1 - margin of safety)
- Margin of safety: default 25%, adjustable

---

### 5. GROWTH ENGINE (`score_growth`)

**Growth Durability Score (0-100), weight: 15% of total**

Sub-scores:
- 5-year revenue CAGR
- 5-year EPS CAGR (if earnings positive across period)
- Revenue acceleration or deceleration (is growth speeding up or slowing?)
- Reinvestment rate: capex / depreciation > 1 = investing for growth
- FCF growth trend

---

### 6. RISK ENGINE (`score_risk`)

**Risk Profile Score (0-100), weight: 15% of total**

Higher score = LOWER risk (safer stock)

Sub-scores:
- Beta: <0.8 = low vol (good), 0.8-1.2 = normal, >1.5 = high vol (risky)
- Debt risk: from financial strength sub-scores
- Earnings volatility: standard deviation of net income over 5 years
- FCF volatility: standard deviation of FCF over 5 years
- Dilution risk: shares outstanding trend
- Cyclicality proxy: revenue standard deviation / mean

---

### 7. SECTOR ENGINE (`apply_sector_overrides`)

After scoring, apply sector-specific adjustments:

**Banks / Insurance** (sector contains "Financial" or "Bank" or "Insurance"):
- Increase weight of P/B and ROE in valuation
- Decrease weight of DCF (less reliable for financials)
- Flag if P/B < 0.8 (potential deep value) or > 2.0 (expensive)

**Technology / Software** (sector contains "Technology"):
- Watch stock-based compensation: if SBC > 10% of revenue, flag
- EV/Revenue may be relevant for high-growth, low-profit companies
- Penalize negative FCF less if revenue growth > 20%

**Consumer Staples** (sector contains "Consumer Defensive" or "Consumer Staples"):
- Emphasize margin stability over growth rate
- Reward dividend consistency

**Energy / Commodities** (sector contains "Energy" or "Basic Materials"):
- Use normalized earnings (average of last 5 years) for P/E, not just trailing
- Higher weight on cyclicality risk

**Industrials / Manufacturing**:
- Capex intensity matters: capex/revenue > 10% = capital heavy
- Cyclicality adjustment

**Healthcare / Pharma**:
- R&D as % of revenue (proxy from operating expenses if available)
- Pipeline risk flag for biotech (no revenue or very low revenue + high R&D)

---

### 8. MOMENTUM ENGINE (`score_momentum`)

Technical timing overlay — this does NOT affect the Ideal/Medium/Risky classification, but adds a "timing signal" to the report.

Indicators:
- **200-day MA**: price above = uptrend, below = downtrend
- **50-day MA**: price above = short-term strength
- **RSI (14-day)**: <30 = oversold (potential buy), >70 = overbought (caution)
- **50/200 MA crossover**: golden cross (bullish) vs death cross (bearish)
- **Distance from 52-week high**: >20% below = potential opportunity, near high = less margin

**Timing Signal Output:**
- "Strong Buy Timing" — oversold + uptrend + below fair value
- "Good Entry" — price near or in buy zone + neutral/positive technicals
- "Wait" — price in buy zone but momentum negative (catching a falling knife risk)
- "Not Yet" — price above buy zone regardless of technicals
- "Overbought" — RSI high + near 52-week high + above fair value

---

### 9. PEER ENGINE (`compare_peers`)

For each stock:
1. Get its sector and industry from yfinance
2. From the batch of tickers being analyzed, find others in the same sector
3. If fewer than 3 peers in the batch, note "limited peer comparison"
4. Compare: P/E, EV/EBIT, EV/FCF, ROE, margin, growth rate
5. Rank within peer group for each metric
6. Output: "ASML trades at a premium to peers on P/E but justified by higher ROIC and growth"

---

### 10. CLASSIFICATION ENGINE (`classify_stock`)

**Composite Score = weighted sum of all pillar scores:**
- Business Quality: 25%
- Financial Strength: 20%
- Valuation Attractiveness: 25%
- Growth Durability: 15%
- Risk Profile: 15%

**Classification Rules:**

**Ideal (score 80-100):**
ALL of these must be true:
- Business quality score ≥ 70
- Financial strength score ≥ 65
- Valuation score ≥ 70
- FCF positive in at least 4 of last 5 years
- Current price ≤ buy-below price OR within 10% of buy-below price
- No critical risk flags

**Medium (score 55-79):**
- Composite score in range AND does not meet all Ideal criteria
- OR: great business but overpriced (quality high, valuation low)
- OR: cheap but lower quality (valuation high, quality mediocre)

**Risky (score below 55):**
ONE OR MORE of these triggers override into Risky regardless of composite:
- FCF negative in 3+ of last 5 years
- Net debt/EBITDA > 5
- Negative earnings with no improvement trend
- Data quality score < 40 (too much missing data to trust)
- Terminal value > 80% of DCF value AND current price > fair value

---

### 11. EXPLANATION ENGINE (`generate_explanation`)

For every stock, generate a plain-English paragraph covering:

1. **Classification reason**: "Classified as [Ideal/Medium/Risky] because..."
2. **Business summary**: "The company has [high/moderate/low] business quality with [X]% operating margins and [X]% ROIC."
3. **Valuation summary**: "Estimated fair value is [X], current price is [Y], representing [Z]% [upside/downside]."
4. **Buy-below**: "The buy-below price with 25% margin of safety is [X]. Current status: [in buy zone / watchlist / overvalued]."
5. **Key risks**: "Main risks include: [list top 2-3]."
6. **Thesis**: "This investment works if [conditions]. It breaks if [conditions]."
7. **Timing**: "Technical timing signal: [signal]. [Brief explanation]."

Also generate:
- One-line summary for the rankings sheet
- Thesis type mapping: earnings compounder / margin expansion / recovery / re-rating / deleveraging / FCF machine

---

### 12. BACKTEST ENGINE (`run_backtest`) — SIMPLIFIED

This is a basic historical validation, not a full backtest framework.

For each stock:
1. Get price 1 year ago and 3 years ago from yfinance
2. Calculate actual 1Y and 3Y return
3. In the output, note: "If you had bought 1Y ago at [price], return would have been [X]%"
4. After classification, add a note: "Historically, this stock returned [X]% over 3 years, which [supports / contradicts] the current classification"

This is NOT a strategy backtest — it's a sanity check. Full backtesting is a future phase.

---

## Excel Output Specification

### Workbook: `stock_analysis_YYYYMMDD.xlsx`

**Sheet 1: "Rankings"**
- All stocks in one table, sorted by composite score (highest first)
- Columns: Rank, Ticker, Company Name, Sector, Currency, Price, Fair Value, Buy-Below Price, Upside %, Composite Score, Classification (Ideal/Medium/Risky), Timing Signal, One-Line Summary
- Conditional formatting: green rows for Ideal, yellow for Medium, red for Risky
- Header row: bold, dark background, white text
- Freeze top row

**Sheet 2: "Scorecard"**
- All stocks, pillar-by-pillar breakdown
- Columns: Ticker, Business Quality (0-100), Financial Strength (0-100), Valuation (0-100), Growth (0-100), Risk (0-100), Composite (0-100), Classification
- Color-code each score cell: green ≥70, yellow 50-69, red <50

**Sheet 3: "Valuation Detail"**
- One section per stock (separated by blank rows)
- Show: DCF inputs, bear/base/bull fair values, weighted fair value, current multiples, historical multiples, sector multiples, blended fair value, margin of safety, buy-below price
- Blue text for input assumptions
- Black text for calculated values

**Sheet 4: "Financials"**
- Raw financial data per stock
- Revenue, net income, FCF, margins, ratios for last 5 years
- One section per stock

**Sheet 5: "Scenarios"**
- Bear / Base / Bull table per stock
- Show revenue growth, margin, terminal growth, resulting fair value
- Highlight the base case

**Sheet 6: "Technicals"**
- Per stock: current price, 50MA, 200MA, RSI, distance from 52w high, timing signal
- Chart-ready data (even if no charts in v1)

**Sheet 7: "Peer Comparison"**
- Grouped by sector
- Side-by-side multiples and quality metrics
- Highlight best-in-class for each metric

**Formatting Standards:**
- Font: Arial 10pt throughout
- Currency values: show currency symbol + 2 decimal places
- Percentages: 1 decimal place
- Ratios/multiples: 1 decimal place with "x" suffix where appropriate
- Column widths: auto-fit to content
- Borders: thin borders for data cells
- Use openpyxl for all formatting
- Run `python /mnt/skills/public/xlsx/scripts/recalc.py` after saving if formulas used

---

## How to Run

```bash
pip install yfinance openpyxl pandas numpy
python stock_analyzer.py
```

Default tickers (for testing):
```python
TICKERS = [
    "OR.PA",      # L'Oréal
    "SAP.DE",     # SAP
    "ASML.AS",    # ASML
    "NESN.SW",    # Nestlé
    "NOVO-B.CO",  # Novo Nordisk
    "MC.PA",      # LVMH
    "SIE.DE",     # Siemens
    "ABI.BR",     # AB InBev
    "SHEL.L",     # Shell
    "SAN.MC",     # Banco Santander
]
```

The ticker list should be easily editable at the top of the file.

---

## Build Phases

**Phase 1: Data Layer**
- `collect_data()` function
- Pull all raw data from yfinance
- Clean and structure into dictionaries/dataframes
- Handle missing data gracefully
- Output: print summary of what was collected per ticker

**Phase 2: Quality + Financial Scoring**
- `score_quality()` and `score_financials()`
- Calculate all sub-scores
- Output: basic Excel with Rankings and Scorecard sheets (partial)

**Phase 3: Valuation Engine**
- `score_valuation()` with full DCF + relative multiples
- WACC calculation
- 3-scenario analysis
- Buy-below price calculation

**Phase 4: Growth + Risk Engines**
- `score_growth()` and `score_risk()`
- Complete the composite score

**Phase 5: Classification + Explanation**
- `classify_stock()` rules engine
- `generate_explanation()` plain-English output

**Phase 6: Sector Overrides + Momentum + Peers**
- `apply_sector_overrides()`
- `score_momentum()`
- `compare_peers()`

**Phase 7: Backtest + Full Excel Output**
- `run_backtest()` sanity check
- Complete all 7 Excel sheets with formatting
- Final polished output

**Phase 8: Testing + Refinement**
- Run on 10+ EU stocks
- Verify data accuracy
- Tune scoring thresholds
- Fix edge cases

---

## Key Design Principles

1. **Fail gracefully**: if yfinance returns no data for a field, use `None` and skip that sub-score. Never crash.
2. **No false precision**: fair values are ranges, not exact numbers.
3. **Separation of concerns**: quality score and valuation score are independent. A great business can be overpriced.
4. **Explain everything**: every classification must have a human-readable reason.
5. **Conservative bias**: when in doubt, classify as Medium or Risky, not Ideal.
6. **Sector awareness**: don't use the same lens for a bank and a SaaS company.
7. **Timing ≠ thesis**: momentum overlay helps with entry timing but never overrides fundamental analysis.
