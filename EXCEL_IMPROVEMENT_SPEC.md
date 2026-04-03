# EXCEL OUTPUT IMPROVEMENT SPEC

## File to modify: excel_output.py
## No changes needed to main.py — all data is already in the `results` list.

---

## CURRENT STATE

excel_output.py has 2 sheet types:
1. Summary — one row per ticker, 19 columns, too wide, hard to scan
2. Per-ticker detail sheets — vertical dump of all metric blocks in identical 6-column format (Metric, Value, Formatted, Assessment, Benchmark, Detail)

Problems:
- Summary has too many columns (19) — need to scroll horizontally
- Detail sheets are just metric dumps — no narrative, no scenario table, no visual hierarchy
- Scenarios (bear/base/bull) are calculated but never written to Excel
- Peer comparison is calculated but never written to Excel
- Momentum/technicals are calculated but never written to Excel
- Explanation engine generates great narrative paragraphs but they never appear in Excel
- Sensitivity grid is calculated but never written to Excel
- No conditional formatting beyond classification cell color

---

## NEW SHEET STRUCTURE (7 sheets total)

### Sheet 1: "Dashboard" (replaces current Summary)

Purpose: Open the Excel, know what to buy in 30 seconds.

Columns (only 10, not 19):
| Ticker | Company | Sector | Price (native) | Fair Value | Upside % | Classification | Timing Signal | Key Opportunity | Key Risk |

- Sort rows by Upside % descending (best opportunities first)
- Classification cell: colored background (green=STRONG BUY/BUY, yellow=WATCHLIST/HOLD, red=AVOID/STRONG AVOID), white bold text for dark backgrounds
- Upside % cell: green fill if >15%, red fill if <-15%
- Key Opportunity and Key Risk come from classification_result["key_opportunity"] and classification_result["key_risk"]
- Timing Signal comes from momentum analysis — derive from momentum_metrics: if RSI <30 and price < MA200 → "Oversold", if price > MA50 > MA200 → "Uptrend", if MA50 < MA200 → "Downtrend", else "Neutral"
- Freeze top row (headers)
- Row height: 22px for readability
- Font: Arial 10pt throughout
- Column widths: Ticker 10, Company 30, Sector 20, Price 12, Fair Value 12, Upside 9, Classification 14, Timing 14, Key Opportunity 45, Key Risk 45

### Sheet 2: "Scorecard"

Purpose: Side-by-side numerical comparison of all tickers.

Columns:
| Ticker | Company | Currency | Price | ROIC | WACC | ROIC/WACC Spread | Rev CAGR 5Y | Op Margin | Net Margin | Gross Margin | Debt/Equity | ND/EBITDA | Interest Coverage | FCF Yield | Beta | Data Quality |

- One row per ticker
- Conditional formatting per cell:
  - ROIC: green ≥15%, yellow 8-15%, red <8%
  - ROIC/WACC Spread: green >3%, yellow 0-3%, red <0%
  - ND/EBITDA: green <1x, yellow 1-3x, red >3x
  - Interest Coverage: green >8x, yellow 3-8x, red <3x
  - FCF Yield: green >5%, yellow 2-5%, red <2%
- Freeze first two columns (Ticker, Company)
- All percentage values formatted as "12.3%" not "0.123"
- All multiples formatted as "2.1x"

### Sheet 3: "Scenarios"

Purpose: Bear/Base/Bull DCF side-by-side for every ticker.

Layout: One block per ticker, separated by 2 blank rows.

Each block:
Row 1: Ticker + Company name (merged across columns, bold, colored header)
Row 2: Column headers → | Parameter | Bear | Base | Bull |
Row 3: Probability weight
Row 4: Starting revenue growth
Row 5: Starting operating margin  
Row 6: Steady-state margin (yr 10)
Row 7: Terminal growth rate
Row 8: WACC
Row 9: PV of 10Y FCFs
Row 10: PV of terminal value
Row 11: Terminal value % of EV
Row 12: Enterprise value
Row 13: Equity value
Row 14: Fair value per share (bold, larger font)
Row 15: Probability-weighted fair value (bold, highlighted)
Row 16: Current price
Row 17: Upside/downside vs weighted fair value

Data source: record["scenarios"] contains "bear", "base", "bull" dicts each with these keys:
- growth_rate, operating_margin, steady_state_margin, terminal_growth, wacc
- pv_fcfs, pv_terminal_value, tv_pct_of_ev, enterprise_value, equity_value, per_share_value
- Weights from record["scenarios"]["_weights"]["bear"/"base"/"bull"]
- Weighted fair value from record["fair_value_weighted"]

Also add the sensitivity grid below each block if record.get("valuation_result", {}).get("tv_sensitivity") exists:
- It has wacc_steps (list of 5 WACC values), tg_steps (list of 5 terminal growth values), and grid (5x5 matrix of per-share values)
- Format as a proper grid with WACC on rows and terminal growth on columns
- Highlight the cell closest to base WACC + base terminal growth

### Sheet 4: "Peer Comparison"

Purpose: How does each stock compare to others in its sector?

Data source: Run analyze_peers(results) — this is already called in main.py. Pass the peer_result into write_excel or reconstruct it.

Actually, the simplest approach: call analyze_peers(results) inside write_excel since it needs the full results list.

Layout: One section per sector group.
Row 1: Sector name (merged, colored header)
Row 2: Column headers → | Ticker | Company | Classification | Rev CAGR 5Y | Op Margin | ROIC | ND/EBITDA | Int Coverage | FCF Yield | Fair Value | Overall Score |
Then one row per ticker in that sector group, sorted by overall_percentile descending.

- Bold the #1 ranked stock in each metric column
- Overall Score formatted as percentage (e.g. "78%")
- Color the overall score: green ≥70%, yellow 40-70%, red <40%

### Sheet 5: "Technicals"

Purpose: Momentum and timing signals for all tickers.

Columns:
| Ticker | Price | MA50 | MA200 | Price vs MA50 | Price vs MA200 | Trend | RSI (14d) | 52W High | 52W Low | % of 52W Range | 1M Return | 3M Return | 6M Return | 12M Return | Timing Signal |

Data sources:
- Price, MA50, MA200, RSI from data_layer fields
- Trend from momentum_metrics["moving_averages"]["formatted"]
- 52W data from data_layer
- Returns from momentum_metrics (return_1m, return_3m, return_6m, return_12m values)
- Timing signal: derive same logic as Dashboard

Conditional formatting:
- RSI: green <30 (oversold opportunity), red >70 (overbought risk), neutral otherwise
- Trend: green "UPTREND", red "DOWNTREND", yellow "SIDEWAYS"
- Returns: green if positive, red if negative

### Sheet 6: Per-ticker detail sheets (IMPROVED)

Keep one sheet per ticker but restructure the layout:

**Section A: Narrative Summary (rows 3-20 approx)**
- Pull from record["explanation"]["paragraphs"] — this is a list of (heading, text) tuples
- Each paragraph gets: heading in bold (merged across columns), then the text below it (merged across columns, wrapped)
- Paragraphs: Business Overview, Financial Health, Valuation, Key Risks, Verdict
- Also add the one_liner from record["explanation"]["one_liner"] at the top in italic

**Section B: Key Numbers (compact, 2-column layout)**
A compact block with the most important numbers, NOT the full metric dump:
- Left side: Price, Fair Value, Buy-Below, Upside %, Classification
- Right side: ROIC, WACC, Spread, Rev CAGR, Op Margin, ND/EBITDA

**Section C: Scenario Summary (compact bear/base/bull table)**
Same format as the Scenarios sheet but just for this ticker.

**Section D: Red Flags**
- Only if red flags exist
- Severity | Pattern | Detail
- Color-coded by severity (red=HIGH, amber=MEDIUM, blue=LOW)

**Remove**: The current approach of dumping ALL metrics from ALL engines (quality_metrics, financial_metrics, valuation_metrics, growth_metrics, risk_metrics) in identical 6-column tables. This makes each detail sheet 200+ rows of metrics that nobody reads. The narrative + key numbers + scenarios + red flags is much more useful.

### Sheet 7: "Raw Data" (optional, for power users)

If you want to keep the full metric dumps, put them all on one "Raw Data" sheet rather than on each detail sheet. One section per ticker with all metrics from all engines. This is the "appendix" — there for those who want to dig in, but not front and center.

---

## FORMATTING STANDARDS

- Font: Arial 10pt for data, Arial 11pt bold for section headers, Arial 14pt bold for sheet titles
- Header rows: dark navy background (#1F3864), white bold text
- Sub-headers: mid blue (#2E5D9E), white text
- Alternating row colors: white and very light blue (#F2F7FF)
- All borders: thin, light blue (#B8CCE4)
- Number formats:
  - Percentages: "12.3%" (not 0.123)
  - Multiples: "2.1x"
  - Prices: 2 decimal places with currency where known
  - Large numbers: use _bn() format (e.g. "12.34B", "456.7M")
- Column widths: auto-sized to content where possible, minimum 8, maximum 50
- Row heights: 18-22px for data rows, 28-30px for header rows
- Freeze panes: freeze header row on all sheets, freeze Ticker+Company columns on Dashboard and Scorecard
- Wrap text in assessment/narrative columns

---

## CLASSIFICATION COLOR MAP (already exists, keep it)

STRONG BUY  → dark green  (#1E7B34), white text
BUY         → green       (#70AD47), white text  
WATCHLIST   → light green (#92D050), black text
HOLD        → amber       (#FFD966), black text
AVOID       → orange-red  (#FF7043), white text
STRONG AVOID→ dark red    (#C00000), white text

---

## IMPLEMENTATION NOTES

1. The `write_excel(results)` function signature stays the same
2. Call `analyze_peers(results)` inside write_excel to get peer data for Sheet 4
3. All data needed is already in each record dict — no new calculations needed
4. Build each sheet in its own function (_build_dashboard, _build_scorecard, _build_scenarios, _build_peers, _build_technicals, _build_detail, _build_raw_data)
5. Keep the existing style helpers (_fill, _font, _border, _style_header, _style_data) — they work fine
6. Test with mixed tickers (EU + US) and verify currency handling
7. Handle None values gracefully everywhere — some tickers will have missing data
8. Sheet names max 31 characters — truncate ticker names if needed
