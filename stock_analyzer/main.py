"""
main.py  —  Stock Analyzer entry point
Orchestrates all engines and produces the final Excel report.
"""

import sys
from datetime import datetime

# ── engines ────────────────────────────────────────────────────────────────────
from data_layer            import collect_data
from quality_engine        import analyze_quality,       print_quality
from financial_strength    import analyze_financials,    print_financials
from valuation_engine      import analyze_valuation,     print_valuation
from growth_engine         import analyze_growth,        print_growth
from risk_engine           import analyze_risk,          print_risk
from red_flags             import analyze_red_flags,     print_red_flags
from classification_engine import classify_stock,        print_classification
from sector_engine         import apply_sector_context,  print_sector_context
from momentum_engine       import analyze_momentum,      print_momentum
from backtest_engine       import analyze_price_history, print_price_history
from explanation_engine    import generate_explanation,  print_explanation
from peer_engine           import analyze_peers,         print_peers
from excel_output          import write_excel

def _configure_console_output() -> None:
    """Avoid Windows console crashes when printing Unicode report headers."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# ── ticker list ──────────────────────────────────────────────────────────────
# Edit this list to analyze any combination of exchanges:
#
#   EU exchanges  →  suffix required:  OR.PA  SAP.DE  ASML.AS  NESN.SW
#                                      SHEL.L  NOVO-B.CO  SIE.DE  ABI.BR
#   NYSE / NASDAQ →  no suffix:        AAPL  MSFT  GOOGL  JPM  BRK-B
#   Mixed lists are fine — yfinance handles all of them.
#
#   Note: London (.L) prices are quoted in GBp (pence); the data layer
#         automatically divides by 100 to convert to GBP.

TICKERS = [
    "SXR8.DE",    # iShares Core S&P 500 UCITS ETF USD (Acc)
    "VWCE.DE",    # Vanguard FTSE All-World UCITS ETF USD Accumulation
    "MSFT",       # Microsoft
    "ITB",        # iShares U.S. Home Construction ETF
]

MARGIN_OF_SAFETY = 0.3   # 30 % default

# ── pipeline ──────────────────────────────────────────────────────────────────

def analyze(tickers: list[str]) -> list[dict]:
    """Run the full analysis pipeline for each ticker and return results."""
    results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Analyzing {ticker} ...", flush=True)

        # Phase 1  data collection
        data = collect_data(ticker)
        if data.get("error"):
            print(f"    Data error: {data['error']}")

        # Skip ETFs — no income statement / balance sheet to analyse
        if data.get("quote_type") == "ETF":
            print(f"    Skipping ETF — fundamental analysis not applicable for {ticker}")
            continue

        if data.get("data_quality_score", 0) < 40:
            print(f"    Low data quality ({data['data_quality_score']}/100) results may be unreliable")

        # Sector context first — WACC adjustment must feed into valuation
        sector_result = apply_sector_context(data)

        # Phase 2  quality & financial strength
        quality_result   = analyze_quality(data)
        financial_result = analyze_financials(data)

        # Phase 3  valuation (WACC + DCF + buy-below), with sector WACC override
        valuation_result = analyze_valuation(
            data,
            margin_of_safety=MARGIN_OF_SAFETY,
            wacc_adjustment=sector_result["wacc_adjustment"],
        )

        # Phase 4  growth trajectory + risk profile
        growth_result    = analyze_growth(data)
        risk_result      = analyze_risk(data)

        # Red flags  pass WACC from valuation result for capital destruction check
        wacc = valuation_result.get("wacc_data", {}).get("wacc")
        red_flag_result = analyze_red_flags(data, wacc=wacc)

        # Merge critical flags from all engines
        all_critical = (
            (financial_result.get("critical_flags") or []) +
            (risk_result.get("critical_flags") or [])
        )

        record = {
            **data,
            "quality_metrics":    quality_result["quality_metrics"],
            "quality_flags":      quality_result["quality_flags"],
            "financial_metrics":  financial_result["financial_metrics"],
            "financial_flags":    financial_result["financial_flags"],
            "valuation_metrics":  valuation_result["valuation_metrics"],
            "valuation_flags":    valuation_result["valuation_flags"],
            "fair_value_weighted":valuation_result["fair_value_weighted"],
            "buy_below_price":    valuation_result["buy_below_price"],
            "wacc_data":          valuation_result["wacc_data"],
            "scenarios":          valuation_result["scenarios"],
            "tv_sensitivity":    valuation_result.get("tv_sensitivity"),
            "growth_metrics":     growth_result["growth_metrics"],
            "growth_flags":       growth_result["growth_flags"],
            "risk_metrics":       risk_result["risk_metrics"],
            "risk_flags":         risk_result["risk_flags"],
            "red_flags":          red_flag_result["red_flags"],
            "red_flag_summary":   red_flag_result["summary"],
            "critical_flags":     all_critical,
        }

        # Momentum + price history — computed before classify/explain so timing signal is available
        momentum_result = analyze_momentum(data)
        backtest_result = analyze_price_history(data)

        # Merge momentum into record so classify and explain can see it
        record["momentum_metrics"]   = momentum_result["momentum_metrics"]
        record["momentum_flags"]     = momentum_result["momentum_flags"]
        record["momentum_trend"]     = momentum_result["trend"]
        record["backtest_metrics"]   = backtest_result["backtest_metrics"]
        record["backtest_flags"]     = backtest_result["backtest_flags"]

        # Classification — needs the full record assembled above
        clf_result = classify_stock(record)
        record["classification_result"] = clf_result
        record["classification"]        = clf_result["classification"]

        # Narrative explanation (momentum_trend / momentum_metrics now in record)
        explanation_result = generate_explanation(record)

        # Store remaining results in the record for peer comparison and Excel
        record["sector_result"]      = sector_result
        record["explanation"]        = explanation_result

        results.append(record)
        _print_summary(record)
        print_explanation(explanation_result, ticker)
        print_sector_context(sector_result, ticker)
        print_quality(quality_result, ticker)
        print_financials(financial_result, ticker)
        print_valuation(valuation_result, ticker)
        print_growth(growth_result, ticker)
        print_risk(risk_result, ticker)
        print_momentum(momentum_result, ticker)
        print_price_history(backtest_result, ticker)
        print_red_flags(red_flag_result, ticker)
        print_classification(clf_result, ticker)

    # Peer comparison — runs once across all tickers after individual analysis
    peer_result = analyze_peers(results)
    print_peers(peer_result)

    return results

def _print_summary(r: dict):
    """Print a one-line header for each stock before the detailed output."""
    name  = r.get("company_name") or r["ticker"]
    price = r.get("current_price")
    curr  = r.get("currency", "")
    price_eur = r.get("price_eur")
    eur_rate  = r.get("eur_rate")
    dq    = r.get("data_quality_score", 0)
    sect  = r.get("sector") or "Unknown sector"
    crit  = r.get("critical_flags") or []

    price_str = f"{curr} {price:.2f}" if price else "price N/A"
    if price_eur and curr != "EUR":
        price_str += f"  (EUR {price_eur:.2f}, rate {eur_rate:.4f})"
    crit_str  = f"  *** CRITICAL FLAGS: {', '.join(crit)}" if crit else ""
    print(f"\n{'=' * 70}")
    print(f"  {name}  ({r['ticker']})  —  {sect}")
    print(f"  Price: {price_str}   Data quality: {dq}/100{crit_str}")
    print(f"{'=' * 70}")

def _print_index(results: list[dict]):
    """Print a compact summary table of all tickers analyzed."""
    print("\n" + "─" * 90)
    print("  ANALYZED TICKERS")
    print("─" * 90)
    print(f"  {'Ticker':<12}  {'Company':<30}  {'Price':<18}  {'Fair Value':<12}  {'Upside':<8}  {'Classification'}")
    print("─" * 90)
    for r in results:
        name      = (r.get("company_name") or r["ticker"])[:30]
        curr      = r.get("currency", "")
        price     = r.get("current_price")
        price_eur = r.get("price_eur")
        fv        = r.get("fair_value_weighted")
        clf       = r.get("classification", "N/A")

        native    = f"{curr} {price:.2f}" if price else "N/A"
        if price_eur and curr != "EUR":
            native += f" (€{price_eur:.2f})"
        fv_str    = f"€{fv:.2f}" if fv else "N/A"
        upside    = f"{(fv-price)/price:+.0%}" if (fv and price and price > 0) else "N/A"
        crit      = "  " if r.get("critical_flags") else ""

        print(f"  {r['ticker']:<12}  {name:<30}  {native:<18}  {fv_str:<12}  {upside:<8}  {clf}{crit}")
    print("─" * 90)

def main():
    _configure_console_output()

    tickers = TICKERS
    if len(sys.argv) > 1:
        # Allow overriding tickers from command line: python main.py AAPL MSFT
        tickers = sys.argv[1:]

    print(f"\nStock Analyzer —  {datetime.today().strftime('%Y-%m-%d')}")
    print(f"Tickers: {tickers}\n")

    results = analyze(tickers)
    _print_index(results)

    # Excel export
    try:
        path = write_excel(results)
        print(f"\nExcel report saved → {path}")
    except ImportError as e:
        print(f"\nExcel export skipped: {e}")

if __name__ == "__main__":
    main()
