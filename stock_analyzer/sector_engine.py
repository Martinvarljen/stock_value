"""
sector_engine.py  —  Sector-specific context and adjustments

apply_sector_context(data) → dict with:
  - wacc_adjustment:        float to add to base WACC (e.g. +0.005 = +0.5%)
  - terminal_growth_range:  (min, max) realistic terminal growth for this sector
  - sector_flags:           sector-specific risk and opportunity notes
  - key_metrics:            which metrics matter most for this sector
  - sector_note:            one-line description of typical sector characteristics
"""

from utils import _pct

# ── sector configuration table ─────────────────────────────────────────────────
#
# wacc_adj:    added to computed WACC — positive for riskier sectors
# tg_range:    (floor, ceiling) for realistic terminal growth
# key_metrics: metric keys from quality/financial/valuation engines to highlight
# risks:       typical sector-specific risks to flag
# note:        brief sector context shown in output

SECTOR_CONFIGS = {
    "Technology": {
        "wacc_adj":   +0.005,
        "tg_range":   (0.020, 0.035),
        "key_metrics": ["revenue_cagr_5y", "gross_margin", "roic", "sbc_burden", "ev_fcf"],
        "risks": [
            "Rapid product obsolescence — moats erode faster than in other sectors",
            "High SBC common — GAAP earnings overstate true owner earnings",
            "Regulatory scrutiny (antitrust, data privacy) increasing globally",
            "Customer concentration risk in enterprise software",
        ],
        "opportunities": [
            "High gross margins enable significant operating leverage as scale grows",
            "Network effects and switching costs can create durable competitive moats",
        ],
        "note": "High-growth, high-margin sector; P/E and EV/FCF more relevant than P/B or EV/EBITDA",
    },

    "Healthcare": {
        "wacc_adj":   0.000,
        "tg_range":   (0.018, 0.030),
        "key_metrics": ["revenue_cagr_5y", "operating_margin", "roic", "net_debt_ebitda", "fcf_yield"],
        "risks": [
            "Patent cliffs — blockbuster drugs lose exclusivity, revenue falls sharply",
            "Regulatory approval risk (FDA/EMA) on pipeline products",
            "Pricing pressure from government drug negotiations",
            "Clinical trial failure risk for early-stage R&D investments",
        ],
        "opportunities": [
            "Aging demographics drive structural long-term demand",
            "High barriers to entry from R&D cost, trials, and regulatory hurdles",
        ],
        "note": "Defensive characteristics; EV/EBIT and FCF yield most relevant; pipeline optionality not captured in DCF",
    },

    "Financial Services": {
        "wacc_adj":   +0.005,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["roe", "pb_ratio", "net_margin", "dividend_yield", "revenue_cagr_5y"],
        "risks": [
            "Credit cycle risk — loan losses spike in downturns, hard to model with DCF",
            "Interest rate sensitivity — net interest margin compresses in flat/inverted yield curves",
            "Regulatory capital requirements limit capital return flexibility",
            "DCF less reliable for banks — use P/B and ROE vs cost of equity instead",
        ],
        "opportunities": [
            "Rising rate environments typically expand net interest margins",
            "Well-capitalised banks can return significant capital via buybacks and dividends",
        ],
        "note": "DCF unreliable for financials — P/B vs ROE and dividend yield are primary valuation tools",
    },

    "Consumer Cyclical": {
        "wacc_adj":   +0.003,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "operating_margin", "net_debt_ebitda", "fcf_yield", "ev_ebitda"],
        "risks": [
            "Earnings highly sensitive to consumer confidence and economic cycles",
            "Brand erosion risk — consumer preferences shift, private labels gain share",
            "High fixed cost structures amplify margin swings in downturns",
        ],
        "opportunities": [
            "Strong brands command pricing power through cycles",
            "E-commerce transition creating new distribution efficiencies",
        ],
        "note": "EV/EBITDA and FCF yield most relevant; margins and leverage are key cycle survival metrics",
    },

    "Consumer Defensive": {
        "wacc_adj":   -0.003,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "gross_margin", "dividend_yield", "net_debt_ebitda", "fcf_yield"],
        "risks": [
            "Private label competition compressing branded product margins",
            "Input cost inflation (commodities, packaging) difficult to pass through quickly",
            "Slow volume growth — sector is mature; revenue growth often price-driven",
        ],
        "opportunities": [
            "Recession-resistant demand — non-discretionary products hold up in downturns",
            "Consistent free cash flow generation supports reliable dividends",
        ],
        "note": "Lower WACC justified by defensive earnings; P/E, dividend yield, and FCF yield most relevant",
    },

    "Industrials": {
        "wacc_adj":   +0.002,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "operating_margin", "roic", "capex_pct_revenue", "net_debt_ebitda"],
        "risks": [
            "Capital-intensive — high capex requirements limit free cash flow",
            "Cyclical end-markets (construction, manufacturing) cause revenue volatility",
            "Supply chain disruptions and commodity costs hard to forecast",
        ],
        "opportunities": [
            "Infrastructure spending cycles create multi-year revenue visibility",
            "High switching costs in engineered components and aftermarket services",
        ],
        "note": "ROIC vs WACC spread is critical — must earn returns above cost of capital to justify capex",
    },

    "Basic Materials": {
        "wacc_adj":   +0.010,
        "tg_range":   (0.010, 0.020),
        "key_metrics": ["revenue_cagr_5y", "operating_margin", "net_debt_ebitda", "fcf_yield", "ev_ebitda"],
        "risks": [
            "Commodity price cycles — earnings highly volatile, hard to normalise",
            "DCF unreliable due to price cycle dependency; use mid-cycle earnings",
            "ESG pressure increasing regulatory and operating costs for miners",
        ],
        "opportunities": [
            "Energy transition driving structural demand for battery metals (lithium, copper, nickel)",
            "Supply constraints in many commodities due to underinvestment post-2015",
        ],
        "note": "Use normalised mid-cycle margins for DCF; EV/EBITDA on trough earnings is key for downside",
    },

    "Energy": {
        "wacc_adj":   +0.008,
        "tg_range":   (0.010, 0.020),
        "key_metrics": ["revenue_cagr_5y", "fcf_yield", "net_debt_ebitda", "ev_ebitda", "dividend_yield"],
        "risks": [
            "Oil/gas price cycles — revenue and earnings highly volatile",
            "Energy transition risk — long-term demand for fossil fuels structurally declining",
            "Stranded asset risk — reserves may lose value if carbon prices rise sharply",
            "Capex-intensive replacement of depleting reserves",
        ],
        "opportunities": [
            "Strong FCF generation at current energy prices funds buybacks and dividends",
            "Integrated majors have diverse downstream businesses providing earnings stability",
        ],
        "note": "Use flat-price DCF assumptions; FCF yield and dividend yield more reliable than earnings-based multiples",
    },

    "Utilities": {
        "wacc_adj":   -0.008,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "net_debt_ebitda", "dividend_yield", "interest_coverage", "fcf_yield"],
        "risks": [
            "Very high leverage — regulated returns require significant debt financing",
            "Interest rate sensitivity — rate rises increase cost of debt and compress valuations",
            "Regulatory risk — allowed returns set by regulators, not market forces",
        ],
        "opportunities": [
            "Regulated earnings provide predictable, bond-like cash flows",
            "Renewable energy build-out driving capital investment and rate base growth",
        ],
        "note": "Lower WACC appropriate; EV/EBITDA, dividend yield, and regulated asset base are primary metrics",
    },

    "Real Estate": {
        "wacc_adj":   -0.005,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "net_debt_ebitda", "fcf_yield", "dividend_yield", "pb_ratio"],
        "risks": [
            "High leverage typical — vulnerable to rising interest rates",
            "GAAP earnings unreliable due to depreciation — use FFO/AFFO instead",
            "Property-type specific risks: office vacancy, retail disruption",
        ],
        "opportunities": [
            "Inflation hedge — rental income and asset values often track CPI",
            "Long-lease structures provide revenue visibility",
        ],
        "note": "GAAP earnings misleading — FCF and dividend yield are primary metrics; NAV discount/premium matters",
    },

    "Communication Services": {
        "wacc_adj":   +0.003,
        "tg_range":   (0.015, 0.025),
        "key_metrics": ["revenue_cagr_5y", "operating_margin", "ev_ebitda", "fcf_yield", "roic"],
        "risks": [
            "Subscriber/user growth slowing — mature streaming/social media markets",
            "Heavy content investment for media companies compresses near-term FCF",
            "Ad-revenue cyclicality — digital ad spend cuts sharply in recessions",
        ],
        "opportunities": [
            "Dominant platforms benefit from strong network effects and switching costs",
            "Monetisation of existing user bases can grow ARPU without added user growth",
        ],
        "note": "EV/EBITDA and FCF yield most relevant; subscriber metrics and ARPU trends key leading indicators",
    },
}

_DEFAULT_CONFIG = {
    "wacc_adj":    0.000,
    "tg_range":    (0.015, 0.025),
    "key_metrics": ["revenue_cagr_5y", "operating_margin", "roic", "net_debt_ebitda", "fcf_yield"],
    "risks":       [],
    "opportunities": [],
    "note":        "No sector-specific configuration — using generic defaults",
}


# ── main function ──────────────────────────────────────────────────────────────

def apply_sector_context(data: dict) -> dict:
    """
    Returns sector-specific context: WACC adjustment, terminal growth range,
    risk flags, and key metrics to highlight.
    """
    sector = data.get("sector") or ""
    cfg    = SECTOR_CONFIGS.get(sector, _DEFAULT_CONFIG)

    flags = []

    # Flag if current WACC assumption seems misaligned with sector norms
    wacc = (data.get("wacc_data") or {}).get("wacc")
    if wacc is not None and cfg["wacc_adj"] != 0:
        adj_note = (
            f"Sector ({sector}) carries a {cfg['wacc_adj']:+.1%} WACC adjustment "
            f"vs generic assumptions — base WACC {_pct(wacc)} → adjusted {_pct(wacc + cfg['wacc_adj'])}"
        )
        flags.append(adj_note)

    # Flag sector-specific risks
    for risk in cfg.get("risks", []):
        flags.append(f"Sector risk: {risk}")

    tg_min, tg_max = cfg["tg_range"]
    return {
        "sector":               sector,
        "wacc_adjustment":      cfg["wacc_adj"],
        "terminal_growth_range": cfg["tg_range"],
        "key_metrics":          cfg["key_metrics"],
        "sector_flags":         flags,
        "opportunities":        cfg.get("opportunities", []),
        "sector_note":          cfg.get("note", ""),
        "tg_floor":             tg_min,
        "tg_ceiling":           tg_max,
    }


# ── display helper ─────────────────────────────────────────────────────────────

def print_sector_context(result: dict, ticker: str = ""):
    header = f"SECTOR CONTEXT — {ticker}" if ticker else "SECTOR CONTEXT"
    print(f"\n{'─' * 70}")
    print(f"  {header}  [{result['sector'] or 'Unknown sector'}]")
    print(f"{'─' * 70}")
    print(f"  {result['sector_note']}")
    print(f"  WACC adjustment:      {result['wacc_adjustment']:+.1%}")
    print(f"  Terminal growth range: {result['tg_floor']:.1%} – {result['tg_ceiling']:.1%}")

    if result["opportunities"]:
        print(f"\n  Sector opportunities:")
        for o in result["opportunities"]:
            print(f"    +  {o}")

    print(f"\n  Sector-specific flags:")
    sector_risks = [f for f in result["sector_flags"] if "Sector risk:" in f]
    wacc_notes   = [f for f in result["sector_flags"] if "Sector risk:" not in f]
    for f in wacc_notes + sector_risks:
        label = "  " if "Sector risk:" not in f else "  ⚠ "
        print(f"  {label}{f.replace('Sector risk: ', '')}")
