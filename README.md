# Stock Value

Python-based stock analysis project that combines data collection, valuation, quality, risk, momentum, peer comparison, and Excel report generation.

## Project layout

- `stock_analyzer/`: analysis engines and the main entry point
- `STOCK_ANALYZER_SPEC.md`: broader project specification
- `EXCEL_IMPROVEMENT_SPEC.md`: Excel output enhancement notes

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python stock_analyzer\main.py
```

Edit the ticker list in `stock_analyzer/main.py` to analyze different symbols.
