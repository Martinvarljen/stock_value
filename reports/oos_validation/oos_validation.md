# OOS validation — research_ls

- Frozen config: `C:\Users\ElenaAdmin\OneDrive - Univerza v Ljubljani\Desktop\Projekti\Finance\portfolio\config.frozen.json`
- Train window: 2019–2022 (IS run skipped)
- OOS window: **2023–2026** (parameters locked)

## Out-of-sample (primary)

- Strategy CAGR: 46.5%
- SPY CAGR: 21.6%
- Strategy max DD: -20.4%
- Beat SPY: **True**

| Year | Strat return | SPY return | Beat SPY | Days |
|------|-------------|------------|----------|------|
| 2023 | +50.2% | +21.3% | yes | 241 |
| 2024 | +16.9% | +25.6% | no | 252 |
| 2025 | +74.7% | +18.0% | yes | 250 |
| 2026 | +16.4% | +7.5% | yes | 95 |

Beat SPY in **3/4** calendar years.

## Forward test (6–12 months)

Run daily paper and refresh OOS report:

```powershell
python portfolio/daily_run.py
python -c "from portfolio.config_loader import load_config; from portfolio.paper_oos import write_oos_report; print(write_oos_report(load_config()))"
```

Set `paper_oos.oos_start_date` in config when you start the forward track.

HTML: `C:\Users\ElenaAdmin\OneDrive - Univerza v Ljubljani\Desktop\Projekti\Finance\reports\oos_validation\oos_2023_2026.html`