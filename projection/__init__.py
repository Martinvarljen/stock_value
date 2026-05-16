"""
projection/ — ML projection layer (LightGBM + rule-based composite).

Modules:
  projection_engine  : rule-based + ML hybrid forward projections
  news_engine        : news fetching + FinBERT / Claude sentiment
  ml_model/          : LightGBM training, feature extraction, inference
"""

import sys
from pathlib import Path

# Make stock_analyzer importable from projection modules
_root = Path(__file__).resolve().parents[1]
_sa = str(_root / "stock_analyzer")
if _sa not in sys.path:
    sys.path.insert(0, _sa)
