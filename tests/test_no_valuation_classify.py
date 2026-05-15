"""Ensure --no-valuation path skips valuation_engine."""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

import backtesting.strategy_backtest as st


class TestNoValuationClassify(unittest.TestCase):
    def test_classify_at_skips_analyze_valuation_when_disabled(self) -> None:
        with patch.object(st, "analyze_valuation", new=MagicMock()) as mock_val:
            st.classify_at({"ticker": "X", "current_price": 1.0}, use_valuation=False)
        mock_val.assert_not_called()


if __name__ == "__main__":
    unittest.main()
