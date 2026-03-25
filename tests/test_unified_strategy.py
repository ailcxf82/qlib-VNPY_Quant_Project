"""
Unit tests for unified_strategy module.
Tests cover:
- UnifiedStrategyBase abstract class
- UnifiedMomentumStrategy
- UnifiedMeanReversionStrategy
- UnifiedValueStrategy
- UnifiedQualityStrategy
- create_strategy factory function
"""

import unittest
from unittest.mock import MagicMock,import pandas as pd

from monitor.unified_strategy import (
    UnifiedStrategyBase,
    UnifiedMomentumStrategy,
    UnifiedMeanReversionStrategy,
    UnifiedValueStrategy,
    UnifiedQualityStrategy,
    create_strategy,
)


class TestUnifiedStrategyBase(unittest.TestCase):
    def test_base_strategy_initialization(self):
        config = {"lookback_period": 20, "top_k": 10}
        strategy = UnifiedMomentumStrategy(config)
        
        self.assertEqual(strategy.STRATEGY_NAME, "momentum")
        self.assertEqual(strategy.lookback_period, 20)
        self.assertEqual(strategy.top_k, 10)
    
    def test_base_strategy_default_factors(self):
        config = {}
        strategy = UnifiedMomentumStrategy(config)
        
        default_factors = strategy._get_default_factors()
        
        self.assertEqual(default_factors["pb"], 1.0)
        self.assertEqual(default_factors["pe"], 15.0)
        self.assertEqual(default_factors["roe"], 0.10)
    
    def test_base_strategy_with_data_source(self):
        mock_data_source = MagicMock()
        mock_data_source.get_stock_factors.return_value = {
            "pb": 2.0,
            "pe": 20.0,
            "roe": 0.15,
        }
        
        config = {}
        strategy = UnifiedMomentumStrategy(config, data_source=mock_data_source)
        
        factors = strategy.fetch_factors("000001")
        
        mock_data_source.get_stock_factors.assert_called_once_with("000001")
        self.assertEqual(factors["pb"], 2.0)
        self.assertEqual(factors["pe"], 20.0)
    
    def test_base_strategy_clear_cache(self):
        config = {}
        strategy = UnifiedMomentumStrategy(config)
        
        strategy._factor_cache["test"] = {"test": 1}
        
        strategy.clear_cache()
        
        self.assertEqual(strategy._factor_cache, {})


class TestUnifiedMomentumStrategy(unittest.TestCase):
    def test_momentum_score_calculation(self):
        config = {}
        strategy = UnifiedMomentumStrategy(config)
        
        factors = {
            "price_momentum_5d": 0.05,
            "price_momentum_20d": 0.10,
            "volume_momentum": 1.5,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertGreaterEqual(score, 0.5)
    
    def test_momentum_generate_signals(self):
        config = {"lookback_period": 10, "min_momentum": 0.03}
        strategy = UnifiedMomentumStrategy(config)
        
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        data = {}
        for date in dates:
            df = pd.DataFrame({
                "date": [date] * 3,
                "close": [10 + i for i in range(20)],
            })
            df = df.sort_values("date").reset_index(drop=True)
            data[f"stock_{i}"] = df
        
        signals = strategy.generate_signals(
            date="2024-01-15",
            data=data,
            positions={}
        )
        
        self.assertEqual(len(signals), 0)


class TestUnifiedMeanReversionStrategy(unittest.TestCase):
    def test_mean_reversion_score_calculation(self):
        config = {}
        strategy = UnifiedMeanReversionStrategy(config)
        
        factors = {
            "rsi_14": 30.0,
            "price_deviation_ma20": -0.10,
            "volume_spike": 2.0,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertGreaterEqual(score, 0.5)
    
    def test_mean_reversion_generate_signals(self):
        config = {"lookback_period": 20, "oversold_threshold": -0.10}
        strategy = UnifiedMeanReversionStrategy(config)
        
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        data = {}
        for date in dates:
            df = pd.DataFrame({
                "date": [date] * 3,
                "close": [8 + i for i in range(20)],
            })
            df = df.sort_values("date").reset_index(drop=True)
            data[f"stock_{i}"] = df
        
        signals = strategy.generate_signals(
            date="2024-01-15",
            data=data,
            positions={}
        )
        
        self.assertEqual(len(signals), 0)


class TestUnifiedValueStrategy(unittest.TestCase):
    def test_value_score_calculation(self):
        config = {}
        strategy = UnifiedValueStrategy(config)
        
        factors = {
            "pb": 1.5,
            "pe": 10.0,
            "roe": 0.15,
            "dividend_yield": 0.03,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertGreaterEqual(score, 0.5)
    
    def test_value_score_low_valuation(self):
        config = {}
        strategy = UnifiedValueStrategy(config)
        
        factors = {
            "pb": 5.0,
            "pe": 50.0,
            "roe": 0.05,
            "dividend_yield": 0.01,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertLess(score, 0.5)


class TestUnifiedQualityStrategy(unittest.TestCase):
    def test_quality_score_calculation(self):
        config = {}
        strategy = UnifiedQualityStrategy(config)
        
        factors = {
            "roe": 0.20,
            "roa": 0.08,
            "debt_ratio": 0.30,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertGreaterEqual(score, 0.7)
    
    def test_quality_score_high_debt(self):
        config = {}
        strategy = UnifiedQualityStrategy(config)
        
        factors = {
            "roe": 0.10,
            "roa": 0.05,
            "debt_ratio": 0.70,
        }
        
        score = strategy.calculate_score("000001", factors)
        
        self.assertLess(score, 0.5)


class TestCreateStrategy(unittest.TestCase):
    def test_create_momentum_strategy(self):
        config = {"lookback_period": 15}
        strategy = create_strategy("momentum", config)
        
        self.assertIsInstance(strategy, UnifiedMomentumStrategy)
        self.assertEqual(strategy.lookback_period, 15)
    
    def test_create_mean_reversion_strategy(self):
        config = {"lookback_period": 15}
        strategy = create_strategy("mean_reversion", config)
        
        self.assertIsInstance(strategy, UnifiedMeanReversionStrategy)
    
    def test_create_value_strategy(self):
        config = {}
        strategy = create_strategy("value", config)
        
        self.assertIsInstance(strategy, UnifiedValueStrategy)
    
    def test_create_quality_strategy(self):
        config = {}
        strategy = create_strategy("quality", config)
        
        self.assertIsInstance(strategy, UnifiedQualityStrategy)
    
    def test_create_unknown_strategy_defaults_to_momentum(self):
        config = {}
        strategy = create_strategy("unknown_strategy", config)
        
        self.assertIsInstance(strategy, UnifiedMomentumStrategy)


if __name__ == "__main__":
    unittest.main()
