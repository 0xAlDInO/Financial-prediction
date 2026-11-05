"""Unit and integration tests for sequential next-candle prediction and streaming modules."""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor
from model.live_streamer import MarketDataStreamer, HistoricalReplayStreamer, resample_klines
from model.sequential_predictor import SequentialNextCandlePredictor, CandlePrediction


class TestSequentialPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load lightweight Kronos-mini on CPU for testing
        cls.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        cls.model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        cls.predictor = KronosPredictor(cls.model, cls.tokenizer, device="cpu", max_context=64)

    def setUp(self):
        # Create a mock 5-minute dataframe
        start_time = datetime(2025, 1, 1, 9, 30)
        times = [start_time + timedelta(minutes=5 * i) for i in range(25)]

        prices = [100.0]
        for _ in range(24):
            prices.append(prices[-1] + np.random.uniform(-0.5, 0.5))

        data = []
        for t, p in zip(times, prices):
            data.append({
                "timestamps": t,
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p + 0.1,
                "volume": 100.0,
                "amount": 10000.0
            })
        self.mock_df = pd.DataFrame(data)

    def test_market_streamer_fallback_and_parsing(self):
        streamer = MarketDataStreamer()
        df = streamer.fetch_klines("BTCUSDT", interval="5m", limit=10)
        self.assertEqual(len(df), 10)
        for col in ["timestamps", "open", "high", "low", "close", "volume", "amount"]:
            self.assertIn(col, df.columns)

    def test_resample_m15(self):
        resampled = resample_klines(self.mock_df, timeframe="15min")
        self.assertGreater(len(resampled), 0)
        self.assertLess(len(resampled), len(self.mock_df))
        self.assertIn("close", resampled.columns)

    def test_predict_next_candle_m5(self):
        engine = SequentialNextCandlePredictor(self.predictor, timeframe="5m", max_context=64, sample_count=1)
        pred = engine.predict_next(self.mock_df[:-1])

        self.assertIsInstance(pred, CandlePrediction)
        self.assertIn(pred.direction, ["ASCENDANT", "DESCENDANT"])
        self.assertIn(pred.direction_symbol, ["🟢", "🔴"])
        self.assertGreaterEqual(pred.predicted_high, pred.predicted_low)

        # Check target timestamp step is +5 minutes
        last_ts = pd.to_datetime(self.mock_df.iloc[-2]["timestamps"])
        expected_target = (last_ts + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(pred.target_timestamp, expected_target)

    def test_predict_next_candle_m15(self):
        engine = SequentialNextCandlePredictor(self.predictor, timeframe="15m", max_context=64, sample_count=1)
        pred = engine.predict_next(self.mock_df[:-1])

        last_ts = pd.to_datetime(self.mock_df.iloc[-2]["timestamps"])
        expected_target = (last_ts + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(pred.target_timestamp, expected_target)

    def test_evaluate_and_stats(self):
        engine = SequentialNextCandlePredictor(self.predictor, timeframe="5m", max_context=64, sample_count=1)
        pred = engine.predict_next(self.mock_df[:-1])
        actual_candle = self.mock_df.iloc[-1]

        evaluated = engine.evaluate_and_record(pred, actual_candle)
        self.assertIsNotNone(evaluated.is_correct)

        stats = engine.get_stats()
        self.assertEqual(stats["total_predictions"], 1)
        self.assertIn(stats["correct_predictions"], [0, 1])
        self.assertIn(stats["accuracy_pct"], [0.0, 100.0])

    def test_replay_streamer(self):
        replay = HistoricalReplayStreamer(self.mock_df, initial_lookback=10)
        window = replay.get_current_window()
        self.assertEqual(len(window), 10)
        self.assertTrue(replay.has_next())

        next_c = replay.step_forward()
        self.assertIsNotNone(next_c)
        new_window = replay.get_current_window()
        self.assertEqual(len(new_window), 11)


if __name__ == "__main__":
    unittest.main()
