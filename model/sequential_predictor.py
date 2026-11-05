"""Sequential next-candle prediction engine using Kronos.
Predicts candle (t+1) from history up to t.
Classifies direction as ASCENDANT (BULLISH) vs DESCENDANT (BEARISH).
Tracks sequential accuracy and prediction vs actual comparisons.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from model.kronos import Kronos, KronosTokenizer, KronosPredictor


@dataclass
class CandlePrediction:
    target_timestamp: str
    direction: str            # "ASCENDANT" or "DESCENDANT"
    direction_symbol: str     # "🟢" or "🔴"
    confidence: float         # 0.0 - 1.0 (estimated probability of the direction)
    current_close: float      # Last known close price
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float
    expected_change_pct: float
    expected_change_abs: float
    predicted_volume: float
    actual_candle: Optional[Dict[str, float]] = None
    is_correct: Optional[bool] = None


class SequentialNextCandlePredictor:
    """Specialized engine for sequential 1-step-ahead (next candle) forecasting."""

    def __init__(
        self,
        predictor: KronosPredictor,
        timeframe: str = "5m",
        max_context: int = 400,
        temperature: float = 0.8,
        top_p: float = 0.9,
        sample_count: int = 3
    ):
        self.predictor = predictor
        self.timeframe = timeframe
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.sample_count = sample_count

        self.step_delta = timedelta(minutes=15 if timeframe in ["15m", "15min", "M15"] else 5)
        self.history_predictions: List[CandlePrediction] = []

    def set_timeframe(self, timeframe: str):
        self.timeframe = timeframe
        self.step_delta = timedelta(minutes=15 if timeframe in ["15m", "15min", "M15"] else 5)

    def predict_next(
        self,
        df_history: pd.DataFrame,
        sample_count: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> CandlePrediction:
        """Takes historical candles up to now (at least 20 candles recommended, ideally 100-400),
        predicts the single next candle, its direction, variation and confidence.
        """
        if len(df_history) < 5:
            raise ValueError("df_history must have at least 5 candles.")

        df = df_history.copy()
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # Truncate context if longer than max_context
        if len(df) > self.max_context:
            df = df.iloc[-self.max_context:].reset_index(drop=True)

        samples = sample_count if sample_count is not None else self.sample_count
        temp = temperature if temperature is not None else self.temperature

        last_known_candle = df.iloc[-1]
        last_timestamp = pd.to_datetime(last_known_candle['timestamps'])
        target_timestamp = last_timestamp + self.step_delta

        x_timestamp = df['timestamps']
        y_timestamp = pd.Series([target_timestamp])

        x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()

        # Generate prediction using KronosPredictor
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=1,
            T=temp,
            top_k=0,
            top_p=self.top_p,
            sample_count=samples,
            verbose=False
        )

        pred_row = pred_df.iloc[0]
        curr_close = float(last_known_candle['close'])
        pred_open = float(pred_row['open'])
        pred_close = float(pred_row['close'])
        pred_high = float(pred_row['high'])
        pred_low = float(pred_row['low'])
        pred_vol = float(max(0.0, pred_row['volume']))

        # Ensure high >= max(open, close) and low <= min(open, close) for candle consistency
        pred_high = max(pred_high, pred_open, pred_close)
        pred_low = min(pred_low, pred_open, pred_close)

        diff_abs = pred_close - curr_close
        diff_pct = (diff_abs / curr_close) * 100.0

        is_bullish = pred_close >= curr_close
        direction = "ASCENDANT" if is_bullish else "DESCENDANT"
        direction_symbol = "🟢" if is_bullish else "🔴"

        # Confidence heuristic: based on relative movement strength
        # Normalized between 55% and 95%
        rel_move = abs(diff_pct)
        confidence = round(min(0.95, max(0.55, 0.50 + rel_move * 0.15)), 3)

        prediction = CandlePrediction(
            target_timestamp=target_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            direction=direction,
            direction_symbol=direction_symbol,
            confidence=confidence,
            current_close=round(curr_close, 4),
            predicted_open=round(pred_open, 4),
            predicted_high=round(pred_high, 4),
            predicted_low=round(pred_low, 4),
            predicted_close=round(pred_close, 4),
            expected_change_pct=round(diff_pct, 3),
            expected_change_abs=round(diff_abs, 4),
            predicted_volume=round(pred_vol, 2)
        )

        return prediction

    def evaluate_and_record(
        self,
        prediction: CandlePrediction,
        actual_candle: pd.Series
    ) -> CandlePrediction:
        """Evaluates previous prediction against actual candle that just closed."""
        actual_close = float(actual_candle['close'])
        actual_open = float(actual_candle['open'])
        actual_high = float(actual_candle['high'])
        actual_low = float(actual_candle['low'])
        actual_vol = float(actual_candle.get('volume', 0.0))

        actual_bullish = actual_close >= prediction.current_close
        pred_bullish = prediction.direction == "ASCENDANT"

        is_correct = (actual_bullish == pred_bullish)

        prediction.actual_candle = {
            "timestamps": str(actual_candle.get('timestamps')),
            "open": round(actual_open, 4),
            "high": round(actual_high, 4),
            "low": round(actual_low, 4),
            "close": round(actual_close, 4),
            "volume": round(actual_vol, 2)
        }
        prediction.is_correct = is_correct

        self.history_predictions.append(prediction)
        return prediction

    def get_stats(self) -> Dict[str, Any]:
        """Returns cumulative statistics on sequential predictions."""
        evaluated = [p for p in self.history_predictions if p.is_correct is not None]
        total = len(evaluated)
        if total == 0:
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy_pct": 0.0,
                "history": []
            }

        correct = sum(1 for p in evaluated if p.is_correct)
        accuracy = (correct / total) * 100.0

        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_pct": round(accuracy, 2),
            "history": [asdict(p) for p in self.history_predictions[-30:]]
        }
