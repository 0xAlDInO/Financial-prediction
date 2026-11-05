"""Flask web application providing the visual interface for sequential next-candle forecasting.
Supports M5 and M15 timeframes, live Binance streaming, and replay mode.
"""

import os
import sys
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch

# Ensure repo root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import Kronos, KronosTokenizer, KronosPredictor
from model.live_streamer import MarketDataStreamer, HistoricalReplayStreamer
from model.sequential_predictor import SequentialNextCandlePredictor, CandlePrediction

app = Flask(__name__, template_folder="templates")
CORS(app)

# Global model state
predictor: Optional[KronosPredictor] = None
seq_engine: Optional[SequentialNextCandlePredictor] = None
streamer: MarketDataStreamer = MarketDataStreamer("BTC/USD")
replay_streamer: Optional[HistoricalReplayStreamer] = None
last_pending_prediction: Optional[CandlePrediction] = None


def init_kronos():
    """Initializes Kronos model on demand with CPU/GPU auto-detection."""
    global predictor, seq_engine
    if predictor is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[WebUI] Loading Kronos-mini on device: {device}...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=128)
        seq_engine = SequentialNextCandlePredictor(
            predictor=predictor,
            timeframe="5m",
            max_context=128,
            temperature=0.8,
            top_p=0.9,
            sample_count=1
        )
        print("[WebUI] Kronos model ready!")


@app.route("/")
def index():
    return render_template("sequential_next_candle.html")


@app.route("/api/predict_next_step", methods=["POST"])
def predict_next_step():
    global last_pending_prediction, replay_streamer
    init_kronos()

    data = request.get_json() or {}
    symbol = data.get("symbol", "BTC/USD")
    timeframe = data.get("timeframe", "5m")
    mode = data.get("mode", "live")

    seq_engine.set_timeframe(timeframe)

    if mode == "replay":
        # Load sample replay data from repository
        sample_path = os.path.join(ROOT_DIR, "finetune_csv", "data", "HK_ali_09988_kline_5min_all.csv")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(ROOT_DIR, "tests", "data", "regression_input.csv")

        if replay_streamer is None:
            replay_streamer = HistoricalReplayStreamer.from_csv(sample_path, initial_lookback=40)

        df_window = replay_streamer.get_current_window(max_context=60)
        # Check if we can evaluate the previous prediction
        actual_candle = replay_streamer.step_forward()
        if last_pending_prediction is not None and actual_candle is not None:
            seq_engine.evaluate_and_record(last_pending_prediction, actual_candle)

        # Predict next candle
        next_pred = seq_engine.predict_next(df_window)
        last_pending_prediction = next_pred

        recent_candles = df_window.tail(30).to_dict(orient="records")
        for c in recent_candles:
            c['timestamps'] = pd.to_datetime(c['timestamps']).strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "status": "success",
            "prediction": as_dict_prediction(next_pred),
            "stats": seq_engine.get_stats(),
            "history_candles": recent_candles
        })

    else:
        # Live streaming from Binance
        df = streamer.fetch_klines(symbol=symbol, interval=timeframe, limit=40)

        # If we had a previous pending prediction whose timestamp has passed or arrived in df
        if last_pending_prediction is not None and len(df) > 0:
            last_candle = df.iloc[-1]
            last_ts_str = pd.to_datetime(last_candle['timestamps']).strftime("%Y-%m-%d %H:%M:%S")
            if last_pending_prediction.target_timestamp <= last_ts_str:
                seq_engine.evaluate_and_record(last_pending_prediction, last_candle)
                last_pending_prediction = None

        # Predict next candle from current full df
        next_pred = seq_engine.predict_next(df)
        last_pending_prediction = next_pred

        recent_candles = df.tail(30).to_dict(orient="records")
        for c in recent_candles:
            c['timestamps'] = pd.to_datetime(c['timestamps']).strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "status": "success",
            "prediction": as_dict_prediction(next_pred),
            "stats": seq_engine.get_stats(),
            "history_candles": recent_candles
        })


@app.route("/api/reset_stats", methods=["POST"])
def reset_stats():
    global last_pending_prediction, replay_streamer
    if seq_engine:
        seq_engine.history_predictions.clear()
    last_pending_prediction = None
    if replay_streamer:
        replay_streamer.reset()
    return jsonify({"status": "success"})


def as_dict_prediction(pred: CandlePrediction) -> Dict[str, Any]:
    return {
        "target_timestamp": pred.target_timestamp,
        "direction": pred.direction,
        "direction_symbol": pred.direction_symbol,
        "confidence": pred.confidence,
        "current_close": pred.current_close,
        "predicted_open": pred.predicted_open,
        "predicted_high": pred.predicted_high,
        "predicted_low": pred.predicted_low,
        "predicted_close": pred.predicted_close,
        "expected_change_pct": pred.expected_change_pct,
        "expected_change_abs": pred.expected_change_abs,
        "predicted_volume": pred.predicted_volume
    }


if __name__ == "__main__":
    init_kronos()
    port = int(os.environ.get("PORT", 7070))
    print(f"🚀 Kronos Sequential Visual Interface running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
