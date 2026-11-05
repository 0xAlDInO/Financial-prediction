"""Script to run next-candle prediction sequentially from CLI with M5 and M15 support.

Usage:
    python examples/predict_next_candle_cli.py --symbol BTCUSDT --timeframe 5m
    python examples/predict_next_candle_cli.py --symbol ETHUSDT --timeframe 15m --mode live
    python examples/predict_next_candle_cli.py --mode replay --steps 5
"""

import sys
import os
import argparse
import time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Kronos, KronosTokenizer, KronosPredictor
from model.live_streamer import MarketDataStreamer, HistoricalReplayStreamer
from model.sequential_predictor import SequentialNextCandlePredictor


def main():
    parser = argparse.ArgumentParser(description="Kronos Sequential Next-Candle Predictor")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Asset symbol (default: BTC/USD)")
    parser.add_argument("--timeframe", type=str, default="5m", choices=["5m", "15m"], help="Timeframe (5m or 15m)")
    parser.add_argument("--mode", type=str, default="live", choices=["live", "replay"], help="Stream mode")
    parser.add_argument("--steps", type=int, default=1, help="Number of sequential steps to predict")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"🚀 Kronos Next-Candle Predictor [{args.timeframe.upper()}] - {args.symbol}")
    print(f"Mode: {args.mode.upper()} | Device: {args.device}")
    print(f"Paire: BTC/USD Spot")
    print(f"=======================================================\n")

    print("Chargement du modèle Kronos-mini...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=128)
    seq_engine = SequentialNextCandlePredictor(
        predictor=predictor,
        timeframe=args.timeframe,
        max_context=128,
        sample_count=1
    )

    if args.mode == "replay":
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune_csv", "data", "HK_ali_09988_kline_5min_all.csv")
        replay = HistoricalReplayStreamer.from_csv(sample_path, initial_lookback=50)

        for step in range(args.steps):
            window = replay.get_current_window(max_context=50)
            pred = seq_engine.predict_next(window)

            print(f"--- Étape {step + 1}/{args.steps} ---")
            print(f"Horodatage cible: {pred.target_timestamp}")
            print(f"Direction       : {pred.direction_symbol} {pred.direction}")
            print(f"Prix actuel     : {pred.current_close}")
            print(f"Prix prédit     : {pred.predicted_close} ({'+' if pred.expected_change_pct >= 0 else ''}{pred.expected_change_pct}%)")
            print(f"Fourchette      : [{pred.predicted_low} - {pred.predicted_high}]")

            actual = replay.step_forward()
            if actual is not None:
                eval_pred = seq_engine.evaluate_and_record(pred, actual)
                status = "✓ GAGNANT" if eval_pred.is_correct else "✗ PERDANT"
                print(f"Réel clôture    : {actual['close']} --> {status}\n")

        print("Stats globales:", seq_engine.get_stats())

    else:
        streamer = MarketDataStreamer()
        df = streamer.fetch_klines(symbol=args.symbol, interval=args.timeframe, limit=50)
        pred = seq_engine.predict_next(df)

        print(f"Horodatage cible: {pred.target_timestamp}")
        print(f"Direction       : {pred.direction_symbol} {pred.direction}")
        print(f"Prix actuel     : {pred.current_close}")
        print(f"Prix prédit     : {pred.predicted_close} ({'+' if pred.expected_change_pct >= 0 else ''}{pred.expected_change_pct}%)")
        print(f"Fourchette      : [{pred.predicted_low} - {pred.predicted_high}]")
        print(f"Confiance       : {round(pred.confidence * 100)}%")


if __name__ == "__main__":
    main()
