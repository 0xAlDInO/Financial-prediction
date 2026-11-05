"""Market data streamer and historical replay module supporting M5 (5m) and M15 (15m).
Supports:
- Binance public API (with Binance US fallback)
- Resampling (e.g. converting 5m data into 15m candles)
- Offline replay / simulation from historical data (CSV or preloaded)
"""

import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List


def resample_klines(df: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    """Resample OHLCV dataframe to a higher timeframe like '15min'.
    Expects columns: ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    """
    df = df.copy()
    if 'timestamps' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamps' column.")

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').drop_duplicates(subset=['timestamps']).reset_index(drop=True)
    df = df.set_index('timestamps')

    rule = "15min" if timeframe in ["15m", "15min", "M15"] else timeframe
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(rule, label='left', closed='left').first()
    resampled['high'] = df['high'].resample(rule, label='left', closed='left').max()
    resampled['low'] = df['low'].resample(rule, label='left', closed='left').min()
    resampled['close'] = df['close'].resample(rule, label='left', closed='left').last()

    if 'volume' in df.columns:
        resampled['volume'] = df['volume'].resample(rule, label='left', closed='left').sum()
    else:
        resampled['volume'] = 0.0

    if 'amount' in df.columns:
        resampled['amount'] = df['amount'].resample(rule, label='left', closed='left').sum()
    else:
        resampled['amount'] = 0.0

    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
    return resampled


class MarketDataStreamer:
    """Fetches live or historical k-lines from Coinbase (BTC/USD native) or Binance / Binance US."""

    COINBASE_GRANULARITY = {
        "5m": 300,
        "M5": 300,
        "5min": 300,
        "15m": 900,
        "M15": 900,
        "15min": 900
    }

    INTERVAL_MAP = {
        "5m": "5m",
        "M5": "5m",
        "5min": "5m",
        "15m": "15m",
        "M15": "15m",
        "15min": "15m"
    }

    def __init__(self, default_symbol: str = "BTC/USD"):
        self.default_symbol = default_symbol

    def fetch_klines(
        self,
        symbol: Optional[str] = None,
        interval: str = "5m",
        limit: int = 400
    ) -> pd.DataFrame:
        """Fetch latest k-lines for a symbol and interval (5m or 15m).
        Specialized for BTC/USD with multi-exchange fallback:
        1. Coinbase Pro (direct BTC-USD spot)
        2. Binance US (BTCUSD)
        3. Binance (BTCUSDT)
        4. Simulated buffer (fallback)
        """
        symbol_raw = (symbol or self.default_symbol).strip().upper()
        norm_interval = self.INTERVAL_MAP.get(interval, "5m")

        # 1. Try Coinbase API (standard for genuine BTC/USD spot pair worldwide)
        if "USD" in symbol_raw:
            cb_df = self._fetch_coinbase("BTC-USD", interval, limit)
            if cb_df is not None and not cb_df.empty:
                return cb_df

        # 2. Try Binance US BTCUSD
        binance_symbols = ["BTCUSD", "BTCUSDT"] if "USD" in symbol_raw else [symbol_raw.replace("/", "").replace("-", "")]
        for b_sym in binance_symbols:
            for base_url in ["https://api.binance.us/api/v3/klines", "https://api.binance.com/api/v3/klines"]:
                url = f"{base_url}?symbol={b_sym}&interval={norm_interval}&limit={limit}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        parsed = json.loads(resp.read().decode('utf-8'))
                        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                            return self._parse_binance_klines(parsed)
                except Exception:
                    continue

        # 3. Fallback
        return self._generate_fallback_klines("BTC/USD", norm_interval, limit)

    def _fetch_coinbase(self, product_id: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        granularity = self.COINBASE_GRANULARITY.get(interval, 300)
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity={granularity}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Coinbase-Client)"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    # Coinbase returns [time, low, high, open, close, volume] ordered descending by time
                    rows = []
                    for item in data[:limit]:
                        ts = pd.to_datetime(item[0], unit='s', utc=True).tz_convert(None)
                        open_p = float(item[3])
                        high_p = float(item[2])
                        low_p = float(item[1])
                        close_p = float(item[4])
                        vol = float(item[5])
                        amt = vol * close_p
                        rows.append({
                            "timestamps": ts,
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": vol,
                            "amount": amt
                        })
                    df = pd.DataFrame(rows)
                    df = df.sort_values('timestamps').reset_index(drop=True)
                    return df
        except Exception as e:
            pass
        return None

    def _parse_binance_klines(self, raw_list: List[list]) -> pd.DataFrame:
        """Parses Binance raw kline array:
        [
          0: Open time
          1: Open
          2: High
          3: Low
          4: Close
          5: Volume
          6: Close time
          7: Quote asset volume (Amount)
          ...
        ]
        """
        rows = []
        for item in raw_list:
            open_time = pd.to_datetime(item[0], unit='ms', utc=True).tz_convert(None)
            rows.append({
                "timestamps": open_time,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "amount": float(item[7])
            })
        df = pd.DataFrame(rows)
        return df

    def _generate_fallback_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generates realistic market candles for offline demo / fallback if external network is blocked."""
        step_min = 15 if interval == "15m" else 5
        base_price = 80000.0 if "BTC" in symbol else 3000.0
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0, tzinfo=None)

        timestamps = [now - timedelta(minutes=step_min * (limit - i)) for i in range(limit)]
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.002, limit)
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1.0 + r))

        rows = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            volatility = price * 0.0015
            open_p = price + np.random.uniform(-volatility, volatility) * 0.5
            close_p = price + np.random.uniform(-volatility, volatility) * 0.5
            high_p = max(open_p, close_p) + abs(np.random.uniform(0, volatility))
            low_p = min(open_p, close_p) - abs(np.random.uniform(0, volatility))
            volume = np.random.uniform(5.0, 50.0)
            amount = volume * close_p
            rows.append({
                "timestamps": ts,
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": round(volume, 4),
                "amount": round(amount, 2)
            })
        return pd.DataFrame(rows)


class HistoricalReplayStreamer:
    """Simulates live streaming step-by-step from an existing historical DataFrame or CSV.
    Useful for testing next-candle prediction sequentially without waiting 5m or 15m per candle.
    """

    def __init__(self, df: pd.DataFrame, initial_lookback: int = 150):
        self.df = df.copy()
        self.df['timestamps'] = pd.to_datetime(self.df['timestamps'])
        self.df = self.df.sort_values('timestamps').reset_index(drop=True)
        self.current_idx = min(initial_lookback, len(self.df) - 1)
        self.initial_lookback = initial_lookback

    @classmethod
    def from_csv(cls, filepath: str, initial_lookback: int = 150):
        df = pd.read_csv(filepath)
        return cls(df, initial_lookback=initial_lookback)

    def get_current_window(self, max_context: int = 400) -> pd.DataFrame:
        """Returns the history window up to current_idx."""
        start_idx = max(0, self.current_idx - max_context)
        return self.df.iloc[start_idx:self.current_idx].copy().reset_index(drop=True)

    def peek_next_actual_candle(self) -> Optional[pd.Series]:
        """Returns the actual next candle if available, for verification."""
        if self.current_idx < len(self.df):
            return self.df.iloc[self.current_idx]
        return None

    def step_forward(self) -> Optional[pd.Series]:
        """Advances pointer to next candle (as if time passed and next candle closed)."""
        if self.current_idx < len(self.df):
            candle = self.df.iloc[self.current_idx]
            self.current_idx += 1
            return candle
        return None

    def has_next(self) -> bool:
        return self.current_idx < len(self.df)

    def reset(self):
        self.current_idx = min(self.initial_lookback, len(self.df) - 1)
