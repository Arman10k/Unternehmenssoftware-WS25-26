"""
Improved LSTM Deployment Script for QQQ with Strategy System
=============================================================
Usage:
  python lstm_deploy.py --strategy aggressive_long --dry-run
  python lstm_deploy.py --strategy cfd_2x_leverage --loop
  python lstm_deploy.py --list-strategies
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import pytz
import requests
import joblib
import importlib.util

import yfinance as yf
import torch
from torch import nn

from news_features import NewsFeatureProvider

# ===== STRATEGY IMPORTS =====
from strategies.strategy_config import  StrategyConfig
from strategies.strategies import (
    LongOnlyMomentumStrategy,
    ShortOnlyStrategy,
    CFDLeveragedStrategy
)

# -----------------------------
# Paths / Imports (unchanged)
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

FEATURES_PY_PATH = os.path.join(PROJECT_ROOT, "scripts", "03_pre_split_prep", "features.py")
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)
else:
    raise RuntimeError(f"Could not load features.py from {FEATURES_PY_PATH}")

FeatureBuilder = getattr(features_module, "FeatureBuilder")

CONF_DIR = os.path.join(PROJECT_ROOT, "conf")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCALER_X_PATH = os.path.join(DATA_DIR, "scaler_X.joblib")
STRATEGY_JSON_PATH = os.path.join(THIS_DIR, "strategies", "strategies.json")

# -----------------------------
# Load configs
# -----------------------------
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

with open(os.path.join(CONF_DIR, "keys.yaml"), "r") as f:
    keys = yaml.safe_load(f)

# -----------------------------
# Constants (unchanged)
# -----------------------------
TICKER = "QQQ"
SEQUENCE_LENGTH = 50
INPUT_SIZE = 14
HIDDEN_SIZE = 384
NUM_LAYERS = 2
OUTPUT_SIZE = 5
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASTERN = pytz.timezone("US/Eastern")

# Alpaca
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", keys["KEYS"].get("APCA-API-KEY-ID-Paper"))
ALPACA_SECRET = os.getenv("ALPACA_SECRET", keys["KEYS"].get("APCA-API-SECRET-KEY-Paper"))
ALPACA_BASE = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "features_clean.txt")

# Cooldown tracking
last_trade_time: Dict[str, datetime] = {}


# -----------------------------
# Model (unchanged)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last_layer_h = h_n[-self.num_directions :, :, :]
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(last_layer_h)


def create_last_sequence(X: np.ndarray, seq_len: int) -> np.ndarray:
    if len(X) < seq_len:
        return np.array([])
    return np.array([X[-seq_len:]])


# -----------------------------
# Broker Interface Implementation
# -----------------------------
class AlpacaBroker:
    """Broker interface matching strategy_config.BrokerInterface"""

    def __init__(self):
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_KEY_ID,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
            "Content-Type": "application/json",
        }

    def get_account_info(self) -> dict:
        r = requests.get(f"{ALPACA_BASE}/v2/account", headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> List[dict]:
        r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=self.headers, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    def get_position(self, symbol: str) -> Optional[dict]:
        r = requests.get(f"{ALPACA_BASE}/v2/positions/{symbol}", headers=self.headers, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def submit_order(self, order_params: dict) -> Optional[dict]:
        try:
            r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=self.headers,
                            json=order_params, timeout=30)
            r.raise_for_status()
            od = r.json()
            print(f"[ORDER] {order_params['side'].upper()} {order_params['qty']} "
                  f"{order_params['symbol']} | id={od.get('id')}")
            return od
        except Exception as e:
            print(f"[ERROR] submit_order failed: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            r = requests.delete(f"{ALPACA_BASE}/v2/positions/{symbol}",
                              headers=self.headers, timeout=30)
            r.raise_for_status()
            print(f"[CLOSE] Closed {symbol}")
            return True
        except Exception as e:
            print(f"[ERROR] close_position failed for {symbol}: {e}")
            return False

    def get_last_fill_time(self, symbol: str, side: str) -> Optional[datetime]:
        try:
            params_q = {"status": "closed", "limit": "200", "direction": "desc"}
            r = requests.get(f"{ALPACA_BASE}/v2/orders", headers=self.headers,
                           params=params_q, timeout=30)
            r.raise_for_status()
            orders = r.json()

            last_dt: Optional[datetime] = None
            for o in orders:
                if (str(o.get("status", "")).lower() != "filled" or
                    str(o.get("symbol", "")).upper() != symbol.upper() or
                    str(o.get("side", "")).lower() != side.lower()):
                    continue

                filled_at = o.get("filled_at")
                if not filled_at:
                    continue
                try:
                    dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
                except Exception:
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                if last_dt is None or dt > last_dt:
                    last_dt = dt

            return last_dt
        except Exception as e:
            print(f"[WARN] cannot fetch orders for fill-time: {e}")
            return None


# -----------------------------
# Calendar / Data functions (unchanged)
# -----------------------------
def build_calendar_map(start_dt: datetime, end_dt: datetime) -> Dict[datetime.date, Tuple[datetime, datetime]]:
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params_q = {"start": start_dt.strftime("%Y-%m-%d"), "end": end_dt.strftime("%Y-%m-%d")}
    r = requests.get(f"{ALPACA_BASE}/v2/calendar", headers=headers, params=params_q, timeout=30)
    r.raise_for_status()
    days = r.json()
    cal_map: Dict[datetime.date, Tuple[datetime, datetime]] = {}
    for d in days:
        date_str = d.get("date")
        open_str = d.get("open")
        close_str = d.get("close")
        if not date_str or not open_str or not close_str:
            continue
        y, m, dd = map(int, date_str.split("-"))
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        open_dt = EASTERN.localize(datetime(y, m, dd, oh, om))
        close_dt = EASTERN.localize(datetime(y, m, dd, ch, cm))
        cal_map[open_dt.date()] = (open_dt, close_dt)
    return cal_map


def is_rth(ts: pd.Timestamp, cal_map: Dict[datetime.date, Tuple[datetime, datetime]]) -> bool:
    if ts.tzinfo is None:
        ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    else:
        try:
            ts_eastern = ts.tz_convert(EASTERN)
        except Exception:
            ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)

    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


def download_qqq_data(days: int = 5) -> pd.DataFrame:
    print(f"[DATA] Downloading {days}d of 1m for {TICKER} via yfinance...")
    df = yf.download(TICKER, period=f"{days}d", interval="1m", auto_adjust=True,
                     prepost=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def load_feature_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature list: {path}")
    feats: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    return feats


def build_features_no_news(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    df = df_raw.copy()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df["timestamp"] = df.index
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
    slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]

    builder = FeatureBuilder(
        df=df,
        ema_windows=ema_periods,
        return_windows=slope_periods,
        price_col="vwap",
        timestamp_col="timestamp",
    )
    df_feat = builder.build_features_before_split()

    if "avg_volume_per_trade" not in df_feat.columns:
        df_feat["avg_volume_per_trade"] = df_feat["volume"] / 100.0

    pd.set_option('future.no_silent_downcasting', True)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    if df_feat.empty:
        raise RuntimeError("All features NaN after rolling windows (insufficient history?).")

    last_ts = df_feat.index[-1]
    return df_feat, last_ts


def load_lstm_model() -> Tuple[LSTMModel, object, object]:
    model_path = os.path.join(MODELS_DIR, "best_lstm_model.pth")
    scaler_y_path = os.path.join(DATA_DIR, "scaler_y.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler Y not found: {scaler_y_path}")
    if not os.path.exists(SCALER_X_PATH):
        raise FileNotFoundError(f"Scaler X not found: {SCALER_X_PATH}")

    scaler_y = joblib.load(scaler_y_path)
    scaler_X = joblib.load(SCALER_X_PATH)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        bidirectional=False,
        dropout=DROPOUT,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler_y, scaler_X


# -----------------------------
# Load strategy configs
# -----------------------------
def load_strategies_from_json() -> Dict[str, dict]:
    """Load all strategies from strategies.json"""
    if not os.path.exists(STRATEGY_JSON_PATH):
        raise FileNotFoundError(f"Strategy config not found: {STRATEGY_JSON_PATH}")

    with open(STRATEGY_JSON_PATH, 'r') as f:
        strategies = json.load(f)

    print(f"[STRATEGIES] Loaded {len(strategies)} strategies from {STRATEGY_JSON_PATH}")
    return strategies


def initialize_strategy_manager() -> StrategyManager:
    """Initialize strategy manager with all available strategies"""
    manager = StrategyManager()

    # Load all strategies from JSON
    strategies_dict = load_strategies_from_json()

    # Map strategy types to classes
    strategy_classes = {
        'long_only': LongOnlyMomentumStrategy,
        'short_only': ShortOnlyStrategy,
        'cfd_leveraged': CFDLeveragedStrategy,
        'long_short': CFDLeveragedStrategy,  # Can use same class or create specific one
    }

    # Register each strategy
    for name, config_dict in strategies_dict.items():
        strategy_type = config_dict['strategy_type']
        strategy_class = strategy_classes.get(strategy_type)

        if not strategy_class:
            print(f"[WARN] Unknown strategy type '{strategy_type}' for '{name}', skipping")
            continue

        # Create config object
        config = StrategyConfig(**config_dict)

        # Instantiate strategy
        strategy = strategy_class(config)

        # Add to manager
        manager.strategies[name] = strategy
        print(f"[STRATEGY] Registered: {name} ({strategy_type}, leverage={config.leverage}x)")

    return manager


# -----------------------------
# Main trading logic with strategy
# -----------------------------
def run_once(strategy_name: str, dry_run: bool = False, use_news: bool = True):
    print("=" * 70)
    print(f"LSTM QQQ Bot - Strategy: {strategy_name}")
    print("=" * 70)

    # Initialize strategy manager and set active strategy
    strategy_manager = initialize_strategy_manager()

    if strategy_name not in strategy_manager.strategies:
        available = ', '.join(strategy_manager.strategies.keys())
        raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")

    strategy_manager.set_active_strategy(strategy_name)
    strategy = strategy_manager.get_active_strategy()

    print(f"[STRATEGY] Active: {strategy.config.name}")
    print(f"           Type: {strategy.config.strategy_type}")
    print(f"           Entry: {strategy.config.entry_threshold:.6f}")
    print(f"           Leverage: {strategy.config.leverage}x")
    print(f"           Max Positions: {strategy.config.max_positions}")

    # Initialize broker
    broker = AlpacaBroker()
    acct = broker.get_account_info()
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", 0))
    print(f"[ACCOUNT] Equity=${equity:,.2f} Cash=${cash:,.2f}")

    # Load model
    model, scaler_y, scaler_X = load_lstm_model()
    feat_list = load_feature_list(FEATURE_LIST_PATH)
    print(f"[MODEL] Input={INPUT_SIZE} Features={len(feat_list)}")

    if len(feat_list) != INPUT_SIZE:
        raise ValueError(f"Feature mismatch! Found {len(feat_list)} but model expects {INPUT_SIZE}")

    # News provider
    news_provider = None
    if use_news:
        try:
            news_provider = NewsFeatureProvider(decay_lambda=0.001, cache_minutes=5)
            print("[NEWS] Alpha Vantage news enabled")
        except ValueError as e:
            print(f"[NEWS WARNING] {e}")
            print("[NEWS] Falling back to neutral news features")
            use_news = False

    # Download data
    df_raw = download_qqq_data(days=5)
    if df_raw.empty:
        print("[ERROR] No yfinance data.")
        return

    # RTH filter
    end_dt = datetime.now(tz=EASTERN)
    start_dt = end_dt - timedelta(days=10)
    cal_map = build_calendar_map(start_dt, end_dt)

    df_rth = df_raw[df_raw.index.to_series().map(lambda ts: is_rth(ts, cal_map))]
    if df_rth.empty:
        print("[WARN] No RTH bars.")
        return

    if len(df_rth) < SEQUENCE_LENGTH + 2:
        print("[ERROR] Not enough bars.")
        return

    # Last completed minute bar
    bar_time = df_rth.index[-2]
    val = df_rth["Close"].iloc[-2]
    last_completed_price = float(val.item() if hasattr(val, "item") else val)

    # Build features
    df_feat, last_ts = build_features_no_news(df_rth)

    if len(df_feat.columns) > 0 and isinstance(df_feat.columns[0], tuple):
        df_feat.columns = [col[0] if isinstance(col, tuple) else col for col in df_feat.columns]

    # News
    news_val = {
        "last_news_sentiment": 0.0,
        "news_age_minutes": 0.0,
        "effective_sentiment_t": 0.0
    }

    if use_news and news_provider is not None:
        try:
            current_time = bar_time.to_pydatetime()
            news_val = news_provider.get_news_features_dict(current_time, tickers=["QQQ"])
            print(f"[NEWS] S={news_val['last_news_sentiment']:.4f} "
                  f"Age={news_val['news_age_minutes']:.1f} "
                  f"Eff={news_val['effective_sentiment_t']:.4f}")
        except Exception as e:
            print(f"[NEWS FAIL] {e}")

    # Add news to features
    df_feat["last_news_sentiment"] = news_val["last_news_sentiment"]
    df_feat["news_age_minutes"] = news_val["news_age_minutes"]
    df_feat["effective_sentiment_t"] = news_val["effective_sentiment_t"]

    # Build feature matrix
    X_list = []
    for feat in feat_list:
        if feat in df_feat.columns:
            X_list.append(df_feat[feat].values.astype(np.float32))
        else:
            raise ValueError(f"Feature '{feat}' missing from live DF!")

    X_raw = np.column_stack(X_list).astype(np.float32)
    X_df_raw = pd.DataFrame(X_raw, columns=feat_list)
    X = scaler_X.transform(X_df_raw)

    # Create sequence
    X_seq = create_last_sequence(X, SEQUENCE_LENGTH)
    if X_seq.size == 0:
        print("[ERROR] not enough data for sequence")
        return

    X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)

    # Predict
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()[0]
    pred = scaler_y.inverse_transform([pred_scaled])[0]
    pred = pred / 100.0  # Convert to decimal

    prediction_dict = {
        '1m': pred[0],
        '3m': pred[1],
        '5m': pred[2],
        '10m': pred[3],
        '15m': pred[4]
    }

    print(f"[PRED] 1m={pred[0]*100:.3f}% 3m={pred[1]*100:.3f}% "
          f"5m={pred[2]*100:.3f}% 10m={pred[3]*100:.3f}% 15m={pred[4]*100:.3f}%")

    # Calculate market data
    market_data = {
        'current_price': last_completed_price,
        'volatility': df_rth['Close'].pct_change().std() * np.sqrt(252 * 390),  # Annualized
        'volume': float(df_rth['Volume'].iloc[-1])
    }

    # ===== USE STRATEGY =====
    signal, metadata = strategy.calc_signal(prediction_dict, market_data)

    print(f"[SIGNAL] {signal:.6f} (threshold={strategy.config.entry_threshold:.6f})")
    print(f"[META] {metadata}")

    # Check position
    position = broker.get_position(TICKER)

    if position is None:
        # No position - check entry
        can_enter, reason = strategy.can_enter(TICKER, signal, metadata, broker)

        if can_enter:
            qty = strategy.calculate_position_size(TICKER, signal, equity, last_completed_price)

            if qty > 0:
                # Determine side based on signal
                side = 'buy' if signal > 0 else 'sell'
                order_params = strategy.get_order_params(TICKER, qty, side, last_completed_price)

                print(f"[ENTRY] {order_params['side'].upper()} {qty} {TICKER} @ ${last_completed_price:.2f}")
                print(f"        SL={order_params['stop_loss']['stop_price']} "
                      f"TP={order_params['take_profit']['limit_price']}")

                if not dry_run:
                    broker.submit_order(order_params)
                    last_trade_time[TICKER] = datetime.now(timezone.utc)
                else:
                    print("[DRY RUN] Order not submitted")
            else:
                print("[WARN] Position size calculated as 0")
        else:
            print(f"[NO ENTRY] {reason}")

    else:
        # Have position - check exit
        should_exit, reason = strategy.should_exit(TICKER, position, signal, metadata, broker)

        if should_exit:
            print(f"[EXIT] {reason}")
            if not dry_run:
                broker.close_position(TICKER)
            else:
                print("[DRY RUN] Not closing position")
        else:
            qty = position.get('qty', 0)
            entry_price = float(position.get('avg_entry_price', 0))
            cur_price = float(position.get('current_price', 0))
            uplpc = float(position.get('unrealized_plpc', 0))
            print(f"[HOLD] {TICKER} qty={qty} entry=${entry_price:.2f} "
                  f"cur=${cur_price:.2f} upl={uplpc*100:.2f}% | {reason}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="LSTM Trading Bot with Strategy Selection")
    ap.add_argument("--strategy", type=str, required=False,
                    help="Strategy name (e.g., 'aggressive_long', 'cfd_2x_leverage')")
    ap.add_argument("--list-strategies", action="store_true",
                    help="List all available strategies")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run without executing trades")
    ap.add_argument("--loop", action="store_true",
                    help="Run continuously")
    ap.add_argument("--interval", type=int, default=300,
                    help="Loop interval in seconds (default: 300)")
    ap.add_argument("--skip-market-hours", action="store_true",
                    help="Skip market hours check")
    ap.add_argument("--no-news", action="store_true",
                    help="Disable news features")

    args = ap.parse_args()

    # List strategies
    if args.list_strategies:
        print("\n" + "="*70)
        print("AVAILABLE STRATEGIES")
        print("="*70)

        strategies_dict = load_strategies_from_json()

        for name, config in strategies_dict.items():
            print(f"\n  {name}:")
            print(f"    Name: {config['name']}")
            print(f"    Type: {config['strategy_type']}")
            print(f"    Entry Threshold: {config['entry_threshold']:.6f}")
            print(f"    Position Size: {config['position_size_pct']*100:.1f}%")
            print(f"    Max Positions: {config['max_positions']}")
            print(f"    Leverage: {config['leverage']}x")
            print(f"    Stop Loss: {config['stop_loss_pct']*100:.2f}%")
            print(f"    Take Profit: {config['take_profit_pct']*100:.2f}%")

        print("\n" + "="*70)
        print("Usage: python lstm_deploy.py --strategy <name> [options]")
        print("="*70 + "\n")
        return

    # Require strategy if not listing
    if not args.strategy:
        print("[ERROR] --strategy is required (or use --list-strategies)")
        return

    # Run
    if args.loop:
        print(f"\n[LOOP] Running every {args.interval}s with strategy: {args.strategy}")
        i = 0
        try:
            while True:
                i += 1
                now = datetime.now(EASTERN)

                is_weekday = now.weekday() < 5
                market_open = now.time() >= datetime.strptime("09:30", "%H:%M").time()
                market_close = now.time() <= datetime.strptime("16:00", "%H:%M").time()
                is_market_hours = is_weekday and market_open and market_close

                print(f"\n--- RUN #{i} {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

                if args.skip_market_hours or is_market_hours:
                    if args.skip_market_hours and not is_market_hours:
                        print("[WARNING] Market CLOSED but running anyway")

                    run_once(
                        strategy_name=args.strategy,
                        dry_run=args.dry_run,
                        use_news=not args.no_news
                    )
                else:
                    print("[SKIP] Market CLOSED - Waiting for market hours")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n[STOPPED] By user. Total runs: {i}")
    else:
        run_once(
            strategy_name=args.strategy,
            dry_run=args.dry_run,
            use_news=not args.no_news
        )


if __name__ == "__main__":
    main()