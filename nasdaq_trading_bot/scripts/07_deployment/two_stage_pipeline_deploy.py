"""
Two Stage Pipeline Deployment - Continuous Event Monitor
=========================================================
Treats every time point as a potential "event" and extracts pre-window features.

Strategy:
- Every N minutes, extract last 20min as "pre-window"  
- Build event features + get news sentiment
- Apply Stage 1 filter + Stage 2 direction classifier
- Trade LONG if both stages agree
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
import pytz
import requests
import joblib
import json

import yfinance as yf
import torch
from torch import nn

# Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

CONF_DIR = os.path.join(PROJECT_ROOT, "conf")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "two_stage_pipeline")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCALER_X_PATH = os.path.join(DATA_DIR, "event_scaler_X.joblib")

# Load configs
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

with open(os.path.join(CONF_DIR, "keys.yaml"), "r") as f:
    keys = yaml.safe_load(f)

# Trading params
TICKER = "QQQ"
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.02
COOLDOWN_MINUTES = 60

STOP_LOSS_PCT = -0.01
TAKE_PROFIT_PCT = 0.015

PRE_WINDOW_MINUTES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EASTERN = pytz.timezone("US/Eastern")

# Alpaca
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", keys["KEYS"].get("APCA-API-KEY-ID-Paper"))
ALPACA_SECRET = os.getenv("ALPACA_SECRET", keys["KEYS"].get("APCA-API-SECRET-KEY-Paper"))
ALPACA_BASE = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

last_trade_time: Dict[str, datetime] = {}


# Stage 1 Model
class Stage1TradeFilter(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout_p=0.30):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)


def alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }


def get_account_info() -> dict:
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=alpaca_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def get_position(symbol: str) -> Optional[dict]:
    r = requests.get(f"{ALPACA_BASE}/v2/positions/{symbol}", headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def submit_bracket_market(symbol: str, qty: int, sl_price: float, tp_price: float) -> Optional[dict]:
    payload = {
        "symbol": symbol, "qty": qty, "side": "buy", "type": "market",
        "time_in_force": "day", "order_class": "bracket",
        "take_profit": {"limit_price": f"{tp_price:.2f}"},
        "stop_loss": {"stop_price": f"{sl_price:.2f}"},
    }
    try:
        r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=alpaca_headers(), json=payload, timeout=30)
        r.raise_for_status()
        od = r.json()
        print(f"[ORDER] BUY {qty} {symbol} @ SL={sl_price:.2f} TP={tp_price:.2f}")
        return od
    except Exception as e:
        print(f"[ERROR] Order failed: {e}")
        return None


def download_qqq_data(days: int = 2) -> pd.DataFrame:
    df = yf.download(TICKER, period=f"{days}d", interval="1m", auto_adjust=True, prepost=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def extract_event_features_from_window(pre_window: pd.DataFrame, news_sentiment: float = 0.0, event_time: Optional[datetime] = None) -> Dict:
    """Extract event features from pre-window"""
    features = {}
    
    if len(pre_window) < 2:
        return None
    
    prices = pre_window['open'].values
    
    # Returns
    if len(pre_window) >= 5:
        features['pre_return_5m'] = (prices[-1] / prices[-5] - 1) * 100
    else:
        features['pre_return_5m'] = 0.0
    
    if len(pre_window) >= 10:
        features['pre_return_10m'] = (prices[-1] / prices[-10] - 1) * 100
    else:
        features['pre_return_10m'] = 0.0
    
    features['pre_return_20m'] = (prices[-1] / prices[0] - 1) * 100
    
    # Trend
    x = np.arange(len(prices))
    if len(x) > 1:
        slope, _ = np.polyfit(x, prices, 1)
        features['pre_price_trend'] = slope
    else:
        features['pre_price_trend'] = 0.0
    
    features['pre_price_std'] = np.std(prices)
    
    # Volatility features
    if len(pre_window) >= 2:
        # Realized volatility (std of log returns)
        log_returns = np.log(prices[1:] / prices[:-1])
        features['pre_realized_vol'] = np.std(log_returns)
        
        # High-Low span
        if 'high' in pre_window.columns and 'low' in pre_window.columns:
            hl_spans = pre_window['high'] - pre_window['low']
            features['pre_hl_span_mean'] = np.mean(hl_spans)
        else:
            features['pre_hl_span_mean'] = 0.0
    else:
        features['pre_realized_vol'] = 0.0
        features['pre_hl_span_mean'] = 0.0
    
    # Volume
    if 'volume' in pre_window.columns:
        volumes = pre_window['volume'].values
        features['pre_volume_mean'] = np.mean(volumes)
        features['pre_volume_std'] = np.std(volumes)
        features['pre_volume_spike'] = np.max(volumes) / features['pre_volume_mean'] if features['pre_volume_mean'] > 0 else 1.0
        if len(volumes) > 1:
            slope_v, _ = np.polyfit(x, volumes, 1)
            features['pre_volume_trend'] = slope_v
        else:
            features['pre_volume_trend'] = 0.0
    else:
        features['pre_volume_mean'] = 0.0
        features['pre_volume_std'] = 0.0
        features['pre_volume_spike'] = 1.0
        features['pre_volume_trend'] = 0.0
    
    features['pre_trade_count_mean'] = features['pre_volume_mean'] / 100.0
    features['pre_avg_trade_size'] = 100.0
    
    # Time features
    if event_time is not None:
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        else:
            event_time = event_time.astimezone(timezone.utc)
        
        features['news_hour'] = event_time.hour
        features['news_minute'] = event_time.minute
        features['news_day_of_week'] = event_time.weekday()
        
        # Minutes since market open (9:30 ET = 14:30 UTC)
        market_open_utc = event_time.replace(hour=14, minute=30, second=0, microsecond=0)
        if event_time >= market_open_utc:
            features['minutes_since_market_open'] = (event_time - market_open_utc).total_seconds() / 60.0
        else:
            features['minutes_since_market_open'] = 0.0
    else:
        # Fallback to current time
        now = datetime.now(timezone.utc)
        features['news_hour'] = now.hour
        features['news_minute'] = now.minute
        features['news_day_of_week'] = now.weekday()
        features['minutes_since_market_open'] = 60.0
    
    features['news_sentiment'] = news_sentiment
    features['pre_bars_count'] = len(pre_window)
    features['pre_coverage_pct'] = (len(pre_window) / PRE_WINDOW_MINUTES) * 100.0
    
    return features


def create_sentiment_features(sentiment: float) -> np.ndarray:
    """Create 4 sentiment features for Stage 2"""
    return np.array([
        sentiment,
        abs(sentiment),
        1.0 if sentiment > 0 else 0.0,
        1.0 if sentiment < 0 else 0.0
    ]).reshape(1, -1)


def get_news_sentiment() -> float:
    """Get news sentiment - placeholder for now"""
    try:
        # Integrate your NewsFeatureProvider here
        return 0.0
    except:
        return 0.0


def load_models():
    """Load Stage 1 and Stage 2"""
    config_path = os.path.join(MODEL_DIR, "final_results.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Stage 1
    stage1_path = os.path.join(MODEL_DIR, "best_stage1_trade_filter.pt")
    checkpoint = torch.load(stage1_path, map_location=DEVICE)
    hidden_dims = checkpoint.get("hidden_dims", (64, 32))
    dropout_p = checkpoint.get("dropout_p", 0.30)
    
    feature_names_path = os.path.join(DATA_DIR, "event_feature_names.txt")
    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    stage1_model = Stage1TradeFilter(input_dim=len(feature_names), hidden_dims=hidden_dims, dropout_p=dropout_p).to(DEVICE)
    stage1_model.load_state_dict(checkpoint["state_dict"])
    stage1_model.eval()
    
    # Stage 2
    stage2_path = os.path.join(MODEL_DIR, "best_stage2_direction.joblib")
    stage2_model = joblib.load(stage2_path)
    
    scaler_X = joblib.load(SCALER_X_PATH)
    
    return stage1_model, stage2_model, scaler_X, config, feature_names


def run_once(dry_run: bool = False):
    print("=" * 70)
    print("TWO STAGE PIPELINE - Continuous Event Monitor")
    print("=" * 70)

    acct = get_account_info()
    equity = float(acct.get("equity", 0))
    print(f"[ACCOUNT] Equity=${equity:,.2f}")

    stage1_model, stage2_model, scaler_X, config, feature_names = load_models()
    stage1_threshold = config["stage1"]["threshold"] if config["stage1"] else 0.5
    stage2_threshold = config["stage2"]["threshold"] if config["stage2"] else 0.5
    
    print(f"[CONFIG] Stage1={stage1_threshold:.3f}, Stage2={stage2_threshold:.3f}")

    # Download data
    df_raw = download_qqq_data(days=2)
    if df_raw.empty:
        print("[ERROR] No data")
        return
    
    df_raw = df_raw.rename(columns={"Open": "open", "High": "high", "Low": "low", 
                                     "Close": "close", "Volume": "volume"})
    
    if len(df_raw) < PRE_WINDOW_MINUTES + 2:
        print(f"[ERROR] Not enough bars")
        return
    
    pre_window = df_raw.iloc[-PRE_WINDOW_MINUTES-1:-1]
    current_price = float(df_raw['close'].iloc[-1])
    
    print(f"[DATA] Pre-window: {len(pre_window)} bars, Current: ${current_price:.2f}")
    
    # Get news sentiment
    news_sentiment = get_news_sentiment()
    
    # Extract features
    event_time = datetime.now(timezone.utc)
    features_dict = extract_event_features_from_window(pre_window, news_sentiment, event_time)
    
    if features_dict is None:
        print("[ERROR] Could not extract features")
        return
    
    # Build feature vector
    X_list = []
    for feat_name in feature_names:
        if feat_name in features_dict:
            X_list.append(features_dict[feat_name])
        else:
            X_list.append(0.0)
    
    X_raw = np.array([X_list], dtype=np.float32)
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    X = scaler_X.transform(X_df)
    
    # Stage 1: Trade Filter
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        stage1_logits = stage1_model(X_tensor).cpu().numpy()[0]
    stage1_prob = 1.0 / (1.0 + np.exp(-stage1_logits))
    
    print(f"[STAGE 1] Filter prob: {stage1_prob:.4f} (threshold: {stage1_threshold:.3f})")
    
    if stage1_prob < stage1_threshold:
        print("[NO TRADE] Stage 1 filtered out")
        return
    
    # Stage 2: Direction (sentiment-based)
    X_stage2 = create_sentiment_features(news_sentiment)
    stage2_prob = stage2_model.predict_proba(X_stage2)[0, 1]  # P(UP)
    
    print(f"[STAGE 2] UP prob: {stage2_prob:.4f}, Sentiment: {news_sentiment:+.4f}")
    
    if stage2_prob < stage2_threshold:
        print(f"[NO TRADE] Stage 2 prob < threshold")
        return
    
    print(f"[SIGNAL] LONG with UP prob={stage2_prob:.4f}")
    
    # Position management
    pos = get_position(TICKER)
    
    if pos is None:
        if TICKER in last_trade_time:
            dt = datetime.now(timezone.utc) - last_trade_time[TICKER]
            if dt < timedelta(minutes=COOLDOWN_MINUTES):
                print(f"[COOLDOWN] {(COOLDOWN_MINUTES - dt.total_seconds()/60):.1f} min left")
                return
        
        target_value = equity * POSITION_SIZE_PCT
        qty = int(target_value / current_price)
        if qty <= 0:
            print("[WARN] qty=0")
            return

        sl = current_price * (1 + STOP_LOSS_PCT)
        tp = current_price * (1 + TAKE_PROFIT_PCT)

        print(f"[ENTRY] BUY {TICKER} qty={qty} @ ${current_price:.2f}")
        if not dry_run:
            od = submit_bracket_market(TICKER, qty, sl, tp)
            if od:
                last_trade_time[TICKER] = datetime.now(timezone.utc)
        else:
            print("[DRY RUN]")
    
    elif pos is not None:
        qty = pos.get("qty")
        entry = float(pos.get("avg_entry_price", 0))
        upl = float(pos.get("unrealized_plpc", 0))
        print(f"[HOLD] qty={qty} entry=${entry:.2f} P&L={upl*100:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()

    if args.loop:
        i = 0
        try:
            while True:
                i += 1
                now = datetime.now(EASTERN)
                print(f"\n{'='*70}")
                print(f"RUN #{i} - {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"{'='*70}")
                
                is_weekday = now.weekday() < 5
                market_open = now.time() >= datetime.strptime("09:30", "%H:%M").time()
                market_close = now.time() <= datetime.strptime("16:00", "%H:%M").time()
                
                if is_weekday and market_open and market_close:
                    try:
                        run_once(dry_run=args.dry_run)
                    except Exception as e:
                        print(f"[ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[SKIP] Market closed")
                
                print(f"\n[SLEEP] Waiting {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n[STOPPED] Total runs: {i}")
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
