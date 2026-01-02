"""
Backtest for Final 2-Stage Pipeline (Quantile q=0.35 + Sentiment Direction)

This script:
1. Loads trained Stage 1 (trade filter) and Stage 2 (direction) models
2. Applies 2-stage filtering to test data
3. Simulates LONG trades on predicted Up events
4. Calculates profitability after transaction costs (10 bps)
5. Reports Sharpe ratio, mean returns, win rate
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ======================================================================================
# MODEL LOADING
# ======================================================================================

class Stage1TradeFilter(torch.nn.Module):
    """Binary FFN for Stage 1 trade filtering."""
    
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout_p=0.30):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout_p)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def create_sentiment_features(sentiment_series):
    """Create 4 sentiment features."""
    sent = sentiment_series.fillna(0.0).values
    return np.column_stack([
        sent,
        np.abs(sent),
        (sent > 0).astype(float),
        (sent < 0).astype(float)
    ])


# ======================================================================================
# BACKTEST
# ======================================================================================

def run_backtest(
    split_name='test', 
    transaction_cost_bps=2,
    filter_mode='none',  # 'none', 'top_k', 'min_prob', 'expected_value'
    top_k_pct=0.20,  # Top K% of predictions (for top_k mode)
    stage2_min_prob=0.60,  # Min probability (for min_prob mode)
    avg_move_estimate=0.15,  # Average move size in % (for EV mode)
    enable_short_trades=True,  # Enable SHORT trades (NEW)
    up_threshold=0.60,  # LONG if p_up >= this (NEW)
    down_threshold=0.60  # SHORT if p_up <= (1 - this) (NEW)
):
    """
    Run backtest on specified data split with advanced profitability filters and SHORT trades.
    
    Args:
        split_name: 'train', 'val', or 'test'
        transaction_cost_bps: Transaction cost in basis points (default 2 = 0.02%)
        filter_mode: Profitability filter strategy
        top_k_pct: Percentage of top predictions to trade (0.0-1.0)
        stage2_min_prob: Minimum Stage-2 probability for min_prob mode
        avg_move_estimate: Estimated average move size in % for EV calculation
        enable_short_trades: If True, trade both LONG and SHORT
        up_threshold: LONG if Stage-2 p_up >= this value
        down_threshold: SHORT if Stage-2 p_up <= (1 - this value)
    """
    
    print("\n" + "="*90)
    print(f"FINAL 2-STAGE PIPELINE BACKTEST ({split_name.upper()})")
    print("="*90)
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'models', 'two_stage_pipeline')
    
    # Load configuration
    with open(os.path.join(model_dir, 'final_results.json'), 'r') as f:
        config = json.load(f)
    
    q = config['q']
    abs_threshold = config['abs_threshold']
    stage1_threshold = config['stage1']['threshold']
    stage2_threshold = config['stage2']['threshold'] if config['stage2'] else 0.5
    
    print(f"\nConfiguration:")
    print(f"  Quantile: q={q}")
    print(f"  Stage 1 threshold: |return| >= {abs_threshold:.6f}%")
    print(f"  Stage 1 prob threshold: {stage1_threshold:.3f}")
    print(f"  Stage 2 prob threshold: {stage2_threshold:.3f}")
    print(f"  Profitability filter mode: {filter_mode}")
    if filter_mode == 'top_k':
        print(f"    → Top {top_k_pct*100:.0f}% of Stage-2 predictions")
    elif filter_mode == 'min_prob':
        print(f"    → Stage-2 prob >= {stage2_min_prob:.2f}")
    elif filter_mode == 'expected_value':
        print(f"    → Expected edge > {transaction_cost_bps} bps (avg move: {avg_move_estimate:.2f}%)")
    print(f"  SHORT trades enabled: {enable_short_trades}")
    if enable_short_trades:
        print(f"    → LONG if p_up >= {up_threshold:.2f}")
        print(f"    → SHORT if p_up <= {1-down_threshold:.2f}")
    print(f"  Transaction cost: {transaction_cost_bps} bps ({transaction_cost_bps/100:.2f}%)")
    
    # Load data WITH ALIGNMENT - use CSV for guaranteed order consistency
    # This ensures X, y, and metadata (sentiment) are properly aligned by row index
    if split_name == 'train':
        df = pd.read_csv(os.path.join(data_dir, 'event_train.csv'))
        X_scaled_df = pd.read_csv(os.path.join(data_dir, 'event_X_train_scaled.csv'))
        y_df = pd.read_csv(os.path.join(data_dir, 'event_y_train.csv'))
    elif split_name == 'val':
        df = pd.read_csv(os.path.join(data_dir, 'event_validation.csv'))
        X_scaled_df = pd.read_csv(os.path.join(data_dir, 'event_X_val_scaled.csv'))
        y_df = pd.read_csv(os.path.join(data_dir, 'event_y_val.csv'))
    else:  # test
        df = pd.read_csv(os.path.join(data_dir, 'event_test.csv'))
        X_scaled_df = pd.read_csv(os.path.join(data_dir, 'event_X_test_scaled.csv'))
        y_df = pd.read_csv(os.path.join(data_dir, 'event_y_test.csv'))
    
    # Verify alignment (critical for correct backtest results)
    assert len(df) == len(X_scaled_df) == len(y_df), \
        f"Data alignment error! df={len(df)}, X={len(X_scaled_df)}, y={len(y_df)}"
    
    # Extract arrays in guaranteed aligned order
    X = X_scaled_df.values
    y_cont = y_df.values.flatten()
    
    print(f"\nLoaded {split_name} data: {len(X)} events")
    print(f"Verified alignment: X, y, and metadata all have {len(df)} rows")
    
    # Load Stage 1 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stage1_model = Stage1TradeFilter(input_dim=X.shape[1])
    stage1_ckpt = torch.load(os.path.join(model_dir, 'best_stage1_trade_filter.pt'), map_location=device)
    stage1_model.load_state_dict(stage1_ckpt['state_dict'])
    stage1_model.eval()
    
    # Stage 1: Predict tradeable events
    with torch.no_grad():
        logits = stage1_model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    stage1_probs = sigmoid(logits)
    stage1_preds = (stage1_probs >= stage1_threshold).astype(np.int64)
    
    n_stage1_trade = stage1_preds.sum()
    print(f"\nStage 1 filtered: {n_stage1_trade}/{len(X)} events ({n_stage1_trade/len(X)*100:.1f}%)")
    
    if n_stage1_trade == 0:
        print("No tradeable events after Stage 1 filtering!")
        return
    
    # Load Stage 2 model
    stage2_model = joblib.load(os.path.join(model_dir, 'best_stage2_direction.joblib'))
    
    # Extract sentiment features for Stage 2
    sent_features = create_sentiment_features(df.get('news_sentiment', pd.Series([0.0]*len(df))))
    
    # Apply Stage 2 only to Stage 1 filtered events
    mask_trade = stage1_preds == 1
    X_stage2 = sent_features[mask_trade]
    y_stage2_actual = (y_cont[mask_trade] > 0).astype(np.int64)
    returns_stage2 = y_cont[mask_trade]
    
    # Stage 2: Predict direction
    stage2_probs = stage2_model.predict_proba(X_stage2)[:, 1]
    stage2_preds = (stage2_probs >= stage2_threshold).astype(np.int64)  # 1 = Up, 0 = Down
    
    n_stage2_up = stage2_preds.sum()
    print(f"Stage 2 predicted UP: {n_stage2_up}/{n_stage1_trade} events ({n_stage2_up/n_stage1_trade*100:.1f}%)")
    
    if n_stage2_up == 0:
        print("No UP predictions from Stage 2!")
        return
    
    # ======================================================================================
    # BACKTEST: LONG-ONLY or LONG/SHORT Strategy
    # ======================================================================================
    
    print("\n" + "="*90)
    if enable_short_trades:
        print("BACKTEST: LONG/SHORT STRATEGY (Trade UP and DOWN predictions)")
    else:
        print("BACKTEST: LONG-ONLY STRATEGY (Trade UP predictions)")
    print("="*90)
    
    cost_per_trade = transaction_cost_bps / 100.0  # Convert bps to percentage points
    
    # Initialize trade lists
    all_gross_pnl = []
    all_net_pnl = []
    all_actual_dir = []
    
    n_long = 0
    n_short = 0
    
    if enable_short_trades:
        # ========== LONG/SHORT MODE ==========
        # Classify trades based on Stage-2 probabilities
        long_mask_initial = stage2_probs >= up_threshold
        short_mask_initial = stage2_probs <= (1 - down_threshold)
        
        print(f"\nTrade Classification:")
        print(f"  Initial LONG signals: {long_mask_initial.sum()} (p_up >= {up_threshold:.2f})")
        print(f"  Initial SHORT signals: {short_mask_initial.sum()} (p_up <= {1-down_threshold:.2f})")
        print(f"  Neutral (no trade): {(~(long_mask_initial | short_mask_initial)).sum()}")
        
        # Apply filters (simplified - no top_k for now, just min_prob)
        long_mask_final = long_mask_initial.copy()
        short_mask_final = short_mask_initial.copy()
        
        if filter_mode == 'min_prob':
            # Filter LONG: keep only high prob
            long_high_prob = stage2_probs >= stage2_min_prob
            long_mask_final = long_mask_initial & long_high_prob
            filtered_long = long_mask_initial.sum() - long_mask_final.sum()
            print(f"\nMin Prob Filter (LONG >= {stage2_min_prob:.2f}): Filtered out {filtered_long} trades")
            
            # Filter SHORT: keep only low prob  
            short_low_prob = stage2_probs <= (1 - stage2_min_prob)
            short_mask_final = short_mask_initial & short_low_prob
            filtered_short = short_mask_initial.sum() - short_mask_final.sum()
            print(f"Min Prob Filter (SHORT <= {1-stage2_min_prob:.2f}): Filtered out {filtered_short} trades")
        
        # Calculate PnL for LONG trades
        if long_mask_final.sum() > 0:
            long_returns = returns_stage2[long_mask_final]
            long_actual_dir = y_stage2_actual[long_mask_final]
            
            long_gross = long_returns
            long_net = long_gross - cost_per_trade
            
            all_gross_pnl.append(long_gross)
            all_net_pnl.append(long_net)
            all_actual_dir.append(long_actual_dir)
            n_long = len(long_returns)
        
        # Calculate PnL for SHORT trades
        if short_mask_final.sum() > 0:
            short_returns = returns_stage2[short_mask_final]
            short_actual_dir = y_stage2_actual[short_mask_final]
            
            # SHORT: Profit when return is negative (inverse)
            short_gross = -short_returns
            short_net = short_gross - cost_per_trade
            
            # For SHORT, "correct direction" = actual went DOWN
            short_correct_dir = 1 - short_actual_dir  # If actual=1 (UP), wrong for SHORT. If actual=0 (DOWN), right for SHORT.
            
            all_gross_pnl.append(short_gross)
            all_net_pnl.append(short_net)
            all_actual_dir.append(short_correct_dir)
            n_short = len(short_returns)
    
    else:
        # ========== LONG-ONLY MODE ==========
        trade_mask_initial = stage2_preds == 1
        
        print(f"\nTrade Classification:")
        print(f"  Initial LONG signals: {trade_mask_initial.sum()}")
        
        # Apply filters
        trade_mask_final = trade_mask_initial.copy()
        
        if filter_mode == 'top_k' and trade_mask_initial.sum() > 0:
            n_top = max(1, int(trade_mask_initial.sum() * top_k_pct))
            up_indices = np.where(trade_mask_initial)[0]
            up_probs = stage2_probs[up_indices]
            top_k_local = np.argsort(up_probs)[-n_top:]
            top_k_global = up_indices[top_k_local]
            
            new_mask = np.zeros(len(trade_mask_initial), dtype=bool)
            new_mask[top_k_global] = True
            trade_mask_final = new_mask
            
            print(f"\nTop-K Filter: Selected {n_top}/{trade_mask_initial.sum()} trades (top {top_k_pct*100:.0f}%)")
        
        elif filter_mode == 'min_prob' and trade_mask_initial.sum() > 0:
            high_prob = stage2_probs >= stage2_min_prob
            trade_mask_final = trade_mask_initial & high_prob
            filtered = trade_mask_initial.sum() - trade_mask_final.sum()
            print(f"\nMin Prob Filter (>= {stage2_min_prob:.2f}): Filtered out {filtered} trades")
        
        # Calculate PnL for LONG trades
        if trade_mask_final.sum() > 0:
            traded_returns = returns_stage2[trade_mask_final]
            traded_actual_dir = y_stage2_actual[trade_mask_final]
            
            traded_gross = traded_returns
            traded_net = traded_gross - cost_per_trade
            
            all_gross_pnl.append(traded_gross)
            all_net_pnl.append(traded_net)
            all_actual_dir.append(traded_actual_dir)
            n_long = len(traded_returns)
    
    # ========== COMBINE RESULTS ==========
    if len(all_gross_pnl) == 0:
        print("\nNo trades executed. Adjust filter settings.")
        return
    
    all_gross_pnl = np.concatenate(all_gross_pnl)
    all_net_pnl = np.concatenate(all_net_pnl)
    all_actual_dir = np.concatenate(all_actual_dir)
    
    n_trades = len(all_gross_pnl)
    
    # Calculate metrics
    total_gross_pnl = all_gross_pnl.sum()
    total_net_pnl = all_net_pnl.sum()
    mean_gross_return = all_gross_pnl.mean()
    mean_net_return = all_net_pnl.mean()
    
    # Separate GROSS and NET outcomes for clarity
    gross_wins = (all_gross_pnl > 0).sum()
    gross_win_rate = gross_wins / n_trades if n_trades > 0 else 0
    
    net_wins = (all_net_pnl > 0).sum()
    net_losses = (all_net_pnl <= 0).sum()
    net_win_rate = net_wins / n_trades if n_trades > 0 else 0
    
    # Sharpe ratio (annualized, assuming ~252 trading days, ~6.5 hours per day, events every ~20 mins → ~195 events/day)
    if all_net_pnl.std() > 0:
        sharpe = (mean_net_return / all_net_pnl.std()) * np.sqrt(195 * 252)
    else:
        sharpe = 0.0
    
    # Direction accuracy: Percentage that actually went in the predicted direction (on gross returns)
    direction_accuracy = all_actual_dir.mean() if n_trades > 0 else 0.0
    
    print(f"\nTrading Statistics:")
    print(f"  Total events: {len(X)}")
    print(f"  Stage 1 filtered: {n_stage1_trade}")
    print(f"  Stage 2 UP predicted: {n_stage2_up}")
    if enable_short_trades:
        print(f"  LONG trades: {n_long}")
        print(f"  SHORT trades: {n_short}")
    print(f"  Total trades: {n_trades}")
    
    print(f"\nPerformance:")
    print(f"  Mean gross return: {mean_gross_return:.4f}% ({mean_gross_return*100:.2f} bps)")
    print(f"  Transaction cost: {cost_per_trade*100:.4f}% ({transaction_cost_bps:.0f} bps)")
    print(f"  Mean net return: {mean_net_return:.4f}% ({mean_net_return*100:.2f} bps)")
    print(f"  Total gross PnL: {total_gross_pnl:.4f}%")
    print(f"  Total net PnL: {total_net_pnl:.4f}%")
    
    print(f"\nTrade Outcomes (GROSS - direction only):")
    print(f"  Correct direction: {gross_wins} ({gross_win_rate*100:.1f}%)")
    print(f"  Wrong direction: {n_trades - gross_wins} ({(1-gross_win_rate)*100:.1f}%)")
    print(f"  Direction accuracy: {direction_accuracy*100:.1f}%")
    
    print(f"\nTrade Outcomes (NET - after {transaction_cost_bps}bps cost):")
    print(f"  Profitable: {net_wins} ({net_win_rate*100:.1f}%)")
    print(f"  Unprofitable: {net_losses} ({(1-net_win_rate)*100:.1f}%)")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe ratio (annualized): {sharpe:.3f}")
    print(f"  Std dev (net): {all_net_pnl.std():.4f}%")
    
    # Verdict
    print("\n" + "="*90)
    if mean_net_return > 0:
        print(f"PROFITABLE: Mean net return = {mean_net_return:.4f}% ({mean_net_return*100:.2f} bps)")
    else:
        print(f"UNPROFITABLE: Mean net return = {mean_net_return:.4f}% ({mean_net_return*100:.2f} bps)")
    
    if sharpe > 1.0:
        print(f"GOOD SHARPE: {sharpe:.3f} (target > 1.0)")
    elif sharpe > 0:
        print(f"LOW SHARPE: {sharpe:.3f} (target > 1.0)")
    else:
        print(f"NEGATIVE SHARPE: {sharpe:.3f}")
    
    print("="*90)
    
    # Save results
    backtest_results = {
        'split': split_name,
        'config': {
            'q': q,
            'abs_threshold': abs_threshold,
            'stage1_threshold': stage1_threshold,
            'stage2_threshold': stage2_threshold,
            'transaction_cost_bps': transaction_cost_bps,
            'filter_mode': filter_mode,
            'top_k_pct': top_k_pct,
            'stage2_min_prob': stage2_min_prob,
            'avg_move_estimate': avg_move_estimate,
            'enable_short_trades': enable_short_trades,
            'up_threshold': up_threshold,
            'down_threshold': down_threshold
        },
        'events': {
            'total': len(X),
            'stage1_filtered': int(n_stage1_trade),
            'stage2_up_predicted': int(n_stage2_up),
            'long_trades_executed': int(n_long),
            'short_trades_executed': int(n_short),
            'total_trades_executed': int(n_trades)
        },
        'performance': {
            'mean_gross_return_pct': float(mean_gross_return),
            'mean_net_return_pct': float(mean_net_return),
            'total_gross_pnl_pct': float(total_gross_pnl),
            'total_net_pnl_pct': float(total_net_pnl),
            'gross_win_rate': float(gross_win_rate),
            'net_win_rate': float(net_win_rate),
            'direction_accuracy': float(direction_accuracy),
            'sharpe_ratio': float(sharpe),
            'std_dev_pct': float(all_net_pnl.std())
        },
        'outcomes': {
            'gross_correct_direction': int(gross_wins),
            'gross_wrong_direction': int(n_trades - gross_wins),
            'net_profitable': int(net_wins),
            'net_unprofitable': int(net_losses)
        }
    }
    
    output_path = os.path.join(model_dir, f'backtest_{split_name}.json')
    with open(output_path, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    print(f"\nBacktest results saved: {output_path}")
    
    return backtest_results


if __name__ == "__main__":
    # Run backtest on all splits with configurable filters
    print("\n" + "="*90)
    print("FINAL 2-STAGE PIPELINE BACKTESTING")
    print("="*90)
    
    # ==================================================================================
    # CONFIGURATION: Choose your profitability filter strategy
    # ==================================================================================
    # Configuration for 60min horizon with SHORT trades
    TRANSACTION_COST_BPS = 2  # 2 bps = 0.02% roundtrip cost
    
    # Filter Mode
    FILTER_MODE = 'top_k'
    TOP_K_PCT = 0.60  # Top 60% of predictions
    
    # SHORT trades (NEW!)
    ENABLE_SHORT_TRADES = True  # Enable bidirectional trading
    UP_THRESHOLD = 0.55  # LONG if p_up >= 0.55 (lowered from 0.60)
    DOWN_THRESHOLD = 0.55  # SHORT if p_up <= 0.45 (lowered from 0.60)
    
    print("\nConfiguration:")
    print(f"  Transaction cost: {TRANSACTION_COST_BPS} bps")
    print(f"  Filter mode: {FILTER_MODE}")
    if FILTER_MODE == 'top_k':
        print(f"    → Top {TOP_K_PCT*100:.0f}% of predictions")
    print(f"  SHORT trades: {ENABLE_SHORT_TRADES}")
    if ENABLE_SHORT_TRADES:
        print(f"    → LONG: p_up >= {UP_THRESHOLD:.2f}")
        print(f"    → SHORT: p_up <= {1-DOWN_THRESHOLD:.2f}")
    print()
    
    for split in ['train', 'val', 'test']:
        run_backtest(
            split_name=split,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            filter_mode=FILTER_MODE,
            top_k_pct=TOP_K_PCT,
            enable_short_trades=ENABLE_SHORT_TRADES,
            up_threshold=UP_THRESHOLD,
            down_threshold=DOWN_THRESHOLD
        )
        print("\n")
