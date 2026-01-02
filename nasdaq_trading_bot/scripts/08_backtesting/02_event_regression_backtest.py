"""
Regression Pipeline Backtest

Tests the 2-stage pipeline with return regression:
- Stage 1: Tradeable filter
- Stage 2: Return regression
- Trading: LONG if pred >= +τ, SHORT if pred <= -τ
"""

import sys
import os

# Add model training directory to path
model_training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '06_model_training', 'event_feed_forward')
sys.path.append(model_training_dir)

import torch
import numpy as np
import pandas as pd
import json

import joblib  # Added for RF
# from event_return_regression import ReturnRegressor # Removed NN


# Simple Binary Classifier for Stage-1
class EventBinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(0.2)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)


def run_regression_backtest(
    split_name='test',
    transaction_cost_bps=2,
    tau=0.10  # Threshold for trading
):
    """
    Run backtest with regression model.
    
    Args:
        split_name: 'train', 'val', or 'test'
        transaction_cost_bps: Transaction cost in bps
        tau: Threshold for trading (e.g., 0.10 = 0.10%)
            - LONG if predicted_return >= +tau
            - SHORT if predicted_return <= -tau
    """
    print("\n" + "="*90)
    print(f"REGRESSION PIPELINE BACKTEST ({split_name.upper()})")
    print("="*90)
    
    # Paths
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models', 'regression_pipeline_60m')
    
    device = 'cpu'
    
    # Load configuration
    with open(os.path.join(model_dir, 'regression_results.json'), 'r') as f:
        config = json.load(f)
    
    stage1_threshold = config['stage1']['threshold']
    abs_threshold = config['abs_threshold']
    
    print(f"\nConfiguration:")
    print(f"  Stage-1 threshold: {stage1_threshold:.3f}")
    print(f"  Abs threshold: {abs_threshold:.6f}%")
    print(f"  Trading threshold τ: {tau:.4f}%")
    print(f"  Transaction cost: {transaction_cost_bps} bps ({transaction_cost_bps/100:.2f}%)")
    
    # Load data
    if split_name == 'train':
        X = np.load(os.path.join(data_dir, 'event_X_train_scaled.npy'))
        y = np.load(os.path.join(data_dir, 'event_y_train.npy')).flatten()
    elif split_name == 'val':
        X = np.load(os.path.join(data_dir, 'event_X_val_scaled.npy'))
        y = np.load(os.path.join(data_dir, 'event_y_val.npy')).flatten()
    else:  # test
        X = np.load(os.path.join(data_dir, 'event_X_test_scaled.npy'))
        y = np.load(os.path.join(data_dir, 'event_y_test.npy')).flatten()
    
    print(f"\nLoaded {split_name} data: {len(X)} events")
    
    # Load Stage-1 model
    stage1_model = EventBinaryClassifier(input_dim=X.shape[1]).to(device)
    stage1_model.load_state_dict(torch.load(os.path.join(model_dir, 'stage1_model.pt')))
    stage1_model.eval()
    
    # Stage-1 predictions
    with torch.no_grad():
        stage1_probs = torch.sigmoid(stage1_model(torch.FloatTensor(X).to(device))).cpu().numpy()
    
    stage1_mask = stage1_probs >= stage1_threshold
    n_stage1 = stage1_mask.sum()
    
    print(f"\nStage-1 filtered: {n_stage1}/{len(X)} events ({n_stage1/len(X)*100:.1f}%)")
    
    if n_stage1 == 0:
        print("No events passed Stage-1 filter!")
        return
    
    # Prepare Stage-2 data
    X_stage2 = X[stage1_mask]
    y_stage2 = y[stage1_mask]

    # Load Stage-2 Random Forest model
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    # If using joblib
    import joblib
    try:
        stage2_model = joblib.load(model_path)
        print(f"Loaded RF model from {model_path}")
    except Exception as e:
        print(f"Error loading RF model: {e}")
        return
    
    # Stage-2 predictions (continuous returns)
    predicted_returns = stage2_model.predict(X_stage2)
    
    print(f"\nStage-2 predictions:")
    print(f"  Mean: {predicted_returns.mean():.4f}%")
    print(f"  Std: {predicted_returns.std():.4f}%")
    print(f"  Min: {predicted_returns.min():.4f}%")
    print(f"  Max: {predicted_returns.max():.4f}%")
    
    # Trading logic
    print("\n" + "="*90)
    print("BACKTEST: LONG/SHORT (Return Regression)")
    print("="*90)
    
    long_mask = predicted_returns >= tau
    short_mask = predicted_returns <= -tau
    
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    n_trades = n_long + n_short
    
    print(f"\nTrade Classification:")
    print(f"  LONG signals: {n_long} (pred >= +{tau:.4f}%)")
    print(f"  SHORT signals: {n_short} (pred <= -{tau:.4f}%)")
    print(f"  Neutral (no trade): {n_stage1 - n_trades}")
    print(f"  Total trades: {n_trades}")
    
    if n_trades == 0:
        print("\nNo trades executed. Lower threshold τ!")
        return
    
    # Calculate PnL
    cost_per_trade = transaction_cost_bps / 100.0
    
    # LONG trades
    long_returns = y_stage2[long_mask] if n_long > 0 else np.array([])
    long_pred = predicted_returns[long_mask] if n_long > 0 else np.array([])
    long_gross = long_returns
    long_net = long_gross - cost_per_trade
    
    # SHORT trades (inverse returns)
    short_returns = y_stage2[short_mask] if n_short > 0 else np.array([])
    short_pred = predicted_returns[short_mask] if n_short > 0 else np.array([])
    short_gross = -short_returns  # Inverse for short
    short_net = short_gross - cost_per_trade
    
    # Combined
    all_gross = np.concatenate([long_gross, short_gross])
    all_net = np.concatenate([long_net, short_net])
    
    # Metrics
    total_gross_pnl = all_gross.sum()
    total_net_pnl = all_net.sum()
    mean_gross = all_gross.mean()
    mean_net = all_net.mean()
    
    gross_wins = (all_gross > 0).sum()
    gross_win_rate = gross_wins / n_trades
    
    net_wins = (all_net > 0).sum()
    net_win_rate = net_wins / n_trades
    
    sharpe = (mean_net / all_net.std()) * np.sqrt(195 * 252) if all_net.std() > 0 else 0
    
    print(f"\nPerformance:")
    print(f"  Mean gross return: {mean_gross:.4f}% ({mean_gross*100:.2f} bps)")
    print(f"  Transaction cost: {cost_per_trade:.4f}% ({transaction_cost_bps:.0f} bps)")
    print(f"  Mean net return: {mean_net:.4f}% ({mean_net*100:.2f} bps)")
    print(f"  Total gross PnL: {total_gross_pnl:.4f}%")
    print(f"  Total net PnL: {total_net_pnl:.4f}%")
    
    print(f"\nTrade Outcomes (GROSS):")
    print(f"  Correct direction: {gross_wins} ({gross_win_rate*100:.1f}%)")
    print(f"  Wrong direction: {n_trades - gross_wins} ({(1-gross_win_rate)*100:.1f}%)")
    
    print(f"\nTrade Outcomes (NET - after {transaction_cost_bps}bps cost):")
    print(f"  Profitable: {net_wins} ({net_win_rate*100:.1f}%)")
    print(f"  Unprofitable: {n_trades - net_wins} ({(1-net_win_rate)*100:.1f}%)")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe ratio (annualized): {sharpe:.3f}")
    print(f"  Std dev (net): {all_net.std():.4f}%")
    
    # Prediction quality
    if n_long > 0:
        long_corr = np.corrcoef(long_pred, long_returns)[0, 1] if len(long_returns) > 1 else 0
        print(f"\nLONG Prediction Quality:")
        print(f"  Mean predicted: {long_pred.mean():.4f}%")
        print(f"  Mean actual: {long_returns.mean():.4f}%")
        print(f"  Correlation: {long_corr:.3f}")
    
    if n_short > 0:
        short_corr = np.corrcoef(short_pred, short_returns)[0, 1] if len(short_returns) > 1 else 0
        print(f"\nSHORT Prediction Quality:")
        print(f"  Mean predicted: {short_pred.mean():.4f}%")
        print(f"  Mean actual: {short_returns.mean():.4f}%")
        print(f"  Correlation: {short_corr:.3f}")
    
    # Verdict
    print("\n" + "="*90)
    if mean_net > 0 and net_win_rate > 0.50:
        print("PROFITABLE STRATEGY")
    elif mean_net > 0:
        print("MARGINALLY PROFITABLE (low win rate)")
    else:
        print("UNPROFITABLE STRATEGY")
    print("="*90)
    
    # Save results
    results = {
        'split': split_name,
        'config': {
            'tau': tau,
            'transaction_cost_bps': transaction_cost_bps,
            'stage1_threshold': stage1_threshold
        },
        'events': {
            'total': len(X),
            'stage1_filtered': int(n_stage1),
            'long_trades': int(n_long),
            'short_trades': int(n_short),
            'total_trades': int(n_trades)
        },
        'performance': {
            'mean_gross_pnl': float(mean_gross),
            'mean_net_pnl': float(mean_net),
            'total_gross_pnl': float(total_gross_pnl),
            'total_net_pnl': float(total_net_pnl),
            'gross_win_rate': float(gross_win_rate),
            'net_win_rate': float(net_win_rate),
            'sharpe': float(sharpe),
            'std_dev': float(all_net.std())
        }
    }
    
    output_file = os.path.join(model_dir, f'backtest_{split_name}_tau{int(tau*100)}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_file}")
    
    return results


if __name__ == "__main__":
    print("="*90)
    print("REGRESSION PIPELINE BACKTESTING")
    print("="*90)
    
    # Configuration
    TRANSACTION_COST_BPS = 2  # 2 bps = 0.02%
    
    # Test multiple thresholds
    THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    print(f"\nConfiguration:")
    print(f"  Transaction cost: {TRANSACTION_COST_BPS} bps")
    print(f"  Testing thresholds: {THRESHOLDS}")
    print()
    
    # Backtest each split with each threshold
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*90}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'='*90}")
        
        for tau in THRESHOLDS:
            try:
                run_regression_backtest(
                    split_name=split,
                    transaction_cost_bps=TRANSACTION_COST_BPS,
                    tau=tau
                )
            except Exception as e:
                print(f"\nError with τ={tau:.2f}: {e}")
                continue
        
        print("\n")
