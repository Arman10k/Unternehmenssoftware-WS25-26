"""
2-Stage Pipeline with Return Regression (Stage-2)

Stage 1: Tradeable Filter (Binary Classification)
Stage 2: Return Regression (Continuous Prediction)

Trading Rule:
- if predicted_return >= +τ: LONG
- if predicted_return <= -τ: SHORT  
- else: NO TRADE
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Dynamic imports for numbered files
current_dir = os.path.dirname(__file__)

# Import 05_event_return_regression
mod_05_path = os.path.join(current_dir, "05_event_return_regression.py")
mod_05 = import_module_from_path("event_return_regression", mod_05_path)
train_return_regressor = mod_05.train_return_regressor
ReturnRegressor = mod_05.ReturnRegressor

# Import 06_event_random_forest
mod_06_path = os.path.join(current_dir, "06_event_random_forest.py")
mod_06 = import_module_from_path("event_random_forest", mod_06_path)
train_rf_regressor = mod_06.train_rf_regressor



# ======================================================================================
# PLOTTING UTILS
# ======================================================================================

def plot_loss(history, outpath):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    plt.title("Stage 1 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_accuracy_bar(cm, title, outpath, labels=('Correct', 'Incorrect')):
    # cm structure for binary/directional: [[TN, FP], [FN, TP]] (or similar)
    # We just need Correct (TN+TP) vs Incorrect (FP+FN)
    cm = np.array(cm, dtype=int)
    correct = cm[0, 0] + cm[1, 1]
    incorrect = cm[0, 1] + cm[1, 0]
    total = correct + incorrect
    
    if total == 0:
        return

    pct_correct = (correct / total) * 100
    pct_incorrect = (incorrect / total) * 100
    
    vals = [pct_correct, pct_incorrect]
    bar_labels = labels
    colors = ['#2ca02c', '#d62728']  # Green, Red
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(bar_labels, vals, color=colors, alpha=0.8)
    
    plt.title(title)
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    # Add count labels
    counts = [correct, incorrect]
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'({counts[i]})',
                 ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def compute_confusion_matrix(y_true, y_pred):
    # Binary/Directional confusion matrix
    # y_true, y_pred are 0/1 integers
    # Returns [[TN, FP], [FN, TP]]
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return np.array([[tn, fp], [fn, tp]])

# Simple Binary Classifier for Stage-1
class EventBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    print("="*90)
    print("2-STAGE PIPELINE: TRADEABLE FILTER + RETURN REGRESSION")
    print("="*90)
    
    # Configuration
    # From: scripts/06_model_training/event_feed_forward/regression_pipeline.py
    # To: nasdaq_trading_bot/ (4 levels up)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models', 'regression_pipeline_60m')
    os.makedirs(model_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ==================================================================================
    # LOAD DATA
    # ==================================================================================
    print("\nLoading data...")
    
    X_train = np.load(os.path.join(data_dir, 'event_X_train_scaled.npy'))
    X_val = np.load(os.path.join(data_dir, 'event_X_val_scaled.npy'))
    X_test = np.load(os.path.join(data_dir, 'event_X_test_scaled.npy'))
    
    y_train = np.load(os.path.join(data_dir, 'event_y_train.npy')).flatten()
    y_val = np.load(os.path.join(data_dir, 'event_y_val.npy')).flatten()
    y_test = np.load(os.path.join(data_dir, 'event_y_test.npy')).flatten()
    
    print(f"  Train: {len(X_train)} events")
    print(f"  Val: {len(X_val)} events")
    print(f"  Test: {len(X_test)} events")
    
    # FIX 5: Verify 60min target
    print(f"\nTarget Statistics (verifying 60min):")
    print(f"  Mean: {y_train.mean():.4f}%")
    print(f"  Std: {y_train.std():.4f}%")
    print(f"  |y| quantiles:")
    print(f"    90%: {np.quantile(np.abs(y_train), 0.90):.4f}%")
    print(f"    95%: {np.quantile(np.abs(y_train), 0.95):.4f}%")
    print(f"    99%: {np.quantile(np.abs(y_train), 0.99):.4f}%")
    print(f"  (20m: 99%~0.3-0.5%, 60m: 99%~0.8-1.5%)")
    
    # ==================================================================================
    # STAGE 1: TRADEABLE FILTER
    # ==================================================================================
    print("\n" + "="*90)
    print("STAGE 1: QUANTILE TRADE FILTER")
    print("="*90)
    
    # FIX 1: Quantile labeling (CORRECTED: 1-q for top q%)
    q = 0.35
    abs_threshold = np.quantile(np.abs(y_train), 1.0 - q)  # FIX: was q, now 1-q
    
    y_train_s1 = (np.abs(y_train) >= abs_threshold).astype(int)
    y_val_s1 = (np.abs(y_val) >= abs_threshold).astype(int)
    y_test_s1 = (np.abs(y_test) >= abs_threshold).astype(int)
    
    print(f"\nq={q} → |return| >= {abs_threshold:.6f}%")
    print(f"Train tradeable: {y_train_s1.sum()}/{len(y_train)} ({y_train_s1.mean()*100:.1f}%)")
    print(f"Val tradeable: {y_val_s1.sum()}/{len(y_val)} ({y_val_s1.mean()*100:.1f}%)")
    print(f"Test tradeable: {y_test_s1.sum()}/{len(y_test)} ({y_test_s1.mean()*100:.1f}%)")
    
    # Train Stage 1
    pos_weight = (y_train_s1 == 0).sum() / y_train_s1.sum()
    print(f"\npos_weight={pos_weight:.3f}")
    
    stage1_model = EventBinaryClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(stage1_model.parameters(), lr=1e-3)
    
    # Simple training loop
    loss_history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    for epoch in range(50):
        stage1_model.train()
        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train_s1).to(device)
        
        optimizer.zero_grad()
        logits = stage1_model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        
        # Track loss
        train_loss_val = loss.item()
        
        # Validation
        stage1_model.eval()
        with torch.no_grad():
            val_logits = stage1_model(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_logits, torch.FloatTensor(y_val_s1).to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(stage1_model.state_dict(), os.path.join(model_dir, 'stage1_model.pt'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss:.4f}")
        
        loss_history['train_loss'].append(train_loss_val)
        loss_history['val_loss'].append(val_loss)
    
    # Load best and get probabilities
    stage1_model.load_state_dict(torch.load(os.path.join(model_dir, 'stage1_model.pt')))
    stage1_model.eval()
    
    with torch.no_grad():
        stage1_probs_train = torch.sigmoid(stage1_model(torch.FloatTensor(X_train).to(device))).cpu().numpy()
        stage1_probs_val = torch.sigmoid(stage1_model(torch.FloatTensor(X_val).to(device))).cpu().numpy()
        stage1_probs_test = torch.sigmoid(stage1_model(torch.FloatTensor(X_test).to(device))).cpu().numpy()
    
    # Fixed threshold
    stage1_threshold = 0.65
    print(f"\nStage-1 threshold: {stage1_threshold:.3f}")
    
    # ==================================================================================
    # STAGE 2: RETURN REGRESSION
    # ==================================================================================
    print("\n" + "="*90)
    print("STAGE 2: RETURN REGRESSION")
    print("="*90)
    
    # FIX 2: Train/Val on GROUND-TRUTH tradeable (not Stage-1 predictions)
    # This eliminates selection bias
    
    # Ground-truth tradeable masks
    train_true_mask = y_train_s1 == 1  # True tradeable events
    val_true_mask = y_val_s1 == 1
    
    # TEST uses Stage-1 predictions (realistic deployment)
    test_pred_mask = stage1_probs_test >= stage1_threshold
    
    X_train_s2 = X_train[train_true_mask]
    y_train_s2 = y_train[train_true_mask]
    
    X_val_s2 = X_val[val_true_mask]
    y_val_s2 = y_val[val_true_mask]
    
    X_test_s2 = X_test[test_pred_mask]
    y_test_s2 = y_test[test_pred_mask]
    
    print(f"\nStage-2 training samples (ground-truth tradeable):")
    print(f"  Train: {len(X_train_s2)} (ground-truth)")
    print(f"  Val: {len(X_val_s2)} (ground-truth)")
    print(f"  Test: {len(X_test_s2)} (Stage-1 predictions)")
    
    # FIX 4: Clip extreme outliers before training
    y_train_s2_clipped = np.clip(y_train_s2, -2.0, 2.0)
    y_val_s2_clipped = np.clip(y_val_s2, -2.0, 2.0)
    
    print(f"\nTarget clipping (robust training):")
    print(f"  Before: Train std={y_train_s2.std():.4f}%, Val std={y_val_s2.std():.4f}%")
    print(f"  After: Train std={y_train_s2_clipped.std():.4f}%, Val std={y_val_s2_clipped.std():.4f}%")
    
    # Train Random Forest model
    stage2_model, stage2_metrics = train_rf_regressor(
        X_train_s2, y_train_s2_clipped,
        X_val_s2, y_val_s2_clipped,
        model_dir=model_dir,
        n_estimators=200,
        max_depth=10
    )

    # ==================================================================================
    # PLOTS (Images)
    # ==================================================================================
    img_out_dir = os.path.join(base_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    
    print(f"\nGeneratings plots to {img_out_dir}...")
    
    # 1. Stage 1 Loss
    plot_loss(loss_history, os.path.join(img_out_dir, "06_reg_stage1_loss.png"))
    
    # 2. Stage 1 Accuracy (Test)
    # Re-predict using best model
    stage1_pred = (stage1_probs_test >= stage1_threshold).astype(int)
    cm_s1 = compute_confusion_matrix(y_test_s1, stage1_pred)
    plot_accuracy_bar(cm_s1, "Stage 1: Prediction Accuracy (Trade Filter)", 
                      os.path.join(img_out_dir, "06_reg_stage1_accuracy.png"))
                      
    # 3. Stage 2 Directional Accuracy (Test)
    # We have y_test_s2 (ground truth returns) and need predictions
    # Note: Test set for Stage 2 is "Stage-1 predictions" (realistic)
    # But we can also evaluate on all tradeable-filtered items
    
    # Let's predict on the actual test set used for Stage 2 (X_test_s2)
    stage2_pred_returns = stage2_model.predict(X_test_s2)
    
    # Direction: +1 if ret > 0, -1 if ret < 0 (or 0)
    # We can just check signs
    true_dir = np.sign(y_test_s2)
    pred_dir = np.sign(stage2_pred_returns)
    
    # Binarize for confusion matrix (Up vs Down/Flat)
    # Let's treat > 0 as 1, <= 0 as 0
    t_bin = (true_dir > 0).astype(int)
    p_bin = (pred_dir > 0).astype(int)
    
    cm_s2 = compute_confusion_matrix(t_bin, p_bin)
    plot_accuracy_bar(cm_s2, "Stage 2: Directional Accuracy (Regression Sign)", 
                      os.path.join(img_out_dir, "06_reg_stage2_directional_accuracy.png"))
    
    print("Plots created:")
    print("  06_reg_stage1_loss.png")
    print("  06_reg_stage1_accuracy.png")
    print("  06_reg_stage2_directional_accuracy.png")
    
    # ==================================================================================
    # THRESHOLD OPTIMIZATION (VAL)
    # ==================================================================================
    print("\n" + "="*90)
    print("THRESHOLD OPTIMIZATION")
    print("="*90)
    
    # stage2_model.eval() - Not needed for RF
    # with torch.no_grad():
    val_pred_returns = stage2_model.predict(X_val_s2)
    
    # FIX 3: Improved τ-optimization with min_trades constraint
    MIN_TRADES = 20
    cost_pct = 2.0 / 100.0  # 2 bps in %
    
    thresholds = np.linspace(0.02, 0.30, 50)  # Finer grid, lower start
    results = []
    
    for tau in thresholds:
        # Trading rule
        long_mask = val_pred_returns >= tau
        short_mask = val_pred_returns <= -tau
        
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        n_total = n_long + n_short
        
        # FIX: Skip if too few trades (avoid overfitting)
        if n_total < MIN_TRADES:
            continue
        
        # Calculate NET returns (after cost)
        long_net = y_val_s2[long_mask] - cost_pct if n_long > 0 else np.array([])
        short_net = -y_val_s2[short_mask] - cost_pct if n_short > 0 else np.array([])
        
        all_net = np.concatenate([long_net, short_net])
        
        mean_net = all_net.mean()
        median_net = np.median(all_net)
        std_net = all_net.std()
        sharpe = (mean_net / std_net) * np.sqrt(252) if std_net > 0 else 0
        win_rate = (all_net > 0).mean()
        
        results.append({
            'threshold': tau,
            'n_trades': n_total,
            'n_long': n_long,
            'n_short': n_short,
            'mean_net': mean_net,
            'median_net': median_net,
            'sharpe': sharpe,
            'win_rate': win_rate
        })
    
    # Find best by mean net return (more robust than Sharpe with limited data)
    if len(results) == 0:
        print(f"\nNo threshold found with >={MIN_TRADES} trades!")
        print(f"  Signal too weak or need Top-K instead of hard threshold")
        return None
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_net'].idxmax()  # Optimize on mean net return
    best = results_df.iloc[best_idx]
    
    print(f"\nOptimal Threshold (min {MIN_TRADES} trades): τ = {best['threshold']:.4f}%")
    print(f"  Trades: {best['n_trades']:.0f} (LONG: {best['n_long']:.0f}, SHORT: {best['n_short']:.0f})")
    print(f"  Mean NET Return: {best['mean_net']:.4f}%")
    print(f"  Median NET Return: {best['median_net']:.4f}%")
    print(f"  Win Rate: {best['win_rate']*100:.1f}%")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    
    # ==================================================================================
    # SAVE RESULTS
    # ==================================================================================
    final_results = {
        'q': q,
        'abs_threshold': abs_threshold,
        'stage1': {
            'threshold': stage1_threshold,
            'train_tradeable_pct': float(y_train_s1.mean()),
            'val_tradeable_pct': float(y_val_s1.mean()),
            'test_tradeable_pct': float(y_test_s1.mean())
        },
        'stage2': {
            'val_mae': float(stage2_metrics['val_mae']),
            'val_rmse': float(stage2_metrics['val_rmse']),
            'val_r2': float(stage2_metrics['val_r2']),
            'optimal_threshold': float(best['threshold']),
            'val_trades': int(best['n_trades']),
            'val_mean_net': float(best['mean_net']),  # Fixed: was mean_return
            'val_median_net': float(best['median_net']),  # Added
            'val_sharpe': float(best['sharpe']),
            'val_win_rate': float(best['win_rate'])
        }
    }
    
    with open(os.path.join(model_dir, 'regression_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nSaved results to {model_dir}/regression_results.json")
    print("\n" + "="*90)
    print("PIPELINE TRAINING COMPLETE!")
    print("="*90)


if __name__ == "__main__":
    main()
