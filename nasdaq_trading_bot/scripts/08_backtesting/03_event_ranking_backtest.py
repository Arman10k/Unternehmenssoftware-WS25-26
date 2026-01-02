
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_daily_top_k_simulation(k=1, min_threshold=0.12, transaction_cost_bps=2.0):
    print("="*80)
    print(f"DAILY TOP-{k} TRADING SIMULATION (Min Threshold: {min_threshold}%)")
    print("="*80)
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models', 'regression_pipeline_60m')
    
    # 2. Load Model & Data
    try:
        model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
        
        # Load TEST data (Unscaled CSV to get proper dates, then NPY for features)
        test_df = pd.read_csv(os.path.join(data_dir, 'event_test.csv'))
        X_test = np.load(os.path.join(data_dir, 'event_X_test_scaled.npy'))
        y_test = np.load(os.path.join(data_dir, 'event_y_test.npy')).flatten()
        
    except Exception as e:
        print(f"Error loading data/model: {e}")
        return

    # 3. Predict
    print(f"Predicting on {len(test_df)} test events...")
    test_df['predicted_return'] = model.predict(X_test)
    test_df['actual_return'] = y_test
    test_df['abs_pred'] = test_df['predicted_return'].abs()
    
    # Ensure datetime
    test_df['event_date'] = pd.to_datetime(test_df['event_time']).dt.date
    
    # 4. Simulation Loop
    cost_pct = transaction_cost_bps / 100.0
    daily_results = []
    trades = []
    
    unique_dates = sorted(test_df['event_date'].unique())
    print(f"Simulating across {len(unique_dates)} trading days...")
    
    for date in unique_dates:
        # Get events for this day
        day_events = test_df[test_df['event_date'] == date].copy()
        
        if day_events.empty:
            continue
            
        # Filter by minimum threshold
        strong_events = day_events[day_events['abs_pred'] >= min_threshold]
        
        if strong_events.empty:
            continue

        # Sort by absolute prediction strength (Magnitude)
        strong_events = strong_events.sort_values('abs_pred', ascending=False)
        
        # Select Top-K
        selected = strong_events.head(k)
        
        # Execute Trades
        for _, row in selected.iterrows():
            # Strategy: Long if pred > 0, Short if pred < 0
            direction = 1 if row['predicted_return'] > 0 else -1
            
            # Net Return = (Actual * Direction) - Cost
            gross_pnl = row['actual_return'] * direction
            net_pnl = gross_pnl - cost_pct
            
            trades.append({
                'date': date,
                'symbol': 'NQ',
                'direction': 'LONG' if direction==1 else 'SHORT',
                'pred': row['predicted_return'],
                'actual': row['actual_return'],
                'net_pnl': net_pnl
            })
    
    # 5. Analysis
    trades_df = pd.DataFrame(trades)
    
    if trades_df.empty:
        print("No trades executed!")
        return

    total_net = trades_df['net_pnl'].sum()
    mean_net = trades_df['net_pnl'].mean()
    win_rate = (trades_df['net_pnl'] > 0).mean()
    
    # Sharpe (assuming daily aggregation for correctness, or per-trade if simplified)
    # Correct way: Daily PnL curve
    daily_pnl = trades_df.groupby('date')['net_pnl'].sum()
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std()
    sharpe_daily = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    
    print("\n" + "-"*60)
    print(f"RESULTS: Top-{k} per Day Strategy")
    print("-" * 60)
    print(f"  Total Trades:     {len(trades_df)}")
    print(f"  Win Rate:         {win_rate*100:.1f}%")
    print(f"  Total Net PnL:    {total_net:.4f}%")
    print(f"  Avg Net per Day:  {daily_mean:.4f}%")
    print(f"  Sharpe (Daily):   {sharpe_daily:.2f}")
    
    print("\nCumulative Return Curve:")
    trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()
    
    # ASCII Plot
    y = trades_df['cum_pnl'].values
    min_y, max_y = y.min(), y.max()
    range_y = max_y - min_y
    if range_y == 0: range_y = 1
    
    print(f"Start: 0.00% -> End: {y[-1]:.2f}%")
    
    # Save results
    res_path = os.path.join(model_dir, f'daily_top{k}_simulation.csv')
    trades_df.to_csv(res_path, index=False)
    print(f"\nSaved trade log to: {res_path}")

if __name__ == "__main__":
    # Simulate Top-1 with Min Threshold
    run_daily_top_k_simulation(k=1, min_threshold=0.12)
