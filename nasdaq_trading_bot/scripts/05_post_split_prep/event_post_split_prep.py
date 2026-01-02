"""
Step 5: Event Post-Split Preparation
----------------------------------------
- Separates X (features) and y (target)
- Fits StandardScaler on train data only
- Transforms all splits
- Saves scaled/unscaled data with unique event names

Note: Does NOT overwrite existing X_train.csv, y_train.csv etc.
      Uses event_X_train.csv, event_y_train.csv etc.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_event_data():
    """Prepare event data for ML training."""
    print("="*80)
    print("EVENT POST-SPLIT PREPARATION")
    print("="*80)
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    
    # Input files (from split_event_data.py)
    train_file = os.path.join(data_dir, 'event_train.csv')
    validation_file = os.path.join(data_dir, 'event_validation.csv')
    test_file = os.path.join(data_dir, 'event_test.csv')
    
    # Load splits
    print("\nLoading splits...")
    train_df = pd.read_csv(train_file)
    validation_df = pd.read_csv(validation_file)
    test_df = pd.read_csv(test_file)
    
    print(f"  Train: {len(train_df):,} events")
    print(f"  Validation: {len(validation_df):,} events")
    print(f"  Test: {len(test_df):,} events")
    
    # Define column types
    metadata_cols = ['event_id', 'event_time', 'news_id', 'news_headline']
    target_cols = [col for col in train_df.columns if col.startswith('target_')]
    post_metadata_cols = ['post_bars_count', 'post_coverage_pct']
    
    # Features: everything except metadata, targets, and post-window metadata
    exclude_cols = metadata_cols + target_cols + post_metadata_cols
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"\nColumn Classification:")
    print(f"  Metadata: {len(metadata_cols)} columns")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Targets: {len(target_cols)} columns")
    
    # Primary target - CHANGED for 60min horizon deployment
    primary_target = 'target_vwap_return_60m'  # Was: target_vwap_return_20m
    print(f"\nPrimary target: {primary_target}")
    
    # -------------------------------------------
    # Separate X and y
    # -------------------------------------------
    print(f"\nSeparating features and targets...")
    
    X_train = train_df[feature_cols]
    X_val = validation_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # For y, we use only the primary target
    y_train = train_df[[primary_target]]
    y_val = validation_df[[primary_target]]
    y_test = test_df[[primary_target]]
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    
    # Check for NaN
    print(f"\nNaN Check:")
    print(f"  NaN in X_train: {X_train.isna().sum().sum()}")
    print(f"  NaN in y_train: {y_train.isna().sum().sum()}")
    
    if X_train.isna().any().any():
        print("  Filling NaN in features with 0...")
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
    
    # -------------------------------------------
    # Scale features (X)
    # -------------------------------------------
    print(f"\nScaling features...")
    scaler_X = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.values)
    X_val_scaled = scaler_X.transform(X_val.values)
    X_test_scaled = scaler_X.transform(X_test.values)
    
    # Convert back to DataFrame for saving as CSV
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    print(f"  Scaler fitted on train data")
    print(f"  Mean (first 3 features): {scaler_X.mean_[:3]}")
    print(f"  Std (first 3 features): {scaler_X.scale_[:3]}")
    
    # -------------------------------------------
    # Save all data with EVENT prefix
    # -------------------------------------------
    print(f"\nSaving prepared data...")
    
    # Unscaled X (features)
    X_train.to_csv(os.path.join(data_dir, "event_X_train.csv"), index=False)
    X_val.to_csv(os.path.join(data_dir, "event_X_val.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "event_X_test.csv"), index=False)
    
    # Scaled X (features)
    X_train_scaled_df.to_csv(os.path.join(data_dir, "event_X_train_scaled.csv"), index=False)
    X_val_scaled_df.to_csv(os.path.join(data_dir, "event_X_val_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(data_dir, "event_X_test_scaled.csv"), index=False)
    
    # -------------------------------------------
    # Scale targets (y)
    # -------------------------------------------
    print(f"\nScaling targets...")
    scaler_y = StandardScaler()
    
    y_train_scaled = scaler_y.fit_transform(y_train.values)
    y_val_scaled = scaler_y.transform(y_val.values)
    y_test_scaled = scaler_y.transform(y_test.values)
    
    # Convert back to DataFrame
    y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=[primary_target], index=y_train.index)
    y_val_scaled_df = pd.DataFrame(y_val_scaled, columns=[primary_target], index=y_val.index)
    y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=[primary_target], index=y_test.index)

    # -------------------------------------------
    # Save all data with EVENT prefix
    # -------------------------------------------
    print(f"\nSaving prepared data...")
    
    # Unscaled X (features)
    X_train.to_csv(os.path.join(data_dir, "event_X_train.csv"), index=False)
    X_val.to_csv(os.path.join(data_dir, "event_X_val.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "event_X_test.csv"), index=False)
    
    # Scaled X (features)
    X_train_scaled_df.to_csv(os.path.join(data_dir, "event_X_train_scaled.csv"), index=False)
    X_val_scaled_df.to_csv(os.path.join(data_dir, "event_X_val_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(data_dir, "event_X_test_scaled.csv"), index=False)
    
    # Unscaled y (targets)
    y_train.to_csv(os.path.join(data_dir, "event_y_train.csv"), index=False)
    y_val.to_csv(os.path.join(data_dir, "event_y_val.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "event_y_test.csv"), index=False)
    
    # Scaled y (targets)
    y_train_scaled_df.to_csv(os.path.join(data_dir, "event_y_train_scaled.csv"), index=False)
    y_val_scaled_df.to_csv(os.path.join(data_dir, "event_y_val_scaled.csv"), index=False)
    y_test_scaled_df.to_csv(os.path.join(data_dir, "event_y_test_scaled.csv"), index=False)
    
    # Also save as numpy arrays for easier loading in PyTorch
    np.save(os.path.join(data_dir, "event_X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(data_dir, "event_X_val_scaled.npy"), X_val_scaled)
    np.save(os.path.join(data_dir, "event_X_test_scaled.npy"), X_test_scaled)
    
    np.save(os.path.join(data_dir, "event_y_train.npy"), y_train.values)
    np.save(os.path.join(data_dir, "event_y_val.npy"), y_val.values)
    np.save(os.path.join(data_dir, "event_y_test.npy"), y_test.values)
    
    np.save(os.path.join(data_dir, "event_y_train_scaled.npy"), y_train_scaled)
    np.save(os.path.join(data_dir, "event_y_val_scaled.npy"), y_val_scaled)
    np.save(os.path.join(data_dir, "event_y_test_scaled.npy"), y_test_scaled)
    
    # Save scalers
    joblib.dump(scaler_X, os.path.join(data_dir, "event_scaler_X.joblib"))
    joblib.dump(scaler_y, os.path.join(data_dir, "event_scaler_y.joblib"))
    
    # Save feature names
    with open(os.path.join(data_dir, "event_feature_names.txt"), 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"  Saved unscaled X: event_X_train.csv, event_X_val.csv, event_X_test.csv")
    print(f"  Saved scaled X: event_X_train_scaled.csv, event_X_val_scaled.csv, event_X_test_scaled.csv")
    print(f"  Saved y: event_y_train.csv, event_y_val.csv, event_y_test.csv")
    print(f"  Saved numpy arrays: event_X_*_scaled.npy, event_y_*.npy")
    print(f"  Saved scaler: event_scaler_X.joblib")
    print(f"  Saved feature names: event_feature_names.txt")
    
    print("\n" + "="*80)
    print("EVENT POST-SPLIT PREPARATION COMPLETE!")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {primary_target}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")


def main():
    """Main execution."""
    prepare_event_data()


if __name__ == "__main__":
    main()
