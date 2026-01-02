
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_rf_regressor(
    X_train, y_train, X_val, y_val,
    model_dir='../../models/return_regression_rf',
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
):
    """
    Train Random Forest Regressor for Event Return Prediction.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nTraining Random Forest Regressor...")
    print(f"  Samples: {len(X_train)} | Features: {X_train.shape[1]}")
    print(f"  Params: n_est={n_estimators}, depth={max_depth}, leaf={min_samples_leaf}")

    # Initialize RF
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1  # Use all cores
    )
    
    # Train
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'val_r2': r2_score(y_val, val_pred)
    }
    
    print(f"\nRF Training Complete:")
    print(f"  Train R²: {metrics['train_r2']:.4f} | MAE: {metrics['train_mae']:.4f}%")
    print(f"  Val   R²: {metrics['val_r2']:.4f} | MAE: {metrics['val_mae']:.4f}%")
    
    # Save model
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    joblib.dump(rf, model_path)
    print(f"  Saved model to {model_path}")
    
    return rf, metrics
