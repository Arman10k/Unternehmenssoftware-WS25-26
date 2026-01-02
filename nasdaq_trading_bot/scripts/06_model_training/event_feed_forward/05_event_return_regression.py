"""
Event Return Regression - Stage 2 Replacement
Predicts continuous returns instead of direction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os


class ReturnRegressor(nn.Module):
    """
    Simple feedforward network for return regression.
    """
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout=0.2):
        super().__init__()
        
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        
        # Output: Single continuous value (predicted return)
        layers.append(nn.Linear(prev, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(1)  # Output shape: (batch_size,)


def train_return_regressor(
    X_train, y_train,
    X_val, y_val,
    model_dir='../../models/return_regression',
    device='cpu',
    epochs=50,
    lr=1e-3,
    batch_size=32
):
    """
    Train return regression model.
    
    Args:
        X_train, y_train: Training data (y in percentage points)
        X_val, y_val: Validation data
        model_dir: Where to save model
        
    Returns:
        model, best_val_metrics, training_history
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Initialize model
    model = ReturnRegressor(input_dim=X_train.shape[1]).to(device)
    
    # Huber Loss (robust to outliers)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    # Early stopping
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining Return Regressor:")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Early stopping patience: {patience} epochs")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Mini-batch training
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(idx)
        
        train_loss /= len(X_train_t)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_mae = torch.abs(val_preds - y_val_t).mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}%")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    
    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy()
    
    val_actual = y_val
    
    metrics = {
        'val_loss': best_val_loss,
        'val_mae': mean_absolute_error(val_actual, val_preds),
        'val_mse': mean_squared_error(val_actual, val_preds),
        'val_rmse': np.sqrt(mean_squared_error(val_actual, val_preds)),
        'val_r2': np.corrcoef(val_actual, val_preds)[0, 1]**2 if len(val_actual) > 1 else 0.0
    }
    
    print(f"\nBest Validation Metrics:")
    print(f"  MAE: {metrics['val_mae']:.4f}%")
    print(f"  RMSE: {metrics['val_rmse']:.4f}%")
    print(f"  R²: {metrics['val_r2']:.4f}")
    
    # Save plots
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True)
    
    # MAE curve
    plt.subplot(1, 3, 2)
    plt.plot(history['val_mae'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE (%)')
    plt.title('Validation MAE')
    plt.grid(True)
    
    # Predictions vs Actual
    plt.subplot(1, 3, 3)
    plt.scatter(val_actual, val_preds, alpha=0.5, s=10)
    plt.plot([val_actual.min(), val_actual.max()], 
             [val_actual.min(), val_actual.max()], 
             'r--', label='Perfect')
    plt.xlabel('Actual Return (%)')
    plt.ylabel('Predicted Return (%)')
    plt.title(f'Predictions (R²={metrics["val_r2"]:.3f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_plots.png'), dpi=150)
    print(f"\nSaved plots to {model_dir}/training_plots.png")
    
    return model, metrics, history


if __name__ == "__main__":
    # Test with dummy data
    print("Testing Return Regressor...")
    
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000) * 0.2  # Returns in %
    
    X_val = np.random.randn(200, 20)
    y_val = np.random.randn(200) * 0.2
    
    model, metrics, history = train_return_regressor(
        X_train, y_train,
        X_val, y_val,
        model_dir='./test_regression',
        epochs=30
    )
    
    print("\nTest complete!")
