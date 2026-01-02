"""
Event Classification Model Training Script

Implements improved classification approach:
1. Converts VWAP return to 3-class classification (bullish/neutral/bearish)
2. Trains baseline models (LogReg, LightGBM)
3. Trains improved FFN [64, 32] with stronger regularization
4. Compares all models

Target: 3-class with epsilon threshold
- Bullish (1): VWAP return > ε
- Neutral (0): -ε <= VWAP return <= ε  
- Bearish (-1): VWAP return < -ε
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Optional: LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

import matplotlib.pyplot as plt
import seaborn as sns
import json


class EventClassificationFFN(nn.Module):
    """Improved Feed Forward for 3-class classification."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_p=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = hidden_dim
        
        # Output: 3 classes
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def convert_to_classification(y, epsilon=0.1):
    """
    Convert continuous VWAP returns to 3-class labels.
    
    Args:
        y: VWAP returns in percent
        epsilon: Threshold for neutral class (default 0.1%)
        
    Returns:
        labels: 0=bearish, 1=neutral, 2=bullish
    """
    labels = np.zeros(len(y), dtype=int)
    labels[y > epsilon] = 2  # Bullish
    labels[(y >= -epsilon) & (y <= epsilon)] = 1  # Neutral
    labels[y < -epsilon] = 0  # Bearish
    
    return labels


def load_and_prepare_data(data_dir, epsilon=0.1):
    """Load data and convert to classification."""
    print("Loading data...")
    
    # Load scaled features
    X_train = np.load(os.path.join(data_dir, 'event_X_train_scaled.npy'))
    X_val = np.load(os.path.join(data_dir, 'event_X_val_scaled.npy'))
    X_test = np.load(os.path.join(data_dir, 'event_X_test_scaled.npy'))
    
    # Load targets (VWAP returns)
    y_train_cont = np.load(os.path.join(data_dir, 'event_y_train.npy')).flatten()
    y_val_cont = np.load(os.path.join(data_dir, 'event_y_val.npy')).flatten()
    y_test_cont = np.load(os.path.join(data_dir, 'event_y_test.npy')).flatten()
    
    # Convert to classification
    y_train = convert_to_classification(y_train_cont, epsilon)
    y_val = convert_to_classification(y_val_cont, epsilon)
    y_test = convert_to_classification(y_test_cont, epsilon)
    
    print(f"\nClassification Distribution (epsilon={epsilon}%):")
    print(f"  Train: Bearish={np.sum(y_train==0)} | Neutral={np.sum(y_train==1)} | Bullish={np.sum(y_train==2)}")
    print(f"  Val:   Bearish={np.sum(y_val==0)} | Neutral={np.sum(y_val==1)} | Bullish={np.sum(y_val==2)}")
    print(f"  Test:  Bearish={np.sum(y_test==0)} | Neutral={np.sum(y_test==1)} | Bullish={np.sum(y_test==2)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression baseline."""
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION BASELINE")
    print("="*80)
    
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nResults:")
    print(f"  Val Accuracy: {val_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Val F1: {val_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Bearish', 'Neutral', 'Bullish']))
    
    return model, {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1, 'test_f1': test_f1}


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train LightGBM baseline."""
    print("\n" + "="*80)
    print("LIGHTGBM BASELINE")
    print("="*80)
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters (shallow tree to avoid overfitting)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Shallow tree
        'max_depth': 4,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
    )
    
    # Predictions
    y_pred_val = np.argmax(model.predict(X_val), axis=1)
    y_pred_test = np.argmax(model.predict(X_test), axis=1)
    
    # Metrics
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\nResults:")
    print(f"  Val Accuracy: {val_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Val F1: {val_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Bearish', 'Neutral', 'Bullish']))
    
    return model, {'val_acc': val_acc, 'test_acc': test_acc, 'val_f1': val_f1, 'test_f1': test_f1}


def train_improved_ffn(X_train, y_train, X_val, y_val, X_test, y_test, 
                       model_dir, img_dir):
    """Train improved FFN [64, 32] with stronger regularization."""
    print("\n" + "="*80)
    print("IMPROVED FEED FORWARD [64, 32]")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = EventClassificationFFN(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout_p=0.5  # Higher dropout
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Input: {X_train.shape[1]} features")
    print(f"  Hidden: [64, 32]")
    print(f"  Output: 3 classes")
    print(f"  Dropout: 0.5")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3  # Stronger L2 regularization
    )
    
    # Training
    epochs = 200
    patience = 15
    best_val_acc = 0
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\nTraining...")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  Weight decay (L2): 1e-3")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_event_classification_ffn.pt'))
            print(f"  New best model saved (val_acc: {val_acc*100:.2f}%)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_event_classification_ffn.pt')))
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test F1: {test_f1:.4f}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Bearish', 'Neutral', 'Bullish']))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, img_dir)
    
    return model, {'val_acc': best_val_acc, 'test_acc': test_acc, 'test_f1': test_f1}


def plot_training_history(train_losses, val_losses, train_accs, val_accs, img_dir):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot([x*100 for x in train_accs], label='Train Acc')
    ax2.plot([x*100 for x in val_accs], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '06_event_classification_ffn_history.png'), dpi=150)
    plt.close()


def main():
    """Main execution."""
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'models', 'event_classification')
    img_dir = os.path.join(project_root, 'images')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(data_dir, epsilon=0.1)
    
    # Train models
    results = {}
    
    # 1. Logistic Regression
    logreg_model, logreg_results = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    results['LogisticRegression'] = logreg_results
    
    # 2. LightGBM (optional)
    if HAS_LIGHTGBM:
        lgbm_model, lgbm_results = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
        results['LightGBM'] = lgbm_results
    else:
        print("\nSkipping LightGBM (not installed)")
        print("   Install with: pip install lightgbm")
    
    # 3. Improved FFN
    ffn_model, ffn_results = train_improved_ffn(X_train, y_train, X_val, y_val, X_test, y_test, model_dir, img_dir)
    results['ImprovedFFN'] = ffn_results
    
    # Save results
    results_path = os.path.join(model_dir, 'classification_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION MODELS COMPARISON")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {metrics['test_acc']*100:.2f}%")
        print(f"  Test F1: {metrics['test_f1']:.4f}")
    
    print("\nAll models trained and evaluated!")


if __name__ == "__main__":
    main()
