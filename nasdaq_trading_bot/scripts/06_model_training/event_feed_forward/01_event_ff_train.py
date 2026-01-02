"""
Event Model Feed Forward Training Script

Trains a Feed Forward neural network to predict VWAP returns from news events.

Input: Prepared event data (from event_data_preparation.py)
Output: Trained model, training history, evaluation metrics
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json


class EventFeedForwardModel(nn.Module):
    """
    Feed Forward model for event-based VWAP return prediction.
    
    Architecture: Simpler than the existing FF model due to smaller dataset size.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_p=0.3):
        """
        Initialize model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            dropout_p: Dropout probability
        """
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
        
        # Output layer (single value: VWAP return prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EventModelTrainer:
    """
    Trainer for event-based Feed Forward model.
    """
    
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        img_dir: str,
        hidden_dims=[128, 64, 32],
        dropout_p=0.3,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=200,
        patience=20
    ):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory with prepared data
            model_dir: Directory to save models
            img_dir: Directory to save plots
            hidden_dims: Hidden layer dimensions
            dropout_p: Dropout probability
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            epochs: Maximum epochs
            patience: Early stopping patience
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_dir = img_dir
        
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout_p
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load prepared data."""
        print("\nLoading prepared event data...")
        
        # Load training data with event_ prefix
        X_train = np.load(os.path.join(self.data_dir, 'event_X_train_scaled.npy'))
        y_train = np.load(os.path.join(self.data_dir, 'event_y_train.npy'))
        
        # Load validation data
        X_val = np.load(os.path.join(self.data_dir, 'event_X_val_scaled.npy'))
        y_val = np.load(os.path.join(self.data_dir, 'event_y_val.npy'))
        
        # Load test data
        X_test = np.load(os.path.join(self.data_dir, 'event_X_test_scaled.npy'))
        y_test = np.load(os.path.join(self.data_dir, 'event_y_test.npy'))
        
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val: X={X_val.shape}, y={y_val.shape}")
        print(f"  Test: X={X_test.shape}, y={y_test.shape}")
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        # Flatten y to (n,) then unsqueeze to (n, 1) for single target
        y_train = torch.tensor(y_train.flatten(), dtype=torch.float32).unsqueeze(1)
        
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val.flatten(), dtype=torch.float32).unsqueeze(1)
        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.flatten(), dtype=torch.float32).unsqueeze(1)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.input_dim = X_train.shape[1]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self):
        """Build and initialize model."""
        print(f"\nBuilding model...")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Hidden dims: {self.hidden_dims}")
        print(f"  Dropout: {self.dropout_p}")
        
        model = EventFeedForwardModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_p=self.dropout_p
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        return epoch_loss / len(self.train_loader.dataset)
    
    def validate(self, model, criterion, loader):
        """Validate model."""
        model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                epoch_loss += loss.item() * X_batch.size(0)
        
        return epoch_loss / len(loader.dataset)
    
    def train(self):
        """Train the model."""
        print("\n" + "="*80)
        print("TRAINING EVENT FEED FORWARD MODEL")
        print("="*80)
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        # Build model
        model = self.build_model()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop
        print(f"\nStarting training...")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Patience: {self.patience}")
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(model, criterion, self.val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch:3d}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model
                model_path = os.path.join(self.model_dir, 'best_event_ff_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'hidden_dims': self.hidden_dims,
                    'input_dim': self.input_dim
                }, model_path)
                
                print(f"  New best model saved (val_loss: {val_loss:.6f})")
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': len(train_losses) - self.patience if epochs_without_improvement >= self.patience else len(train_losses),
            'best_val_loss': best_val_loss
        }
        
        history_path = os.path.join(self.model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        # Load best model and evaluate
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.evaluate(model, X_test, y_test)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        return model, history
    
    def plot_training_history(self, train_losses, val_losses):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Event FF Model Training History', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.img_dir, '06_event_ff_training_loss.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"\nTraining plot saved to: {plot_path}")
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model on test set."""
        print("\nEvaluating on test set...")
        
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(self.device)
            predictions = model(X_test_device).cpu().numpy()
        
        y_test_np = y_test.numpy()
        
        # Metrics
        mse = mean_squared_error(y_test_np, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, predictions)
        r2 = r2_score(y_test_np, predictions)
        
        print(f"\nTest Set Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
        
        # Directional accuracy
        actual_direction = np.sign(y_test_np)
        pred_direction = np.sign(predictions)
        directional_accuracy = (actual_direction == pred_direction).mean() * 100
        
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Save metrics
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy)
        }
        
        metrics_path = os.path.join(self.model_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Scatter plot
        self.plot_predictions(y_test_np, predictions)
        
        return metrics
    
    def plot_predictions(self, y_actual, y_pred):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual VWAP Return (%)', fontsize=12)
        plt.ylabel('Predicted VWAP Return (%)', fontsize=12)
        plt.title('Event FF Model: Actual vs Predicted', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.img_dir, '06_event_ff_predictions.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"  Predictions plot saved to: {plot_path}")


def main():
    """Main execution."""
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    data_dir = os.path.join(project_root, 'data')  # Direct data directory
    model_dir = os.path.join(project_root, 'models', 'event_feed_forward')
    img_dir = os.path.join(project_root, 'images')
    
    # Create trainer
    trainer = EventModelTrainer(
        data_dir=data_dir,
        model_dir=model_dir,
        img_dir=img_dir,
        hidden_dims=[128, 64, 32],  # Smaller network for smaller dataset
        dropout_p=0.3,               # Higher dropout to prevent overfitting
        batch_size=32,               # Smaller batch size
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=200,
        patience=20
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
