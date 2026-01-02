"""
Event Binary Classification: Trade vs No-Trade (Improved)

Goal: High precision trade filter (Stage 1 of 2-stage approach)

Label:
  Tradeable (1): |VWAP_Return_%| > epsilon
  No-Trade (0):  otherwise

Improvements vs previous:
- Use 1-logit binary head + BCEWithLogitsLoss (cleaner than 2-class CE)
- pos_weight to counter class imbalance
- Threshold selection on VAL to maximize precision (with constraints)
- Early stopping based on selected precision objective (tradingading-aligned)
- Reports PR curve stats and number of trades
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)


# -----------------------------
# Utilities
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def convert_to_trade_notrade(y_cont: np.ndarray, epsilon: float) -> np.ndarray:
    """
    y_cont is VWAP return in percent points, e.g. 0.10 means +0.10%
    epsilon also in percent points.
    """
    return (np.abs(y_cont) > epsilon).astype(np.int64)


def load_and_prepare_data(data_dir: str, epsilon: float):
    print(f"Loading data from: {data_dir}")

    X_train = np.load(os.path.join(data_dir, "event_X_train_scaled.npy"))
    X_val   = np.load(os.path.join(data_dir, "event_X_val_scaled.npy"))
    X_test  = np.load(os.path.join(data_dir, "event_X_test_scaled.npy"))

    y_train_cont = np.load(os.path.join(data_dir, "event_y_train.npy")).flatten()
    y_val_cont   = np.load(os.path.join(data_dir, "event_y_val.npy")).flatten()
    y_test_cont  = np.load(os.path.join(data_dir, "event_y_test.npy")).flatten()

    y_train = convert_to_trade_notrade(y_train_cont, epsilon)
    y_val   = convert_to_trade_notrade(y_val_cont, epsilon)
    y_test  = convert_to_trade_notrade(y_test_cont, epsilon)

    def dist(name, y):
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        print(f"  {name}: No-Trade={n0} ({n0/len(y)*100:.1f}%) | Trade={n1} ({n1/len(y)*100:.1f}%)")

    print(f"\nTrade vs No-Trade Distribution (epsilon={epsilon:.3f}%):")
    dist("Train", y_train)
    dist("Val  ", y_val)
    dist("Test ", y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_precision: float = 0.60,
    min_recall: float = 0.20,
    min_trades: int = 10,
):
    """
    Choose a probability threshold on VAL that satisfies constraints and maximizes precision, then recall.

    Returns:
        dict with keys: threshold, precision, recall, trades
        or None if no threshold meets constraints.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision, recall include one extra point (thresholds length is -1)
    precision = precision[:-1]
    recall = recall[:-1]

    best = None
    for p, r, t in zip(precision, recall, thresholds):
        y_pred = (y_prob >= t).astype(np.int64)
        trades = int(y_pred.sum())
        if trades < min_trades:
            continue
        if p >= min_precision and r >= min_recall:
            # maximize precision, then recall, then trades
            score = (p, r, trades)
            if best is None or score > best["score"]:
                best = {"score": score, "threshold": float(t), "precision": float(p), "recall": float(r), "trades": trades}

    if best is None:
        return None
    best.pop("score", None)
    return best


def threshold_sweep_summary(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None):
    """
    Print a small summary table for a few thresholds.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        rows.append({
            "thr": t,
            "trades": int(y_pred.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    print("\nThreshold sweep (quick view):")
    for r in rows:
        print(f"  thr={r['thr']:.2f} | trades={r['trades']:4d} | P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")


# -----------------------------
# Baseline: Logistic Regression
# -----------------------------

def train_logistic_regression_binary(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION BASELINE (BINARY)")
    print("=" * 80)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Use probabilities to allow thresholding
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    threshold_sweep_summary(y_val, val_prob)

    picked = pick_threshold(y_val, val_prob, min_precision=0.60, min_recall=0.20, min_trades=10)
    if picked is None:
        print("\nNo VAL threshold met constraints; using 0.5.")
        best_thr = 0.5
    else:
        best_thr = picked["threshold"]
        print(f"\nPicked VAL threshold: {best_thr:.3f} (P={picked['precision']:.3f}, R={picked['recall']:.3f}, trades={picked['trades']})")

    y_pred_test = (test_prob >= best_thr).astype(np.int64)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    print(f"\nTest Results @ thr={best_thr:.3f}:")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall:    {test_recall:.3f}")
    print(f"  F1-Score:  {test_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["No-Trade", "Trade"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           No-Trade  Trade")
    print(f"Actual")
    print(f"No-Trade    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Trade       {cm[1,0]:4d}    {cm[1,1]:4d}")

    return model, {"acc": test_acc, "precision": test_precision, "recall": test_recall, "f1": test_f1, "threshold": best_thr}


# -----------------------------
# Improved FFN
# -----------------------------

class TradeBinaryFFN(nn.Module):
    """
    Binary classifier with 1-logit output.
    """
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout_p=0.30):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev = h
        layers.append(nn.Linear(prev, 1))  # 1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch,)


def train_binary_ffn(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_dir: str,
    *,
    hidden_dims=(64, 32),
    dropout_p=0.30,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-3,
    epochs=200,
    patience=20,
    # Trading-aligned threshold selection constraints
    min_precision=0.60,
    min_recall=0.20,
    min_trades=10,
):
    print("\n" + "=" * 80)
    print(f"IMPROVED BINARY FFN {list(hidden_dims)} - TRADE vs NO-TRADE")
    print("=" * 80)

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)  # BCE wants float
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # Model
    model = TradeBinaryFFN(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nArchitecture:")
    print(f"  Input: {X_train.shape[1]} features")
    print(f"  Hidden: {list(hidden_dims)}")
    print(f"  Dropout: {dropout_p}")
    print(f"  Params: {total_params:,}")

    # pos_weight
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    print(f"\npos_weight = {pos_weight.item():.3f} (neg/pos = {n_neg}/{n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val_precision = -1.0
    best_val_threshold = 0.5
    epochs_no_improve = 0

    print(f"\nTraining (early stop on VAL precision with constraints)...")
    print(f"  epochs={epochs}, batch={batch_size}, lr={lr}, wd={weight_decay}, patience={patience}")
    print(f"  Constraints: min_precision={min_precision}, min_recall={min_recall}, min_trades={min_trades}")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # --- Validate: compute probs + select threshold ---
        model.eval()
        val_logits = []
        val_labels = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb_dev = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb_dev)
                val_loss_sum += loss.item() * xb.size(0)

                val_logits.append(logits.cpu().numpy())
                val_labels.append(yb.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels).astype(np.int64)
        val_prob = sigmoid(val_logits)

        picked = pick_threshold(
            val_labels, val_prob,
            min_precision=min_precision,
            min_recall=min_recall,
            min_trades=min_trades
        )

        if picked is None:
            # fallback: use 0.5 threshold metrics
            val_pred = (val_prob >= 0.5).astype(np.int64)
            val_precision = precision_score(val_labels, val_pred, zero_division=0)
            val_recall = recall_score(val_labels, val_pred, zero_division=0)
            val_f1 = f1_score(val_labels, val_pred, zero_division=0)
            chosen_thr = 0.5
            trades = int(val_pred.sum())
        else:
            chosen_thr = picked["threshold"]
            val_pred = (val_prob >= chosen_thr).astype(np.int64)
            val_precision = picked["precision"]
            val_recall = picked["recall"]
            val_f1 = f1_score(val_labels, val_pred, zero_division=0)
            trades = picked["trades"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"VAL@thr={chosen_thr:.3f}: P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f} trades={trades}"
        )

        # Early stopping criterion: maximize precision first (trading-aligned), tie-break by recall
        improved = (val_precision > best_val_precision + 1e-6)
        if improved:
            best_val_precision = val_precision
            best_val_threshold = chosen_thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(best_state, os.path.join(model_dir, "best_trade_binary_ffn.pt"))
            print(f"  Saved best (VAL precision={best_val_precision:.3f}, thr={best_val_threshold:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best
    if best_state is None:
        best_state = torch.load(os.path.join(model_dir, "best_trade_binary_ffn.pt"))
    model.load_state_dict(best_state)

    # Evaluate on TEST using best_val_threshold
    model.eval()
    test_logits = []
    test_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            test_logits.append(logits)
            test_labels.append(yb.numpy())

    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels).astype(np.int64)
    test_prob = sigmoid(test_logits)
    test_pred = (test_prob >= best_val_threshold).astype(np.int64)

    test_acc = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, zero_division=0)
    test_recall = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)

    print("\nFinal TEST Results (using VAL-picked threshold):")
    print(f"  Threshold: {best_val_threshold:.3f}")
    print(f"  Trades:    {int(test_pred.sum())}/{len(test_pred)}")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall:    {test_recall:.3f}")
    print(f"  F1-Score:  {test_f1:.3f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, target_names=["No-Trade", "Trade"], zero_division=0))

    cm = confusion_matrix(test_labels, test_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           No-Trade  Trade")
    print(f"Actual")
    print(f"No-Trade    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Trade       {cm[1,0]:4d}    {cm[1,1]:4d}")

    return model, {
        "acc": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "threshold": best_val_threshold,
    }


# -----------------------------
# Main (supports epsilon sweep)
# -----------------------------

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    data_dir = os.path.join(project_root, "data")
    model_root = os.path.join(project_root, "models", "event_binary_improved")

    os.makedirs(model_root, exist_ok=True)

    # ---- Choose epsilons to test ----
    epsilons = [0.10, 0.15, 0.20]  # in percent points

    results = {}

    for eps in epsilons:
        print("\n" + "#" * 90)
        print(f"RUN FOR epsilon={eps:.3f}%")
        print("#" * 90)

        X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(data_dir, epsilon=eps)

        run_dir = os.path.join(model_root, f"eps_{str(eps).replace('.','p')}")
        os.makedirs(run_dir, exist_ok=True)

        # Baseline
        logreg_model, logreg_metrics = train_logistic_regression_binary(X_train, y_train, X_val, y_val, X_test, y_test)

        # Improved FFN
        ffn_model, ffn_metrics = train_binary_ffn(
            X_train, y_train, X_val, y_val, X_test, y_test,
            model_dir=run_dir,
            hidden_dims=(64, 32),
            dropout_p=0.30,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-3,
            epochs=200,
            patience=20,
            min_precision=0.60,
            min_recall=0.20,
            min_trades=10,
        )

        results[str(eps)] = {
            "LogisticRegression": logreg_metrics,
            "ImprovedFFN": ffn_metrics,
        }

        # Save per-epsilon results
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results[str(eps)], f, indent=2)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY (Binary Trade Filter)")
    print("=" * 90)

    for eps, res in results.items():
        print(f"\nε={eps}%:")
        for name, m in res.items():
            print(
                f"  {name:16s} | thr={m['threshold']:.3f} | "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['acc']*100:.2f}%"
            )

    with open(os.path.join(model_root, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
