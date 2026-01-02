"""
Option B - Stage 1 Trade Filter via Quantiles (instead of fixed epsilon)

Label:
  Tradeable (1): |VWAP_Return_%| in top q-quantile of TRAIN distribution
  No-Trade (0):  otherwise

Why:
- Avoids vanishing positives when epsilon is too high
- Keeps class balance controllable and learnable
- More stable across regimes than fixed absolute thresholds

Inputs expected in data_dir:
- event_X_train_scaled.npy
- event_X_val_scaled.npy
- event_X_test_scaled.npy
- event_y_train.npy (continuous VWAP returns in % points)
- event_y_val.npy
- event_y_test.npy

Outputs:
- models/optionB_trade_quantile/.../best_trade_quantile_ffn.pt
- results json summaries

Run:
python event_trade_filter_quantile.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)


# -----------------------------
# Helpers
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_trade_labels_by_quantile(
    y_train_cont: np.ndarray,
    y_cont: np.ndarray,
    q: float
):
    """
    Build binary labels using a TRAIN-derived absolute-return threshold.

    Args:
      y_train_cont: continuous returns from TRAIN only
      y_cont: continuous returns for split (train/val/test)
      q: fraction of events to label as tradeable (e.g. 0.30 => top 30%)

    Returns:
      labels (0/1), abs_threshold_used
    """
    assert 0 < q < 1, "q must be in (0,1)"
    abs_train = np.abs(y_train_cont)

    # threshold such that top-q are tradeable:
    # e.g. q=0.30 -> threshold at 70th percentile
    thr = np.quantile(abs_train, 1.0 - q)

    labels = (np.abs(y_cont) >= thr).astype(np.int64)
    return labels, float(thr)


def pick_threshold_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_precision: float = 0.60,
    min_recall: float = 0.20,
    min_trades: int = 10,
):
    """
    Choose probability threshold on VAL to satisfy constraints and maximize precision then recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    best = None
    for p, r, t in zip(precision, recall, thresholds):
        y_pred = (y_prob >= t).astype(np.int64)
        trades = int(y_pred.sum())
        if trades < min_trades:
            continue
        if p >= min_precision and r >= min_recall:
            score = (p, r, trades)
            if best is None or score > best["score"]:
                best = {"score": score, "threshold": float(t), "precision": float(p), "recall": float(r), "trades": trades}

    if best is None:
        return None
    best.pop("score", None)
    return best


def quick_threshold_sweep(y_true, y_prob, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    print("\nThreshold sweep (VAL quick view):")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  thr={t:.2f} | trades={int(y_pred.sum()):4d} | P={p:.3f} R={r:.3f} F1={f1:.3f}")


# -----------------------------
# Data loading
# -----------------------------

def load_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "event_X_train_scaled.npy"))
    X_val   = np.load(os.path.join(data_dir, "event_X_val_scaled.npy"))
    X_test  = np.load(os.path.join(data_dir, "event_X_test_scaled.npy"))

    y_train_cont = np.load(os.path.join(data_dir, "event_y_train.npy")).flatten()
    y_val_cont   = np.load(os.path.join(data_dir, "event_y_val.npy")).flatten()
    y_test_cont  = np.load(os.path.join(data_dir, "event_y_test.npy")).flatten()

    return X_train, X_val, X_test, y_train_cont, y_val_cont, y_test_cont


def prepare_quantile_labels(data_dir: str, q: float):
    X_train, X_val, X_test, y_train_cont, y_val_cont, y_test_cont = load_data(data_dir)

    y_train, thr = make_trade_labels_by_quantile(y_train_cont, y_train_cont, q)
    y_val, _     = make_trade_labels_by_quantile(y_train_cont, y_val_cont, q)
    y_test, _    = make_trade_labels_by_quantile(y_train_cont, y_test_cont, q)

    def dist(name, y):
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        print(f"  {name}: No-Trade={n0} ({n0/len(y)*100:.1f}%) | Trade={n1} ({n1/len(y)*100:.1f}%)")

    print("\n" + "=" * 90)
    print(f"QUANTILE LABELING: q={q:.2f} (top {int(q*100)}% by |return| are Tradeable)")
    print(f"Absolute-return threshold from TRAIN: |ret| >= {thr:.6f} (% points)")
    print("=" * 90)
    dist("Train", y_train)
    dist("Val  ", y_val)
    dist("Test ", y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, thr


# -----------------------------
# Baseline: Logistic Regression
# -----------------------------

def train_logreg(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION BASELINE (Quantile Trade Filter)")
    print("=" * 80)

    model = LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    quick_threshold_sweep(y_val, val_prob)

    picked = pick_threshold_precision(y_val, val_prob, min_precision=0.60, min_recall=0.20, min_trades=10)
    best_thr = picked["threshold"] if picked else 0.5
    if picked:
        print(f"\nPicked VAL threshold: {best_thr:.3f} (P={picked['precision']:.3f}, R={picked['recall']:.3f}, trades={picked['trades']})")
    else:
        print("\nNo VAL threshold met constraints; using 0.5.")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_thr).astype(np.int64)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    print(f"\nTest Results @ thr={best_thr:.3f}:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["No-Trade", "Trade"], zero_division=0))

    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": best_thr}


# -----------------------------
# Improved FFN (Binary 1-logit)
# -----------------------------

class TradeQuantileFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout_p=0.30):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # 1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_ffn(
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
    min_precision=0.60,
    min_recall=0.20,
    min_trades=10,
):
    print("\n" + "=" * 80)
    print(f"IMPROVED FFN {list(hidden_dims)} (Quantile Trade Filter)")
    print("=" * 80)

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    model = TradeQuantileFFN(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)

    # pos_weight to correct imbalance
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    print(f"pos_weight={pos_weight.item():.3f} (neg/pos={n_neg}/{n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val_precision = -1.0
    best_val_thr = 0.5
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # Validate
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

        quick_threshold_sweep(val_labels, val_prob, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))

        picked = pick_threshold_precision(
            val_labels, val_prob,
            min_precision=min_precision,
            min_recall=min_recall,
            min_trades=min_trades
        )

        if picked is None:
            chosen_thr = 0.5
            val_pred = (val_prob >= chosen_thr).astype(np.int64)
            val_precision = precision_score(val_labels, val_pred, zero_division=0)
            val_recall = recall_score(val_labels, val_pred, zero_division=0)
            val_f1 = f1_score(val_labels, val_pred, zero_division=0)
            trades = int(val_pred.sum())
        else:
            chosen_thr = picked["threshold"]
            val_precision = picked["precision"]
            val_recall = picked["recall"]
            val_pred = (val_prob >= chosen_thr).astype(np.int64)
            val_f1 = f1_score(val_labels, val_pred, zero_division=0)
            trades = picked["trades"]

        print(
            f"\nEpoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"VAL@thr={chosen_thr:.3f}: P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f} trades={trades}"
        )

        # Early stop on VAL precision (trading-aligned)
        if val_precision > best_val_precision + 1e-6:
            best_val_precision = val_precision
            best_val_thr = chosen_thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(model_dir, "best_trade_quantile_ffn.pt"))
            no_improve = 0
            print(f"  Saved best (VAL precision={best_val_precision:.3f}, thr={best_val_thr:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best
    if best_state is None:
        best_state = torch.load(os.path.join(model_dir, "best_trade_quantile_ffn.pt"))
    model.load_state_dict(best_state)

    # Test
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
    test_pred = (test_prob >= best_val_thr).astype(np.int64)

    acc = accuracy_score(test_labels, test_pred)
    prec = precision_score(test_labels, test_pred, zero_division=0)
    rec = recall_score(test_labels, test_pred, zero_division=0)
    f1 = f1_score(test_labels, test_pred, zero_division=0)

    print("\nFinal TEST Results (VAL threshold):")
    print(f"  Threshold: {best_val_thr:.3f}")
    print(f"  Trades:    {int(test_pred.sum())}/{len(test_pred)}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, target_names=["No-Trade", "Trade"], zero_division=0))

    cm = confusion_matrix(test_labels, test_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           No-Trade  Trade")
    print(f"Actual")
    print(f"No-Trade    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Trade       {cm[1,0]:4d}    {cm[1,1]:4d}")

    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": best_val_thr}


# -----------------------------
# Main: sweep q values
# -----------------------------

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    data_dir = os.path.join(project_root, "data")
    out_root = os.path.join(project_root, "models", "optionB_trade_quantile")
    os.makedirs(out_root, exist_ok=True)

    # Try multiple q values (fraction tradeable)
    q_list = [0.20, 0.25, 0.30, 0.35]  # start here

    all_results = {}

    for q in q_list:
        X_train, y_train, X_val, y_val, X_test, y_test, thr = prepare_quantile_labels(data_dir, q=q)

        run_dir = os.path.join(out_root, f"q_{int(q*100):02d}")
        os.makedirs(run_dir, exist_ok=True)

        res = {"abs_return_threshold_train": thr}

        # Baseline
        _, logreg_metrics = train_logreg(X_train, y_train, X_val, y_val, X_test, y_test)
        res["LogisticRegression"] = logreg_metrics

        # FFN
        _, ffn_metrics = train_ffn(
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
        res["ImprovedFFN"] = ffn_metrics

        all_results[str(q)] = res

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=2)

    print("\n" + "=" * 90)
    print("SUMMARY (Option B Quantile Trade Filter)")
    print("=" * 90)

    for q, res in all_results.items():
        print(f"\nq={q} (tradeable top {int(float(q)*100)}% |return|): thr_abs={res['abs_return_threshold_train']:.6f}")
        for name in ["LogisticRegression", "ImprovedFFN"]:
            m = res[name]
            print(f"  {name:16s} | thr={m['threshold']:.3f} | P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['acc']*100:.2f}%")

    with open(os.path.join(out_root, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
