"""
Stage 2: Direction (Up vs Down) for Option B Quantile Pipeline

Train ONLY on events selected by Stage-1 quantile rule (top q by |return|),
using TRAIN-derived absolute-return threshold (no leakage).

Direction label:
  Down (0): return < 0
  Up   (1): return > 0

Key:
- Filter subset must match Stage 1 selection logic (quantile threshold), NOT epsilon.
- Threshold on VAL tuned for precision (trading-aligned), not max F1.

Inputs in data_dir:
- event_X_train_scaled.npy, event_X_val_scaled.npy, event_X_test_scaled.npy
- event_y_train.npy, event_y_val.npy, event_y_test.npy  (returns in % points)
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_arrays(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "event_X_train_scaled.npy"))
    X_val   = np.load(os.path.join(data_dir, "event_X_val_scaled.npy"))
    X_test  = np.load(os.path.join(data_dir, "event_X_test_scaled.npy"))

    y_train = np.load(os.path.join(data_dir, "event_y_train.npy")).flatten()
    y_val   = np.load(os.path.join(data_dir, "event_y_val.npy")).flatten()
    y_test  = np.load(os.path.join(data_dir, "event_y_test.npy")).flatten()
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_abs_threshold_from_train(y_train_cont: np.ndarray, q: float) -> float:
    # q=0.35 => threshold at 65th percentile of |return|
    return float(np.quantile(np.abs(y_train_cont), 1.0 - q))


def filter_by_abs_threshold(X: np.ndarray, y_cont: np.ndarray, abs_thr: float):
    mask = np.abs(y_cont) >= abs_thr
    return X[mask], y_cont[mask], mask


def make_direction_labels(y_cont: np.ndarray) -> np.ndarray:
    # define 1=Up, 0=Down; assume filtered y_cont contains no zeros-ish
    return (y_cont > 0).astype(np.int64)


def pick_threshold_precision(y_true: np.ndarray, y_prob: np.ndarray, min_precision=0.60, min_recall=0.20, min_trades=10):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]

    best = None
    for p, r, t in zip(precision, recall, thresholds):
        pred = (y_prob >= t).astype(np.int64)
        trades = int(pred.sum())
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


# -----------------------------
# Baseline: Logistic Regression
# -----------------------------

def train_logreg_direction(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION (Direction on Stage-1-selected events)")
    print("=" * 80)

    model = LogisticRegression(max_iter=4000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    picked = pick_threshold_precision(y_val, val_prob, min_precision=0.58, min_recall=0.25, min_trades=10)
    thr = picked["threshold"] if picked else 0.5

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= thr).astype(np.int64)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)

    print(f"VAL-picked thr={thr:.3f}" + (f" (VAL P={picked['precision']:.3f} R={picked['recall']:.3f} trades={picked['trades']})" if picked else ""))
    print(f"TEST: Acc={acc*100:.2f}% P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    print(classification_report(y_test, test_pred, target_names=["Down","Up"], zero_division=0))

    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": thr}


# -----------------------------
# FFN (1-logit)
# -----------------------------

class DirectionFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout_p=0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_direction_ffn(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_dir: str,
    hidden_dims=(64, 32),
    dropout_p=0.25,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-3,
    epochs=200,
    patience=20,
):
    print("\n" + "=" * 80)
    print("IMPROVED FFN (Direction on Stage-1-selected events)")
    print("=" * 80)

    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | train_samples={len(X_train)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    model = DirectionFFN(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    print(f"pos_weight={pos_weight.item():.3f} (neg/pos={n_neg}/{n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val_score = -1.0
    best_val_thr = 0.5
    no_improve = 0

    # For direction, we tune for higher precision than your current 0.56
    MIN_P = 0.60
    MIN_R = 0.25
    MIN_TRADES = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_logits = []
        val_labels = []
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb_dev = yb.to(device)
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

        picked = pick_threshold_precision(val_labels, val_prob, min_precision=MIN_P, min_recall=MIN_R, min_trades=MIN_TRADES)
        if picked is None:
            # fallback score = F1@0.5 (just for early stopping to behave)
            pred05 = (val_prob >= 0.5).astype(np.int64)
            p05 = precision_score(val_labels, pred05, zero_division=0)
            r05 = recall_score(val_labels, pred05, zero_division=0)
            f105 = f1_score(val_labels, pred05, zero_division=0)
            score = f105
            thr = 0.5
            p_sel, r_sel = p05, r05
        else:
            thr = picked["threshold"]
            pred = (val_prob >= thr).astype(np.int64)
            p_sel = picked["precision"]
            r_sel = picked["recall"]
            f1_sel = f1_score(val_labels, pred, zero_division=0)
            # score prioritizes precision, then recall (trading-aligned)
            score = p_sel + 0.1 * r_sel

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | VAL thr={thr:.3f} P={p_sel:.3f} R={r_sel:.3f} score={score:.3f}")

        if score > best_val_score + 1e-6:
            best_val_score = score
            best_val_thr = thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(model_dir, "best_direction_ffn.pt"))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = torch.load(os.path.join(model_dir, "best_direction_ffn.pt"))
    model.load_state_dict(best_state)

    # TEST
    model.eval()
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            test_logits.append(model(xb).cpu().numpy())
            test_labels.append(yb.numpy())
    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels).astype(np.int64)

    test_prob = sigmoid(test_logits)
    test_pred = (test_prob >= best_val_thr).astype(np.int64)

    acc = accuracy_score(test_labels, test_pred)
    prec = precision_score(test_labels, test_pred, zero_division=0)
    rec = recall_score(test_labels, test_pred, zero_division=0)
    f1 = f1_score(test_labels, test_pred, zero_division=0)

    print("\nTEST using VAL-picked precision threshold:")
    print(f"  thr={best_val_thr:.3f} | Acc={acc*100:.2f}% P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    print(classification_report(test_labels, test_pred, target_names=["Down","Up"], zero_division=0))

    cm = confusion_matrix(test_labels, test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": best_val_thr}


def main():
    # --- config ---
    q = 0.35  # FIXED per your decision
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    data_dir = os.path.join(project_root, "data")
    model_root = os.path.join(project_root, "models", "optionB_direction_quantile_q35")
    os.makedirs(model_root, exist_ok=True)

    # Load
    X_train, X_val, X_test, y_train_cont, y_val_cont, y_test_cont = load_arrays(data_dir)

    # Stage-1 quantile threshold from TRAIN
    abs_thr = compute_abs_threshold_from_train(y_train_cont, q=q)
    print(f"\nStage-1 abs threshold (from TRAIN, q={q:.2f}): |ret| >= {abs_thr:.6f} (% points)")

    # Filter same way on each split
    X_train_f, y_train_f_cont, _ = filter_by_abs_threshold(X_train, y_train_cont, abs_thr)
    X_val_f,   y_val_f_cont,   _ = filter_by_abs_threshold(X_val,   y_val_cont,   abs_thr)
    X_test_f,  y_test_f_cont,  _ = filter_by_abs_threshold(X_test,  y_test_cont,  abs_thr)

    y_train_dir = make_direction_labels(y_train_f_cont)
    y_val_dir   = make_direction_labels(y_val_f_cont)
    y_test_dir  = make_direction_labels(y_test_f_cont)

    print("\nFiltered sizes:")
    print(f"  Train={len(y_train_dir)} Val={len(y_val_dir)} Test={len(y_test_dir)}")

    if len(y_train_dir) < 80 or len(y_val_dir) < 30 or len(y_test_dir) < 30:
        print("\nToo few samples after filtering. Consider increasing q or adding more data.")
        return

    # Distribution
    def dist(name, y):
        d0 = int((y == 0).sum()); d1 = int((y == 1).sum())
        print(f"  {name}: Down={d0} ({d0/len(y)*100:.1f}%) Up={d1} ({d1/len(y)*100:.1f}%)")
    print("\nDirection distribution:")
    dist("Train", y_train_dir); dist("Val", y_val_dir); dist("Test", y_test_dir)

    results = {}

    # Baseline
    _, m1 = train_logreg_direction(X_train_f, y_train_dir, X_val_f, y_val_dir, X_test_f, y_test_dir)
    results["LogisticRegression"] = m1

    # FFN
    run_dir = os.path.join(model_root, "ffn")
    _, m2 = train_direction_ffn(X_train_f, y_train_dir, X_val_f, y_val_dir, X_test_f, y_test_dir, model_dir=run_dir)
    results["ImprovedFFN"] = m2

    with open(os.path.join(model_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Saved:", os.path.join(model_root, "results.json"))


if __name__ == "__main__":
    main()
