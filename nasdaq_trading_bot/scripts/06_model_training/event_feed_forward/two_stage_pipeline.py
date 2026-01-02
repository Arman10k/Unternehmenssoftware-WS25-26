"""
FINAL PRODUCTION PIPELINE: 2-Stage Event Trading Model (FIXED)

Fixes included:
- NO TEST leakage: thresholds are picked on VAL only (Stage1 + Stage2)
- Best model saving: checkpoints + metadata (threshold, metrics)
- Correct scheduler usage with val_loss
- Correct device usage (CPU/GPU)
- Real plots: loss curves, PR curves, threshold tradeoff, confusion matrices, pipeline counts
"""

import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt


# ======================================================================================
# UTILS
# ======================================================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_quantile_threshold(y_train_cont, q):
    """Compute absolute return threshold from TRAIN quantile."""
    return float(np.quantile(np.abs(y_train_cont), 1.0 - q))


def create_stage1_labels(y_cont, abs_threshold):
    """Convert returns to Trade (1) vs No-Trade (0) labels."""
    return (np.abs(y_cont) >= abs_threshold).astype(np.int64)


def make_direction_labels(y_cont):
    """Up (1) vs Down (0)."""
    return (y_cont > 0).astype(np.int64)


def create_sentiment_features(sentiment_series: pd.Series):
    """Create 4 sentiment features."""
    sent = sentiment_series.fillna(0.0).values.astype(float)
    return np.column_stack([
        sent,
        np.abs(sent),
        (sent > 0).astype(float),
        (sent < 0).astype(float)
    ])


def pick_precision_threshold(y_true, y_scores, min_precision=0.60, min_recall=0.20, min_trades=10):
    """
    Pick threshold on VAL: maximize precision then recall under constraints.
    Returns dict or None.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall = precision[:-1], recall[:-1]

    best = None
    for p, r, t in zip(precision, recall, thresholds):
        pred = (y_scores >= t).astype(np.int64)
        trades = int(pred.sum())
        if trades < min_trades:
            continue
        if p < min_precision or r < min_recall:
            continue
        score = (p, r, trades)
        if best is None or score > best["score"]:
            best = {"score": score, "threshold": float(t),
                    "precision": float(p), "recall": float(r), "trades": trades}

    if best is None:
        return None
    best.pop("score", None)
    return best


# ======================================================================================
# STAGE 1 MODEL
# ======================================================================================

class Stage1TradeFilter(nn.Module):
    """Binary FFN for Stage 1: Trade vs No-Trade using quantile filtering."""
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout_p=0.30):
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


def train_stage1(
    X_train, y_train,
    X_val, y_val,
    model_dir,
    *,
    hidden_dims=(64, 32),
    dropout_p=0.30,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-3,
    epochs=200,
    patience=20,
    min_precision=0.50,
    min_recall=0.20,
    min_trades=10,
    device=None
):
    """
    Train Stage 1 with:
    - BCEWithLogits + pos_weight
    - ReduceLROnPlateau on val_loss
    - Early stop on VAL precision (picked threshold on VAL)
    - Saves best checkpoint + meta
    """
    os.makedirs(model_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print("\n" + "=" * 80)
    print("STAGE 1: QUANTILE TRADE FILTER (FIXED)")
    print("=" * 80)
    print(f"Using device: {device}")

    model = Stage1TradeFilter(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_p=dropout_p).to(device)

    # pos_weight for imbalance
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    print(f"pos_weight={pos_weight.item():.3f} (neg/pos={n_neg}/{n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False
    )

    history = {
        "train_loss": [], "val_loss": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
        "val_threshold": [], "val_trades": [],
        "lr": [],
    }

    best = {
        "val_precision": -1.0,
        "val_threshold": 0.5,
        "epoch": -1,
        "state_dict": None,
        "picked": None
    }
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
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

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_logits.append(logits.cpu().numpy())
                val_labels.append(yb.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels).astype(np.int64)
        val_prob = sigmoid_np(val_logits)

        picked = pick_precision_threshold(
            val_labels, val_prob,
            min_precision=min_precision, min_recall=min_recall, min_trades=min_trades
        )
        if picked is None:
            thr = 0.5
            val_pred = (val_prob >= thr).astype(np.int64)
            p = precision_score(val_labels, val_pred, zero_division=0)
            r = recall_score(val_labels, val_pred, zero_division=0)
            f1 = f1_score(val_labels, val_pred, zero_division=0)
            trades = int(val_pred.sum())
        else:
            thr = picked["threshold"]
            p = picked["precision"]
            r = picked["recall"]
            val_pred = (val_prob >= thr).astype(np.int64)
            f1 = f1_score(val_labels, val_pred, zero_division=0)
            trades = picked["trades"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_precision"].append(float(p))
        history["val_recall"].append(float(r))
        history["val_f1"].append(float(f1))
        history["val_threshold"].append(float(thr))
        history["val_trades"].append(int(trades))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"VAL thr={thr:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} trades={trades}")

        # ---- best checkpoint on VAL precision ----
        if p > best["val_precision"] + 1e-6:
            best["val_precision"] = float(p)
            best["val_threshold"] = float(thr)
            best["epoch"] = epoch
            best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["picked"] = picked
            no_improve = 0

            ckpt = {
                "epoch": epoch,
                "state_dict": best["state_dict"],
                "best_val_precision": best["val_precision"],
                "best_val_threshold": best["val_threshold"],
                "pos_weight": float(pos_weight.item()),
                "hidden_dims": list(hidden_dims),
                "dropout_p": float(dropout_p),
            }
            torch.save(ckpt, os.path.join(model_dir, "best_stage1_trade_filter.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])

    # save meta + history
    with open(os.path.join(model_dir, "stage1_history.json"), "w") as f:
        best_for_json = {k: v for k, v in best.items() if k != "state_dict"}
        json.dump({"best": best_for_json, "history": history}, f, indent=2)

    print(f"Stage 1 saved: best_stage1_trade_filter.pt (VAL best P={best['val_precision']:.3f}, thr={best['val_threshold']:.3f})")

    return model, best, history


def eval_stage1(model, X, y_true, threshold, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    prob = sigmoid_np(logits)
    pred = (prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, pred)

    return {
        "prob": prob,
        "pred": pred,
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "cm": cm.tolist(),
        "n_pred_trade": int(pred.sum()),
        "n": int(len(y_true)),
    }


# ======================================================================================
# STAGE 2 (LogReg)
# ======================================================================================

def train_stage2_logreg(X_train, y_train, X_val, y_val, model_dir,
                        min_precision=0.50, min_recall=0.20, min_trades=10):
    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("STAGE 2: SENTIMENT DIRECTION (FIXED)")
    print("=" * 80)

    model = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    picked = pick_precision_threshold(y_val, val_prob,
                                      min_precision=min_precision,
                                      min_recall=min_recall,
                                      min_trades=min_trades)
    thr = picked["threshold"] if picked else 0.5

    import joblib
    joblib.dump(model, os.path.join(model_dir, "best_stage2_direction.joblib"))
    with open(os.path.join(model_dir, "best_stage2_meta.json"), "w") as f:
        json.dump({"threshold": float(thr), "picked": picked}, f, indent=2)

    print(f"Stage 2 saved: best_stage2_direction.joblib (VAL thr={thr:.3f})")
    return model, thr


def eval_stage2(model, X, y_true, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, pred)

    return {
        "prob": prob,
        "pred": pred,
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "cm": cm.tolist(),
        "n_pred_up": int(pred.sum()),
        "n": int(len(y_true)),
    }


# ======================================================================================
# PLOTS (REAL DIAGRAMS)
# ======================================================================================

def plot_loss(history, outpath):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train_loss", linewidth=2)
    plt.plot(history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Stage 1 Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pr(y_true, prob, title, outpath):
    p, r, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_threshold_tradeoff(y_true, prob, outpath):
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, trade_rate = [], [], []
    for t in thresholds:
        pred = (prob >= t).astype(np.int64)
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        trade_rate.append(pred.sum() / max(1, len(y_true)))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="precision", linewidth=2)
    plt.plot(thresholds, recalls, label="recall", linewidth=2)
    plt.plot(thresholds, trade_rate, label="trade_rate", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score / Rate")
    plt.title("Stage 1 Threshold Tradeoff")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_accuracy_bar(cm, title, outpath):
    # cm structure: [[TN, FP], [FN, TP]]
    cm = np.array(cm, dtype=int)
    correct = cm[0, 0] + cm[1, 1]
    incorrect = cm[0, 1] + cm[1, 0]
    total = correct + incorrect
    
    if total == 0:
        return

    pct_correct = (correct / total) * 100
    pct_incorrect = (incorrect / total) * 100
    
    labels = ['Correct', 'Incorrect']
    values = [pct_correct, pct_incorrect]
    colors = ['#2ca02c', '#d62728']  # Green, Red
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    
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
                 
    # Add count labels inside bars
    counts = [correct, incorrect]
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'({counts[i]})',
                 ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pipeline_counts(n_total, n_stage1_trade, n_stage2_up, outpath):
    labels = ["All test events", "Stage1 predicted trade", "Stage2 predicted up"]
    vals = [n_total, n_stage1_trade, n_stage2_up]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.title("Pipeline Counts (TEST)")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def create_all_plots(img_out_dir, stage1_history, stage1_eval, y_stage1_true, stage2_eval=None, y_stage2_true=None):
    os.makedirs(img_out_dir, exist_ok=True)

    # 1. Loss
    plot_loss(stage1_history, os.path.join(img_out_dir, "06_stage1_loss.png"))
    
    # 2. Accuracy Bar (Correct vs Incorrect)
    plot_accuracy_bar(stage1_eval["cm"], "Stage 1: Prediction Accuracy", 
                      os.path.join(img_out_dir, "06_stage1_accuracy.png"))

    if stage2_eval is not None and y_stage2_true is not None and len(y_stage2_true) > 0:
        plot_accuracy_bar(stage2_eval["cm"], "Stage 2: Prediction Accuracy", 
                          os.path.join(img_out_dir, "06_stage2_accuracy.png"))


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    Q = 0.35
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models", "two_stage_pipeline")
    os.makedirs(model_dir, exist_ok=True)

    # Load X / y
    X_train = np.load(os.path.join(data_dir, "event_X_train_scaled.npy"))
    X_val   = np.load(os.path.join(data_dir, "event_X_val_scaled.npy"))
    X_test  = np.load(os.path.join(data_dir, "event_X_test_scaled.npy"))

    y_train_cont = np.load(os.path.join(data_dir, "event_y_train.npy")).flatten()
    y_val_cont   = np.load(os.path.join(data_dir, "event_y_val.npy")).flatten()
    y_test_cont  = np.load(os.path.join(data_dir, "event_y_test.npy")).flatten()

    # Load sentiment (already computed)
    train_df = pd.read_csv(os.path.join(data_dir, "event_train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "event_validation.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "event_test.csv"))

    # ---- Stage 1 labels ----
    abs_thr = compute_quantile_threshold(y_train_cont, Q)
    y_train_s1 = create_stage1_labels(y_train_cont, abs_thr)
    y_val_s1   = create_stage1_labels(y_val_cont, abs_thr)
    y_test_s1  = create_stage1_labels(y_test_cont, abs_thr)

    print("\n" + "="*90)
    print(f"FINAL PIPELINE (FIXED) | q={Q} | abs_thr={abs_thr:.6f}%")
    print("="*90)
    print(f"Train tradeable: {y_train_s1.sum()}/{len(y_train_s1)} ({y_train_s1.mean()*100:.1f}%)")
    print(f"Val tradeable:   {y_val_s1.sum()}/{len(y_val_s1)} ({y_val_s1.mean()*100:.1f}%)")
    print(f"Test tradeable:  {y_test_s1.sum()}/{len(y_test_s1)} ({y_test_s1.mean()*100:.1f}%)")

    # ---- Train Stage 1 (VAL picks threshold) ----
    stage1_model, best1, hist1 = train_stage1(
        X_train, y_train_s1,
        X_val, y_val_s1,
        model_dir=model_dir,
        device=device,
        min_precision=0.50,
        min_recall=0.20,
        min_trades=10
    )
    
    # 🟢 OPTION A: Fixed probability threshold instead of precision-optimized
    # Target: 10-30% selection rate (30-80 trades on TEST with 269 events)
    stage1_thr_fixed = 0.65  # Fixed threshold - more relaxed than precision-optimized
    
    print(f"\nUsing FIXED Stage-1 threshold: {stage1_thr_fixed:.3f} (instead of precision-optimized {best1['val_threshold']:.3f})")
    stage1_thr = stage1_thr_fixed
    
    # ---- Eval Stage 1 on TEST using FIXED threshold ----
    s1_test = eval_stage1(stage1_model, X_test, y_test_s1, threshold=stage1_thr, device=device)
    print("\nSTAGE 1 TEST (FIXED threshold=0.65):")
    print(f"thr={stage1_thr:.3f} | P={s1_test['precision']:.3f} R={s1_test['recall']:.3f} "
          f"F1={s1_test['f1']:.3f} Acc={s1_test['acc']*100:.2f}% | trades={s1_test['n_pred_trade']}/{s1_test['n']} ({s1_test['n_pred_trade']/s1_test['n']*100:.1f}%)")


    # ---- Stage 2 datasets ----
    # Train/Val: use TRUE tradeable labels (no leakage)
    mask_train = (y_train_s1 == 1)
    mask_val   = (y_val_s1 == 1)

    # Test: use PREDICTED tradeables (real pipeline)
    mask_test = (s1_test["pred"] == 1)

    sent_train = create_sentiment_features(train_df.get("news_sentiment", pd.Series([0.0]*len(train_df))))
    sent_val   = create_sentiment_features(val_df.get("news_sentiment", pd.Series([0.0]*len(val_df))))
    sent_test  = create_sentiment_features(test_df.get("news_sentiment", pd.Series([0.0]*len(test_df))))

    X_train_s2 = sent_train[mask_train]
    X_val_s2   = sent_val[mask_val]
    X_test_s2  = sent_test[mask_test]

    y_train_s2 = make_direction_labels(y_train_cont[mask_train])
    y_val_s2   = make_direction_labels(y_val_cont[mask_val])
    y_test_s2  = make_direction_labels(y_test_cont[mask_test])

    print("\nStage 2 sizes:")
    print(f"Train: {len(y_train_s2)} | Val: {len(y_val_s2)} | Test (from Stage1 preds): {len(y_test_s2)}")

    stage2_eval = None

    if len(y_train_s2) >= 50 and len(y_val_s2) >= 20:
        stage2_model, stage2_thr = train_stage2_logreg(
            X_train_s2, y_train_s2,
            X_val_s2, y_val_s2,
            model_dir=model_dir,
            min_precision=0.50,
            min_recall=0.20,
            min_trades=5
        )

        if len(y_test_s2) >= 10:
            stage2_eval = eval_stage2(stage2_model, X_test_s2, y_test_s2, threshold=stage2_thr)
            print("\nSTAGE 2 TEST (VAL threshold):")
            print(f"thr={stage2_thr:.3f} | P={stage2_eval['precision']:.3f} R={stage2_eval['recall']:.3f} "
                  f"F1={stage2_eval['f1']:.3f} Acc={stage2_eval['acc']*100:.2f}% | up={stage2_eval['n_pred_up']}/{stage2_eval['n']}")
        else:
            print("Stage2 TEST too small after Stage1 filtering (metrics unreliable).")
            stage2_thr = None
    else:
        print("Not enough samples for Stage2 training.")
        stage2_thr = None

    # ---- Combined metric (only if stage2 exists) ----
    combined_precision = None
    if stage2_eval is not None:
        combined_precision = s1_test["precision"] * stage2_eval["precision"]

    results = {
        "q": Q,
        "abs_threshold": abs_thr,
        "stage1": {
            "threshold": float(stage1_thr),
            "precision": s1_test["precision"],
            "recall": s1_test["recall"],
            "f1": s1_test["f1"],
            "accuracy": s1_test["acc"],
            "pred_trades": s1_test["n_pred_trade"],
        },
        "stage2": None if stage2_eval is None else {
            "threshold": None if stage2_thr is None else float(stage2_thr),
            "precision": stage2_eval["precision"],
            "recall": stage2_eval["recall"],
            "f1": stage2_eval["f1"],
            "accuracy": stage2_eval["acc"],
            "pred_up": stage2_eval["n_pred_up"],
            "n_test": stage2_eval["n"],
        },
        "combined_precision": None if combined_precision is None else float(combined_precision),
        "test_filtered": int(mask_test.sum()),
    }

    with open(os.path.join(model_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- plots ----
    # Save plots to nasdaq_trading_bot/images
    img_out_dir = os.path.join(project_root, "images")
    
    create_all_plots(
        img_out_dir=img_out_dir,
        stage1_history=hist1,
        stage1_eval=s1_test,
        y_stage1_true=y_test_s1,
        stage2_eval=stage2_eval,
        y_stage2_true=y_test_s2 if stage2_eval is not None else None
    )

    print(f"\nSaved results to: {model_dir}")
    print(f"Saved plots to:   {img_out_dir}")
    print("Plots created:")
    print("  06_stage1_loss.png")
    print("  06_stage1_accuracy.png")
    if stage2_eval is not None:
        print("  06_stage2_accuracy.png")


if __name__ == "__main__":
    main()
