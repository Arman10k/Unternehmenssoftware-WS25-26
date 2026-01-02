"""
Stage 2: Content-Based Direction Model (Text + Sentiment)

Pipeline:
1) Stage-1 filter: q=0.35 quantile on |VWAP_return| (from TRAIN)
2) Stage-2 direction: predict Up/Down using NEWS CONTENT (headline + sentiment)

Features:
- Sentiment (float): raw score from Alpha Vantage
- Derived: abs_sent, sent_pos, sent_neg
- Headline Text: TF-IDF (1-2 grams)
- Optional: Keyword flags (beats, misses, raises, lowers, acquires, SEC, etc.)

Models:
- Logistic Regression (baseline)
- Linear SVM (often better for text + sparse features)

Threshold: Optimized on VAL for PRECISION (not F1) - trading-aligned

Inputs:
- data/news_events_metadata.csv  (event_id, news_headline, news_sentiment)
- data/event_y_train.npy, event_y_val.npy, event_y_test.npy  (VWAP returns)
- data/event_train.csv, event_val.csv, event_test.csv  (for event_id mapping)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
import scipy.sparse as sp


# -----------------------------
# Helpers
# -----------------------------

def compute_abs_threshold_from_train(y_train_cont: np.ndarray, q: float) -> float:
    """Compute absolute return threshold from TRAIN quantile."""
    return float(np.quantile(np.abs(y_train_cont), 1.0 - q))


def filter_by_abs_threshold(y_cont: np.ndarray, abs_thr: float):
    """Filter events by absolute return threshold."""
    mask = np.abs(y_cont) >= abs_thr
    return mask


def make_direction_labels(y_cont: np.ndarray) -> np.ndarray:
    """Convert returns to binary direction: 1=Up (ret>0), 0=Down (ret<=0)."""
    return (y_cont > 0).astype(np.int64)


def create_sentiment_features(sentiment: pd.Series) -> pd.DataFrame:
    """
    Create derived sentiment features.
    
    Returns DataFrame with columns: sentiment, abs_sent, sent_pos, sent_neg
    """
    df = pd.DataFrame()
    df['sentiment'] = sentiment.fillna(0.0)  # handle missing
    df['abs_sent'] = np.abs(df['sentiment'])
    df['sent_pos'] = (df['sentiment'] > 0).astype(float)
    df['sent_neg'] = (df['sentiment'] < 0).astype(float)
    return df


def create_keyword_features(headlines: pd.Series) -> pd.DataFrame:
    """
    Optional: Create binary keyword flags for common trading terms.
    
    Returns DataFrame with columns for each keyword pattern.
    """
    df = pd.DataFrame()
    
    # Convert to lowercase for matching
    headlines_lower = headlines.fillna('').str.lower()
    
    # Earnings-related
    df['kw_beats'] = headlines_lower.str.contains(r'\bbeat[s]?\b', regex=True).astype(float)
    df['kw_misses'] = headlines_lower.str.contains(r'\bmiss(es|ed)?\b', regex=True).astype(float)
    df['kw_earnings'] = headlines_lower.str.contains(r'\bearnings?\b', regex=True).astype(float)
    
    # Guidance-related
    df['kw_raises'] = headlines_lower.str.contains(r'\braise[sd]?\b', regex=True).astype(float)
    df['kw_lowers'] = headlines_lower.str.contains(r'\blower[sd]?\b', regex=True).astype(float)
    df['kw_guidance'] = headlines_lower.str.contains(r'\bguidance\b', regex=True).astype(float)
    
    # Corporate actions
    df['kw_acquires'] = headlines_lower.str.contains(r'\bacquir(e[sd]?|ition)\b', regex=True).astype(float)
    df['kw_merger'] = headlines_lower.str.contains(r'\bmerger?\b', regex=True).astype(float)
    df['kw_acquisition'] = headlines_lower.str.contains(r'\bacquisition\b', regex=True).astype(float)
    
    # Regulatory/legal
    df['kw_sec'] = headlines_lower.str.contains(r'\bsec\b', regex=True).astype(float)
    df['kw_probe'] = headlines_lower.str.contains(r'\bprobe[sd]?\b', regex=True).astype(float)
    df['kw_investigation'] = headlines_lower.str.contains(r'\binvestigation\b', regex=True).astype(float)
    
    # General sentiment
    df['kw_surprise'] = headlines_lower.str.contains(r'\bsurprise[sd]?\b', regex=True).astype(float)
    df['kw_warning'] = headlines_lower.str.contains(r'\bwarning\b', regex=True).astype(float)
    df['kw_upgrade'] = headlines_lower.str.contains(r'\bupgrade[sd]?\b', regex=True).astype(float)
    df['kw_downgrade'] = headlines_lower.str.contains(r'\bdowngrade[sd]?\b', regex=True).astype(float)
    
    return df


def pick_threshold_precision(y_true: np.ndarray, y_scores: np.ndarray, 
                             min_precision=0.60, min_recall=0.20, min_trades=10):
    """
    Pick threshold on VAL to maximize precision with constraints.
    
    Returns dict with threshold, precision, recall, trades or None if no threshold meets constraints.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall = precision[:-1], recall[:-1]
    
    best = None
    for p, r, t in zip(precision, recall, thresholds):
        pred = (y_scores >= t).astype(np.int64)
        trades = int(pred.sum())
        if trades < min_trades:
            continue
        if p >= min_precision and r >= min_recall:
            # Prioritize precision, then recall
            score = (p, r, trades)
            if best is None or score > best["score"]:
                best = {"score": score, "threshold": float(t), 
                       "precision": float(p), "recall": float(r), "trades": trades}
    
    if best is None:
        return None
    best.pop("score", None)
    return best


# -----------------------------
# Data Loading
# -----------------------------

def load_event_data(project_root: str, q: float):
    """
    Load event data, apply Stage-1 filter, and prepare content features.
    
    Returns:
        X_text_train, y_dir_train, X_text_val, y_dir_val, X_text_test, y_dir_test, sentiment_features, keyword_features
    """
    data_dir = os.path.join(project_root, 'data')
    
    # Load returns
    y_train_cont = np.load(os.path.join(data_dir, 'event_y_train.npy')).flatten()
    y_val_cont = np.load(os.path.join(data_dir, 'event_y_val.npy')).flatten()
    y_test_cont = np.load(os.path.join(data_dir, 'event_y_test.npy')).flatten()
    
    # Load event CSVs (already have news_headline and news_sentiment)
    train_data = pd.read_csv(os.path.join(data_dir, 'event_train.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'event_validation.csv'))  # Fixed
    test_data = pd.read_csv(os.path.join(data_dir, 'event_test.csv'))
    
    # Stage-1 filter: q=0.35 threshold from TRAIN
    abs_thr = compute_abs_threshold_from_train(y_train_cont, q)
    print(f"\nStage-1 Filter (q={q:.2f}): |return| >= {abs_thr:.6f} (% points)")
    
    mask_train = filter_by_abs_threshold(y_train_cont, abs_thr)
    mask_val = filter_by_abs_threshold(y_val_cont, abs_thr)
    mask_test = filter_by_abs_threshold(y_test_cont, abs_thr)
    
    print(f"  Train: {mask_train.sum()}/{len(y_train_cont)} ({mask_train.mean()*100:.1f}%)")
    print(f"  Val:   {mask_val.sum()}/{len(y_val_cont)} ({mask_val.mean()*100:.1f}%)")
    print(f"  Test:  {mask_test.sum()}/{len(y_test_cont)} ({mask_test.mean()*100:.1f}%)")
    
    # Filter datasets
    train_filt = train_data[mask_train].reset_index(drop=True)
    val_filt = val_data[mask_val].reset_index(drop=True)
    test_filt = test_data[mask_test].reset_index(drop=True)
    
    y_train_filt_cont = y_train_cont[mask_train]
    y_val_filt_cont = y_val_cont[mask_val]
    y_test_filt_cont = y_test_cont[mask_test]
    
    # Direction labels
    y_train_dir = make_direction_labels(y_train_filt_cont)
    y_val_dir = make_direction_labels(y_val_filt_cont)
    y_test_dir = make_direction_labels(y_test_filt_cont)
    
    print(f"\nDirection Distribution:")
    for name, y in [("Train", y_train_dir), ("Val", y_val_dir), ("Test", y_test_dir)]:
        d0 = int((y == 0).sum())
        d1 = int((y == 1).sum())
        print(f"  {name}: Down={d0} ({d0/len(y)*100:.1f}%) | Up={d1} ({d1/len(y)*100:.1f}%)")
    
    # Text features (headlines) - check if column exists
    if 'news_headline' not in train_filt.columns:
        print("\nnews_headline column not found in event CSVs. Using empty strings.")
        X_text_train = pd.Series([''] * len(train_filt))
        X_text_val = pd.Series([''] * len(val_filt))
        X_text_test = pd.Series([''] * len(test_filt))
    else:
        X_text_train = train_filt['news_headline'].fillna('')
        X_text_val = val_filt['news_headline'].fillna('')
        X_text_test = test_filt['news_headline'].fillna('')
    
    # Sentiment features
    if 'news_sentiment' not in train_filt.columns:
        print("\nnews_sentiment column not found in event CSVs. Using zeros.")
        sent_train = create_sentiment_features(pd.Series([0.0] * len(train_filt)))
        sent_val = create_sentiment_features(pd.Series([0.0] * len(val_filt)))
        sent_test = create_sentiment_features(pd.Series([0.0] * len(test_filt)))
    else:
        sent_train = create_sentiment_features(train_filt['news_sentiment'])
        sent_val = create_sentiment_features(val_filt['news_sentiment'])
        sent_test = create_sentiment_features(test_filt['news_sentiment'])
    
    # Keyword features
    kw_train = create_keyword_features(X_text_train)
    kw_val = create_keyword_features(X_text_val)
    kw_test = create_keyword_features(X_text_test)
    
    return (X_text_train, y_train_dir, X_text_val, y_val_dir, X_text_test, y_test_dir,
            (sent_train, sent_val, sent_test), (kw_train, kw_val, kw_test), abs_thr)



# -----------------------------
# Model Training
# -----------------------------

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Logistic Regression with precision-optimized threshold."""
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION (Content-Based Direction)")
    print("="*80)
    
    model = LogisticRegression(max_iter=5000, C=1.0, random_state=42, solver='saga')
    model.fit(X_train, y_train)
    
    # Get probabilities for Up class
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Pick threshold on VAL for max precision
    picked = pick_threshold_precision(y_val, val_probs, min_precision=0.60, min_recall=0.20, min_trades=10)
    
    if picked:
        thr = picked['threshold']
        print(f"\nVAL threshold: {thr:.3f} (P={picked['precision']:.3f}, R={picked['recall']:.3f}, trades={picked['trades']})")
    else:
        thr = 0.5
        print(f"\nNo VAL threshold met constraints, using 0.5")
    
    test_pred = (test_probs >= thr).astype(np.int64)
    
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    
    print(f"\nTEST: Acc={acc*100:.2f}% | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f}")
    print(classification_report(y_test, test_pred, target_names=['Down', 'Up'], zero_division=0))
    
    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": thr}


def train_linear_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Linear SVM with precision-optimized threshold."""
    print("\n" + "="*80)
    print("LINEAR SVM (Content-Based Direction)")
    print("="*80)
    
    model = LinearSVC(max_iter=5000, C=1.0, random_state=42, dual=False)
    model.fit(X_train, y_train)
    
    # Get decision function scores
    val_scores = model.decision_function(X_val)
    test_scores = model.decision_function(X_test)
    
    # Pick threshold on VAL
    picked = pick_threshold_precision(y_val, val_scores, min_precision=0.60, min_recall=0.20, min_trades=10)
    
    if picked:
        thr = picked['threshold']
        print(f"\nVAL threshold: {thr:.3f} (P={picked['precision']:.3f}, R={picked['recall']:.3f}, trades={picked['trades']})")
    else:
        thr = 0.0
        print(f"\nNo VAL threshold met constraints, using 0.0")
    
    test_pred = (test_scores >= thr).astype(np.int64)
    
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    
    print(f"\nTEST: Acc={acc*100:.2f}% | P={prec:.3f} | R={rec:.3f} | F1={f1:.3f}")
    print(classification_report(y_test, test_pred, target_names=['Down', 'Up'], zero_division=0))
    
    return model, {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": thr}


# -----------------------------
# Main
# -----------------------------

def main():
    q = 0.35  # Stage-1 quantile threshold
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    print("\n" + "="*90)
    print("CONTENT-BASED DIRECTION MODEL (Stage 2 with Text + Sentiment)")
    print("="*90)
    
    # Load data
    (X_text_train, y_train, X_text_val, y_val, X_text_test, y_test,
     (sent_train, sent_val, sent_test),
     (kw_train, kw_val, kw_test), abs_thr) = load_event_data(project_root, q=q)
    
    # Create TF-IDF features from headlines
    print("\nChecking headlines for TF-IDF features...")
    
    # Check if we have any non-empty headlines
    non_empty_count = sum(1 for text in X_text_train if text.strip())
    
    if non_empty_count < 10:
        print(f"  Only {non_empty_count} non-empty headlines found. Skipping TF-IDF.")
        print("  → Using only Sentiment + Keyword features")
        
        # No TF-IDF features - just use sentiment + keywords
        X_train_comb = sp.hstack([sp.csr_matrix(sent_train.values), sp.csr_matrix(kw_train.values)])
        X_val_comb = sp.hstack([sp.csr_matrix(sent_val.values), sp.csr_matrix(kw_val.values)])
        X_test_comb = sp.hstack([sp.csr_matrix(sent_test.values), sp.csr_matrix(kw_test.values)])
        
        total_features = X_train_comb.shape[1]
        print(f"\nCombined features: {total_features} ({sent_train.shape[1]} sentiment + {kw_train.shape[1]} keywords)")
        
    else:
        print(f"  Found {non_empty_count} non-empty headlines. Creating TF-IDF (1-2 grams)...")
        
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2, max_df=0.8)
        
        X_tfidf_train = tfidf.fit_transform(X_text_train)
        X_tfidf_val = tfidf.transform(X_text_val)
        X_tfidf_test = tfidf.transform(X_text_test)
        
        print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
        print(f"  TF-IDF shape: {X_tfidf_train.shape}")
        
        # Combine features: TF-IDF + Sentiment + Keywords
        X_train_comb = sp.hstack([X_tfidf_train, sp.csr_matrix(sent_train.values), sp.csr_matrix(kw_train.values)])
        X_val_comb = sp.hstack([X_tfidf_val, sp.csr_matrix(sent_val.values), sp.csr_matrix(kw_val.values)])
        X_test_comb = sp.hstack([X_tfidf_test, sp.csr_matrix(sent_test.values), sp.csr_matrix(kw_test.values)])
        
        total_features = X_train_comb.shape[1]
        print(f"\nCombined features: {total_features} (TF-IDF + {sent_train.shape[1]} sentiment + {kw_train.shape[1]} keywords)")

    
    # Check for sufficient samples
    if len(y_train) < 80 or len(y_val) < 30 or len(y_test) < 30:
        print(f"\nToo few tradeable samples after Stage-1 filter: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
        print("Consider lowering q or adding more data.")
        return
    
    results = {}
    
    # Train Logistic Regression
    logreg, logreg_metrics = train_logistic_regression(X_train_comb, y_train, X_val_comb, y_val, X_test_comb, y_test)
    results['LogisticRegression'] = logreg_metrics
    
    # Train Linear SVM
    svm, svm_metrics = train_linear_svm(X_train_comb, y_train, X_val_comb, y_val, X_test_comb, y_test)
    results['LinearSVM'] = svm_metrics
    
    # Save results
    out_dir = os.path.join(project_root, 'models', 'content_direction_q35')
    os.makedirs(out_dir, exist_ok=True)
    
    results['q'] = q
    results['abs_threshold'] = abs_thr
    results['n_features'] = total_features
    
    with open(os.path.join(out_dir, 'content_direction_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    for name in ['LogisticRegression', 'LinearSVM']:
        m = results[name]
        print(f"{name}: thr={m['threshold']:.3f} | P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['acc']*100:.2f}%")
    
    print(f"\nSaved to: {os.path.join(out_dir, 'content_direction_results.json')}")


if __name__ == "__main__":
    main()
