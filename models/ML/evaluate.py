"""Evaluate model performance and fairness."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_recall_curve, 
    roc_auc_score, 
    precision_score, 
    recall_score
)
import pandas as pd
from joblib import dump
from typing import Dict, Any, Union, List, Tuple
from sklearn.base import BaseEstimator
import warnings

# Define identity terms
identity_terms = [
    "muslim", "christian", "jewish", "black", "white", "asian",
    "latino", "gay", "lesbian", "trans", "female", "male",
    "woman", "man", "nonbinary", "disabled"
]

def find_best_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    """Find optimal thresholds for each label using precision-recall curves.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        
    Returns:
        Array of optimal thresholds
    """
    thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        thresholds.append(thresh[np.argmax(f1_scores)])
    return np.array(thresholds)

def evaluate_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    df: pd.DataFrame,
    label_names: List[str],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """Evaluate model performance and fairness.
    
    Args:
        model: Trained model
        X: Input features
        y_true: True labels
        df: DataFrame with original data
        label_names: List of label names
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directories
    output_path = Path(output_dir)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    (output_path / "results").mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    y_probs = np.column_stack([
        estimator.predict_proba(X)[:, 1] for estimator in model.estimators_
    ])
    
    # Find optimal thresholds
    best_thresholds = find_best_thresholds(y_true, y_probs)
    y_pred = (y_probs > best_thresholds).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    
    # Fairness assessment
    mask_id = df['has_identity'] == 1
    mask_noid = df['has_identity'] == 0
    
    print("\n Fairness Check - Macro F1 Scores")
    print(f"With identity terms: {f1_score(y_true[mask_id], y_pred[mask_id], average='macro'):.4f}")
    print(f"Without identity terms: {f1_score(y_true[mask_noid], y_pred[mask_noid], average='macro'):.4f}")
    
    metrics['fairness'] = {
        'with_identity': f1_score(y_true[mask_id], y_pred[mask_id], average='macro'),
        'without_identity': f1_score(y_true[mask_noid], y_pred[mask_noid], average='macro')
    }
    
    # Identity-term specific F1 scores
    identity_f1 = {}
    for term in identity_terms:
        mask = df['comment_text'].str.lower().str.contains(term)
        if mask.sum() == 0:
            continue
        y_term = y_true[mask]
        y_term_pred = y_pred[mask]
        identity_f1[term] = f1_score(y_term, y_term_pred, average='macro')
    
    # Sort and plot identity-term F1 scores
    identity_f1 = dict(sorted(identity_f1.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(identity_f1.keys()), y=list(identity_f1.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Macro F1")
    plt.title("Fairness Assessment: F1 Score by Identity Term")
    plt.tight_layout()
    plt.savefig(output_path / "figures" / "per_label_f1_ml.png")
    plt.close()
    
    metrics['identity_f1'] = identity_f1
    
    # Save metrics
    with open(output_path / "results" / "eval_results.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics, best_thresholds

def evaluate_model_old(df, Y, Y_probs, best_thresholds, label_names, model_path="models/ML/opt_model.joblib"):
    Y_pred = (Y_probs > best_thresholds).astype(int)
    print(classification_report(Y, Y_pred, target_names=label_names))

    print("\n Fairness Check - Macro F1 Scores")
    mask_id = df['has_identity'] == 1
    mask_noid = df['has_identity'] == 0
    print(f"With identity terms: {f1_score(Y[mask_id], Y_pred[mask_id], average='macro'):.4f}")
    print(f"Without identity terms: {f1_score(Y[mask_noid], Y_pred[mask_noid], average='macro'):.4f}")

    identity_terms = df.columns[df.columns.str.startswith("has_") | df.columns.str.contains("identity")].tolist()
    identity_f1 = {}
    for term in identity_terms:
        mask = df['comment_text'].str.lower().str.contains(term.replace("has_", ""))
        if mask.sum() == 0:
            continue
        Y_term = Y[mask]
        Y_term_pred = Y_pred[mask]
        identity_f1[term] = f1_score(Y_term, Y_term_pred, average='macro')

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(identity_f1.keys()), y=list(identity_f1.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Macro F1")
    plt.title("Fairness Assessment: F1 Score by Identity Term")
    plt.tight_layout()
    plt.savefig("outputs/per_label_f1_ml.png")
    plt.close()

    np.save("outputs/X_bert_test.npy", Y_probs)
    pd.DataFrame(Y, columns=label_names).to_csv("outputs/Y_test.csv", index=False)
    
    # Save model
    dump(model, model_path)
    print(f"[INFO] Model saved to {model_path}")
    
    return best_thresholds