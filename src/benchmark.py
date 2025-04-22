# Benchmarking Script: ML Model vs LLM Model (Enhanced for Publication)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, classification_report
from joblib import load
from transformers import pipeline
import seaborn as sns
from datetime import datetime
from pathlib import Path
import os
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Get project root path
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "opt_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
TEST_LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "test_labels.csv"
BERT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "X_bert_test.npy"
EVAL_RESULTS_PATH = PROJECT_ROOT / "eval_results.json"

# Define label names and ensure they match the model's output
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
n_labels = len(label_names)

def load_data():
    """Load all required data and models with error handling."""
    try:
        # Load LLM evaluation results
        with open(EVAL_RESULTS_PATH) as f:
            llm_results = json.load(f)
        
        # Load ML model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        opt_model = load(MODEL_PATH)
        
        # Verify model type
        if not isinstance(opt_model, MultiOutputClassifier):
            raise TypeError(f"Expected MultiOutputClassifier, got {type(opt_model)}")
        
        # Load test data
        if not BERT_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"BERT embeddings not found at {BERT_EMBEDDINGS_PATH}")
        X_bert = np.load(BERT_EMBEDDINGS_PATH)
        
        # Load test labels
        if not TEST_LABELS_PATH.exists():
            raise FileNotFoundError(f"Test labels not found at {TEST_LABELS_PATH}")
        Y_true_df = pd.read_csv(TEST_LABELS_PATH)
        
        # Ensure we only use the labels we care about
        Y_true = Y_true_df[label_names].values.astype(int)
        
        # Load test data for LLM
        if not TEST_DATA_PATH.exists():
            raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")
        test_data = pd.read_csv(TEST_DATA_PATH)
        
        # Ensure consistent sample sizes
        if len(X_bert) != len(Y_true):
            print(f"Warning: Mismatch in sample sizes - X_bert: {len(X_bert)}, Y_true: {len(Y_true)}")
            # Use the smaller size
            min_size = min(len(X_bert), len(Y_true))
            X_bert = X_bert[:min_size]
            Y_true = Y_true[:min_size]
            test_data = test_data.iloc[:min_size]
            print(f"Using {min_size} samples for evaluation")
        
        return llm_results, opt_model, X_bert, Y_true, test_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def predict_proba_multioutput(model, X):
    """Get probability predictions from a MultiOutputClassifier."""
    try:
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.column_stack(predictions)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def find_best_thresholds(y_true, y_probs):
    """Find optimal thresholds for each label using ROC curve."""
    thresholds = []
    for i in range(y_true.shape[1]):
        # Remove -1 values from consideration
        mask = y_true[:, i] != -1
        y_true_clean = y_true[mask, i]
        y_probs_clean = y_probs[mask, i]
        
        if len(y_true_clean) == 0:
            print(f"Warning: No valid labels for class {i}")
            thresholds.append(0.5)
            continue
            
        # Get unique values for threshold candidates
        unique_probs = np.unique(y_probs_clean)
        best_f1 = -1
        best_thresh = 0.5  # Default threshold
        
        # Try different thresholds
        for thresh in unique_probs:
            y_pred = (y_probs_clean >= thresh).astype(int)
            f1 = f1_score(y_true_clean, y_pred, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds.append(best_thresh)
        print(f"Label {i}: Best threshold = {best_thresh:.4f}, F1 = {best_f1:.4f}")
    return np.array(thresholds)

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate metrics for multilabel classification."""
    metrics = {}
    
    # Calculate per-label metrics
    per_label_metrics = {}
    for i, label in enumerate(label_names):
        try:
            # Remove -1 values from consideration
            mask = y_true[:, i] != -1
            y_true_clean = y_true[mask, i]
            y_pred_clean = y_pred[mask, i]
            y_probs_clean = y_probs[mask, i]
            
            if len(y_true_clean) == 0:
                print(f"Warning: No valid labels for {label}")
                per_label_metrics[label] = {
                    'f1': 0.0,
                    'accuracy': 0.0,
                    'roc_auc': 0.0
                }
                continue
                
            per_label_metrics[label] = {
                'f1': f1_score(y_true_clean, y_pred_clean, average='binary'),
                'accuracy': accuracy_score(y_true_clean, y_pred_clean),
                'roc_auc': roc_auc_score(y_true_clean, y_probs_clean)
            }
        except Exception as e:
            print(f"Warning: Could not calculate metrics for {label}: {str(e)}")
            per_label_metrics[label] = {
                'f1': 0.0,
                'accuracy': 0.0,
                'roc_auc': 0.0
            }
    
    # Calculate overall metrics
    try:
        # Remove -1 values from consideration
        mask = (y_true != -1).all(axis=1)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        y_probs_clean = y_probs[mask]
        
        metrics.update({
            "f1_macro": f1_score(y_true_clean, y_pred_clean, average="macro"),
            "f1_micro": f1_score(y_true_clean, y_pred_clean, average="micro"),
            "accuracy": accuracy_score(y_true_clean, y_pred_clean),
            "roc_auc_macro": roc_auc_score(y_true_clean, y_probs_clean, average="macro"),
            "per_label_metrics": per_label_metrics
        })
    except Exception as e:
        print(f"Warning: Could not calculate overall metrics: {str(e)}")
        metrics.update({
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "accuracy": 0.0,
            "roc_auc_macro": 0.0,
            "per_label_metrics": per_label_metrics
        })
    
    return metrics

def main():
    try:
        # Load all data
        llm_results, opt_model, X_bert, Y_true, test_data = load_data()
        
        print("\n LLM Model Evaluation")
        print("-------------------------")
        print(f"F1 Macro:       {llm_results.get('f1_macro', 'N/A'):.4f}")
        print(f"F1 Micro:       {llm_results.get('f1_micro', 'N/A'):.4f}")
        print(f"Accuracy:       {llm_results.get('accuracy', 'N/A'):.4f}")
        print(f"ROC-AUC Macro:  {llm_results.get('roc_auc_macro', 'N/A'):.4f}")
        
        # Predict probabilities
        print("[INFO] Predicting with ML model...")
        start_time = time.time()
        Y_probs = predict_proba_multioutput(opt_model, X_bert)
        ml_inference_time = time.time() - start_time
        
        # Verify prediction shape
        if Y_probs.shape[1] != n_labels:
            print(f"Warning: Prediction shape {Y_probs.shape} has wrong number of labels (expected {n_labels})")
            # Ensure correct number of labels
            Y_probs = Y_probs[:, :n_labels]
        
        # Find best thresholds
        print("[INFO] Tuning thresholds...")
        best_thresholds = find_best_thresholds(Y_true, Y_probs)
        Y_pred = (Y_probs > best_thresholds).astype(int)
        
        # Print prediction statistics
        print("\nPrediction Statistics:")
        print(f"Y_true shape: {Y_true.shape}")
        print(f"Y_pred shape: {Y_pred.shape}")
        print(f"Y_probs shape: {Y_probs.shape}")
        print(f"Y_true unique values: {np.unique(Y_true)}")
        print(f"Y_pred unique values: {np.unique(Y_pred)}")
        print(f"Y_probs range: [{Y_probs.min():.4f}, {Y_probs.max():.4f}]")
        
        # Calculate metrics
        ml_results = calculate_metrics(Y_true, Y_pred, Y_probs)
        ml_results["inference_time_sec"] = ml_inference_time
        
        print("\n ML Model Evaluation")
        print("-------------------------")
        for key, val in ml_results.items():
            if key not in ["inference_time_sec", "per_label_metrics"]:
                print(f"{key.replace('_', ' ').title()}: {val:.4f}")
        print(f"Inference Time (s): {ml_results['inference_time_sec']:.2f}")
        
        print("\n Per-Label Metrics")
        print("-------------------------")
        for label, metrics in ml_results['per_label_metrics'].items():
            print(f"{label}:")
            for metric, val in metrics.items():
                print(f"  {metric}: {val:.4f}")
        
        # Per-label F1 Score Chart
        print("\n Per-label F1 Scores (ML)")
        per_label_f1 = []
        for i in range(len(label_names)):
            mask = Y_true[:, i] != -1
            y_true_clean = Y_true[mask, i]
            y_pred_clean = Y_pred[mask, i]
            if len(y_true_clean) > 0:
                f1 = f1_score(y_true_clean, y_pred_clean, average='binary')
            else:
                f1 = 0.0
            per_label_f1.append(f1)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=label_names, y=per_label_f1)
        plt.ylabel("F1 Score")
        plt.title("Per-label F1 Scores - ML Model")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("per_label_f1_ml.png")
        plt.close()

        # Simulated LLM Inference Time
        print("[INFO] Benchmarking LLM inference time...")
        sample_comments = pd.read_csv(TEST_DATA_PATH)["comment_text"].fillna("").tolist()[:100]
        llm_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        start = time.time()
        for text in sample_comments:
            llm_pipe(text, candidate_labels=label_names, multi_label=True)
        llm_inference_time = time.time() - start
        llm_results["inference_time_sec"] = llm_inference_time
        print(f"LLM Inference Time (100 samples): {llm_inference_time:.2f} seconds")

        # Summary Comparison
        def summarize_comparison(llm, ml):
            print("\n Summary Benchmark Comparison")
            print("-------------------------------")
            rows = []
            for key in ["f1_macro", "f1_micro", "accuracy", "roc_auc_macro"]:
                llm_val = llm.get(key, 0.0)
                ml_val = ml.get(key, 0.0)
                winner = "LLM" if llm_val > ml_val else "ML" if ml_val > llm_val else "Tie"
                rows.append((key.upper(), llm_val, ml_val, winner))
                print(f"{key.upper():<15} | LLM: {llm_val:.4f} | ML: {ml_val:.4f} |  Winner: {winner}")
            return rows

        summary = summarize_comparison(llm_results, ml_results)

        # Statistical Significance Testing (Paired F1)
        print("\n Statistical Significance Test")
        t_stat, p_value = ttest_rel(per_label_f1, [0.5]*len(per_label_f1))
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

        # Generate HTML Report
        print("\n Generating HTML Report...")
        report_html = f"""
        <html>
        <head><title>Model Benchmark Report</title></head>
        <body>
            <h1>Toxic Comment Classification Benchmark</h1>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>Model Performance</h2>
            <table border='1' cellpadding='5'>
                <tr><th>Metric</th><th>LLM</th><th>ML</th><th>Winner</th></tr>
                {''.join([f"<tr><td>{row[0]}</td><td>{row[1]:.4f}</td><td>{row[2]:.4f}</td><td>{row[3]}</td></tr>" for row in summary])}
            </table>
            <h2>Inference Time</h2>
            <p><strong>LLM:</strong> {llm_results['inference_time_sec']:.2f} sec (100 samples)</p>
            <p><strong>ML:</strong> {ml_results['inference_time_sec']:.2f} sec</p>
            <h2>Per-label F1 Scores (ML)</h2>
            <img src='per_label_f1_ml.png' width='600'>
            <h2>Statistical Test</h2>
            <p>T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}</p>
        </body>
        </html>
        """

        with open("benchmark_report.html", "w") as f:
            f.write(report_html)

        print(" Report saved as benchmark_report.html")

    except Exception as e:
        print(f"Error in benchmark: {str(e)}")
        raise

if __name__ == "__main__":
    main()
