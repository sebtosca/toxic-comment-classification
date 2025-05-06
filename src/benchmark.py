# Benchmarking: ML Model vs LLM Model

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import ttest_rel


PROJECT_ROOT = Path(__file__).parent.parent


ML_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "eval_results.json"
LLM_RESULTS_PATH = PROJECT_ROOT / "src" / "eval_results.json"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

def load_results():
    """Load existing evaluation results."""
    with open(ML_RESULTS_PATH) as f:
        ml_results = json.load(f)
    
    with open(LLM_RESULTS_PATH) as f:
        llm_results = json.load(f)
    
    return ml_results, llm_results

def summarize_comparison(ml_results, llm_results):
    """Compare ML and LLM results."""
    print("\n Summary Benchmark Comparison")
    print("-------------------------------")
    
    # Calculate ROC-AUC Micro for ML model from per-label metrics
    ml_roc_auc_micro = np.mean([metrics["roc_auc"] for metrics in ml_results["per_label_metrics"].values()])
    
    # Compare overall metrics
    metrics = {
        "f1_macro": ("F1 Macro", ml_results["f1_macro"], llm_results["eval_f1_macro"]),
        "f1_micro": ("F1 Micro", ml_results["f1_micro"], llm_results["eval_f1_micro"]),
        "roc_auc_macro": ("ROC-AUC Macro", ml_results["roc_auc_macro"], llm_results["eval_roc_auc_macro"]),
        "roc_auc_micro": ("ROC-AUC Micro", ml_roc_auc_micro, llm_results["eval_roc_auc_micro"])
    }
    
    rows = []
    for key, (name, ml_val, llm_val) in metrics.items():
        winner = "LLM" if llm_val > ml_val else "ML" if ml_val > llm_val else "Tie"
        rows.append((name, llm_val, ml_val, winner))
        print(f"{name:<15} | LLM: {llm_val:.4f} | ML: {ml_val:.4f} | Winner: {winner}")
    
    return rows

def plot_per_label_comparison(ml_results, llm_results):
    """Create visualization comparing per-label metrics."""
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    # Extract F1 scores
    ml_f1 = [ml_results["per_label_metrics"][label]["f1"] for label in labels]
    llm_f1 = [llm_results[f"eval_{label}_f1"] for label in labels]
    
    # Create bar plot
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, ml_f1, width, label='ML Model')
    plt.bar(x + width/2, llm_f1, width, label='LLM Model')
    
    plt.xlabel('Labels')
    plt.ylabel('F1 Score')
    plt.title('Per-label F1 Score Comparison')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_label_comparison.png")
    plt.close()

def plot_roc_auc_comparison(ml_results, llm_results):
    """Create visualization comparing ROC-AUC scores."""
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    # Extract ROC-AUC scores
    ml_roc_auc = [ml_results["per_label_metrics"][label]["roc_auc"] for label in labels]
    llm_roc_auc = [llm_results[f"eval_{label}_roc_auc"] for label in labels]
    
    # Create bar plot
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, ml_roc_auc, width, label='ML Model', color='#1f77b4')
    bars2 = plt.bar(x + width/2, llm_roc_auc, width, label='LLM Model', color='#ff7f0e')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Labels')
    plt.ylabel('ROC-AUC Score')
    plt.title('Per-label ROC-AUC Score Comparison')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.ylim(0.9, 1.02)  
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_label_roc_auc_comparison.png")
    plt.close()

def generate_html_report(rows, ml_results, llm_results):
    """Generate HTML report comparing results."""
    # Calculate ROC-AUC Micro for ML model
    ml_roc_auc_micro = np.mean([metrics["roc_auc"] for metrics in ml_results["per_label_metrics"].values()])
    
    report_html = f"""
    <html>
    <head><title>Model Benchmark Report</title></head>
    <body>
        <h1>Toxic Comment Classification Benchmark</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Model Performance</h2>
        <table border='1' cellpadding='5'>
            <tr><th>Metric</th><th>LLM</th><th>ML</th><th>Winner</th></tr>
            {''.join([f"<tr><td>{row[0]}</td><td>{row[1]:.4f}</td><td>{row[2]:.4f}</td><td>{row[3]}</td></tr>" for row in rows])}
        </table>
        
        <h2>Per-label F1 Score Comparison</h2>
        <img src='per_label_comparison.png' width='800'>
        
        <h2>Per-label ROC-AUC Score Comparison</h2>
        <img src='per_label_roc_auc_comparison.png' width='800'>
        
        <h2>ML Model Per-label Metrics</h2>
        <table border='1' cellpadding='5'>
            <tr><th>Label</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th></tr>
            {''.join([f"<tr><td>{label}</td><td>{metrics['f1']:.4f}</td><td>{metrics['precision']:.4f}</td><td>{metrics['recall']:.4f}</td><td>{metrics['roc_auc']:.4f}</td></tr>" 
                     for label, metrics in ml_results['per_label_metrics'].items()])}
        </table>
        
        <h2>LLM Model Per-label Metrics</h2>
        <table border='1' cellpadding='5'>
            <tr><th>Label</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th></tr>
            {''.join([f"<tr><td>{label}</td><td>{llm_results[f'eval_{label}_f1']:.4f}</td><td>{llm_results[f'eval_{label}_precision']:.4f}</td><td>{llm_results[f'eval_{label}_recall']:.4f}</td><td>{llm_results[f'eval_{label}_roc_auc']:.4f}</td></tr>" 
                     for label in ml_results['per_label_metrics'].keys()])}
        </table>
        
        <h2>Note on ROC-AUC Micro</h2>
        <p>The ML model's ROC-AUC Micro score is calculated as the average of per-label ROC-AUC scores, as the micro-average was not directly available in the results.</p>
    </body>
    </html>
    """
    
    with open(RESULTS_DIR / "benchmark_report.html", "w") as f:
        f.write(report_html)

def main():
    try:
        # Load results
        ml_results, llm_results = load_results()
        
        # Compare results
        rows = summarize_comparison(ml_results, llm_results)
        
        # Create visuals
        plot_per_label_comparison(ml_results, llm_results)
        plot_roc_auc_comparison(ml_results, llm_results)
        
        # Generate report
        generate_html_report(rows, ml_results, llm_results)
        
        print("\nReport generated successfully!")
        print("1. benchmark_report.html - Contains detailed comparison")
        print("2. per_label_comparison.png - Visual comparison of F1 scores")
        print("3. per_label_roc_auc_comparison.png - Visual comparison of ROC-AUC scores")
        
    except Exception as e:
        print(f"Error in benchmark: {str(e)}")
        raise

if __name__ == "__main__":
    main()
