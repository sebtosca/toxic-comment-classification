"""Test the evaluation module."""

import pytest
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from models.ML.evaluate import find_best_thresholds, evaluate_model

def test_find_best_thresholds():
    """Test the find_best_thresholds function."""
    # Test with simple binary case
    y_true = np.array([[1, 0], [0, 1]])
    y_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    thresholds = find_best_thresholds(y_true, y_probs)
    assert isinstance(thresholds, np.ndarray)
    assert thresholds.shape == (2,)
    assert np.all(thresholds >= 0) and np.all(thresholds <= 1)
    
    # Test with shape mismatch
    with pytest.raises(ValueError):
        find_best_thresholds(y_true, y_probs[:, :1])

def test_evaluate_model():
    """Test the evaluate_model function."""
    # Create test data
    np.random.seed(42)
    X = np.random.rand(10, 768)
    y = np.random.randint(0, 2, (10, 6))
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test evaluation
    metrics = evaluate_model(model, X, y)
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['roc_auc', 'f1', 'precision', 'recall'])
    assert all(isinstance(metrics[k], dict) for k in metrics)
    
    # Check output files
    assert Path("outputs/figures/per_label_f1_ml.png").exists()
    assert Path("outputs/results/eval_results.json").exists()

def test_evaluate_model_edge_cases():
    """Test edge cases in evaluate_model."""
    # Test with invalid model
    X = np.random.rand(10, 768)
    y = np.random.randint(0, 2, (10, 6))
    
    class InvalidModel:
        pass
    
    with pytest.raises(ValueError):
        evaluate_model(InvalidModel(), X, y)
    
    # Test with shape mismatch
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    with pytest.raises(ValueError):
        evaluate_model(model, X, y[:5])

def test_evaluate_model_custom_labels():
    """Test evaluation with custom label names."""
    X = np.random.rand(10, 768)
    y = np.random.randint(0, 2, (10, 6))
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    metrics = evaluate_model(model, X, y, label_names=label_names)
    assert isinstance(metrics, dict)
    assert Path("outputs/figures/per_label_f1_ml.png").exists() 