"""Test the main module functionality."""

import pytest
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from models.ML.main import main
from models.ML.preprocess import preprocess_data
from models.ML.embed_bert import get_bert_embeddings
from models.ML.train_lightgbm import train_model
from models.ML.evaluate import evaluate_model

def setup_test_directories():
    """Create necessary directories for testing."""
    base_path = Path(".")
    dirs = [
        base_path / "data" / "raw",
        base_path / "models" / "embeddings",
        base_path / "models" / "saved",
        base_path / "outputs" / "figures",
        base_path / "outputs" / "results"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs

def cleanup_test_files(test_file: Path, dirs: list):
    """Clean up test files and empty directories."""
    files_to_remove = [
        Path("models/saved/opt_model.joblib"),
        Path("models/embeddings/bert_embeddings.npy"),
        Path("outputs/figures/per_label_f1_ml.png"),
        Path("outputs/results/eval_results.json"),
        test_file
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()

def test_main_function():
    """Test the main function with a small dataset."""
    # Create test directories
    dirs = setup_test_directories()
    
    # Create a small test dataset
    test_data = pd.DataFrame({
        'comment_text': [
            'This is a toxic comment about woman and muslim people',
            'This is a clean comment',
            'Another toxic comment with hate speech about gay people',
            'A normal comment about technology'
        ],
        'toxic': [1, 0, 1, 0],
        'severe_toxic': [0, 0, 1, 0],
        'obscene': [0, 0, 1, 0],
        'threat': [0, 0, 0, 0],
        'insult': [1, 0, 1, 0],
        'identity_hate': [1, 0, 0, 0]
    })
    
    # Save test data
    test_file = Path("data/raw/test_data.csv")
    test_data.to_csv(test_file, index=False)
    
    try:
        # Run main function
        metrics = main(str(test_file))
        
        # Check if output files were created
        assert Path("models/embeddings/bert_embeddings.npy").exists()
        assert Path("models/saved/opt_model.joblib").exists()
        assert Path("outputs/figures/per_label_f1_ml.png").exists()
        assert Path("outputs/results/eval_results.json").exists()
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['roc_auc', 'f1', 'precision', 'recall'])
        assert all(isinstance(metrics[k], dict) for k in metrics)
        
        # Load and check embeddings
        embeddings = np.load("models/embeddings/bert_embeddings.npy")
        assert embeddings.shape[0] == len(test_data)
        assert embeddings.shape[1] > 0  # Should have some features
        
    finally:
        # Clean up
        cleanup_test_files(test_file, dirs)

def test_main_function_invalid_path():
    """Test main function with invalid data path."""
    with pytest.raises(FileNotFoundError):
        main("nonexistent_file.csv") 