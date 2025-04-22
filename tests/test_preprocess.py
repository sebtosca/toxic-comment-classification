"""Tests for the preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from models.ML.preprocess import clean_text, has_identity_term, preprocess_data

def test_clean_text():
    """Test text cleaning function."""
    # Test cases
    test_cases = [
        ("Hello World!", "hello world"),
        ("UPPERCASE", "uppercase"),
        ("Multiple   Spaces", "multiple spaces"),
        ("Special!@#$%^&*()", "special"),
        ("123 Numbers 456", "numbers"),
        ("", ""),
        ("   ", ""),
        ("\nNew\nLines\n", "new lines"),
        ("Tab\tTab", "tab tab"),
    ]
    
    for input_text, expected_output in test_cases:
        assert clean_text(input_text) == expected_output

def test_has_identity_term():
    """Test identity term detection."""
    # Test cases
    test_cases = [
        ("I am a woman", 1),
        ("He is a man", 1),
        ("No identity terms here", 0),
        ("", 0),
        ("woman man", 1),
        ("WOMAN", 1),  # Case insensitive
        ("womann", 0),  # Not exact match
    ]
    
    for text, expected in test_cases:
        assert has_identity_term(text) == expected

def test_preprocess_data():
    """Test data preprocessing function."""
    # Create test data
    test_data = pd.DataFrame({
        'comment_text': [
            'This is a toxic comment',
            'This is a clean comment',
            'This comment has identity terms'
        ],
        'toxic': [1, 0, 0],
        'severe_toxic': [0, 0, 0],
        'obscene': [0, 0, 0],
        'threat': [0, 0, 0],
        'insult': [0, 0, 0],
        'identity_hate': [0, 0, 0]
    })
    
    # Save test data
    test_data.to_csv('test_data.csv', index=False)
    
    try:
        # Test preprocessing
        df, texts, labels = preprocess_data('test_data.csv')
        
        # Check output types
        assert isinstance(df, pd.DataFrame)
        assert isinstance(texts, list)
        assert isinstance(labels, np.ndarray)
        
        # Check output shapes
        assert len(df) == len(test_data)
        assert len(texts) == len(test_data)
        assert labels.shape == (len(test_data), 6)  # 6 labels
        
        # Check text cleaning
        assert all(isinstance(text, str) for text in texts)
        assert all(text == text.lower() for text in texts)
        
        # Check label values
        assert np.all((labels == 0) | (labels == 1))
        
    finally:
        # Clean up
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

def test_preprocess_data_edge_cases():
    """Test data preprocessing with edge cases."""
    # Test with empty file
    with pytest.raises(FileNotFoundError):
        preprocess_data('nonexistent.csv')
    
    # Test with missing columns
    test_data = pd.DataFrame({
        'comment_text': ['test'],
        'toxic': [1]
    })
    test_data.to_csv('test_data.csv', index=False)
    
    try:
        with pytest.raises(ValueError):
            preprocess_data('test_data.csv')
    finally:
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv') 