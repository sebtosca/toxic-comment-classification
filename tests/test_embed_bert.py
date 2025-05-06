"""Tests for the BERT embedding module."""

import pytest
import torch
import numpy as np
from models.ML.embed_bert import get_bert_embeddings
from models.ML.config import MODEL_CONFIG

def test_get_bert_embeddings():
    """Test BERT embedding generation."""
    # Test data
    test_texts = [
        "This is a test comment",
        "Another test comment with more words",
        "Short"
    ]
    
    # Get embeddings
    embeddings = get_bert_embeddings(test_texts)
    
    # Check output type and shape
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.shape[1] == 768  
    
    # Check for NaN values
    assert not np.isnan(embeddings).any()
    
    # Check embedding values are reasonable
    assert np.all(np.abs(embeddings) < 10)
    
    # Check different length inputs
    embeddings_short = get_bert_embeddings(["Short"])
    assert embeddings_short.shape == (1, 768)
    
    embeddings_long = get_bert_embeddings(["Long " * 100 + "comment"])
    assert embeddings_long.shape == (1, 768)

def test_get_bert_embeddings_edge_cases():
    """Test BERT embedding generation with edge cases."""
    # Empty input
    with pytest.raises(ValueError):
        get_bert_embeddings([])
    
    # Empty string
    embeddings = get_bert_embeddings([""])
    assert embeddings.shape == (1, 768)
    
    # Very long input
    long_text = "Long " * 1000
    embeddings = get_bert_embeddings([long_text])
    assert embeddings.shape == (1, 768)
    
    # Special characters
    special_text = "!@#$%^&*()"
    embeddings = get_bert_embeddings([special_text])
    assert embeddings.shape == (1, 768)
    
    # Unicode characters
    unicode_text = "你好世界"
    embeddings = get_bert_embeddings([unicode_text])
    assert embeddings.shape == (1, 768) 