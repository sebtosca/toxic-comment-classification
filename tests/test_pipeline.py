"""Test the full preprocessing and embedding pipeline."""

import pytest
import pandas as pd
import numpy as np
from models.ML.preprocess import preprocess_data, clean_text, has_identity_term
from models.ML.embed_bert import get_bert_embeddings

def test_full_pipeline():
    """Test the complete preprocessing and embedding pipeline."""
    # Create test data
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
    test_data.to_csv('test_pipeline.csv', index=False)
    
    try:
        # Test preprocessing
        df, texts, labels = preprocess_data('test_pipeline.csv')
        
        # Verify preprocessing results
        assert len(df) == len(test_data)
        assert len(texts) == len(test_data)
        assert labels.shape == (len(test_data), 6)
        assert 'has_identity' in df.columns
        assert df['has_identity'].sum() > 0  # At least one comment has identity terms
        
        # Test BERT embeddings
        embeddings = get_bert_embeddings(texts)
        
        # Verify embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 768  # BERT base hidden size
        assert not np.isnan(embeddings).any()
        assert np.all(np.abs(embeddings) < 10)  # Reasonable embedding values
        
    finally:
        # Clean up
        import os
        if os.path.exists('test_pipeline.csv'):
            os.remove('test_pipeline.csv')

def test_pipeline_edge_cases():
    """Test pipeline with edge cases."""
    # Test with single comment
    single_data = pd.DataFrame({
        'comment_text': ['Test comment'],
        'toxic': [0],
        'severe_toxic': [0],
        'obscene': [0],
        'threat': [0],
        'insult': [0],
        'identity_hate': [0]
    })
    
    single_data.to_csv('test_single.csv', index=False)
    
    try:
        df, texts, labels = preprocess_data('test_single.csv')
        embeddings = get_bert_embeddings(texts)
        
        assert len(df) == 1
        assert len(texts) == 1
        assert labels.shape == (1, 6)
        assert embeddings.shape == (1, 768)
        
    finally:
        import os
        if os.path.exists('test_single.csv'):
            os.remove('test_single.csv')

def test_pipeline_error_handling():
    """Test error handling in the pipeline."""
    # Test with empty DataFrame
    empty_data = pd.DataFrame(columns=[
        'comment_text', 'toxic', 'severe_toxic', 'obscene',
        'threat', 'insult', 'identity_hate'
    ])
    empty_data.to_csv('test_empty.csv', index=False)
    
    try:
        # Should process without errors
        df, texts, labels = preprocess_data('test_empty.csv')
        assert len(df) == 0
        assert len(texts) == 0
        assert labels.shape == (0, 6)
        
        # BERT embeddings should raise error for empty input
        with pytest.raises(ValueError):
            get_bert_embeddings([])
            
    finally:
        import os
        if os.path.exists('test_empty.csv'):
            os.remove('test_empty.csv') 