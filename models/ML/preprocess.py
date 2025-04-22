"""Preprocess data for toxic comment classification."""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple, List

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define identity terms
identity_terms = [
    "muslim", "christian", "jewish", "black", "white", "asian",
    "latino", "gay", "lesbian", "trans", "female", "male",
    "woman", "man", "nonbinary", "disabled"
]

def has_identity_term(text: str) -> int:
    """Check if text contains identity terms.
    
    Args:
        text: Input text
        
    Returns:
        1 if identity terms found, 0 otherwise
    """
    text = str(text).lower()
    return int(any(term in text for term in identity_terms))

def clean_text(text: str) -> str:
    """Clean text by removing special characters and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_data(data_path: str) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """Preprocess the input data.
    
    Args:
        data_path: Path to the input CSV file
        
    Returns:
        Tuple of (DataFrame, list of texts, numpy array of labels)
    """
    # Read data
    df = pd.read_csv(data_path, on_bad_lines='skip')
    
    # Define label names
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Add identity term flag
    df['has_identity'] = df['comment_text'].apply(has_identity_term)
    
    # Clean text
    print("Cleaning text...")
    df['clean_comment'] = df['comment_text'].apply(clean_text)
    
    # Remove very short comments
    df = df[df['clean_comment'].str.split().str.len() > 4]
    
    # Extract texts and labels
    texts = df['clean_comment'].fillna('').tolist()
    labels = df[label_names].values
    
    return df, texts, labels