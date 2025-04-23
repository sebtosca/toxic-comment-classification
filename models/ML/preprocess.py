"""Preprocess data for toxic comment classification."""

import pandas as pd
import numpy as np
import re
import nltk
import unicodedata
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
    """Check if text contains identity terms."""
    text = str(text).lower()
    return int(any(term in text for term in identity_terms))

def clean_text(text: str) -> str:
    """Thoroughly clean text using regex and normalization techniques."""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def calculate_cleaning_effectiveness(original_texts: List[str], cleaned_texts: List[str]) -> float:
    """
    Calculate the percentage of content reduction after cleaning.
    
    Args:
        original_texts: List of original text strings.
        cleaned_texts: List of cleaned text strings.
    
    Returns:
        Float percentage of token reduction across the dataset.
    """
    original_tokens = sum(len(str(text).split()) for text in original_texts)
    cleaned_tokens = sum(len(str(text).split()) for text in cleaned_texts)
    
    if original_tokens == 0:
        return 0.0

    reduction = original_tokens - cleaned_tokens
    return round((reduction / original_tokens) * 100, 2)

def preprocess_data(data_path: str) -> Tuple[pd.DataFrame, List[str], np.ndarray, float]:
    """Preprocess the input data."""
    df = pd.read_csv(data_path, on_bad_lines='skip')
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    df['has_identity'] = df['comment_text'].apply(has_identity_term)
    
    print(" Cleaning text...")
    df['clean_comment'] = df['comment_text'].apply(clean_text)
    
    print(" Calculating cleaning effectiveness...")
    effectiveness = calculate_cleaning_effectiveness(df['comment_text'], df['clean_comment'])
    print(f"\n Text reduction due to cleaning: {effectiveness}%\n")

    # Optional: Show a few cleaned samples
    print("🔍 Sample cleaned comments:")
    print(df[['comment_text', 'clean_comment']].sample(3, random_state=42).to_string(index=False))

    df = df[df['clean_comment'].str.split().str.len() > 4]
    texts = df['clean_comment'].fillna('').tolist()
    labels = df[label_names].values
    
    return df, texts, labels, effectiveness

# Entry point for terminal execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("  Please provide the path to the CSV file as a command-line argument.")
    else:
        data_path = sys.argv[1]
        df, texts, labels, effectiveness = preprocess_data(data_path)
