# evaluate_obfuscation_resilience.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
from joblib import load
from sklearn.metrics import f1_score, classification_report
from transformers import pipeline
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, BertTokenizer, BertModel
from datetime import datetime
from deep_translator import GoogleTranslator
from pathlib import Path
import os
import json
from transformers import RobertaTokenizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Add at the top of the file after imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# -------------------
# 1. Load Obfuscated Test Set
# -------------------
def load_test_data():
    """Load or create test data for obfuscation resilience evaluation."""
    test_file = PROJECT_ROOT / 'security' / 'obfuscation_test_set.csv'
    
    if not test_file.exists():
        # Create more realistic test data with various obfuscation techniques
        test_data = {
            'text': [
                'This is a normal comment',
                'This is a toxic comment',
                'This is an obfuscated comment',
                'Th1s 1s 4n 0bfusc4t3d c0mm3nt',  # Leetspeak
                'This is a toxic comment with spaces',  # Space injection
                'This is a toxic comment with repeated letters',  # Character repetition
                'This is a toxic comment with special characters',  # Special chars
                'This is a toxic comment with mixed case',  # Case variation
                'This is a toxic comment with unicode',  # Unicode chars
                'This is a toxic comment with emojis'  # Emoji injection
            ],
            'label': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'obfuscation_type': [
                'clean',
                'clean',
                'basic',
                'leetspeak',
                'space_injection',
                'char_repetition',
                'special_chars',
                'case_variation',
                'unicode',
                'emoji'
            ]
        }
        df = pd.DataFrame(test_data)
        test_file.parent.mkdir(exist_ok=True)
        df.to_csv(test_file, index=False)
        print(f"Created comprehensive test data at {test_file}")
    else:
        try:
            df = pd.read_csv(test_file)
            print(f"Loaded existing test data from {test_file}")
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            return None
    
    return df

# -------------------
# 2. Defense: Normalize Obfuscated Text
# -------------------
def normalize_obfuscation(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # remove char repetition
    text = text.replace(" ", "")  # remove injected spaces
    return text.lower()

# -------------------
# 3. Synonym + Backtranslation Attacks
# -------------------
def backtranslate(text):
    try:
        return GoogleTranslator(source='en', target='de').translate(text)
    except:
        return text

# -------------------
# 4. Load ML Model and Run Predictions
# -------------------
def load_models():
    """Load ML and LLM models"""
    # Load ML model
    ml_model_path = PROJECT_ROOT / 'models' / 'saved' / 'opt_model.joblib'
    ml_model = joblib.load(ml_model_path)
    print("[ML] Loaded optimized model")
    
    # Load LLM model
    llm_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=6)
    llm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("[LLM] Loaded RoBERTa model")
    
    return ml_model, llm_model, llm_tokenizer

def calculate_resilience_metrics(pred_clean, pred_obf, label_names, model_name):
    """Calculate detailed resilience metrics for a model."""
    metrics = {}
    
    # Convert numpy arrays to Python lists for JSON serialization
    pred_clean = pred_clean.tolist()
    pred_obf = pred_obf.tolist()
    
    # Calculate flip counts per label
    flip_counts = np.sum(np.array(pred_clean) != np.array(pred_obf), axis=0)
    metrics['flip_counts'] = dict(zip(label_names, flip_counts.tolist()))
    
    # Calculate overall flip rate
    total_samples = len(pred_clean)
    total_flips = np.sum(flip_counts)
    metrics['flip_rate'] = float(total_flips / (total_samples * len(label_names)))
    
    # Calculate per-label flip rates
    metrics['label_flip_rates'] = {
        label: float(count / total_samples) for label, count in zip(label_names, flip_counts)
    }
    
    # Calculate agreement rate (1 - flip rate)
    metrics['agreement_rate'] = 1 - metrics['flip_rate']
    
    # Calculate per-label agreement rates
    metrics['label_agreement_rates'] = {
        label: 1 - rate for label, rate in metrics['label_flip_rates'].items()
    }
    
    # Calculate confusion matrix metrics
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.array(pred_clean).flatten(), np.array(pred_obf).flatten())
    metrics['confusion_matrix'] = cm.tolist()
    
    # Calculate per-label F1 scores
    from sklearn.metrics import f1_score
    metrics['label_f1_scores'] = {
        label: float(f1_score(np.array(pred_clean)[:, i], np.array(pred_obf)[:, i], 
                            average='binary', zero_division=0))
        for i, label in enumerate(label_names)
    }
    
    # Calculate overall F1 score
    metrics['overall_f1'] = float(f1_score(np.array(pred_clean).flatten(), 
                                         np.array(pred_obf).flatten(), 
                                         average='weighted', zero_division=0))
    
    return metrics

def main():
    # Load models
    ml_model, llm_model, llm_tokenizer = load_models()
    
    # Generate test data
    test_data = generate_test_data()
    
    # Evaluate resilience
    metrics = evaluate_resilience(ml_model, llm_model, llm_tokenizer, test_data)
    
    # Save metrics
    output_path = PROJECT_ROOT / 'security' / 'evaluation' / 'resilience_metrics.json'
    save_metrics(metrics, output_path)
    
    # Print results
    print_metrics(metrics)

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = ' '.join(text.split())
    return text

def get_bert_embeddings(texts, model_name='bert-base-uncased'):
    """Get BERT embeddings for a list of texts"""
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Tokenize and get BERT embeddings
            inputs = tokenizer(text, return_tensors="pt", 
                             truncation=True, max_length=512,
                             padding=True)
            outputs = model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
    
    return np.array(embeddings)

def evaluate_resilience(ml_model, llm_model, llm_tokenizer, test_data):
    """Evaluate model resilience to obfuscation"""
    # Get BERT embeddings for ML model
    texts = test_data['text'].tolist()
    bert_embeddings = get_bert_embeddings(texts)
    
    # Get ML model predictions and convert to numpy array
    ml_predictions = []
    for embedding in bert_embeddings:
        pred = ml_model.predict_proba(embedding.reshape(1, -1))
        ml_predictions.append(pred[0])
    ml_predictions = np.array(ml_predictions)
    
    # Reshape ML predictions to match LLM predictions shape
    if len(ml_predictions.shape) == 3:
        ml_predictions = ml_predictions.reshape(ml_predictions.shape[0], -1)
    
    # Get LLM model predictions
    llm_predictions = []
    for text in texts:
        inputs = llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = llm_model(**inputs)
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(outputs.logits).detach().numpy()
        llm_predictions.append(pred[0])  # Shape: (6,) for 6 labels
    llm_predictions = np.array(llm_predictions)  # Shape: (n_samples, 6)
    
    # Print shapes for debugging
    print(f"ML predictions shape after reshape: {ml_predictions.shape}")
    print(f"LLM predictions shape: {llm_predictions.shape}")
    
    # Ensure both predictions have the same number of classes
    n_classes = min(ml_predictions.shape[1], llm_predictions.shape[1])
    ml_predictions = ml_predictions[:, :n_classes]
    llm_predictions = llm_predictions[:, :n_classes]
    
    # Calculate metrics
    metrics = {
        'f1_macro': float(f1_score(np.round(ml_predictions), np.round(llm_predictions), average='macro')),
        'f1_micro': float(f1_score(np.round(ml_predictions), np.round(llm_predictions), average='micro')),
        'accuracy': float(np.mean(np.round(ml_predictions) == np.round(llm_predictions))),
        'roc_auc_macro': float(roc_auc_score(np.round(ml_predictions), llm_predictions, average='macro')),
        'per_label_metrics': {}
    }
    
    # Calculate per-label metrics
    for i, label in enumerate(LABEL_NAMES[:n_classes]):
        metrics['per_label_metrics'][label] = {
            'f1': float(f1_score(np.round(ml_predictions[:, i]), 
                                np.round(llm_predictions[:, i]), 
                                average='binary'))
        }
    
    return metrics

def generate_test_data():
    """Generate test data with clean and obfuscated text"""
    test_data = pd.DataFrame({
        'text': [
            # Clean examples
            'This is a normal comment.',
            'You are stupid and worthless.',
            'I hate you and your family.',
            'You should die in a fire.',
            'Your kind should be eliminated.',
            # Obfuscated examples
            'Th1s 1s 4 n0rm4l c0mm3nt.',  # Leetspeak
            'Y o u  a r e  s t u p i d.',  # Space injection
            'I h8888 you and ur fam.',  # Character repetition
            'Y0u $h0uld d13 1n 4 f1r3.',  # Mixed obfuscation
            'Ur k1nd sh0uld b3 3l1m1n4t3d.'  # Mixed obfuscation
        ],
        'is_obfuscated': [False, False, False, False, False,
                         True, True, True, True, True]
    })
    
    # Save test data
    test_data_path = PROJECT_ROOT / 'security' / 'obfuscation_test_set.csv'
    test_data.to_csv(test_data_path, index=False)
    print(f"Created test data file at {test_data_path}")
    
    return test_data

def save_metrics(metrics: dict, output_path: Path = None) -> None:
    """Save evaluation metrics to JSON file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "eval_results.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_path}")

def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\nResilience Metrics:")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"F1 Micro: {metrics['f1_micro']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC Macro: {metrics['roc_auc_macro']:.4f}")
    print("\nPer-Label Metrics:")
    for label, scores in metrics['per_label_metrics'].items():
        print(f"{label}: F1 = {scores['f1']:.4f}")

if __name__ == "__main__":
    main()
