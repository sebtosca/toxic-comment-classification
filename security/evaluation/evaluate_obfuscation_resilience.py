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
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, accuracy_score


PROJECT_ROOT = Path(__file__).parent.parent.parent
LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Thresholds for the classes
CLASS_THRESHOLDS = {
    'toxic': 0.5,        # Balanced threshold for general toxicity
    'severe_toxic': 0.6, # Higher threshold due to severity
    'obscene': 0.5,      # Balanced threshold for obscene content
    'threat': 0.55,      # Slightly higher for threats
    'insult': 0.45,      # Lower threshold to catch subtle insults
    'identity_hate': 0.5 # Balanced threshold for identity-based hate
}

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
def load_models(ml_model=None, llm_model=None, bert_tokenizer=None, bert_model=None, llm_tokenizer=None):
    """Load or use provided ML and LLM models"""
    # Use provided ML model or load from default path
    if ml_model is None:
        ml_model_path = PROJECT_ROOT / 'models' / 'saved' / 'opt_model.joblib'
        if not ml_model_path.exists():
            raise FileNotFoundError(f"ML model not found at {ml_model_path}")
        ml_model = joblib.load(ml_model_path)
        print("Loaded optimized ML model from default path")
    else:
        print("Using provided ML model")
    
    # Using my ML model 
    if bert_tokenizer is None or bert_model is None:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        print("Loaded default BERT tokenizer and ML model")
    else:
        print("Using provided BERT tokenizer and ML model")
    
    # Using my LLM model
    if llm_model is None or llm_tokenizer is None:
        llm_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=len(LABEL_NAMES),
            problem_type="multi_label_classification"
        )
        llm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Loaded default RoBERTa model and tokenizer")
    else:
        print("Using provided LLM model and tokenizer")
    
    return ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer

def calculate_resilience_metrics(y_true, y_pred, label_names, model_name, thresholds):
    """Calculate detailed resilience metrics for a model."""
    metrics = {}
    
    # Convert probabilities to binary predictions using optimal thresholds
    y_pred_binary = np.zeros_like(y_pred)
    for i, label in enumerate(label_names):
        y_pred_binary[:, i] = (y_pred[:, i] > thresholds[label]).astype(int)
    
    # Calculate overall metrics
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred, average='macro')
    except:
        metrics['roc_auc_macro'] = float('nan')
    
    # Calculate per-label metrics
    metrics['per_label_metrics'] = {}
    for i, label in enumerate(label_names):
        label_metrics = {}
        
        # Basic metrics
        label_metrics['f1'] = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        label_metrics['precision'] = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        label_metrics['recall'] = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        
        try:
            label_metrics['roc_auc'] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            label_metrics['roc_auc'] = float('nan')
        
        label_metrics['threshold'] = thresholds[label]
        
        # Confusion matrix elements
        label_metrics['true_positives'] = int(np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1)))
        label_metrics['true_negatives'] = int(np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0)))
        label_metrics['false_positives'] = int(np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1)))
        label_metrics['false_negatives'] = int(np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0)))
        
        metrics['per_label_metrics'][label] = label_metrics
    
    return metrics

def main(ml_model=None, llm_model=None, bert_tokenizer=None, bert_model=None, llm_tokenizer=None):
    """Main function to evaluate model resilience to obfuscation.
    
    Args:
        ml_model: Pre-trained ML model (optional)
        llm_model: Pre-trained LLM model (optional)
        bert_tokenizer: Pre-trained BERT tokenizer (optional)
        bert_model: Pre-trained BERT model (optional)
        llm_tokenizer: Pre-trained LLM tokenizer (optional)
    """
    # Loading the models
    ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer = load_models(
        ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer
    )
    
    # Generating test data
    test_data = generate_test_data()
    
    # Evaluating resilience
    metrics = evaluate_resilience(ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer, test_data)
    
    # Saving metrics
    output_path = PROJECT_ROOT / 'security' / 'evaluation' / 'resilience_metrics.json'
    save_metrics(metrics, output_path)
    
    # Creating visuals
    plot_resilience_metrics(metrics, output_path.parent / 'figures')
    
    # Print results
    print_metrics(metrics)

def preprocess_text(text):
    """Basic text preprocessing"""
    # lowercasing text
    text = text.lower()
    # Remove special characters and space
    text = ' '.join(text.split())
    return text

def get_bert_embeddings(texts, tokenizer, bert_model):
    """Get BERT embeddings for a list of texts"""
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Tokenize and get BERT embeddings
            inputs = tokenizer(text, return_tensors="pt", 
                             truncation=True, max_length=512,
                             padding=True)
            outputs = bert_model(**inputs)
            
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
            
    embeddings = np.array(embeddings)
    
    
    feature_names = [f'bert_feature_{i}' for i in range(embeddings.shape[1])]
    
    # Putting the embeddings in a dataframe
    embeddings_df = pd.DataFrame(embeddings, columns=feature_names)
    
    return embeddings_df, feature_names

def find_best_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    """Find optimal thresholds for each label using precision-recall curves."""
    thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        thresholds.append(thresh[np.argmax(f1_scores)])
    return np.array(thresholds)

def create_multi_label_encoding(df):
    """Create multi-label encoding from the test data."""
    # Initialize labels array
    labels = np.zeros((len(df), len(LABEL_NAMES)))
    
    # Map text to labels based on content
    for i, text in enumerate(df['text']):
        text_lower = text.lower()
        
        # Check for toxic content
        if any(word in text_lower for word in ['stupid', 'worthless', 'idiot']):
            labels[i, LABEL_NAMES.index('toxic')] = 1
            labels[i, LABEL_NAMES.index('insult')] = 1
            
        # Check for severe toxic content
        if any(word in text_lower for word in ['die', 'cancer', 'painfully']):
            labels[i, LABEL_NAMES.index('severe_toxic')] = 1
            labels[i, LABEL_NAMES.index('toxic')] = 1
            
        # Check for obscene content
        if any(word in text_lower for word in ['fuck', 'shit', 'f*ck', 'sh*t']):
            labels[i, LABEL_NAMES.index('obscene')] = 1
            labels[i, LABEL_NAMES.index('toxic')] = 1
            
        # Check for threats
        if 'kill' in text_lower or ('find' in text_lower and 'you' in text_lower):
            labels[i, LABEL_NAMES.index('threat')] = 1
            labels[i, LABEL_NAMES.index('toxic')] = 1
            labels[i, LABEL_NAMES.index('severe_toxic')] = 1
            
        # Check for identity hate
        if '[group]' in text_lower and any(word in text_lower for word in ['eliminate', 'die', 'should']):
            labels[i, LABEL_NAMES.index('identity_hate')] = 1
            labels[i, LABEL_NAMES.index('toxic')] = 1
    
    return labels

def ensemble_predict(models, text, bert_tokenizer, bert_model, llm_tokenizer):
    """Make predictions using an ensemble of models"""
    ml_model, llm_model = models
    
    # Precitions from ML model
    if isinstance(text, str):
        
        embeddings_df, _ = get_bert_embeddings([text], bert_tokenizer, bert_model)
        ml_pred = ml_model.predict_proba(embeddings_df)
        ml_pred = np.array(ml_pred) 
    else:
        
        feature_names = [f'bert_feature_{i}' for i in range(text.shape[1])]
        embeddings_df = pd.DataFrame(text.reshape(1, -1), columns=feature_names)
        ml_pred = ml_model.predict_proba(embeddings_df)
        ml_pred = np.array(ml_pred)  # Convert list to numpy array
    
    # Get predictions from LLM model
    if isinstance(text, str):
        
        inputs = llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = llm_model(**inputs)
            llm_pred = torch.sigmoid(outputs.logits).numpy()
    else:
        
        text = bert_tokenizer.decode(text.argmax(axis=-1))
        inputs = llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = llm_model(**inputs)
            llm_pred = torch.sigmoid(outputs.logits).numpy()
    
    # Ensure both predictions have the same shape
    if len(ml_pred.shape) > 2:
        ml_pred = ml_pred.reshape(ml_pred.shape[0], -1)
    if len(llm_pred.shape) > 2:
        llm_pred = llm_pred.reshape(llm_pred.shape[0], -1)
    
    # Pad predictions if needed
    max_length = max(ml_pred.shape[1], llm_pred.shape[1], len(LABEL_NAMES))
    if ml_pred.shape[1] < max_length:
        padded_pred = np.zeros((ml_pred.shape[0], max_length))
        padded_pred[:, :ml_pred.shape[1]] = ml_pred
        ml_pred = padded_pred
    if llm_pred.shape[1] < max_length:
        padded_pred = np.zeros((llm_pred.shape[0], max_length))
        padded_pred[:, :llm_pred.shape[1]] = llm_pred
        llm_pred = padded_pred
    
    # Average the predictions
    ensemble_pred = (ml_pred + llm_pred) / 2
    
    # Ensure output has the correct shape
    if ensemble_pred.shape[1] < len(LABEL_NAMES):
        padded_pred = np.zeros((ensemble_pred.shape[0], len(LABEL_NAMES)))
        padded_pred[:, :ensemble_pred.shape[1]] = ensemble_pred
        ensemble_pred = padded_pred
    elif ensemble_pred.shape[1] > len(LABEL_NAMES):
        ensemble_pred = ensemble_pred[:, :len(LABEL_NAMES)]
    
    return ensemble_pred.squeeze()

def calculate_additional_metrics(y_true, y_pred, y_probs):
    """Calculate additional evaluation metrics."""
    metrics = {}
    
    # Calculate balanced accuracy
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except:
        metrics['balanced_accuracy'] = float('nan')
    
    # Calculate Matthews correlation coefficient
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics['matthews_corrcoef'] = float('nan')
    
    # Calculate Cohen's kappa
    try:
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    except:
        metrics['cohen_kappa'] = float('nan')
    
    # Calculate per-class metrics
    metrics['per_class_metrics'] = {}
    for i in range(y_true.shape[1]):
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_f1 = np.max(f1_scores)
            metrics['per_class_metrics'][f'class_{i}'] = {
                'best_f1': float(best_f1),
                'precision_at_best_f1': float(precision[np.argmax(f1_scores)]),
                'recall_at_best_f1': float(recall[np.argmax(f1_scores)])
            }
        except:
            metrics['per_class_metrics'][f'class_{i}'] = {
                'best_f1': float('nan'),
                'precision_at_best_f1': float('nan'),
                'recall_at_best_f1': float('nan')
            }
    
    return metrics

def find_optimal_thresholds(y_true, y_pred, label_names):
    """Find optimal thresholds for each class using ROC curves and F1 scores."""
    thresholds = {}
    
    for i, label in enumerate(label_names):
        # Get label-specific predictions and true values
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        
        # Skip if no positive samples
        if len(np.unique(y_true_label)) < 2:
            thresholds[label] = 0.5
            continue
            
        # Calculate precision-recall curve
        precision, recall, thresh = precision_recall_curve(y_true_label, y_pred_label)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Find threshold that maximizes F1 score
        best_f1_thresh = thresh[np.argmax(f1_scores)]
        
        # Calculate ROC curve
        fpr, tpr, roc_thresh = roc_curve(y_true_label, y_pred_label)
        
        # Find threshold that maximizes TPR while keeping FPR reasonable
        optimal_idx = np.argmax(tpr - fpr)
        roc_optimal_thresh = roc_thresh[optimal_idx]
        
        # Use weighted average of F1 and ROC thresholds
        final_threshold = 0.7 * best_f1_thresh + 0.3 * roc_optimal_thresh
        
        # Apply class-specific adjustments
        if label == 'toxic':
            
            final_threshold *= 0.9
        elif label == 'severe_toxic':
            
            final_threshold *= 1.1
        elif label == 'threat':
            
            final_threshold *= 1.15
        
        # Ensure threshold is between 0 and 1
        final_threshold = np.clip(final_threshold, 0.1, 0.9)
        
        thresholds[label] = float(final_threshold)
    
    return thresholds

def evaluate_resilience(ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer, test_data):
    """Evaluate model resilience to obfuscation"""
    # Create multi-label encoding
    labels = create_multi_label_encoding(test_data)
    
    # Get BERT embeddings for ML model
    texts = test_data['text'].tolist()
    embeddings_df, feature_names = get_bert_embeddings(texts, bert_tokenizer, bert_model)
    
    # Get predictions from ML model
    ml_predictions = []
    for _, row in embeddings_df.iterrows():
        pred = ml_model.predict_proba(row.to_frame().T)
        pred = np.array(pred)
        if len(pred.shape) == 3:
            pred = pred.reshape(pred.shape[0], -1)
        if pred.shape[1] < len(LABEL_NAMES):
            padded_pred = np.zeros((pred.shape[0], len(LABEL_NAMES)))
            padded_pred[:, :pred.shape[1]] = pred
            pred = padded_pred
        ml_predictions.append(pred[0])
    ml_predictions = np.array(ml_predictions)
    
    # Get predictions from LLM model
    llm_predictions = []
    for text in texts:
        inputs = llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = llm_model(**inputs)
        probs = torch.sigmoid(outputs.logits).detach().numpy()
        if probs.shape[1] < len(LABEL_NAMES):
            padded_pred = np.zeros((probs.shape[0], len(LABEL_NAMES)))
            padded_pred[:, :probs.shape[1]] = probs
            probs = padded_pred
        elif probs.shape[1] > len(LABEL_NAMES):
            probs = probs[:, :len(LABEL_NAMES)]
        llm_predictions.append(probs[0])
    llm_predictions = np.array(llm_predictions)
    
    # Get ensemble predictions
    ensemble_predictions = []
    for i, text in enumerate(texts):
        pred = ensemble_predict([ml_model, llm_model], text, bert_tokenizer, bert_model, llm_tokenizer)
        if len(pred.shape) == 1:
            pred = pred.reshape(1, -1)
        if pred.shape[1] < len(LABEL_NAMES):
            padded_pred = np.zeros((pred.shape[0], len(LABEL_NAMES)))
            padded_pred[:, :pred.shape[1]] = pred
            pred = padded_pred
        elif pred.shape[1] > len(LABEL_NAMES):
            pred = pred[:, :len(LABEL_NAMES)]
        ensemble_predictions.append(pred[0])
    ensemble_predictions = np.array(ensemble_predictions)
    
    # Find optimal thresholds for each model
    ml_thresholds = find_optimal_thresholds(labels, ml_predictions, LABEL_NAMES)
    llm_thresholds = find_optimal_thresholds(labels, llm_predictions, LABEL_NAMES)
    ensemble_thresholds = find_optimal_thresholds(labels, ensemble_predictions, LABEL_NAMES)
    
    # Calculate metrics using optimal thresholds
    metrics = {
        'ml_model': calculate_resilience_metrics(labels, ml_predictions, LABEL_NAMES, "ML Model", ml_thresholds),
        'llm_model': calculate_resilience_metrics(labels, llm_predictions, LABEL_NAMES, "LLM Model", llm_thresholds),
        'ensemble': calculate_resilience_metrics(labels, ensemble_predictions, LABEL_NAMES, "Ensemble", ensemble_thresholds)
    }
    
    return metrics

def generate_test_data():
    """Generate test data with clean and obfuscated text"""
    test_data = pd.DataFrame({
        'text': [
            # Clean examples for each class
            'This is a normal comment.',  
            'Have a great day!',  
            'Thank you for sharing.',  
            
            # Toxic examples
            'You are a stupid worthless idiot.',  
            'You are completely useless.',  
            'What a dumb thing to say.',  
            
            # Severe toxic examples
            'I hope you die painfully from cancer.',  
            'You deserve to suffer and die.',  
            'Kill yourself, nobody wants you here.',  
            
            # Obscene examples
            'You are a fucking piece of shit.',  
            'Go fuck yourself you piece of garbage.',  
            'What a fucking asshole.',  
            
            # Threat examples
            'I will find you and kill you.',  
            'I am going to hurt you badly.', 
            'You better watch your back.',  
            
            # Identity hate examples
            'All [group] people should be eliminated.',  
            '[group] people are subhuman.',  
            '[group] should all disappear.',  
            
            # Leetspeak obfuscation
            'Y0u 4r3 4 $tup1d w0rthl3$$ 1d10t.',  
            '1 h0p3 y0u d13 p41nfully.',  
            'Y0u 4r3 4 fuck1ng p13c3 0f $h1t.',  
            '1 w1ll f1nd y0u 4nd k1ll y0u.',  
            '4ll [gr0up] p30pl3 $h0uld d13.',  
            
            # Space injection. lol mind the language 
            'Y o u  a r e  a  s t u p i d.',  
            'D i e  i n  a  f i r e.',  
            'F * c k  y o u.', 
            'I  w i l l  h u r t  y o u.', 
            '[G r o u p]  a r e  s c u m.',
            
            # Character repetition
            'Yooouuu aaarrreee stuuupid.',  
            'Diiiieee sloooowly.',  
            'Fuuuck yooouuu.',  
            'I wiiiill huuurt yooou.',  
            '[Groooup] shouuuld diiiie.',  
            
            # Mixed obfuscation
            'Y0u 4r3 4 $tup1d!!!',  
            'D13 1n h3ll!!!',  
            'F*ck1ng tr4$h!!!',  
            'W4tch y0ur b4ck!!!', 
            '[Gr0up] mu$t g0!!!',  
            
            # Unicode homoglyphs
            'Yοu аrе ѕtuріd.',  
            'Dіе ѕlοwly.',  
            'Fυсk yου.',  
            'Ι wіll hυrt yου.',  
            '[Grουр] аrе ѕсυm.',  
            
            # Combined techniques
            'Y0u_4r3_$tup1d_4f!!!',  
            'D.1.3_1.n_h.3.l.l',  
            'F*ck1ng_p13c3_0f_$h1t',  
            'W4tch_y0ur_b4ck_c4r3fully',  
            '[Gr0up]_mu$t_d13_n0w',  
        ],
        'is_obfuscated': [
            False, False, False,  
            False, False, False,  
            False, False, False,  
            False, False, False,  
            False, False, False,  
            False, False, False,  
            True, True, True, True, True,  
            True, True, True, True, True,  
            True, True, True, True, True,  
            True, True, True, True, True,  
            True, True, True, True, True,  
            True, True, True, True, True,  
        ]
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
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Resilience Metrics:")
        print(f"F1 Macro: {model_metrics.get('f1_macro', float('nan')):.4f}")
        print(f"F1 Micro: {model_metrics.get('f1_micro', float('nan')):.4f}")
        print(f"Accuracy: {model_metrics.get('accuracy', float('nan')):.4f}")
        print(f"ROC-AUC Macro: {model_metrics.get('roc_auc_macro', float('nan')):.4f}")
        
        if 'per_label_metrics' in model_metrics:
            print("\nPer-Label Metrics:")
            for label, scores in model_metrics['per_label_metrics'].items():
                print(f"{label}:")
                print(f"  F1 = {scores.get('f1', float('nan')):.4f}")
                print(f"  Precision = {scores.get('precision', float('nan')):.4f}")
                print(f"  Recall = {scores.get('recall', float('nan')):.4f}")
                print(f"  ROC-AUC = {scores.get('roc_auc', float('nan')):.4f}")
                print(f"  Threshold = {scores.get('threshold', float('nan')):.4f}")

def plot_resilience_metrics(metrics, output_dir):
    """Create visualizations of resilience metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot metrics for each model
    for model_name, model_metrics in metrics.items():
        # Plot per-label metrics
        if 'per_label_metrics' in model_metrics:
            labels = list(model_metrics['per_label_metrics'].keys())
            metrics_names = ['f1', 'precision', 'recall', 'roc_auc']
            
            for metric in metrics_names:
                values = []
                valid_labels = []
                for label in labels:
                    value = model_metrics['per_label_metrics'][label].get(metric, float('nan'))
                    if not np.isnan(value):
                        values.append(value)
                        valid_labels.append(label)
                
                if values: 
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(valid_labels, values)
                    plt.title(f'{model_name} - Per-label {metric.upper()} Scores')
                    plt.xticks(rotation=45)
                    plt.ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{model_name}_per_label_{metric}.png')
                    plt.close()
            
            # Plot confusion matrix elements
            for label in labels:
                cm_metrics = model_metrics['per_label_metrics'][label]
                plt.figure(figsize=(8, 6))
                confusion_matrix = np.array([
                    [cm_metrics['true_negatives'], cm_metrics['false_positives']],
                    [cm_metrics['false_negatives'], cm_metrics['true_positives']]
                ])
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model_name} - {label} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(output_dir / f'{model_name}_{label}_confusion_matrix.png')
                plt.close()
        
        # Plot overall metrics
        overall_metrics = {
            'F1 Macro': model_metrics.get('f1_macro', float('nan')),
            'F1 Micro': model_metrics.get('f1_micro', float('nan')),
            'Accuracy': model_metrics.get('accuracy', float('nan')),
            'ROC-AUC Macro': model_metrics.get('roc_auc_macro', float('nan'))
        }
        
        # Filter out NaN values
        valid_metrics = {k: v for k, v in overall_metrics.items() if not np.isnan(v)}
        
        if valid_metrics:  
            plt.figure(figsize=(10, 6))
            bars = plt.bar(valid_metrics.keys(), valid_metrics.values())
            plt.title(f'{model_name} - Overall Metrics')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_overall_metrics.png')
            plt.close()

def get_text_from_embeddings(embeddings, tokenizer):
    """Convert BERT embeddings back to text using the tokenizer"""
    # Decoding tokens back to text
    tokens = tokenizer.decode(embeddings)
    # Remove special tokens and clean up the text
    text = tokens.replace("[CLS]", "").replace("[SEP]", "").strip()
    return text

if __name__ == "__main__":
    main()
