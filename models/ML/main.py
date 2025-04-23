"""Main module for toxic comment classification."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, make_scorer, roc_auc_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from pathlib import Path
import warnings
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main(data_path: str = None):
    """Run the complete toxic comment classification pipeline.
    
    Args:
        data_path: Path to the input CSV file. If None, uses default path in data/raw/train.csv
    """
    try:
        # Define label names
        label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Create base paths relative to project root
        base_path = project_root
        models_path = base_path / "models"
        outputs_path = base_path / "outputs"
        data_path = data_path or (base_path / "data" / "raw" / "train.csv")
        
        # Verify data file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"Data file not found at {data_path}. Please ensure the file exists.\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Project root: {project_root}"
            )
        
        # Create output directories
        (models_path / "embeddings").mkdir(parents=True, exist_ok=True)
        (models_path / "saved").mkdir(parents=True, exist_ok=True)
        (outputs_path / "figures").mkdir(parents=True, exist_ok=True)
        (outputs_path / "results").mkdir(parents=True, exist_ok=True)
        
        # -----------------------
        # 1. Load and Inspect Data
        # -----------------------
        print("[INFO] Loading data...")
        df_train = pd.read_csv(data_path, on_bad_lines='skip')
        
        # Verify required columns exist
        required_columns = ['comment_text'] + label_names
        missing_columns = [col for col in required_columns if col not in df_train.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"[INFO] Loaded {len(df_train)} samples")
        
        # -----------------------
        # 2. Preprocessing & Identity Tagging
        # -----------------------
        print("[INFO] Preprocessing data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        identity_terms = [
            "muslim", "christian", "jewish", "black", "white", "asian",
            "latino", "gay", "lesbian", "trans", "female", "male",
            "woman", "man", "nonbinary", "disabled"
        ]
        
        def has_identity_term(text):
            text = str(text).lower()
            return int(any(term in text for term in identity_terms))
        
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return " ".join(words)
        
        # Clean and preprocess data
        print("[INFO] Cleaning text and adding identity tags...")
        df_train['has_identity'] = df_train['comment_text'].apply(has_identity_term)
        df_train['clean_comment'] = df_train['comment_text'].apply(clean_text)
        
        # Log text lengths before filtering
        text_lengths = df_train['clean_comment'].str.split().str.len()
        print(f"[INFO] Text length distribution before filtering:")
        print(f"Min length: {text_lengths.min()}")
        print(f"Max length: {text_lengths.max()}")
        print(f"Mean length: {text_lengths.mean():.2f}")
        print(f"Median length: {text_lengths.median()}")
        
        # Remove only empty comments, keep all others
        initial_len = len(df_train)
        df_train = df_train[df_train['clean_comment'].str.strip().str.len() > 0]  # Only remove empty strings
        removed_count = initial_len - len(df_train)
        print(f"[INFO] Removed {removed_count} empty comments")
        
        if removed_count == initial_len:
            print("[WARNING] All comments were removed during cleaning. Sample of original comments:")
            print(df_train['comment_text'].head())
            raise ValueError("All comments were removed during cleaning. Please check the input data.")
        
        # Check minimum sample size with a more lenient requirement
        MIN_SAMPLES = 3  # Reduced from 10 to 3 for small datasets
        if len(df_train) < MIN_SAMPLES:
            print(f"[ERROR] Insufficient samples after preprocessing. Need at least {MIN_SAMPLES} samples, got {len(df_train)}")
            print("[INFO] Sample of cleaned comments:")
            print(df_train['clean_comment'].head())
            raise ValueError(f"Insufficient samples after preprocessing. Need at least {MIN_SAMPLES} samples")
        
        # Extract texts and labels
        X_text = df_train['clean_comment'].fillna('').tolist()
        Y = df_train[label_names].values
        
        print(f"[INFO] Final dataset size: {len(X_text)} samples")
        print(f"[INFO] Number of labels: {Y.shape[1]}")
        
        if len(X_text) == 0:
            print("[ERROR] No valid texts found after preprocessing. Sample of cleaned comments:")
            print(df_train['clean_comment'].head())
            raise ValueError("No valid texts found after preprocessing")
        
        # -----------------------
        # 3. BERT Embedding
        # -----------------------
        print("[INFO] Initializing BERT...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        
        @torch.no_grad()
        def get_bert_mean_embeddings(texts, batch_size=32):
            """Generate BERT embeddings for a list of texts.
            
            Args:
                texts: List of input texts
                batch_size: Batch size for processing
                
            Returns:
                Numpy array of BERT embeddings
            """
            if not texts:
                raise ValueError("Input texts list is empty")
            
            # Filter out empty or None texts
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            if not valid_texts:
                raise ValueError("No valid texts found in input")
            
            if len(valid_texts) != len(texts):
                print(f"[WARNING] Filtered out {len(texts) - len(valid_texts)} invalid texts")
            
            all_embeddings = []
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i+batch_size]
                try:
                    # Tokenize
                    inputs = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    )
                    
                    # Generate embeddings
                    outputs = model(**inputs)
                    mean_embeddings = outputs.last_hidden_state.mean(dim=1)
                    all_embeddings.append(mean_embeddings)
                    print(f"[INFO] Processed batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}")
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            if not all_embeddings:
                raise ValueError("No embeddings were generated successfully")
            
            try:
                return torch.cat(all_embeddings).numpy()
            except Exception as e:
                raise ValueError(f"Error concatenating embeddings: {str(e)}")
        
        print("[INFO] Generating BERT embeddings...")
        print(f"[INFO] Processing {len(X_text)} texts...")
        X_bert = get_bert_mean_embeddings(X_text)
        print(f"[INFO] Generated embeddings shape: {X_bert.shape}")
        
        # Save embeddings
        print("[INFO] Saving embeddings...")
        np.save(models_path / "embeddings" / "bert_embeddings.npy", X_bert)
        print("[INFO] Embeddings saved successfully.")
        
        # -----------------------
        # 4. Model Training (LightGBM)
        # -----------------------
        print("[INFO] Initializing model...")
        base_model = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
        param_space = {
            'estimator__num_leaves': Integer(20, 150),
            'estimator__max_depth': Integer(3, 10),
            'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'estimator__n_estimators': Integer(100, 500),
            'estimator__subsample': Real(0.6, 1.0),
            'estimator__colsample_bytree': Real(0.6, 1.0)
        }
        
        multi_target_model = MultiOutputClassifier(base_model)
        
        # Adjust cross-validation strategy based on sample size
        if len(X_bert) < 10:
            print("[WARNING] Small dataset detected. Using simpler validation strategy.")
            cv_strategy = 2  # Use 2-fold CV for small datasets
        else:
            cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        def custom_macro_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro')
        
        opt = BayesSearchCV(
            estimator=multi_target_model,
            search_spaces=param_space,
            cv=cv_strategy,
            n_iter=min(25, len(X_bert)),  # Reduce iterations for small datasets
            scoring=make_scorer(custom_macro_f1),
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        
        print("[INFO] Starting Bayesian hyperparameter optimization...")
        opt.fit(X_bert, Y)
        print("[INFO] Optimization completed.")
        
        # -----------------------
        # 5. Threshold Tuning
        # -----------------------
        print("[INFO] Tuning thresholds...")
        def find_best_thresholds(y_true, y_probs):
            thresholds = []
            for i in range(y_true.shape[1]):
                precision, recall, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                thresholds.append(thresh[np.argmax(f1_scores)])
            return np.array(thresholds)
        
        Y_probs = np.column_stack([
            estimator.predict_proba(X_bert)[:, 1] for estimator in opt.best_estimator_.estimators_
        ])
        
        best_thresholds = find_best_thresholds(Y, Y_probs)
        Y_pred_opt = (Y_probs > best_thresholds).astype(int)
        
        # -----------------------
        # 6. Evaluation + Fairness
        # -----------------------
        print("\nClassification Report:")
        print(classification_report(Y, Y_pred_opt, target_names=label_names))
        
        mask_id = df_train['has_identity'] == 1
        mask_noid = df_train['has_identity'] == 0
        print("\n Fairness Check - Macro F1 Scores")
        print(f"With identity terms: {f1_score(Y[mask_id], Y_pred_opt[mask_id], average='macro'):.4f}")
        print(f"Without identity terms: {f1_score(Y[mask_noid], Y_pred_opt[mask_noid], average='macro'):.4f}")
        
        # -----------------------
        # 7. Identity-Term F1 Visualization
        # -----------------------
        print("[INFO] Generating fairness visualization...")
        identity_f1 = {}
        for term in identity_terms:
            mask = df_train['comment_text'].str.lower().str.contains(term)
            if mask.sum() == 0:
                continue
            Y_term = Y[mask]
            Y_term_pred = Y_pred_opt[mask]
            identity_f1[term] = f1_score(Y_term, Y_term_pred, average='macro')
        
        identity_f1 = dict(sorted(identity_f1.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(identity_f1.keys()), y=list(identity_f1.values()))
        plt.xticks(rotation=45)
        plt.ylabel("Macro F1")
        plt.title("Fairness Assessment: F1 Score by Identity Term")
        plt.tight_layout()
        plt.savefig(outputs_path / "figures" / "per_label_f1_ml.png")
        plt.close()
        
        # -----------------------
        # 8. Save Artifacts for Benchmarking
        # -----------------------
        print("\n[INFO] Saving model and data artifacts...")
        np.save(models_path / "embeddings" / "X_bert_test.npy", X_bert)
        pd.DataFrame(Y, columns=label_names).to_csv(outputs_path / "results" / "Y_test.csv", index=False)
        dump(opt.best_estimator_, models_path / "saved" / "opt_model.joblib")
        
        # Save evaluation metrics
        eval_metrics = {
            'f1_macro': f1_score(Y, Y_pred_opt, average='macro'),
            'f1_micro': f1_score(Y, Y_pred_opt, average='micro'),
            'accuracy': np.mean(Y == Y_pred_opt),
            'roc_auc_macro': np.mean([roc_auc_score(Y[:, i], Y_probs[:, i]) for i in range(Y.shape[1])]),
            'per_label_metrics': {
                label: {
                    'f1': f1_score(Y[:, i], Y_pred_opt[:, i], average='binary'),
                    'precision': precision_score(Y[:, i], Y_pred_opt[:, i], average='binary'),
                    'recall': recall_score(Y[:, i], Y_pred_opt[:, i], average='binary'),
                    'roc_auc': roc_auc_score(Y[:, i], Y_probs[:, i])
                }
                for i, label in enumerate(label_names)
            }
        }
        
        # Save metrics to JSON
        with open(outputs_path / "results" / "eval_results.json", 'w') as f:
            json.dump(eval_metrics, f, indent=4)
        
        print("[INFO] Artifacts saved successfully.")
        
        return {
            'model': opt.best_estimator_,
            'thresholds': best_thresholds,
            'metrics': {
                'classification_report': classification_report(Y, Y_pred_opt, target_names=label_names, output_dict=True),
                'fairness': {
                    'with_identity': f1_score(Y[mask_id], Y_pred_opt[mask_id], average='macro'),
                    'without_identity': f1_score(Y[mask_noid], Y_pred_opt[mask_noid], average='macro')
                }
            }
        }
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()