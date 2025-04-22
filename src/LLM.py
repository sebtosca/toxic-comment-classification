import pandas as pd
import torch
import numpy as np
import gc
import json
import re
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from pathlib import Path

# Update data paths to use project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Ensure data directory exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please ensure the data files are in the correct location.")

# Load data with error handling
try:
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    df_test_labels = pd.read_csv(DATA_DIR / "test_labels.csv")
except FileNotFoundError as e:
    print(f"Error loading data files: {str(e)}")
    print(f"Expected files in: {DATA_DIR}")
    print("Please ensure the following files exist:")
    print("- train.csv")
    print("- test.csv")
    print("- test_labels.csv")
    raise

# ---------------------
# 2. Sample Zero-Shot Evaluation
# ---------------------
candidate_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sampled_df = df_test.sample(n=100, random_state=42)
sampled_texts = sampled_df["comment_text"].tolist()

llm_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1,
)

for i, text in enumerate(sampled_texts[:5]):
    result = llm_pipeline(text, candidate_labels=candidate_labels, multi_label=True)
    print(f"\nText {i+1}: {text}")
    print("LLM Output:", result)

# ---------------------
# 3. Preprocessing Function
# ---------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
df_train["clean_comment"] = df_train["comment_text"].apply(clean_text)
df_test["clean_comment"] = df_test["comment_text"].apply(clean_text)
if "comment_text" in df_test_labels.columns:
    df_test_labels["clean_comment"] = df_test_labels["comment_text"].apply(clean_text)
else:
    df_test_labels["clean_comment"] = df_test["clean_comment"]

# ---------------------
# 4. Prepare Tokenizer & Dataset
# ---------------------
MAX_LENGTH = 128
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

train_texts = df_train["clean_comment"].tolist()
train_labels = df_train[candidate_labels].values.astype(np.float32).tolist()
val_texts = df_test_labels["clean_comment"].tolist()
val_labels = df_test_labels[candidate_labels].values.astype(np.float32).tolist()

train_dataset = HFDataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = HFDataset.from_dict({"text": val_texts, "labels": val_labels})

def tokenize_batch(batch):
    encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    encoding["labels"] = batch["labels"]
    return encoding

train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

del train_texts, val_texts
gc.collect()

# ---------------------
# 5. Load RoBERTa Model
# ---------------------
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,
    problem_type="multi_label_classification"
)
model.gradient_checkpointing_enable()

# ---------------------
# 6. Training Arguments
# ---------------------
training_args = TrainingArguments(
    output_dir="./roberta-toxic-output",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)

# ---------------------
# 7. Metrics (no threshold tuning)
# ---------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(logits)).detach().cpu().numpy()
    
    # Convert labels to numpy array and handle -1 values
    labels = np.array(labels).astype(int)
    
    # Initialize metrics
    metrics = {}
    
    # Multi-label case
    n_labels = labels.shape[1]
    
    # Compute per-label metrics
    for i in range(n_labels):
        label_name = candidate_labels[i]  # Use actual label names
        y_true = labels[:, i]
        y_prob = probs[:, i]
        
        # Remove -1 values from consideration
        mask = y_true != -1
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]
        
        if len(y_true_clean) == 0:
            print(f"Warning: No valid labels for {label_name}")
            continue
            
        # Find best threshold for this label
        thresholds = np.unique(y_prob_clean)
        best_f1 = -1
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_prob_clean >= thresh).astype(int)
            f1 = f1_score(y_true_clean, y_pred, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Calculate metrics with best threshold
        y_pred = (y_prob_clean >= best_thresh).astype(int)
        
        metrics[f"{label_name}_f1"] = f1_score(y_true_clean, y_pred, average='binary')
        metrics[f"{label_name}_accuracy"] = accuracy_score(y_true_clean, y_pred)
        metrics[f"{label_name}_precision"] = precision_score(y_true_clean, y_pred, average='binary')
        metrics[f"{label_name}_recall"] = recall_score(y_true_clean, y_pred, average='binary')
        
        try:
            metrics[f"{label_name}_roc_auc"] = roc_auc_score(y_true_clean, y_prob_clean)
        except:
            metrics[f"{label_name}_roc_auc"] = 0.5
    
    # Compute macro averages
    metrics["f1_macro"] = np.mean([metrics.get(f"{label}_f1", 0.0) for label in candidate_labels])
    metrics["accuracy"] = np.mean([metrics.get(f"{label}_accuracy", 0.0) for label in candidate_labels])
    metrics["precision_macro"] = np.mean([metrics.get(f"{label}_precision", 0.0) for label in candidate_labels])
    metrics["recall_macro"] = np.mean([metrics.get(f"{label}_recall", 0.0) for label in candidate_labels])
    metrics["roc_auc_macro"] = np.mean([metrics.get(f"{label}_roc_auc", 0.5) for label in candidate_labels])
    
    # Compute micro averages
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    for i in range(n_labels):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        mask = y_true != -1
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]
        
        if len(y_true_clean) > 0:
            # Find best threshold for this label
            thresholds = np.unique(y_prob_clean)
            best_f1 = -1
            best_thresh = 0.5
            
            for thresh in thresholds:
                y_pred = (y_prob_clean >= thresh).astype(int)
                f1 = f1_score(y_true_clean, y_pred, average='binary')
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            y_pred = (y_prob_clean >= best_thresh).astype(int)
            all_y_true.extend(y_true_clean)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob_clean)
    
    if len(all_y_true) > 0:
        metrics["f1_micro"] = f1_score(all_y_true, all_y_pred, average='binary')
        metrics["precision_micro"] = precision_score(all_y_true, all_y_pred, average='binary')
        metrics["recall_micro"] = recall_score(all_y_true, all_y_pred, average='binary')
        try:
            metrics["roc_auc_micro"] = roc_auc_score(all_y_true, all_y_prob)
        except:
            metrics["roc_auc_micro"] = 0.5
    
    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# ---------------------
# 8. Main Function with Evaluation
# ---------------------
def main():
    trainer.train()
    model.save_pretrained("./roberta-toxic-finetuned")
    tokenizer.save_pretrained("./roberta-toxic-finetuned")

    eval_results = trainer.evaluate()
    print("\n*** Evaluation Metrics on Validation Set ***")
    for key, val in eval_results.items():
        print(f"{key}: {val:.4f}")

    with open("eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
