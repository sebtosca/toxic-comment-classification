import pandas as pd
import torch
import numpy as np
import gc
import json
import re
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from sklearn.utils import resample

# ---------------------
# 1. Paths and Data Loading
# ---------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found at {DATA_DIR}.")

df_train = pd.read_csv(DATA_DIR / "train.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")
df_test_labels = pd.read_csv(DATA_DIR / "test_labels.csv")

candidate_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------
# 2. Label Distribution & Oversampling
# ---------------------
print("Label distribution:\n", df_train[candidate_labels].sum())

minor_labels = ["threat", "identity_hate", "severe_toxic"]
for label in minor_labels:
    df_minority = df_train[df_train[label] == 1]
    df_upsampled = resample(df_minority, replace=True, n_samples=3000, random_state=42)
    df_train = pd.concat([df_train, df_upsampled])
df_train = df_train.sample(frac=1, random_state=42)

# ---------------------
# 3. Enhanced Cleaning Function
# ---------------------
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Normalize contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Remove emojis and special characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)

    # Replace repeated punctuation
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

df_train["clean_comment"] = df_train["comment_text"].apply(clean_text)
df_test["clean_comment"] = df_test["comment_text"].apply(clean_text)
df_test_labels["clean_comment"] = df_test.get("comment_text", df_test["clean_comment"]).apply(clean_text)

# ---------------------
# 4. Tokenization
# ---------------------
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
MAX_LENGTH = 128

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

# ---------------------
# 5. Weighted Loss Model
# ---------------------
label_freqs = df_train[candidate_labels].sum().values
total = len(df_train)
pos_weights = torch.tensor((total - label_freqs) / label_freqs, dtype=torch.float32)

class RobertaForWeightedMultiLabel(RobertaForSequenceClassification):
    def __init__(self, config, pos_weight):
        super().__init__(config)
        self.pos_weight = pos_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)  
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = output.logits
        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

config = RobertaConfig.from_pretrained("roberta-base", num_labels=6, problem_type="multi_label_classification")
model = RobertaForWeightedMultiLabel.from_pretrained("roberta-base", config=config, pos_weight=pos_weights)
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
# 7. Metrics
# ---------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).detach().cpu().numpy()
    labels = np.array(labels).astype(int)
    n_labels = labels.shape[1]

    metrics = {}
    for i in range(n_labels):
        label_name = candidate_labels[i]
        y_true = labels[:, i]
        y_prob = probs[:, i]
        mask = y_true != -1
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]

        if len(np.unique(y_true_clean)) < 2:
            print(f"[Warning] Skipping ROC AUC for {label_name}: not enough classes.")
            metrics[f"{label_name}_roc_auc"] = 0.5
            continue

        thresholds = np.unique(y_prob_clean)
        best_f1, best_thresh = -1, 0.5
        for thresh in thresholds:
            y_pred = (y_prob_clean >= thresh).astype(int)
            f1 = f1_score(y_true_clean, y_pred, average='binary')
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        y_pred = (y_prob_clean >= best_thresh).astype(int)
        metrics[f"{label_name}_f1"] = f1_score(y_true_clean, y_pred)
        metrics[f"{label_name}_precision"] = precision_score(y_true_clean, y_pred, zero_division=0)
        metrics[f"{label_name}_recall"] = recall_score(y_true_clean, y_pred, zero_division=0)
        metrics[f"{label_name}_accuracy"] = accuracy_score(y_true_clean, y_pred)
        metrics[f"{label_name}_roc_auc"] = roc_auc_score(y_true_clean, y_prob_clean)

    metrics["f1_macro"] = np.mean([metrics.get(f"{label}_f1", 0.0) for label in candidate_labels])
    metrics["precision_macro"] = np.mean([metrics.get(f"{label}_precision", 0.0) for label in candidate_labels])
    metrics["recall_macro"] = np.mean([metrics.get(f"{label}_recall", 0.0) for label in candidate_labels])
    metrics["roc_auc_macro"] = np.mean([metrics.get(f"{label}_roc_auc", 0.5) for label in candidate_labels])

    all_y_true, all_y_pred, all_y_prob = [], [], []
    for i in range(n_labels):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        mask = y_true != -1
        y_true_clean = y_true[mask]
        y_prob_clean = y_prob[mask]

        if len(y_true_clean) > 0:
            thresholds = np.unique(y_prob_clean)
            best_f1, best_thresh = -1, 0.5
            for thresh in thresholds:
                y_pred = (y_prob_clean >= thresh).astype(int)
                f1 = f1_score(y_true_clean, y_pred, average='binary')
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh
            y_pred = (y_prob_clean >= best_thresh).astype(int)
            all_y_true.extend(y_true_clean)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob_clean)

    if len(all_y_true) > 0:
        metrics["f1_micro"] = f1_score(all_y_true, all_y_pred, average='binary')
        metrics["precision_micro"] = precision_score(all_y_true, all_y_pred, average='binary')
        metrics["recall_micro"] = recall_score(all_y_true, all_y_pred, average='binary')
        metrics["roc_auc_micro"] = roc_auc_score(all_y_true, all_y_prob)

    return metrics

# ---------------------
# 8. Trainer
# ---------------------
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
# 9. Main
# ---------------------
def main():
    trainer.train()
    model.save_pretrained("./roberta-toxic-finetuned")
    tokenizer.save_pretrained("./roberta-toxic-finetuned")
    eval_results = trainer.evaluate()
    print("\n*** Evaluation Metrics ***")
    for key, val in eval_results.items():
        print(f"{key}: {val:.4f}")
    output_path = Path(__file__).parent.parent / "outputs" / "results" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
