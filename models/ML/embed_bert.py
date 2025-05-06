"""Generate BERT embeddings for text classification."""

import torch
from transformers import BertTokenizer, BertModel
from typing import List
import numpy as np
import warnings

def get_bert_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Get sentence embeddings using BERT (average over tokens).
    
    texts: list of strings
    batch_size: how many to process at once (might need to lower for small GPUs)
    
    returns: numpy array of embeddings (shape: [n_samples, hidden_dim])
    """
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    @torch.no_grad()
    def get_bert_mean_embeddings(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            outputs = model(**inputs)
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings)
        return torch.cat(all_embeddings).numpy()
    
    print(" Generating BERT embeddings")
    embeddings = get_bert_mean_embeddings(texts, batch_size)
    print("Embeddings ready.")
    
    return embeddings
