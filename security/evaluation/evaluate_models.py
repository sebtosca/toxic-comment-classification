import joblib
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from pathlib import Path
from evaluate_obfuscation_resilience import main

def load_models():
    """Load pre-trained ML and LLM models"""
    
    ml_model_path = Path("models/saved/opt_model.joblib")
    if not ml_model_path.exists():
        raise FileNotFoundError(f"ML model not found at {ml_model_path}")
    ml_model = joblib.load(ml_model_path)
    print("Loaded optimized model, the ML one")
    
    
    llm_model_path = Path("src/roberta-toxic-finetuned")
    if not llm_model_path.exists():
        raise FileNotFoundError(f"LLM model not found at {llm_model_path}")
    
    # Loading the fine-tuned model
    llm_model = RobertaForSequenceClassification.from_pretrained(llm_model_path)
    llm_tokenizer = RobertaTokenizer.from_pretrained(llm_model_path)
    print("Loaded fine-tuned RoBERTa model and tokenizer")
    
    # Load BERT tokenizer and model for ML model
    from transformers import BertTokenizer, BertModel
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    print("Loaded BERT tokenizer and model")
    
    return ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer

if __name__ == "__main__":
    # Load models
    ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer = load_models()
    
    # Run evaluation
    main(ml_model, llm_model, bert_tokenizer, bert_model, llm_tokenizer) 