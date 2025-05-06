"""Basic config + logging stuff for this ML thing."""


import logging
from pathlib import Path
import sys


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "ml_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("toxic_classification")


MODEL_CONFIG = {
    "bert_model": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "random_state": 42,
    "n_jobs": -1,
    "test_size": 0.2
}

LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

PATHS = {
    "data": Path("data"),
    "models": Path("models/ML"),
    "embeddings": Path("models/ML/embeddings"),
    "outputs": Path("outputs")
}

for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True) 