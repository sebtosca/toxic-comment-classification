# Toxic Comment Classification

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security](https://img.shields.io/badge/security-enabled-brightgreen)](SECURITY.md)
[![Model Card](https://img.shields.io/badge/model%20card-available-blue)](docs/model_card.md)

A robust machine learning model for detecting toxic comments using RoBERTa, with built-in security features against text obfuscation attacks.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Security Features](#security-features)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Card](#model-card)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Features

- **Advanced Classification**: Uses RoBERTa for state-of-the-art text classification
- **Multi-label Support**: Detects multiple types of toxicity simultaneously
- **Security Features**: 
  - Resilience against text obfuscation
  - Backtranslation attack detection
  - Synonym substitution handling
- **Comprehensive Metrics**: 
  - Accuracy, F1, Precision, Recall
  - Micro and Macro averages
  - ROC AUC scores

## Quick Start

```python
from models.ML.main import main

# Run the complete pipeline
main()
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sebtosca/toxic-comment-classification.git
cd toxic-comment-classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the datasets from Kaggle
5. Place the data files in `data/raw/`
6. Run the training script

## Usage

### Basic Usage

```python
from models.ML.main import main

# Run the complete pipeline
main()
```

### Advanced Usage

```python
from models.ML.preprocess import preprocess_data
from models.ML.embed_bert import get_bert_embeddings
from models.ML.train_lightgbm import train_model
from models.ML.evaluate import evaluate_model

# Step-by-step pipeline
df, X_text, Y = preprocess_data("train.csv")
X_bert = get_bert_embeddings(X_text)
model = train_model(X_bert, Y)
evaluate_model(df, Y, model.predict_proba(X_bert), None, label_names)
```

## Project Structure

```
project/
├── .github/
│   └── workflows/          # CI/CD configuration
├── data/
│   ├── raw/               # Original data files
│   │   ├── test.csv
│   │   ├── test_labels.csv
│   │   └── .gitkeep
│   └── processed/         # Processed data files
│       └── .gitkeep
├── models/
│   ├── ML/               # ML model code
│   ├── saved/            # Saved model files
│   └── embeddings/       # Generated embeddings
├── outputs/
│   ├── figures/          # Generated plots
│   └── results/          # Evaluation results
├── security/
│   └── evaluation/       # Security evaluation code
├── src/                  # Main source code
│   ├── __init__.py
│   ├── LLM.py
│   └── benchmark.py
├── tests/                # Test files
├── docs/                 # Documentation
├── logs/                 # Log files
├── .gitignore           # Git ignore rules
├── LICENSE              # License file
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── SECURITY.md          # Security documentation
```

## Data Requirements
The following data files are required in the `data/raw` directory:
- `train.csv`
- `test.csv`
- `test_labels.csv`

## Data Source
The datasets can be downloaded from the Kaggle competition:
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

### Steps to Download Data:
1. Create a Kaggle account if you don't have one
2. Go to the competition page: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
3. Click on "Download All" to get the datasets
4. Extract the files and place them in the `data/raw` directory

### Required Files:
- `train.csv` - Training data with labels
- `test.csv` - Test data without labels
- `test_labels.csv` - Labels for test data

## Setup
1. Clone the repository
2. Download the datasets from Kaggle
3. Place the data files in `data/raw/`
4. Install dependencies
5. Run the training script