# Model Card for Toxic Comment Classification

## Model Details

### Model Description
- **Model Name**: Toxic Comment Classification
- **Version**: 1.0.0
- **Type**: Multi-label Text Classification
- **Architecture**: RoBERTa-base with LightGBM classifier
- **Task**: Toxic comment detection and classification

### Model Date
- **Created**: 2024
- **Last Updated**: 2024

### Model Type
- **Architecture**: Transformer-based (RoBERTa) + Gradient Boosting (LightGBM)
- **Input**: Text comments
- **Output**: Multi-label classification probabilities

### Paper or Other Resource
- **Repository**: [GitHub Repository](https://github.com/sebtosca/toxic-comment-classification)
- **License**: MIT

## Intended Use

### Primary Intended Uses
- Content moderation
- Toxic comment detection
- Online community management
- Social media monitoring

### Primary Intended Users
- Social media platforms
- Online communities
- Content moderation teams
- Researchers in NLP and content moderation

### Out-of-Scope Use Cases
- Legal decisions
- Automated content removal
- User profiling
- Sentiment analysis

## Factors

### Relevant Factors
- Text language (English)
- Comment length
- Presence of identity terms
- Text formatting and style

### Evaluation Factors
- Classification accuracy
- Fairness across identity groups
- Resilience to adversarial attacks
- Processing speed

## Metrics

### Model Performance Measures
- Accuracy (micro and macro)
- F1 Score (micro and macro)
- Precision (micro and macro)
- Recall (micro and macro)
- ROC AUC Score

### Decision Thresholds
- Optimized per-label thresholds
- Minimum confidence scores
- Identity term sensitivity

## Evaluation Data

### Datasets
- Training Data: Jigsaw Toxic Comment Classification Dataset
- Evaluation Data: Held-out test set
- Adversarial Test Set: Custom security evaluation dataset

### Motivation
- Improve content moderation
- Reduce harmful content
- Maintain community standards
- Protect vulnerable groups

### Preprocessing
- Text cleaning and normalization
- Identity term detection
- Stop word removal
- Lemmatization

## Training Data

### Training Data
- Source: Jigsaw Toxic Comment Classification Dataset
- Size: ~160,000 comments
- Distribution: Multi-label classification
- Languages: English

### Training Procedure
- Pre-training: RoBERTa-base
- Fine-tuning: Multi-label classification
- Optimization: Bayesian hyperparameter search
- Validation: Stratified k-fold cross-validation

## Quantitative Analyses

### Unitary Results
- Overall accuracy: 0.95 (micro), 0.92 (macro)
- Macro F1: 0.91
- Micro F1: 0.94
- Precision: 0.93 (micro), 0.90 (macro)
- Recall: 0.95 (micro), 0.92 (macro)
- ROC AUC: 0.98

### Per-Label Performance
| Label | F1 Score | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| Toxic | 0.94 | 0.93 | 0.95 | 0.98 |
| Severe Toxic | 0.89 | 0.87 | 0.91 | 0.97 |
| Obscene | 0.92 | 0.91 | 0.93 | 0.98 |
| Threat | 0.90 | 0.88 | 0.92 | 0.97 |
| Insult | 0.93 | 0.92 | 0.94 | 0.98 |
| Identity Hate | 0.88 | 0.86 | 0.90 | 0.96 |

### Intersectional Results

#### Performance by Identity Group
| Group | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| With Identity Terms | 0.90 | 0.89 | 0.91 |
| Without Identity Terms | 0.93 | 0.92 | 0.94 |

#### Performance by Comment Length
| Length (words) | F1 Score | Precision | Recall |
|----------------|----------|-----------|--------|
| < 10 | 0.88 | 0.87 | 0.89 |
| 10-50 | 0.92 | 0.91 | 0.93 |
| > 50 | 0.91 | 0.90 | 0.92 |

#### Performance by Toxicity Level
| Toxicity Level | F1 Score | Precision | Recall |
|----------------|----------|-----------|--------|
| Low (0.0-0.3) | 0.95 | 0.94 | 0.96 |
| Medium (0.3-0.7) | 0.92 | 0.91 | 0.93 |
| High (0.7-1.0) | 0.89 | 0.88 | 0.90 |

### Adversarial Performance
| Attack Type | Original F1 | After Attack | Resilience |
|-------------|-------------|--------------|------------|
| Text Obfuscation | 0.91 | 0.89 | 98% |
| Backtranslation | 0.91 | 0.88 | 97% |
| Synonym Substitution | 0.91 | 0.90 | 99% |

### Computational Performance
| Metric | Value |
|--------|-------|
| Training Time (GPU) | 2 hours |
| Inference Time (CPU) | 0.5s/comment |
| Inference Time (GPU) | 0.1s/comment |
| Batch Processing (1000 comments) | 10s |

## Ethical Considerations

### Data
- Source: Public dataset
- Anonymization: User data anonymized
- Consent: Dataset terms of use

### Human Life
- No direct impact on human life
- Indirect impact through content moderation
- Potential for bias in moderation

### Mitigations
- Regular model updates
- Human-in-the-loop review
- Bias detection and correction
- Adversarial testing

### Risks and Harms
- False positives in moderation
- Bias against certain groups
- Over-moderation
- Under-moderation

### Use Cases
- Content moderation
- Community management
- Research and development
- Educational purposes

## Caveats and Recommendations

### Caveats
- English language only
- Limited to text-based content
- May miss context-dependent toxicity
- Requires regular updates

### Recommendations
- Use as part of larger moderation system
- Implement human review
- Regular model retraining
- Monitor for bias and drift

## Technical Specifications

### Compute Infrastructure
- Training: GPU recommended
- Inference: CPU/GPU
- Memory: 8GB+ RAM
- Storage: 2GB+ disk space

### Model Size
- RoBERTa: ~500MB
- LightGBM: ~100MB
- Total: ~600MB

### Latency
- Inference time: < 1 second per comment
- Batch processing supported
- GPU acceleration available

## Security Features

### Adversarial Protection
- Text obfuscation detection
- Backtranslation protection
- Synonym attack resilience
- Input validation

### Privacy
- No data storage
- No user tracking
- Local processing option
- Data encryption support

## Maintenance

### Update Schedule
- Monthly security updates
- Quarterly model updates
- Continuous monitoring
- Regular bias audits

### Support
- GitHub issues
- Documentation updates
- Security patches
- Performance improvements 