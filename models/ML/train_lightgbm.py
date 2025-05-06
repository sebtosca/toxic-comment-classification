"""Train LightGBM model for toxic comment classification. GradientBoosting"""

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from typing import Tuple, Dict, Any

def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any] = None
) -> Tuple[MultiOutputClassifier, Dict[str, float]]:
    """Train a LightGBM model with Bayesian optimization for multilabel classification.
    
    Args:
        X: Input features (BERT embeddings)
        y: Target labels
        params: LightGBM parameters
        
    Returns:
        Trained model and validation metrics
    """
    # Initialize
    base_model = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
    
    # Parameters and their ranges
    param_space = {
        'estimator__num_leaves': Integer(20, 150),
        'estimator__max_depth': Integer(3, 10),
        'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'estimator__n_estimators': Integer(100, 500),
        'estimator__subsample': Real(0.6, 1.0),
        'estimator__colsample_bytree': Real(0.6, 1.0)
    }
    
    # Multi-output classifier
    multi_target_model = MultiOutputClassifier(base_model)
    
    # I am using stratified cross-validation
    cv_strategy = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Scoring
    def custom_macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')
    
    # Bayesian optimization - more optimal 
    opt = BayesSearchCV(
        estimator=multi_target_model,
        search_spaces=param_space,
        cv=cv_strategy,
        n_iter=25,
        scoring=make_scorer(custom_macro_f1),
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    
    # Train model with optimization
    print("[INFO] Starting Bayesian hyperparameter optimization...")
    opt.fit(X, y)
    print("[INFO] Optimization completed.")
    
    # Get validation metrics
    val_metrics = {
        'best_score': opt.best_score_,
        'best_params': opt.best_params_
    }
    
    return opt.best_estimator_, val_metrics