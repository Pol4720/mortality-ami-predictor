"""Resampling methods for robust model evaluation.

This module implements Bootstrap and Jackknife resampling techniques
for estimating model performance with confidence intervals in the test phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone


@dataclass
class ResamplingResult:
    """Results from resampling evaluation."""
    
    method: str  # "bootstrap" or "jackknife"
    scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_iterations: int
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 70,
            f"Resampling Results ({self.method.capitalize()})",
            "=" * 70,
            f"Iterations: {self.n_iterations}",
            f"Mean Score: {self.mean_score:.4f}",
            f"Std Score: {self.std_score:.4f}",
            f"Confidence Interval ({self.confidence_level*100:.0f}%): "
            f"[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]",
            "=" * 70,
        ]
        return "\n".join(lines)


def bootstrap_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    scoring_func: Optional[Callable] = None,
    retrain: bool = False,
    random_state: Optional[int] = None,
) -> ResamplingResult:
    """Perform bootstrap resampling for model evaluation.
    
    Bootstrap: Resample test set WITH replacement, evaluate model.
    This estimates the variability of the model's performance.
    
    Args:
        model: Fitted model or pipeline
        X_test: Test features
        y_test: Test labels
        X_train: Training features (needed if retrain=True)
        y_train: Training labels (needed if retrain=True)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level for CI
        scoring_func: Scoring function (default: ROC AUC for binary classification)
        retrain: If True, retrain model on bootstrap sample before evaluation
        random_state: Random seed
        
    Returns:
        ResamplingResult with bootstrap statistics
    """
    if scoring_func is None:
        from sklearn.metrics import roc_auc_score
        scoring_func = roc_auc_score
    
    rng = np.random.RandomState(random_state)
    n_samples = len(X_test)
    scores = []
    
    for i in range(n_iterations):
        # Bootstrap sample (with replacement)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test[indices]
        
        if retrain and X_train is not None and y_train is not None:
            # Retrain on bootstrap training sample
            train_indices = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_train_boot = X_train[train_indices]
            y_train_boot = y_train[train_indices]
            
            model_boot = clone(model)
            model_boot.fit(X_train_boot, y_train_boot)
            
            # Predict on bootstrap test sample
            if hasattr(model_boot, "predict_proba"):
                y_pred = model_boot.predict_proba(X_boot)[:, 1]
            else:
                y_pred = model_boot.decision_function(X_boot)
        else:
            # Use already fitted model
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_boot)[:, 1]
            else:
                y_pred = model.decision_function(X_boot)
        
        # Calculate score
        try:
            score = scoring_func(y_boot, y_pred)
            scores.append(score)
        except Exception:
            # Skip if calculation fails (e.g., only one class in bootstrap sample)
            continue
    
    scores = np.array(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1))
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(scores, lower_percentile))
    ci_upper = float(np.percentile(scores, upper_percentile))
    
    return ResamplingResult(
        method="bootstrap",
        scores=scores.tolist(),
        mean_score=mean_score,
        std_score=std_score,
        confidence_interval=(ci_lower, ci_upper),
        confidence_level=confidence_level,
        n_iterations=len(scores),
    )


def jackknife_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    scoring_func: Optional[Callable] = None,
    retrain: bool = False,
    confidence_level: float = 0.95,
) -> ResamplingResult:
    """Perform jackknife (leave-one-out) resampling for model evaluation.
    
    Jackknife: Remove one sample at a time, evaluate model.
    This estimates the bias and variance of the estimator.
    
    Args:
        model: Fitted model or pipeline
        X_test: Test features
        y_test: Test labels
        X_train: Training features (needed if retrain=True)
        y_train: Training labels (needed if retrain=True)
        scoring_func: Scoring function (default: ROC AUC for binary classification)
        retrain: If True, retrain model excluding one sample
        confidence_level: Confidence level for CI
        
    Returns:
        ResamplingResult with jackknife statistics
    """
    if scoring_func is None:
        from sklearn.metrics import roc_auc_score
        scoring_func = roc_auc_score
    
    n_samples = len(X_test)
    scores = []
    
    for i in range(n_samples):
        # Leave-one-out: use all samples except i
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        
        X_loo = X_test[mask]
        y_loo = y_test[mask]
        
        if retrain and X_train is not None and y_train is not None:
            # Retrain excluding one training sample
            # Note: For test-phase Jackknife, we typically don't retrain
            # but this is available if needed
            train_mask = np.ones(len(X_train), dtype=bool)
            if i < len(X_train):
                train_mask[i] = False
            
            X_train_loo = X_train[train_mask]
            y_train_loo = y_train[train_mask]
            
            model_loo = clone(model)
            model_loo.fit(X_train_loo, y_train_loo)
            
            # Predict on leave-one-out test sample
            if hasattr(model_loo, "predict_proba"):
                y_pred = model_loo.predict_proba(X_loo)[:, 1]
            else:
                y_pred = model_loo.decision_function(X_loo)
        else:
            # Use already fitted model
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_loo)[:, 1]
            else:
                y_pred = model.decision_function(X_loo)
        
        # Calculate score
        try:
            score = scoring_func(y_loo, y_pred)
            scores.append(score)
        except Exception:
            # Skip if calculation fails
            continue
    
    scores = np.array(scores)
    n_iterations = len(scores)
    mean_score = float(np.mean(scores))
    
    # Jackknife variance estimation
    # Var = (n-1)/n * sum((score_i - mean)^2)
    jackknife_var = ((n_iterations - 1) / n_iterations) * np.sum((scores - mean_score) ** 2)
    std_score = float(np.sqrt(jackknife_var))
    
    # Calculate confidence interval using t-distribution
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence_level) / 2, n_iterations - 1)
    margin = t_value * std_score / np.sqrt(n_iterations)
    ci_lower = mean_score - margin
    ci_upper = mean_score + margin
    
    return ResamplingResult(
        method="jackknife",
        scores=scores.tolist(),
        mean_score=mean_score,
        std_score=std_score,
        confidence_interval=(float(ci_lower), float(ci_upper)),
        confidence_level=confidence_level,
        n_iterations=n_iterations,
    )


def plot_resampling_results(
    results: List[ResamplingResult],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot resampling results for comparison.
    
    Args:
        results: List of ResamplingResult objects
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(7 * n_results, 5))
    
    if n_results == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        # Histogram of scores
        ax.hist(result.scores, bins=30, alpha=0.7, color='steelblue', 
               edgecolor='black')
        
        # Mean line
        ax.axvline(result.mean_score, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {result.mean_score:.4f}')
        
        # Confidence interval
        ax.axvline(result.confidence_interval[0], color='green', 
                  linestyle=':', linewidth=2, 
                  label=f'CI {result.confidence_level*100:.0f}%')
        ax.axvline(result.confidence_interval[1], color='green', 
                  linestyle=':', linewidth=2)
        
        # Fill CI region
        ax.axvspan(result.confidence_interval[0], result.confidence_interval[1], 
                  alpha=0.2, color='green')
        
        ax.set_xlabel('Score (AUROC)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{result.method.capitalize()} Results\n'
                    f'(n={result.n_iterations})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def combined_resampling_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    scoring_func: Optional[Callable] = None,
    random_state: Optional[int] = None,
) -> Tuple[ResamplingResult, ResamplingResult]:
    """Perform both Bootstrap and Jackknife evaluation.
    
    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test labels
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CIs
        scoring_func: Scoring function
        random_state: Random seed
        
    Returns:
        Tuple of (bootstrap_result, jackknife_result)
    """
    # Bootstrap
    bootstrap_result = bootstrap_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        n_iterations=n_bootstrap,
        confidence_level=confidence_level,
        scoring_func=scoring_func,
        retrain=False,
        random_state=random_state,
    )
    
    # Jackknife
    jackknife_result = jackknife_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scoring_func=scoring_func,
        retrain=False,
        confidence_level=confidence_level,
    )
    
    return bootstrap_result, jackknife_result
