"""Resampling methods for robust model evaluation in test phase.

This module implements Bootstrap and Jackknife resampling techniques
for estimating model performance with confidence intervals on the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone


@dataclass
class ResamplingResult:
    """Results from resampling evaluation with multiple metrics."""
    
    method: str  # "bootstrap" or "jackknife"
    metrics: Dict[str, List[float]]  # metric_name -> list of scores
    mean_scores: Dict[str, float]  # metric_name -> mean
    std_scores: Dict[str, float]  # metric_name -> std
    confidence_intervals: Dict[str, Tuple[float, float]]  # metric_name -> (lower, upper)
    confidence_level: float
    n_iterations: int
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 70,
            f"Resampling Results ({self.method.capitalize()})",
            "=" * 70,
            f"Iterations: {self.n_iterations}",
            f"Confidence Level: {self.confidence_level*100:.0f}%",
            "-" * 70,
        ]
        for metric_name in sorted(self.mean_scores.keys()):
            mean = self.mean_scores[metric_name]
            std = self.std_scores[metric_name]
            ci = self.confidence_intervals[metric_name]
            lines.append(
                f"{metric_name:12s}: {mean:.4f} ± {std:.4f}  "
                f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


def compute_resampling_metrics(y_true, y_pred_proba) -> Dict[str, float]:
    """Compute all classification metrics for resampling.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score, brier_score_loss
    )
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'auprc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'brier': brier_score_loss(y_true, y_pred_proba),
    }
    
    return metrics


def bootstrap_evaluation(
    model,
    X_test,  # Accept DataFrame or array
    y_test,  # Accept Series or array
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ResamplingResult:
    """Perform bootstrap resampling for model evaluation on test set.
    
    Bootstrap: Resample test set WITH replacement, evaluate model.
    This estimates the variability of the model's performance.
    
    Args:
        model: Fitted model or pipeline
        X_test: Test features (DataFrame or array)
        y_test: Test labels (Series or array)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level for CI
        random_state: Random seed
        progress_callback: Optional callback function for progress updates
        
    Returns:
        ResamplingResult with bootstrap statistics for all metrics
    """
    # Convert to pandas if numpy array
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)
    
    rng = np.random.RandomState(random_state)
    n_samples = len(X_test)
    
    # Storage for all metrics across iterations
    all_metrics = {
        'auroc': [], 'auprc': [], 'accuracy': [],
        'precision': [], 'recall': [], 'f1': [], 'brier': []
    }
    
    msg = f"🔄 Bootstrap Evaluation ({n_iterations} iterations)..."
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    for i in range(n_iterations):
        if (i + 1) % 100 == 0:
            msg = f"  Progress: {i+1}/{n_iterations}"
            print(msg)
            if progress_callback:
                progress_callback(msg)
        
        # Bootstrap sample (with replacement)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        # Use already fitted model (no retraining on test set!)
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_boot)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_boot)
            
            # Calculate all metrics for this iteration
            iter_metrics = compute_resampling_metrics(y_boot.values, y_pred_proba)
            
            # Store metrics
            for metric_name, metric_value in iter_metrics.items():
                all_metrics[metric_name].append(metric_value)
                
        except Exception as e:
            # Skip if calculation fails (e.g., only one class in bootstrap sample)
            continue
    
    # Compute statistics for each metric
    mean_scores = {}
    std_scores = {}
    confidence_intervals = {}
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    for metric_name in all_metrics.keys():
        scores_array = np.array(all_metrics[metric_name])
        if len(scores_array) > 0:
            mean_scores[metric_name] = float(np.mean(scores_array))
            std_scores[metric_name] = float(np.std(scores_array, ddof=1))
            ci_lower = float(np.percentile(scores_array, lower_percentile))
            ci_upper = float(np.percentile(scores_array, upper_percentile))
            confidence_intervals[metric_name] = (ci_lower, ci_upper)
    
    result = ResamplingResult(
        method="bootstrap",
        metrics=all_metrics,
        mean_scores=mean_scores,
        std_scores=std_scores,
        confidence_intervals=confidence_intervals,
        confidence_level=confidence_level,
        n_iterations=len(all_metrics['auroc']),
    )
    
    msg = f"✅ Bootstrap complete with {result.n_iterations} successful iterations"
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    # Print summary for all metrics
    for metric_name in sorted(mean_scores.keys()):
        msg = f"   {metric_name}: {mean_scores[metric_name]:.4f} ± {std_scores[metric_name]:.4f} | 95% CI: [{confidence_intervals[metric_name][0]:.4f}, {confidence_intervals[metric_name][1]:.4f}]"
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    return result


def jackknife_evaluation(
    model,
    X_test,  # Accept DataFrame or array
    y_test,  # Accept Series or array
    confidence_level: float = 0.95,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ResamplingResult:
    """Perform jackknife (leave-one-out) resampling for model evaluation.
    
    Jackknife: Remove one sample at a time, evaluate model.
    This estimates the bias and variance of the estimator.
    
    Args:
        model: Fitted model or pipeline
        X_test: Test features (DataFrame or array)
        y_test: Test labels (Series or array)
        confidence_level: Confidence level for CI
        progress_callback: Optional callback function for progress updates
        
    Returns:
        ResamplingResult with jackknife statistics for all metrics
    """
    # Convert to pandas if numpy array
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)
    
    n_samples = len(X_test)
    
    # Storage for all metrics across iterations
    all_metrics = {
        'auroc': [], 'auprc': [], 'accuracy': [],
        'precision': [], 'recall': [], 'f1': [], 'brier': []
    }
    
    msg = f"🔄 Jackknife Evaluation (Leave-One-Out, n={n_samples})..."
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            msg = f"  Progress: {i+1}/{n_samples}"
            print(msg)
            if progress_callback:
                progress_callback(msg)
        
        # Leave-one-out: use all samples except i
        X_loo = X_test.drop(X_test.index[i])
        y_loo = y_test.drop(y_test.index[i])
        
        # Use already fitted model (no retraining on test set!)
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_loo)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_loo)
            
            # Calculate all metrics for this iteration
            iter_metrics = compute_resampling_metrics(y_loo.values, y_pred_proba)
            
            # Store metrics
            for metric_name, metric_value in iter_metrics.items():
                all_metrics[metric_name].append(metric_value)
                
        except Exception:
            # Skip if calculation fails
            continue
    
    # Compute statistics for each metric
    mean_scores = {}
    std_scores = {}
    confidence_intervals = {}
    
    from scipy import stats
    
    for metric_name in all_metrics.keys():
        scores_array = np.array(all_metrics[metric_name])
        if len(scores_array) > 0:
            n_iterations = len(scores_array)
            mean_score = float(np.mean(scores_array))
            
            # Jackknife variance estimation
            # Var = (n-1)/n * sum((score_i - mean)^2)
            jackknife_var = ((n_iterations - 1) / n_iterations) * np.sum((scores_array - mean_score) ** 2)
            std_score = float(np.sqrt(jackknife_var))
            
            # Calculate confidence interval using t-distribution
            t_value = stats.t.ppf((1 + confidence_level) / 2, n_iterations - 1)
            margin = t_value * std_score / np.sqrt(n_iterations)
            ci_lower = mean_score - margin
            ci_upper = mean_score + margin
            
            mean_scores[metric_name] = mean_score
            std_scores[metric_name] = std_score
            confidence_intervals[metric_name] = (float(ci_lower), float(ci_upper))
    
    result = ResamplingResult(
        method="jackknife",
        metrics=all_metrics,
        mean_scores=mean_scores,
        std_scores=std_scores,
        confidence_intervals=confidence_intervals,
        confidence_level=confidence_level,
        n_iterations=len(all_metrics['auroc']),
    )
    
    msg = f"✅ Jackknife complete with {result.n_iterations} successful iterations"
    print(msg)
    if progress_callback:
        progress_callback(msg)
    
    # Print summary for all metrics
    for metric_name in sorted(mean_scores.keys()):
        msg = f"   {metric_name}: {mean_scores[metric_name]:.4f} ± {std_scores[metric_name]:.4f} | 95% CI: [{confidence_intervals[metric_name][0]:.4f}, {confidence_intervals[metric_name][1]:.4f}]"
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    return result


def plot_resampling_results(
    results: List[ResamplingResult],
    metric: str = 'auroc',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot resampling results for comparison.
    
    Args:
        results: List of ResamplingResult objects
        metric: Which metric to plot (default: 'auroc')
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(7 * n_results, 5))
    
    if n_results == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        # Get scores for the selected metric
        scores = result.metrics.get(metric, [])
        mean_score = result.mean_scores.get(metric, 0)
        ci = result.confidence_intervals.get(metric, (0, 0))
        
        if not scores:
            continue
            
        # Histogram of scores
        ax.hist(scores, bins=30, alpha=0.7, color='steelblue', 
               edgecolor='black')
        
        # Mean line
        ax.axvline(mean_score, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_score:.4f}')
        
        # Confidence interval
        ax.axvline(ci[0], color='green', 
                  linestyle=':', linewidth=2, 
                  label=f'CI {result.confidence_level*100:.0f}%')
        ax.axvline(ci[1], color='green', 
                  linestyle=':', linewidth=2)
        
        # Fill CI region
        ax.axvspan(ci[0], ci[1], alpha=0.2, color='green')
        
        ax.set_xlabel(f'Score ({metric.upper()})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{result.method.capitalize()} Results\n'
                    f'(n={result.n_iterations})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Resampling plot saved: {save_path}")
    
    return fig


def combined_resampling_evaluation(
    model,
    X_test,  # Accept DataFrame or array
    y_test,  # Accept Series or array
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[ResamplingResult, ResamplingResult]:
    """Perform both Bootstrap and Jackknife evaluation.
    
    Args:
        model: Fitted model
        X_test: Test features (DataFrame or array)
        y_test: Test labels (Series or array)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CIs
        random_state: Random seed
        progress_callback: Optional callback function for progress updates
        
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
        random_state=random_state,
        progress_callback=progress_callback,
    )
    
    # Jackknife
    jackknife_result = jackknife_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        confidence_level=confidence_level,
        progress_callback=progress_callback,
    )
    
    return bootstrap_result, jackknife_result
