"""Learning curve generation for model evaluation.

This module provides functions to generate and visualize learning curves
to assess model performance as training size varies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import learning_curve as sklearn_learning_curve


@dataclass
class LearningCurveResult:
    """Results from learning curve analysis."""
    
    train_sizes: List[int]
    train_scores_mean: List[float]
    train_scores_std: List[float]
    val_scores_mean: List[float]
    val_scores_std: List[float]
    train_scores_all: np.ndarray  # Shape: (n_sizes, n_cv_folds)
    val_scores_all: np.ndarray
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 70,
            "Learning Curve Results",
            "=" * 70,
            f"Training sizes evaluated: {len(self.train_sizes)}",
            f"Range: {min(self.train_sizes)} to {max(self.train_sizes)} samples",
            "",
            "Final performance:",
            f"  Train score: {self.train_scores_mean[-1]:.4f} ± {self.train_scores_std[-1]:.4f}",
            f"  Val score: {self.val_scores_mean[-1]:.4f} ± {self.val_scores_std[-1]:.4f}",
            f"  Gap: {abs(self.train_scores_mean[-1] - self.val_scores_mean[-1]):.4f}",
            "=" * 70,
        ]
        return "\n".join(lines)


def generate_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
) -> LearningCurveResult:
    """Generate learning curve for a model.
    
    Args:
        model: Model or pipeline to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        train_sizes: Array of training sizes to evaluate (default: 10 points from 10% to 100%)
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_state: Random seed
        
    Returns:
        LearningCurveResult with learning curve data
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Generate learning curve
    train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        shuffle=True,
    )
    
    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    return LearningCurveResult(
        train_sizes=train_sizes_abs.tolist(),
        train_scores_mean=train_scores_mean.tolist(),
        train_scores_std=train_scores_std.tolist(),
        val_scores_mean=val_scores_mean.tolist(),
        val_scores_std=val_scores_std.tolist(),
        train_scores_all=train_scores,
        val_scores_all=val_scores,
    )


def plot_learning_curve(
    result: LearningCurveResult,
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot learning curve.
    
    Args:
        result: LearningCurveResult object
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_sizes = result.train_sizes
    train_mean = result.train_scores_mean
    train_std = result.train_scores_std
    val_mean = result.val_scores_mean
    val_std = result.val_scores_std
    
    # Plot training scores
    ax.plot(train_sizes, train_mean, 'o-', color='blue', 
            label='Training score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes, 
                     np.array(train_mean) - np.array(train_std),
                     np.array(train_mean) + np.array(train_std),
                     alpha=0.2, color='blue')
    
    # Plot validation scores
    ax.plot(train_sizes, val_mean, 'o-', color='green', 
            label='Validation score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes,
                     np.array(val_mean) - np.array(val_std),
                     np.array(val_mean) + np.array(val_std),
                     alpha=0.2, color='green')
    
    # Labels and formatting
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score (AUROC)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final scores
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = abs(final_train - final_val)
    
    textstr = f'Final Train: {final_train:.3f}\n' \
              f'Final Val: {final_val:.3f}\n' \
              f'Gap: {gap:.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_learning_curves(
    results: dict[str, LearningCurveResult],
    title: str = "Learning Curves Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple learning curves for model comparison.
    
    Args:
        results: Dictionary mapping model_name -> LearningCurveResult
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot 1: All training curves
    for (name, result), color in zip(results.items(), colors):
        ax1.plot(result.train_sizes, result.train_scores_mean, 
                'o-', color=color, label=name, linewidth=2, markersize=6)
        ax1.fill_between(result.train_sizes,
                         np.array(result.train_scores_mean) - np.array(result.train_scores_std),
                         np.array(result.train_scores_mean) + np.array(result.train_scores_std),
                         alpha=0.15, color=color)
    
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('Training Score (AUROC)', fontsize=12)
    ax1.set_title('Training Curves', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All validation curves
    for (name, result), color in zip(results.items(), colors):
        ax2.plot(result.train_sizes, result.val_scores_mean, 
                'o-', color=color, label=name, linewidth=2, markersize=6)
        ax2.fill_between(result.train_sizes,
                         np.array(result.val_scores_mean) - np.array(result.val_scores_std),
                         np.array(result.val_scores_mean) + np.array(result.val_scores_std),
                         alpha=0.15, color=color)
    
    ax2.set_xlabel('Training Set Size', fontsize=12)
    ax2.set_ylabel('Validation Score (AUROC)', fontsize=12)
    ax2.set_title('Validation Curves', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def diagnose_learning_curve(result: LearningCurveResult) -> dict:
    """Diagnose issues from learning curve.
    
    Args:
        result: LearningCurveResult to analyze
        
    Returns:
        Dictionary with diagnostic information
    """
    final_train = result.train_scores_mean[-1]
    final_val = result.val_scores_mean[-1]
    gap = abs(final_train - final_val)
    
    # Analyze trends
    train_improving = result.train_scores_mean[-1] > result.train_scores_mean[0]
    val_improving = result.val_scores_mean[-1] > result.val_scores_mean[0]
    
    # Convergence check (is validation score stabilizing?)
    if len(result.val_scores_mean) >= 3:
        recent_val_std = np.std(result.val_scores_mean[-3:])
        converged = recent_val_std < 0.01
    else:
        converged = False
    
    # Diagnose problems
    diagnosis = {
        'final_train_score': final_train,
        'final_val_score': final_val,
        'gap': gap,
        'train_improving': train_improving,
        'val_improving': val_improving,
        'converged': converged,
        'issues': [],
        'recommendations': [],
    }
    
    # High bias (underfitting)
    if final_train < 0.75 and final_val < 0.75:
        diagnosis['issues'].append('High bias (underfitting)')
        diagnosis['recommendations'].append('Use more complex model')
        diagnosis['recommendations'].append('Add more features')
        diagnosis['recommendations'].append('Reduce regularization')
    
    # High variance (overfitting)
    if gap > 0.1:
        diagnosis['issues'].append('High variance (overfitting)')
        diagnosis['recommendations'].append('Get more training data')
        diagnosis['recommendations'].append('Reduce model complexity')
        diagnosis['recommendations'].append('Increase regularization')
        diagnosis['recommendations'].append('Use dropout or other regularization techniques')
    
    # Not converged
    if not converged and val_improving:
        diagnosis['issues'].append('Model not converged')
        diagnosis['recommendations'].append('Collect more training data')
        diagnosis['recommendations'].append('Continue training')
    
    # Good fit
    if gap < 0.05 and final_val > 0.80:
        diagnosis['issues'].append('Good fit!')
        diagnosis['recommendations'].append('Model is performing well')
    
    return diagnosis


def print_diagnosis(diagnosis: dict):
    """Print learning curve diagnosis in readable format.
    
    Args:
        diagnosis: Dictionary from diagnose_learning_curve
    """
    print("=" * 70)
    print("LEARNING CURVE DIAGNOSIS")
    print("=" * 70)
    print(f"\nFinal Scores:")
    print(f"  Training:   {diagnosis['final_train_score']:.4f}")
    print(f"  Validation: {diagnosis['final_val_score']:.4f}")
    print(f"  Gap:        {diagnosis['gap']:.4f}")
    
    print(f"\nTrends:")
    print(f"  Training improving:   {diagnosis['train_improving']}")
    print(f"  Validation improving: {diagnosis['val_improving']}")
    print(f"  Converged:            {diagnosis['converged']}")
    
    if diagnosis['issues']:
        print(f"\nIssues Detected:")
        for issue in diagnosis['issues']:
            print(f"  ⚠️  {issue}")
    
    if diagnosis['recommendations']:
        print(f"\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  💡 {rec}")
    
    print("=" * 70)
