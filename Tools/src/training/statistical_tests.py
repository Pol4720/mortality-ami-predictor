"""Statistical tests for model comparison and validation.

This module implements rigorous statistical tests to compare different
machine learning models following academic best practices:
- Normality tests (Shapiro-Wilk)
- Parametric tests (t-Student)
- Non-parametric tests (Mann-Whitney U)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StatisticalTestResult:
    """Results from statistical comparison between models."""
    
    model1_name: str
    model2_name: str
    model1_scores: List[float]
    model2_scores: List[float]
    model1_mean: float
    model1_std: float
    model2_mean: float
    model2_std: float
    
    # Normality tests
    model1_is_normal: bool
    model2_is_normal: bool
    model1_shapiro_pvalue: float
    model2_shapiro_pvalue: float
    
    # Comparison test
    test_used: str  # "t-test" or "mann-whitney"
    test_statistic: float
    p_value: float
    significant: bool
    alpha: float
    
    # Effect size
    effect_size: float
    effect_size_interpretation: str
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            "=" * 80,
            f"Statistical Comparison: {self.model1_name} vs {self.model2_name}",
            "=" * 80,
            "",
            "Descriptive Statistics:",
            f"  {self.model1_name}: μ = {self.model1_mean:.4f}, σ = {self.model1_std:.4f}",
            f"  {self.model2_name}: μ = {self.model2_mean:.4f}, σ = {self.model2_std:.4f}",
            "",
            "Normality Tests (Shapiro-Wilk):",
            f"  {self.model1_name}: p-value = {self.model1_shapiro_pvalue:.4f} " +
            f"({'Normal' if self.model1_is_normal else 'Non-normal'})",
            f"  {self.model2_name}: p-value = {self.model2_shapiro_pvalue:.4f} " +
            f"({'Normal' if self.model2_is_normal else 'Non-normal'})",
            "",
            f"Comparison Test: {self.test_used}",
            f"  Test Statistic: {self.test_statistic:.4f}",
            f"  P-value: {self.p_value:.4f}",
            f"  Significant (α={self.alpha}): {'Yes' if self.significant else 'No'}",
            "",
            f"Effect Size: {self.effect_size:.4f} ({self.effect_size_interpretation})",
            "=" * 80,
        ]
        return "\n".join(lines)


def test_normality(
    scores: np.ndarray,
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """Test if scores follow normal distribution using Shapiro-Wilk test.
    
    Args:
        scores: Array of scores to test
        alpha: Significance level
        
    Returns:
        Tuple of (is_normal, p_value)
    """
    if len(scores) < 3:
        warnings.warn(f"Too few samples ({len(scores)}) for Shapiro-Wilk test")
        return False, 0.0
    
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(scores)
    
    # Null hypothesis: data comes from normal distribution
    # If p > alpha, we fail to reject H0 (assume normal)
    is_normal = p_value > alpha
    
    return is_normal, float(p_value)


def paired_t_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """Perform paired t-test to compare two models.
    
    Used when both distributions are normal.
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        alpha: Significance level
        
    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    # Paired t-test (assumes normal distributions)
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Two-tailed test
    is_significant = p_value < alpha
    
    return float(t_stat), float(p_value), is_significant


def mann_whitney_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """Perform Mann-Whitney U test to compare two models.
    
    Used when distributions are not normal (non-parametric alternative to t-test).
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        alpha: Significance level
        
    Returns:
        Tuple of (u_statistic, p_value, is_significant)
    """
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    
    is_significant = p_value < alpha
    
    return float(u_stat), float(p_value), is_significant


def cohens_d(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    return float(d)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def compare_models(
    model1_name: str,
    model1_scores: List[float],
    model2_name: str,
    model2_scores: List[float],
    alpha: float = 0.05
) -> StatisticalTestResult:
    """Perform complete statistical comparison between two models.
    
    Pipeline:
    1. Test normality of both distributions (Shapiro-Wilk)
    2. If both normal: use paired t-test (parametric)
    3. If not normal: use Mann-Whitney U test (non-parametric)
    4. Calculate effect size (Cohen's d)
    
    Args:
        model1_name: Name of first model
        model1_scores: Scores from first model
        model2_name: Name of second model
        model2_scores: Scores from second model
        alpha: Significance level for tests
        
    Returns:
        StatisticalTestResult with complete comparison
    """
    scores1 = np.array(model1_scores)
    scores2 = np.array(model2_scores)
    
    # Descriptive statistics
    mean1, std1 = float(np.mean(scores1)), float(np.std(scores1, ddof=1))
    mean2, std2 = float(np.mean(scores2)), float(np.std(scores2, ddof=1))
    
    # Test normality
    is_normal1, pval1 = test_normality(scores1, alpha)
    is_normal2, pval2 = test_normality(scores2, alpha)
    
    # Choose appropriate test
    if is_normal1 and is_normal2:
        # Both normal: use parametric test
        test_stat, p_value, significant = paired_t_test(scores1, scores2, alpha)
        test_used = "Paired t-test"
    else:
        # At least one non-normal: use non-parametric test
        test_stat, p_value, significant = mann_whitney_test(scores1, scores2, alpha)
        test_used = "Mann-Whitney U test"
    
    # Calculate effect size
    effect = cohens_d(scores1, scores2)
    effect_interp = interpret_effect_size(effect)
    
    return StatisticalTestResult(
        model1_name=model1_name,
        model2_name=model2_name,
        model1_scores=model1_scores,
        model2_scores=model2_scores,
        model1_mean=mean1,
        model1_std=std1,
        model2_mean=mean2,
        model2_std=std2,
        model1_is_normal=is_normal1,
        model2_is_normal=is_normal2,
        model1_shapiro_pvalue=pval1,
        model2_shapiro_pvalue=pval2,
        test_used=test_used,
        test_statistic=test_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect,
        effect_size_interpretation=effect_interp,
    )


def compare_all_models(
    model_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[Tuple[str, str], StatisticalTestResult]:
    """Compare all pairs of models statistically.
    
    Args:
        model_scores: Dictionary mapping model_name -> list of scores
        alpha: Significance level
        
    Returns:
        Dictionary mapping (model1, model2) -> StatisticalTestResult
    """
    model_names = list(model_scores.keys())
    results = {}
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i + 1:]:
            result = compare_models(
                model1_name=name1,
                model1_scores=model_scores[name1],
                model2_name=name2,
                model2_scores=model_scores[name2],
                alpha=alpha
            )
            results[(name1, name2)] = result
    
    return results


def plot_model_comparison(
    result: StatisticalTestResult,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive visualization of model comparison.
    
    Args:
        result: Statistical test result
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Comparison: {result.model1_name} vs {result.model2_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    data_to_plot = [result.model1_scores, result.model2_scores]
    bp = ax1.boxplot(data_to_plot, labels=[result.model1_name, result.model2_name],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Score (AUROC)')
    ax1.set_title('Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    positions = [1, 2]
    parts1 = ax2.violinplot([result.model1_scores], positions=[1], 
                            showmeans=True, showmedians=True)
    parts2 = ax2.violinplot([result.model2_scores], positions=[2], 
                            showmeans=True, showmedians=True)
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    for pc in parts2['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([result.model1_name, result.model2_name])
    ax2.set_ylabel('Score (AUROC)')
    ax2.set_title('Score Distribution (Violin Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram comparison
    ax3 = axes[1, 0]
    ax3.hist(result.model1_scores, bins=15, alpha=0.5, label=result.model1_name, 
            color='blue', edgecolor='black')
    ax3.hist(result.model2_scores, bins=15, alpha=0.5, label=result.model2_name, 
            color='green', edgecolor='black')
    ax3.axvline(result.model1_mean, color='blue', linestyle='--', linewidth=2, 
                label=f'{result.model1_name} mean')
    ax3.axvline(result.model2_mean, color='green', linestyle='--', linewidth=2, 
                label=f'{result.model2_name} mean')
    ax3.set_xlabel('Score (AUROC)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Histograms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical test results (text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    text_content = [
        "Statistical Test Results:",
        "",
        f"{result.model1_name}:",
        f"  μ = {result.model1_mean:.4f}, σ = {result.model1_std:.4f}",
        f"  Normal: {result.model1_is_normal} (p={result.model1_shapiro_pvalue:.4f})",
        "",
        f"{result.model2_name}:",
        f"  μ = {result.model2_mean:.4f}, σ = {result.model2_std:.4f}",
        f"  Normal: {result.model2_is_normal} (p={result.model2_shapiro_pvalue:.4f})",
        "",
        f"Test: {result.test_used}",
        f"  Statistic: {result.test_statistic:.4f}",
        f"  P-value: {result.p_value:.4f}",
        f"  Significant (α={result.alpha}): {result.significant}",
        "",
        f"Effect Size (Cohen's d): {result.effect_size:.4f}",
        f"  Interpretation: {result.effect_size_interpretation}",
        "",
        "Conclusion:",
        f"  {'Significant' if result.significant else 'No significant'} difference",
        f"  Winner: {result.model1_name if result.model1_mean > result.model2_mean else result.model2_name}",
    ]
    
    ax4.text(0.1, 0.9, '\n'.join(text_content), transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_matrix(
    results: Dict[Tuple[str, str], StatisticalTestResult]
) -> plt.Figure:
    """Create a matrix showing pairwise model comparisons.
    
    Args:
        results: Dictionary of pairwise comparison results
        
    Returns:
        Matplotlib figure
    """
    # Extract all unique model names
    all_names = set()
    for (name1, name2) in results.keys():
        all_names.add(name1)
        all_names.add(name2)
    model_names = sorted(list(all_names))
    n_models = len(model_names)
    
    # Create matrices for p-values and effect sizes
    p_value_matrix = np.ones((n_models, n_models))
    effect_size_matrix = np.zeros((n_models, n_models))
    
    name_to_idx = {name: i for i, name in enumerate(model_names)}
    
    for (name1, name2), result in results.items():
        i, j = name_to_idx[name1], name_to_idx[name2]
        p_value_matrix[i, j] = result.p_value
        p_value_matrix[j, i] = result.p_value
        effect_size_matrix[i, j] = result.effect_size
        effect_size_matrix[j, i] = -result.effect_size
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # P-value heatmap
    im1 = ax1.imshow(p_value_matrix, cmap='RdYlGn', vmin=0, vmax=0.1)
    ax1.set_xticks(range(n_models))
    ax1.set_yticks(range(n_models))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_yticklabels(model_names)
    ax1.set_title('P-values Matrix\n(Green = Significant difference)')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax1.text(j, i, f'{p_value_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=ax1)
    
    # Effect size heatmap
    im2 = ax2.imshow(effect_size_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_xticks(range(n_models))
    ax2.set_yticks(range(n_models))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_yticklabels(model_names)
    ax2.set_title('Effect Size Matrix (Cohen\'s d)\n(Red = Model 1 better, Blue = Model 2 better)')
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = ax2.text(j, i, f'{effect_size_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    return fig
