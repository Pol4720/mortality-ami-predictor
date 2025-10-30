"""Evaluation report generation utilities."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from ..config import CONFIG
from .metrics import compute_classification_metrics
from .calibration import plot_calibration_curve
from .decision_curves import decision_curve_analysis


REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports"
)
os.makedirs(REPORTS_DIR, exist_ok=True)

FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models"
)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, name: str = "model", threshold: float = 0.5) -> str:
    """Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        threshold: Classification threshold
        
    Returns:
        Path to saved figure
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}', fontsize=14)
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    
    path = os.path.join(FIGURES_DIR, f"confusion_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, name: str = "model") -> str:
    """Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        
    Returns:
        Path to saved figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(FIGURES_DIR, f"roc_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    
    return path



def generate_evaluation_report(
    metrics: Dict[str, float],
    task_name: str = "model",
    calibration_path: Optional[str] = None,
    dca_path: Optional[str] = None,
) -> str:
    """Generate evaluation report and save to CSV.
    
    Args:
        metrics: Dictionary of metrics
        task_name: Task name for filename
        calibration_path: Path to calibration plot (optional)
        dca_path: Path to DCA plot (optional)
        
    Returns:
        Path to saved report CSV
    """
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save metrics
    csv_path = os.path.join(REPORTS_DIR, f"evaluation_metrics_{task_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # Create summary report
    report_lines = [
        f"# Evaluation Report: {task_name}",
        "",
        "## Metrics",
        "",
    ]
    
    for metric_name, value in metrics.items():
        report_lines.append(f"- **{metric_name}**: {value:.4f}")
    
    report_lines.extend(["", "## Plots", ""])
    
    if calibration_path:
        report_lines.append(f"- Calibration curve: `{calibration_path}`")
    
    if dca_path:
        report_lines.append(f"- Decision curve analysis: `{dca_path}`")
    
    # Save markdown report
    md_path = os.path.join(REPORTS_DIR, f"evaluation_report_{task_name}.md")
    with open(md_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return csv_path


def evaluate_main(argv: Optional[list] = None) -> None:
    """Evaluate trained model on test set with Bootstrap and Jackknife (FASE 2).
    
    This function performs:
    1. Standard evaluation metrics (accuracy, AUROC, etc.)
    2. FASE 2: Bootstrap resampling (1000 iterations)
    3. FASE 2: Jackknife resampling (leave-one-out)
    4. Confidence intervals at 95%
    
    Args:
        argv: Command line arguments (for testing)
    """
    parser = argparse.ArgumentParser(description="Evaluate saved best model on hold-out test set.")
    parser.add_argument("--data", type=str, default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--task", type=str, choices=["mortality", "arrhythmia"], default="mortality")
    args = parser.parse_args(argv)

    if not args.data:
        raise ValueError("DATASET_PATH not provided.")

    model_path = os.path.join(MODELS_DIR, f"best_classifier_{args.task}.joblib")
    test_path = os.path.join(MODELS_DIR, f"testset_{args.task}.parquet")
    
    if not (os.path.exists(model_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Model or test set not found. Train first.\n"
            f"Expected: {model_path} and {test_path}"
        )

    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"\n🔍 Debugging Info:")
    print(f"  Test set shape: {test_df.shape}")
    print(f"  Test set columns (first 10): {test_df.columns.tolist()[:10]}")

    target = CONFIG.target_column if args.task == "mortality" else CONFIG.arrhythmia_column
    if target not in test_df.columns:
        raise KeyError(f"Target column '{target}' not found in test set.")
    
    print(f"  Target column: {target}")
    print(f"  Target distribution: {pd.Series(test_df[target]).value_counts().to_dict()}")

    # Prepare features and target
    X = test_df.drop(columns=[target])
    y = test_df[target]  # Keep as Series, not .values
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    # Get predictions
    prob = model.predict_proba(X)[:, 1]
    y_pred = (prob >= 0.5).astype(int)
    
    print(f"  Predictions shape: {prob.shape}")
    print(f"  Probability range: [{prob.min():.4f}, {prob.max():.4f}]")

    # Compute standard metrics (convert to numpy for metrics computation)
    metrics = compute_classification_metrics(y.values, prob)
    
    # =========================================================================
    # FASE 2: Bootstrap and Jackknife Resampling
    # =========================================================================
    print(f"\n📊 FASE 2: Resampling Evaluation (Bootstrap + Jackknife)")
    
    from .resampling import bootstrap_evaluation, jackknife_evaluation, plot_resampling_results
    from ..config import RANDOM_SEED
    
    # Bootstrap (with replacement)
    boot_res = bootstrap_evaluation(
        model=model,
        X_test=X,  # Pass DataFrame directly
        y_test=y,  # Pass Series directly
        n_iterations=1000,
        confidence_level=0.95,
        random_state=RANDOM_SEED,
    )
    
    # Jackknife (leave-one-out)
    jack_res = jackknife_evaluation(
        model=model,
        X_test=X,  # Pass DataFrame directly
        y_test=y,  # Pass Series directly
        confidence_level=0.95,
    )
    
    # Save resampling plots (plot AUROC by default)
    fig = plot_resampling_results([boot_res, jack_res], metric='auroc')
    resampling_path = os.path.join(FIGURES_DIR, f"resampling_{args.task}.png")
    fig.savefig(resampling_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n  📊 Resampling plot saved: {resampling_path}")
    
    # Add ALL resampling metrics to the metrics dict
    # Bootstrap metrics
    for metric_name, mean_val in boot_res.mean_scores.items():
        metrics[f'bootstrap_{metric_name}_mean'] = mean_val
        metrics[f'bootstrap_{metric_name}_std'] = boot_res.std_scores[metric_name]
        metrics[f'bootstrap_{metric_name}_ci_lower'] = boot_res.confidence_intervals[metric_name][0]
        metrics[f'bootstrap_{metric_name}_ci_upper'] = boot_res.confidence_intervals[metric_name][1]
    
    # Jackknife metrics
    for metric_name, mean_val in jack_res.mean_scores.items():
        metrics[f'jackknife_{metric_name}_mean'] = mean_val
        metrics[f'jackknife_{metric_name}_std'] = jack_res.std_scores[metric_name]
        metrics[f'jackknife_{metric_name}_ci_lower'] = jack_res.confidence_intervals[metric_name][0]
        metrics[f'jackknife_{metric_name}_ci_upper'] = jack_res.confidence_intervals[metric_name][1]
    
    # =========================================================================
    # Standard Plots
    # =========================================================================
    print(f"\n📈 Generating standard evaluation plots...")
    
    # Generate plots (convert y to numpy for plotting functions)
    y_np = y.values
    calib_path = plot_calibration_curve(y_np, prob, name=args.task)
    dca_path = decision_curve_analysis(y_np, prob, name=args.task)
    confusion_path = plot_confusion_matrix(y_np, prob, name=args.task)
    roc_path = plot_roc_curve(y_np, prob, name=args.task)

    # Save report
    csv_path = generate_evaluation_report(
        metrics,
        task_name=args.task,
        calibration_path=calib_path,
        dca_path=dca_path,
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Metrics saved to: {csv_path}")
    print(f"📈 Calibration plot: {calib_path}")
    print(f"📉 Decision curve: {dca_path}")
    print(f"🔲 Confusion matrix: {confusion_path}")
    print(f"📊 ROC curve: {roc_path}")
    print(f"📊 Resampling plot: {resampling_path}")
    
    # Print key metrics
    print("\n## Key Metrics:")
    for key in ["auroc", "auprc", "accuracy", "f1", "brier"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\n## Bootstrap & Jackknife Results (AUROC):")
    print(f"  Bootstrap:  {boot_res.mean_scores['auroc']:.4f} ± {boot_res.std_scores['auroc']:.4f}")
    print(f"              95% CI: [{boot_res.confidence_intervals['auroc'][0]:.4f}, {boot_res.confidence_intervals['auroc'][1]:.4f}]")
    print(f"  Jackknife:  {jack_res.mean_scores['auroc']:.4f} ± {jack_res.std_scores['auroc']:.4f}")
    print(f"              95% CI: [{jack_res.confidence_intervals['auroc'][0]:.4f}, {jack_res.confidence_intervals['auroc'][1]:.4f}]")
