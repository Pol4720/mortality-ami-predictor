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
    """Evaluate trained model on test set.
    
    This function provides backward compatibility with the old evaluate.py module.
    
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
    y = test_df[target].values
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target column in X: {target in X.columns}")

    # Get predictions
    prob = model.predict_proba(X)[:, 1]
    y_pred = (prob >= 0.5).astype(int)
    
    print(f"  Predictions shape: {prob.shape}")
    print(f"  Probability range: [{prob.min():.4f}, {prob.max():.4f}]")
    print(f"  Predicted class distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    print(f"  First 5 probabilities: {prob[:5]}")
    print(f"  First 5 true labels: {y[:5]}")

    # Compute metrics
    metrics = compute_classification_metrics(y, prob)
    
    # Generate plots
    calib_path = plot_calibration_curve(y, prob, name=args.task)
    dca_path = decision_curve_analysis(y, prob, name=args.task)
    confusion_path = plot_confusion_matrix(y, prob, name=args.task)
    roc_path = plot_roc_curve(y, prob, name=args.task)

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
    
    # Print key metrics
    print("\n## Key Metrics:")
    for key in ["auroc", "auprc", "accuracy", "f1", "brier"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
