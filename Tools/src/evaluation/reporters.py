"""Evaluation report generation utilities."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import joblib
import pandas as pd

from ..config import CONFIG
from .metrics import compute_classification_metrics
from .calibration import plot_calibration_curve
from .decision_curves import decision_curve_analysis


REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "reports"
)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models"
)


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

    target = CONFIG.target_column if args.task == "mortality" else CONFIG.arrhythmia_column
    if target not in test_df.columns:
        raise KeyError(f"Target column '{target}' not found in test set.")

    # Prepare features and target
    X = test_df.drop(columns=[target])
    y = test_df[target].values

    # Get predictions
    prob = model.predict_proba(X)[:, 1]

    # Compute metrics
    metrics = compute_classification_metrics(y, prob)
    
    # Generate plots
    calib_path = plot_calibration_curve(y, prob, name=args.task)
    dca_path = decision_curve_analysis(y, prob, name=args.task)

    # Save report
    csv_path = generate_evaluation_report(
        metrics,
        task_name=args.task,
        calibration_path=calib_path,
        dca_path=dca_path,
    )
    
    print(f"✅ Evaluation complete!")
    print(f"📊 Metrics saved to: {csv_path}")
    print(f"📈 Calibration plot: {calib_path}")
    print(f"📉 Decision curve: {dca_path}")
    
    # Print key metrics
    print("\n## Key Metrics:")
    for key in ["auroc", "auprc", "accuracy", "f1", "brier"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
