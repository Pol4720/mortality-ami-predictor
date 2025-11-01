"""Evaluation report generation utilities."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc

from ..config import CONFIG
from .metrics import compute_classification_metrics
from .calibration import plot_calibration_curve
from .decision_curves import decision_curve_analysis


# Use new processed structure
ROOT_DIR = Path(__file__).parents[2]
REPORTS_DIR = ROOT_DIR / "processed" / "plots" / "evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = REPORTS_DIR  # Plots go directly in evaluation directory
MODELS_DIR = ROOT_DIR / "processed" / "models"


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    name: str = "model", 
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> Union[go.Figure, str]:
    """Plot confusion matrix using Plotly for interactivity.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        threshold: Classification threshold
        save_path: Optional path to save as PNG (for backward compatibility)
        
    Returns:
        Plotly Figure object (or path if save_path provided)
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create annotated heatmap
    classes = ['Negative', 'Positive']
    
    # Create text annotations
    z_text = [[f"{cm[i, j]}<br>({cm[i, j]/cm[i].sum()*100:.1f}%)" 
               for j in range(cm.shape[1])] 
              for i in range(cm.shape[0])]
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=classes,
        y=classes,
        annotation_text=z_text,
        colorscale='Blues',
        showscale=True,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    )
    
    fig.update_layout(
        title=f'Confusion Matrix: {name}',
        xaxis=dict(title='Predicted label', side='bottom'),
        yaxis=dict(title='True label'),
        width=600,
        height=550,
        template='plotly_white'
    )
    
    # Reverse y-axis to match matplotlib convention
    fig['layout']['yaxis']['autorange'] = "reversed"
    
    if save_path:
        fig.write_image(save_path, width=600, height=550)
        return save_path
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    name: str = "model",
    save_path: Optional[str] = None
) -> Union[go.Figure, str]:
    """Plot ROC curve using Plotly for interactivity.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        name: Model name for filename
        save_path: Optional path to save as PNG (for backward compatibility)
        
    Returns:
        Plotly Figure object (or path if save_path provided)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2),
        hovertemplate=(
            'FPR: %{x:.3f}<br>'
            'TPR: %{y:.3f}<br>'
            'Threshold: %{customdata:.3f}<br>'
            '<extra></extra>'
        ),
        customdata=thresholds
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'ROC Curve: {name}',
        xaxis=dict(
            title='False Positive Rate',
            range=[0, 1],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='True Positive Rate',
            range=[0, 1.05],
            gridcolor='lightgray'
        ),
        width=700,
        height=600,
        template='plotly_white',
        legend=dict(x=0.6, y=0.1),
        hovermode='closest'
    )
    
    if save_path:
        fig.write_image(save_path, width=700, height=600)
        return save_path
    
    return fig



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

    # Import helper function to get latest files
    from ..data_load import get_latest_model_by_task, get_latest_testset
    
    # Try to find the latest model and testset for this task
    model_path = get_latest_model_by_task(args.task, MODELS_DIR)
    test_path = get_latest_testset(args.task, ROOT_DIR / "processed" / "testsets")
    
    if not model_path or not model_path.exists():
        raise FileNotFoundError(
            f"Model not found for task '{args.task}'.\n"
            f"Train a model first. Looking in: {MODELS_DIR}"
        )
    
    if not test_path or not test_path.exists():
        raise FileNotFoundError(
            f"Test set not found for task '{args.task}'.\n"
            f"Train a model first. Looking in: {ROOT_DIR / 'processed' / 'testsets'}"
        )

    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"\n🔍 Debugging Info:")
    print(f"Model loaded from: {model_path}")
    print(f"Test set loaded from: {test_path}")
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
    
    # Create Plotly figures and save as PNG for backward compatibility
    calib_fig = plot_calibration_curve(y_np, prob, name=args.task)
    calib_path = os.path.join(FIGURES_DIR, f"calibration_{args.task}.png")
    if isinstance(calib_fig, go.Figure):
        calib_fig.write_image(calib_path)
    else:
        calib_path = calib_fig  # Already a path
    
    dca_fig = decision_curve_analysis(y_np, prob, name=args.task)
    dca_path = os.path.join(FIGURES_DIR, f"decision_curve_{args.task}.png")
    if isinstance(dca_fig, go.Figure):
        dca_fig.write_image(dca_path)
    else:
        dca_path = dca_fig  # Already a path
    
    confusion_fig = plot_confusion_matrix(y_np, prob, name=args.task)
    confusion_path = os.path.join(FIGURES_DIR, f"confusion_{args.task}.png")
    if isinstance(confusion_fig, go.Figure):
        confusion_fig.write_image(confusion_path)
    else:
        confusion_path = confusion_fig  # Already a path
    
    roc_fig = plot_roc_curve(y_np, prob, name=args.task)
    roc_path = os.path.join(FIGURES_DIR, f"roc_{args.task}.png")
    if isinstance(roc_fig, go.Figure):
        roc_fig.write_image(roc_path)
    else:
        roc_path = roc_fig  # Already a path

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
