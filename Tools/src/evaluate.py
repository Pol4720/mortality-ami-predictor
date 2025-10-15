"""Evaluation module: metrics, calibration, decision curves, and reporting."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from .config import CONFIG
from .data import load_dataset
from .models import make_kmeans


REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> str:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.legend()
    path = os.path.join(FIG_DIR, f"calibration_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, name: str) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    path = os.path.join(FIG_DIR, f"confusion_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> str:
    thresholds = np.linspace(0.01, 0.99, 50)
    net_benefits = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        n = len(y_true)
        pt = t / (1 - t)
        nb = (tp / n) - (fp / n) * pt
        net_benefits.append(nb)
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, net_benefits, label="Model")
    plt.plot(thresholds, np.zeros_like(thresholds), label="Treat None", linestyle="--")
    prevalence = y_true.mean()
    treat_all_nb = thresholds * prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    plt.plot(thresholds, treat_all_nb, label="Treat All", linestyle=":")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.legend()
    path = os.path.join(FIG_DIR, f"decision_curve_{name}.png")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def evaluate_main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved best model on hold-out test set.")
    parser.add_argument("--data", type=str, default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--task", type=str, choices=["mortality", "arrhythmia"], default="mortality")
    args = parser.parse_args(argv)

    if not args.data:
        raise ValueError("DATASET_PATH not provided.")

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"best_classifier_{args.task}.joblib")
    test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"testset_{args.task}.parquet")
    if not (os.path.exists(model_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Model or test set not found. Train first.")

    model = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)

    target = CONFIG.target_column if args.task == "mortality" else CONFIG.arrhythmia_column
    assert target in test_df.columns

    X = test_df.drop(columns=[target])
    y = test_df[target].values

    prob = model.predict_proba(X)[:, 1]

    metrics = compute_classification_metrics(y, prob)
    calib_path = plot_calibration(y, prob, name=args.task)
    dca_path = decision_curve_analysis(y, prob, name=args.task)
    cm_path = plot_confusion(y, prob, threshold=0.5, name=args.task)

    # Save tabular metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = os.path.join(REPORTS_DIR, f"final_metrics_{args.task}.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Optional baseline comparisons if classical scores exist
    baselines: List[Dict[str, float]] = []
    for score_col in ["grace_score", "timi_score"]:
        if score_col in test_df.columns:
            s = test_df[score_col].values.astype(float)
            # normalize to 0-1 for comparison if not already probabilities
            s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
            mb = compute_classification_metrics(y, s_norm)
            mb["baseline"] = score_col
            baselines.append(mb)
    if baselines:
        base_csv = os.path.join(REPORTS_DIR, f"baseline_metrics_{args.task}.csv")
        pd.DataFrame(baselines).to_csv(base_csv, index=False)

    # Unsupervised: KMeans cluster profiling
    try:
        km, _ = make_kmeans()
        feats = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        Xn = X[feats].fillna(X[feats].median())
        labels = km.fit_predict(Xn)
        prof = test_df.copy()
        prof["cluster"] = labels
        profile = prof.groupby("cluster")[target].agg(["mean", "count"]).rename(columns={"mean": "event_rate"}).reset_index()
        profile_csv = os.path.join(REPORTS_DIR, f"kmeans_profile_{args.task}.csv")
        profile.to_csv(profile_csv, index=False)
    except Exception:
        pass

    # Minimal PDF report using matplotlib's saved figures; reportlab optional here
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_path = os.path.join(REPORTS_DIR, "final_evaluation.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, f"Final Evaluation Report ({args.task})")
        c.setFont("Helvetica", 10)
        y_text = 720
        for k, v in metrics.items():
            c.drawString(72, y_text, f"{k}: {v:.4f}")
            y_text -= 14
        c.drawImage(calib_path, 72, 380, width=460, height=300, preserveAspectRatio=True)
        c.drawImage(dca_path, 72, 60, width=460, height=300, preserveAspectRatio=True)
        c.showPage()
        c.save()
    except Exception:
        pass

    print(f"Saved metrics to {metrics_csv}")


if __name__ == "__main__":
    evaluate_main()
