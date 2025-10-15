"""Prediction interface: load saved model and run predictions on new data."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def predict_main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Run predictions with a saved model.")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .joblib")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV/Parquet")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV with predictions")
    args = parser.parse_args(argv)

    model = joblib.load(args.model)
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        X = pd.read_csv(args.input)
    elif ext == ".parquet":
        X = pd.read_parquet(args.input)
    else:
        raise ValueError("Unsupported input format. Use CSV or Parquet.")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        out = X.copy()
        out["prediction_proba"] = proba
    else:
        preds = model.predict(X)
        out = X.copy(); out["prediction"] = preds

    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    predict_main()
