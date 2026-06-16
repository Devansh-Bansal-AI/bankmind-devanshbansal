"""
Model evaluation script for both Logistic Regression (baseline) and XGBoost.

Generates:
  - classification_report for each model
  - Accuracy, Precision, Recall, F1, PR-AUC metrics
  - Feature importance ranking (XGBoost)
  - 5 sample customer predictions with full feature context
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, accuracy_score, precision_recall_curve, auc,
)

from src.config import MODEL_DIR
from src.data_pipeline import load_and_split_data


def evaluate_models():
    print("Loading test data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data()

    baseline_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

    if not os.path.exists(baseline_path) or not os.path.exists(xgb_path):
        print("ERROR: Models not found. Run training scripts first:")
        print("  python -m src.train_baseline")
        print("  python -m src.train_tree")
        return

    print("Loading models...")
    baseline_model = joblib.load(baseline_path)
    xgb_model = joblib.load(xgb_path)

    # ── Evaluate each model ─────────────────────────────────────────────
    for name, model in [("Logistic Regression (Baseline)", baseline_model),
                        ("XGBoost", xgb_model)]:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Classification report (required by Track B spec)
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        ))

        # Individual metrics
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec_vals, prec_vals)

        print("Summary Metrics:")
        print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
        print(f"  PR-AUC    : {pr_auc:.4f}")

    # ── Feature importance (XGBoost only) ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("  XGBoost — Feature Importances (Top 10)")
    print(f"{'=' * 60}")

    xgb_classifier = xgb_model.named_steps["classifier"]
    preprocessor = xgb_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_classifier.feature_importances_

    feat_imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    print(feat_imp.to_string(index=False))

    # ── 5 Sample predictions (spec: ≥2 yes, ≥2 no) ────────────────────
    print(f"\n{'=' * 60}")
    print("  5 Sample Customer Predictions (XGBoost)")
    print(f"{'=' * 60}")

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    pred_df = X_test.copy()
    pred_df["_actual"] = label_encoder.inverse_transform(y_test)
    pred_df["_predicted"] = label_encoder.inverse_transform(y_pred_xgb)
    pred_df["_probability"] = y_prob_xgb

    # Guarantee at least 2 "yes" and at least 2 "no" predictions
    pos = pred_df[pred_df["_predicted"] == "yes"].head(3)
    neg = pred_df[pred_df["_predicted"] == "no"].head(2)

    # Fallback if we somehow have fewer than required
    if len(pos) < 2 or len(neg) < 2:
        samples = pred_df.head(5)
    else:
        samples = pd.concat([pos, neg])

    display_cols = [
        "age", "job", "marital", "education", "balance",
        "housing", "loan", "contact", "duration", "poutcome",
    ]

    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        prob_pct = row["_probability"] * 100
        print(f"\n-- Sample {i} (Index {idx}) --")
        for col in display_cols:
            print(f"  {col:12s}: {row[col]}")
        print(f"  {'':12s}  --------------------")
        print(f"  Actual     : {row['_actual']}")
        print(f"  Predicted  : {row['_predicted']}")
        print(f"  Probability: {prob_pct:.2f}%")


if __name__ == "__main__":
    evaluate_models()
