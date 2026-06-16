"""
Model evaluation script for both Logistic Regression (baseline) and XGBoost.

Generates:
  - classification_report for each model
  - Accuracy, Precision, Recall, F1, PR-AUC metrics
  - Feature importance ranking (XGBoost)
  - 5 sample customer predictions with full feature context
  - Saved plots to images/ for README embedding
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, accuracy_score, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
)

from src.config import MODEL_DIR, BASE_DIR
from src.data_pipeline import load_and_split_data

# Output directory for plots
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Consistent plot styling
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"baseline": "#5B9BD5", "xgb": "#ED7D31"}


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

    # Store results for comparison plot
    all_metrics = {}

    # -- Evaluate each model --------------------------------------------------
    for name, model, key in [
        ("Logistic Regression (Baseline)", baseline_model, "baseline"),
        ("XGBoost", xgb_model, "xgb"),
    ]:
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

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "PR-AUC": pr_auc,
        }
        all_metrics[key] = {"name": name, "metrics": metrics,
                            "y_pred": y_pred, "y_prob": y_prob,
                            "prec_curve": prec_vals, "rec_curve": rec_vals}

        print("Summary Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:11s}: {value:.4f}")

    # -- PLOT 1: Model Comparison Bar Chart -----------------------------------
    print("\nGenerating plots...")

    metric_names = list(all_metrics["baseline"]["metrics"].keys())
    baseline_vals = [all_metrics["baseline"]["metrics"][m] for m in metric_names]
    xgb_vals = [all_metrics["xgb"]["metrics"][m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width,
                   label="Logistic Regression", color=COLORS["baseline"],
                   edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, xgb_vals, width,
                   label="XGBoost", color=COLORS["xgb"],
                   edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Logistic Regression vs XGBoost", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: images/model_comparison.png")

    # -- PLOT 2: Precision-Recall Curves --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(all_metrics["baseline"]["rec_curve"],
            all_metrics["baseline"]["prec_curve"],
            label=f'Logistic Regression (PR-AUC={all_metrics["baseline"]["metrics"]["PR-AUC"]:.3f})',
            color=COLORS["baseline"], linewidth=2)
    ax.plot(all_metrics["xgb"]["rec_curve"],
            all_metrics["xgb"]["prec_curve"],
            label=f'XGBoost (PR-AUC={all_metrics["xgb"]["metrics"]["PR-AUC"]:.3f})',
            color=COLORS["xgb"], linewidth=2)

    # Baseline: proportion of positive class
    pos_rate = y_test.sum() / len(y_test)
    ax.axhline(y=pos_rate, color="gray", linestyle="--", linewidth=1,
               label=f"No-skill baseline ({pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: images/precision_recall_curve.png")

    # -- PLOT 3: Feature Importance (XGBoost) ---------------------------------
    xgb_classifier = xgb_model.named_steps["classifier"]
    preprocessor = xgb_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_classifier.feature_importances_

    feat_imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    print(f"\n{'=' * 60}")
    print("  XGBoost -- Feature Importances (Top 10)")
    print(f"{'=' * 60}")
    print(feat_imp.to_string(index=False))

    # Clean up feature names for display (remove prefixes)
    feat_imp_plot = feat_imp.copy()
    feat_imp_plot["Feature"] = (
        feat_imp_plot["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        feat_imp_plot["Feature"][::-1],
        feat_imp_plot["Importance"][::-1],
        color=COLORS["xgb"], edgecolor="white", linewidth=0.8,
    )

    # Add value labels
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)

    ax.set_xlabel("Importance")
    ax.set_title("XGBoost - Top 10 Feature Importances", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"  Saved: images/feature_importance.png")

    # -- PLOT 4: Confusion Matrix (XGBoost) -----------------------------------
    cm = confusion_matrix(y_test, all_metrics["xgb"]["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )
    disp.plot(ax=ax, cmap="Oranges", values_format="d")
    ax.set_title("XGBoost - Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved: images/confusion_matrix.png")

    # -- 5 Sample predictions (spec: >=2 yes, >=2 no) ------------------------
    print(f"\n{'=' * 60}")
    print("  5 Sample Customer Predictions (XGBoost)")
    print(f"{'=' * 60}")

    y_pred_xgb = all_metrics["xgb"]["y_pred"]
    y_prob_xgb = all_metrics["xgb"]["y_prob"]

    pred_df = X_test.copy()
    pred_df["_actual"] = label_encoder.inverse_transform(y_test)
    pred_df["_predicted"] = label_encoder.inverse_transform(y_pred_xgb)
    pred_df["_probability"] = y_prob_xgb

    # Guarantee at least 2 "yes" and at least 2 "no" predictions
    pos = pred_df[pred_df["_predicted"] == "yes"].head(3)
    neg = pred_df[pred_df["_predicted"] == "no"].head(2)

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

    print(f"\nAll plots saved to images/")


if __name__ == "__main__":
    evaluate_models()
