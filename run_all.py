"""
Convenience script to run the entire pipeline in one command.

Usage:
    python run_all.py
"""

from src.eda import run_eda
from src.train_baseline import train_logistic_regression
from src.train_tree import train_xgboost
from src.evaluate import evaluate_models


def main():
    print("\n" + "=" * 60)
    print("  STEP 1/4 — Exploratory Data Analysis")
    print("=" * 60)
    run_eda()

    print("\n\n" + "=" * 60)
    print("  STEP 2/4 — Training Logistic Regression (Baseline)")
    print("=" * 60)
    train_logistic_regression()

    print("\n\n" + "=" * 60)
    print("  STEP 3/4 — Training XGBoost")
    print("=" * 60)
    train_xgboost()

    print("\n\n" + "=" * 60)
    print("  STEP 4/4 — Evaluation & Comparison")
    print("=" * 60)
    evaluate_models()

    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print("  Model artifacts saved to models/")
    print("  See EXPLANATION.md for analysis")


if __name__ == "__main__":
    main()
