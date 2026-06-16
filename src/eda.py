"""
Focused Exploratory Data Analysis for the UCI Bank Marketing dataset.

Covers Track B requirements:
  - Dataset shape, dtypes, missing values
  - Target class distribution (y = yes/no)
  - Subscription rate by job category
  - "Unknown" value audit (the dataset encodes missing data as the string 'unknown')
  - Duration feature caveat (known data leakage risk)
"""

import pandas as pd
from src.config import DATA_PATH, TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def run_eda():
    # ── 1. Load ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, sep=";")

    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Numeric cols : {NUMERIC_FEATURES}")
    print(f"  Categor. cols: {CATEGORICAL_FEATURES}")
    print(f"  Target col   : '{TARGET_COL}'")

    # ── 2. Missing values ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. MISSING VALUES (NaN)")
    print("=" * 60)
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("  No NaN values found in any column.")
    else:
        print(null_counts[null_counts > 0])

    # The UCI Bank dataset uses 'unknown' as a sentinel instead of NaN.
    print("\n  'unknown' string counts (dataset uses this instead of NaN):")
    for col in CATEGORICAL_FEATURES:
        unk = (df[col] == "unknown").sum()
        if unk > 0:
            print(f"    {col:15s} -> {unk:>6,} ({unk / len(df) * 100:.1f}%)")

    # ── 3. Target distribution ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. TARGET CLASS DISTRIBUTION  (y)")
    print("=" * 60)
    dist = df[TARGET_COL].value_counts()
    pct = df[TARGET_COL].value_counts(normalize=True) * 100
    for label in dist.index:
        print(f"  {label:4s} -> {dist[label]:>6,}  ({pct[label]:.2f}%)")
    print(f"\n  Imbalance ratio (no/yes): {dist['no'] / dist['yes']:.1f}:1")
    print("  WARNING: Heavily imbalanced — accuracy alone is misleading.")

    # ── 4. Subscription rate by job ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. SUBSCRIPTION RATE BY JOB CATEGORY")
    print("=" * 60)
    rates = (
        df.groupby("job")[TARGET_COL]
        .apply(lambda s: (s == "yes").mean() * 100)
        .sort_values(ascending=False)
    )
    for job, rate in rates.items():
        print(f"  {job:15s} -> {rate:5.2f}%")

    # ── 5. Duration caveat ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. DURATION FEATURE CAVEAT")
    print("=" * 60)
    print("  The 'duration' column records call length in seconds.")
    print("  UCI documentation warns: duration is NOT known before a call")
    print("  is made, so it should only be included for benchmark purposes.")
    print("  It is highly correlated with the target — a realistic model")
    print("  deployed pre-call would NOT have access to this feature.")
    print(f"  Mean duration (yes): {df.loc[df['y'] == 'yes', 'duration'].mean():.0f}s")
    print(f"  Mean duration (no) : {df.loc[df['y'] == 'no', 'duration'].mean():.0f}s")

    # -- 6. Numeric feature summary --------------------------------------
    print("\n" + "=" * 60)
    print("6. NUMERIC FEATURE SUMMARY")
    print("=" * 60)
    print(df[NUMERIC_FEATURES].describe().round(2).to_string())


if __name__ == "__main__":
    run_eda()
