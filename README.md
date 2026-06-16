# BankMind ML — Track B: ML Engineer

A production-grade ML pipeline that predicts which bank customers are likely to subscribe to a term deposit, built for the **VITB AI Innovators Hub** community screening task.

## Dataset

[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) — 45,211 customer records from a Portuguese bank's direct marketing campaigns.

- **Target**: `y` (yes/no) — whether the customer subscribed to a term deposit
- **Class split**: 88.3% no / 11.7% yes (heavily imbalanced)

> **Note**: Place `bank-full.csv` in the `data/` directory before running. It is git-ignored due to size.

## Repository Structure

```
bankmind-ml/
├── data/
│   └── bank-full.csv          # Raw dataset (download separately)
├── models/                    # Pre-trained model artifacts (included)
│   ├── baseline_model.pkl     # Logistic Regression pipeline
│   ├── xgb_model.pkl          # XGBoost pipeline
│   └── label_encoder.pkl      # Target label encoder
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralised paths, feature lists, constants
│   ├── data_pipeline.py       # Data loading, stratified split, preprocessing
│   ├── eda.py                 # Focused exploratory data analysis
│   ├── train_baseline.py      # Logistic Regression training
│   ├── train_tree.py          # XGBoost training
│   └── evaluate.py            # Metrics, feature importance, sample predictions
├── run_all.py                 # One-command pipeline runner
├── .gitignore
├── requirements.txt
├── README.md
└── EXPLANATION.md
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bankmind-ml.git
cd bankmind-ml

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    Place bank-full.csv into the data/ directory
```

## Usage

### Quick Start (one command)

```bash
python run_all.py
```

This runs the entire pipeline: EDA → Train Baseline → Train XGBoost → Evaluate.

### Step-by-Step (optional)

```bash
# Step 1 — Exploratory Data Analysis
python -m src.eda

# Step 2 — Train Logistic Regression (baseline)
python -m src.train_baseline

# Step 3 — Train XGBoost (main model)
python -m src.train_tree

# Step 4 — Evaluate both models
python -m src.evaluate
```

> **Note:** Pre-trained model files are included in `models/`. You can skip to Step 4 to see evaluation results immediately without retraining.

## Pipeline Architecture

```
bank-full.csv
     │
     ▼
 data_pipeline.py ──► Stratified train/test split (80/20)
     │                 ColumnTransformer (StandardScaler + OneHotEncoder)
     │
     ├─► train_baseline.py ──► Pipeline(preprocessor + LogisticRegression) ──► baseline_model.pkl
     │
     └─► train_tree.py    ──► Pipeline(preprocessor + XGBClassifier)       ──► xgb_model.pkl
                                                                                    │
                                                                evaluate.py ◄───────┘
                                                                    │
                                                                    ▼
                                                        Classification Report
                                                        Accuracy, Precision, Recall, F1, PR-AUC
                                                        Feature Importances
                                                        5 Sample Customer Predictions
```

## Output

### Model Comparison

| Metric    | Logistic Regression | XGBoost |
|-----------|--------------------:|--------:|
| Accuracy  |              0.8455 |  0.8744 |
| Precision |              0.4177 |  0.4780 |
| Recall    |              0.8129 |  0.8015 |
| F1-Score  |              0.5518 |  0.5989 |
| PR-AUC    |              0.5366 |  0.6080 |

XGBoost outperforms the baseline across all metrics, particularly in F1-Score (+0.047) and PR-AUC (+0.071).

### Top Features (XGBoost)

| Feature            | Importance |
|--------------------|------------|
| contact_unknown    | 0.153      |
| poutcome_success   | 0.144      |
| poutcome_unknown   | 0.093      |
| month_mar          | 0.047      |
| housing_yes        | 0.044      |

## Key Design Decisions

- **Scikit-learn Pipelines**: Preprocessing and model are encapsulated in a single serialisable Pipeline, preventing train/test data leakage and simplifying inference.
- **Stratified splitting**: Maintains the 88/12 class ratio in both train and test sets.
- **Class imbalance handling**: `class_weight='balanced'` (Logistic Regression), `scale_pos_weight` (XGBoost).
- **`handle_unknown='ignore'`**: OneHotEncoder gracefully handles unseen categories at inference time.

## License

For educational/screening purposes only. Dataset is from the UCI Machine Learning Repository.
