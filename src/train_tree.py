import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, RANDOM_STATE
from src.data_pipeline import load_and_split_data, create_preprocessor

def train_xgboost():
    print("Loading data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data()
    
    # Calculate scale_pos_weight for handling data imbalance dynamically
    # Formula: count(negative examples) / count(positive examples)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count
    
    print(f"Class imbalance calculated. scale_pos_weight: {scale_weight:.2f}")
    
    print("Building XGBoost pipeline...")
    preprocessor = create_preprocessor()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ))
    ])
    
    print("Training XGBoost...")
    pipeline.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    joblib.dump(pipeline, model_path)
    
    # We also need to save the LabelEncoder to decode predictions later
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(label_encoder, le_path)
    
    print(f"XGBoost model saved to {model_path}")
    print(f"Label encoder saved to {le_path}")

if __name__ == "__main__":
    train_xgboost()