import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, RANDOM_STATE
from src.data_pipeline import load_and_split_data, create_preprocessor

def train_logistic_regression():
    print("Loading data...")
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data()
    
    print("Building baseline pipeline...")
    preprocessor = create_preprocessor()
    
    # The pipeline encapsulates both data transformations and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=RANDOM_STATE
        ))
    ])
    
    print("Training Logistic Regression...")
    pipeline.fit(X_train, y_train)
    
    # Serialize the entire pipeline (preprocessor + model)
    model_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Baseline model saved to {model_path}")
    
    # Save the LabelEncoder so evaluate.py can decode predictions
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(label_encoder, le_path)
    print(f"Label encoder saved to {le_path}")

if __name__ == "__main__":
    train_logistic_regression()