import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Import constants from our config
from src.config import (
    DATA_PATH, TARGET_COL, NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES, TEST_SIZE, RANDOM_STATE
)

def load_and_split_data():
    """
    Loads raw CSV, encodes the binary target, and performs a stratified split.
    Stratification is non-negotiable here due to the imbalanced target (y).
    """
    # Note: UCI Bank dataset uses semicolons
    df = pd.read_csv(DATA_PATH, sep=';')
    
    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]
    
    # Binarize target ('no' -> 0, 'yes' -> 1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Stratify ensures the train and test sets have the same ratio of 'yes'/'no'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y 
    )
    
    return X_train, X_test, y_train, y_test, le

def create_preprocessor():
    """
    Builds the stateless preprocessing engine. 
    Numeric features are standardized; Categorical features are One-Hot Encoded.
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        # handle_unknown='ignore' prevents crashes in production if a new job type appears
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    return preprocessor