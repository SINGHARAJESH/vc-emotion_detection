import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)

# ---------------- Load Parameters ----------------
def load_params(param_path: str) -> int:
    try:
        params = yaml.safe_load(open(param_path, 'r'))
        max_features = params['feature_engineering']['max_features']
        logging.info(f"Max features loaded from params.yaml: {max_features}")
        return max_features
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

# ---------------- Load Data ----------------
def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path).fillna('')
        test_data = pd.read_csv(test_path).fillna('')
        logging.info("Train and test data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# ---------------- Vectorize Text ----------------
def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Text vectorization (BoW) completed.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Text vectorization failed: {e}")
        raise

# ---------------- Save Features ----------------
def save_features(X_train_bow, X_test_bow, y_train, y_test, path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        train_df.to_csv(os.path.join(path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(path, 'test_bow.csv'), index=False)

        logging.info("Feature-engineered datasets saved successfully.")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

# ---------------- Main Function ----------------
def main():
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = load_data('./data/processed/train_processed.csv', './data/processed/test_processed.csv')

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test, max_features)
        save_features(X_train_bow, X_test_bow, y_train, y_test, os.path.join('data', 'features'))

        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.exception("Feature engineering pipeline failed.")

# ---------------- Entry Point ----------------
if __name__ == '__main__':
    main()
