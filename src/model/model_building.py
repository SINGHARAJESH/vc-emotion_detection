import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging
import os

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("model_building.log"),
        logging.StreamHandler()
    ]
)

# ---------- Load Parameters ----------
def load_params(param_file):
    try:
        with open(param_file, 'r') as file:
            param_data = yaml.safe_load(file)['model_building']
        logging.info("Parameters loaded successfully.")
        return param_data
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

# ---------- Load Data ----------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Training data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

# ---------- Split Data ----------
def split_data(train_data):
    try:
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logging.info("Training data split into features and target.")
        return X_train, y_train
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# ---------- Build and Save Model ----------
def model_building(X_train, y_train, params):
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        #pickle.dump(clf, open('model.pkl', 'wb'))
        logging.info("Model trained successfully.")
        return clf
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise
def save_model(model,file_path):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info('Model saved successfully as model.pkl')
    except Exception as e:
        logging.error(f'model saved failed: {e}')
# ---------- Main Pipeline ----------
def main():
    try:
        params = load_params('params.yaml')
        train_df = load_data(r'.\data\features\train_bow.csv')
        X_train, y_train = split_data(train_df)

        clf = model_building(X_train, y_train, params)
        save_model(clf,'models/model.pkl')
        logging.info("Model pipeline completed successfully.")
    except Exception as e:
        logging.exception("Pipeline failed.")

if __name__ == '__main__':
    main()
