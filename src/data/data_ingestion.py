import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# ---------- Logging Configuration ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('pipeline.log')
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---------- Functions ----------
def load_parms(param_path: str) -> float:
    try:
        with open(param_path, 'r') as file:
            test_size = yaml.safe_load(file)['data_ingestion']['test_size']
        logging.info(f"Successfully loaded parameter from {param_path}")
        return test_size
    except FileNotFoundError:
        logging.error(f"Parameter file not found: {param_path}")
        raise
    except KeyError:
        logging.error("Missing key in parameter file. Expected path: ['data_ingestion']['text_size']")
        raise
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error: {e}")
        raise

def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info(f"Data loaded successfully from {url}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {url}: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        if 'sentiment' not in final_df.columns:
            raise Exception("Missing 'sentiment' column in DataFrame.")
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 2}, inplace=True)
        logging.info("Data processed successfully.")
        return final_df
    except KeyError as e:
        logging.error(f"Missing column in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise

def save_path(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logging.info(f"Data saved successfully to {data_path}")
    except Exception as e:
        logging.error(f"Error saving data to path {data_path}: {e}")
        raise

def main() -> None:
    try:
        logging.info("Pipeline started.")
        test_size = load_parms('params.yaml')
        df = load_data(r'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join('data', 'raw')
        save_path(data_path, train_data, test_data)
        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.exception(f"An error occurred in the main pipeline: {e}")

if __name__ == '__main__':
    main()

