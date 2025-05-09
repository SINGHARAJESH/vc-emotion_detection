import os
import re
import string
import logging
import numpy as np
import pandas as pd
import nltk

from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

# ----------------- NLTK Downloads -----------------
nltk.download('wordnet')
nltk.download('stopwords')

# ----------------- Text Cleaning Functions -----------------
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def removing_numbers(text: str) -> str:
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return text.lower()

def removing_punctuations(text: str) -> str:
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    df['content'] = df['content'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
    return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        df = remove_small_sentences(df)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

# ----------------- Data Loading and Saving -----------------
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def make_directory(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory created: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")
        raise

# ----------------- Main Function -----------------
def main() -> None:
    try:
        # Load data
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        # Normalize data
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # Save processed data
        output_dir = os.path.join('data', 'processed')
        make_directory(output_dir)
        save_data(train_processed, os.path.join(output_dir, 'train_processed.csv'))
        save_data(test_processed, os.path.join(output_dir, 'test_processed.csv'))

        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.exception("Pipeline failed.")

# ----------------- Entry Point -----------------
if __name__ == '__main__':
    main()
