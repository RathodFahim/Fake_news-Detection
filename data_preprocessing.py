"""
Data Preprocessing Module for Fake News Detection
Handles FakeNewsNet.csv dataset with columns: title, news_url, source_domain, tweet_num, real
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


class DataPreprocessor:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

    @staticmethod
    def clean_text(text):
        """Clean and normalise a single text string."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # urls
        text = re.sub(r"<.*?>", "", text)  # html tags
        text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
        text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
        return text

    def load_and_preprocess(self, file_path):
        """Load FakeNewsNet.csv and return a clean DataFrame."""
        df = pd.read_csv(file_path)

        # Map columns to standard names
        col_map = {}
        if "title" in df.columns:
            col_map["title"] = "text"
        if "real" in df.columns:
            col_map["real"] = "label"
        df = df.rename(columns=col_map)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'title'/'text' and 'real'/'label' columns")

        # In FakeNewsNet, real=1 means real news.  We want label 1 = Fake.
        df["label"] = df["label"].apply(lambda x: 0 if x == 1 else 1)

        df = df.dropna(subset=["text"])
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
        return df

    def prepare_features(self, df, fit=True):
        """TF-IDF vectorisation. Set fit=False for transform-only."""
        if fit:
            X = self.vectorizer.fit_transform(df["cleaned_text"])
        else:
            X = self.vectorizer.transform(df["cleaned_text"])
        y = df["label"].values
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=42):
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def save(self, path="tfidf_vectorizer.pkl"):
        joblib.dump(self.vectorizer, path)

    def load(self, path="tfidf_vectorizer.pkl"):
        self.vectorizer = joblib.load(path)


if __name__ == "__main__":
    pp = DataPreprocessor()
    df = pp.load_and_preprocess("FakeNewsNet.csv")
    print(f"Loaded {len(df)} samples  |  Fake: {df['label'].sum()}  |  Real: {(df['label']==0).sum()}")
    X, y = pp.prepare_features(df)
    print(f"Feature matrix: {X.shape}")