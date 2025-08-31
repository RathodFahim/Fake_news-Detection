import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            
            # Handle missing values
            df = df.dropna(subset=['text', 'label'])
            
            # Clean text data
            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            # Remove empty texts after cleaning
            df = df[df['cleaned_text'].str.len() > 0]
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features using TF-IDF vectorization"""
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['label'].values
        
        # Save vectorizer for later use
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Note: You need to download the Kaggle dataset and place it in the project directory
    # Dataset should be named 'fake_news.csv' with columns: 'text' and 'label'
    print("Data preprocessing module ready!")
    print("To use: Place your 'fake_news.csv' dataset in the project directory")