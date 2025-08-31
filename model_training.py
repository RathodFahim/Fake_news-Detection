import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
from data_preprocessing import DataPreprocessor

class FakeNewsModel:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        self.metrics = {}
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'logistic':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'naive_bayes':
            return MultinomialNB()
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'logistic', 'naive_bayes', or 'random_forest'")
    
    def train_model(self, dataset_path):
        """Train the model on the dataset"""
        print("Loading and preprocessing data...")
        df = self.preprocessor.load_and_preprocess_data(dataset_path)
        
        if df is None:
            print("Failed to load dataset!")
            return False
        
        print(f"Dataset loaded: {len(df)} samples")
        print(f"Real news: {len(df[df['label'] == 0])}, Fake news: {len(df[df['label'] == 1])}")
        
        # Prepare features
        X, y = self.preprocessor.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        print(f"Training model: {self.model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print("\nModel Performance:")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1-Score: {self.metrics['f1_score']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Save model
        model_filename = f'{self.model_type}_model.pkl'
        joblib.dump(self.model, model_filename)
        
        # Save metrics
        joblib.dump(self.metrics, 'model_metrics.pkl')
        
        print(f"\nModel saved as: {model_filename}")
        self.is_trained = True
        
        return True
    
    def predict(self, text):
        """Predict if a single text is fake or real"""
        if not self.is_trained:
            # Try to load pre-trained model
            try:
                self.model = joblib.load(f'{self.model_type}_model.pkl')
                self.preprocessor.vectorizer = joblib.load('tfidf_vectorizer.pkl')
                self.metrics = joblib.load('model_metrics.pkl')
                self.is_trained = True
            except:
                raise Exception("No trained model found. Please train the model first.")
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Vectorize
        text_vector = self.preprocessor.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': max(probability),
            'probabilities': {'Real': probability[0], 'Fake': probability[1]}
        }

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = {
        'text': [
            # Real news examples
            "Scientists discover breakthrough in renewable energy technology",
            "Government announces new healthcare policy changes",
            "Stock market shows steady growth in technology sector",
            "University research shows benefits of regular exercise",
            "New study reveals impact of climate change on agriculture",
            "Technology company reports quarterly earnings increase",
            "Medical researchers develop new treatment for diabetes",
            "Education department launches new literacy program",
            "Economic indicators show steady job market growth",
            "Environmental agency announces new conservation initiative",
            "Sports team wins championship after successful season",
            "Local community center opens new facility for seniors",
            "Transportation department completes highway improvement project",
            "Public health officials recommend seasonal flu vaccination",
            "Archaeological team discovers ancient artifacts in excavation",
            "Weather service issues storm warning for coastal areas",
            "Banking sector shows stability in recent financial reports",
            "Agricultural department announces crop yield improvements",
            "Space agency successfully launches new satellite mission",
            "International trade agreement signed between countries",
            
            # Fake news examples
            "Celebrity spotted with alien at local restaurant",
            "Miracle cure discovered that doctors don't want you to know",
            "Breaking: World leaders secretly controlled by lizard people",
            "Local man loses 50 pounds with this one weird trick",
            "Scientists hide truth about flat earth conspiracy",
            "Billionaire reveals secret to instant wealth in shocking interview",
            "Government covers up evidence of time travel experiments",
            "Ancient aliens built pyramids using advanced technology",
            "Pharmaceutical companies suppress natural cancer cure",
            "Celebrity endorses magical weight loss tea that melts fat",
            "Breaking: Moon landing was filmed in Hollywood studio",
            "Shocking discovery: Vaccines contain mind control chips",
            "Local woman discovers fountain of youth in backyard",
            "Politicians secretly worship ancient demon in basement ritual",
            "Scientists confirm earth is actually hollow with underground cities",
            "Miracle fruit cures all diseases but big pharma hides it",
            "Breaking: Social media apps can read your thoughts",
            "Government uses weather machines to control population",
            "Celebrity reveals shocking truth about reptilian shapeshifters",
            "Ancient prophecy predicts end of world next Tuesday"
        ],
        'label': [0]*20 + [1]*20  # 0 = Real, 1 = Fake
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_fake_news.csv', index=False)
    print(f"Sample dataset created: sample_fake_news.csv ({len(df)} samples)")

if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_dataset()
    
    # Train model
    model = FakeNewsModel('logistic')
    
    # Use sample dataset if main dataset not available
    if os.path.exists('fake_news.csv'):
        print("Using main dataset: fake_news.csv")
        success = model.train_model('fake_news.csv')
    else:
        print("Main dataset not found, using sample dataset...")
        success = model.train_model('sample_fake_news.csv')
    
    if success:
        # Test prediction
        test_text = "Breaking news: Celebrity endorses miracle cure"
        result = model.predict(test_text)
        print(f"\nTest prediction:")
        print(f"Text: {test_text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")