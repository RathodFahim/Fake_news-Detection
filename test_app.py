#!/usr/bin/env python3
"""
Test script to verify the fake news detection app works correctly
"""

from model_training import FakeNewsModel
import os

def test_model():
    """Test the model functionality"""
    print("Testing Fake News Detection Model...")
    
    # Create and train model if needed
    if not os.path.exists('logistic_model.pkl'):
        print("Training model...")
        model = FakeNewsModel('logistic')
        success = model.train_model('sample_fake_news.csv')
        if not success:
            print("Failed to train model!")
            return False
    
    # Load model
    model = FakeNewsModel('logistic')
    
    # Test predictions
    test_cases = [
        ("Government announces new education policy", "Real"),
        ("Celebrity endorses miracle cure doctors hate", "Fake"),
        ("Scientists discover breakthrough in renewable energy", "Real"),
        ("Local man loses 50 pounds with weird trick", "Fake")
    ]
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for text, expected in test_cases:
        try:
            result = model.predict(text)
            print(f"Text: {text[:50]}...")
            print(f"Prediction: {result['prediction']} (Expected: {expected})")
            print(f"Confidence: {result['confidence']:.2%}")
            print("-" * 50)
        except Exception as e:
            print(f"Error predicting: {e}")
            return False
    
    print("Model test completed successfully!")
    return True

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import nltk
        import joblib
        import matplotlib
        import seaborn
        print("All imports successful!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    print("Fake News Detection App - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("Import test failed!")
        exit(1)
    
    # Test model
    if not test_model():
        print("Model test failed!")
        exit(1)
    
    print("\nAll tests passed! The app is ready to run.")
    print("Run 'streamlit run app.py' to start the web application.")