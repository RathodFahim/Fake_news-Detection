import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import FakeNewsModel
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
}
.real-news {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #c3e6cb;
}
.fake-news {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = FakeNewsModel('logistic')
        # Try to load existing model
        model.model = joblib.load('logistic_model.pkl')
        model.preprocessor.vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model.metrics = joblib.load('model_metrics.pkl')
        model.is_trained = True
        return model
    except:
        return None

def train_model_if_needed():
    """Train model if not already trained"""
    if not os.path.exists('logistic_model.pkl'):
        st.warning("No trained model found. Training model...")
        
        model = FakeNewsModel('logistic')
        
        # Try to use main dataset, fallback to sample
        dataset_path = 'fake_news.csv' if os.path.exists('fake_news.csv') else 'sample_fake_news.csv'
        
        if not os.path.exists(dataset_path):
            # Create sample dataset
            from model_training import create_sample_dataset
            create_sample_dataset()
            dataset_path = 'sample_fake_news.csv'
        
        with st.spinner('Training model... This may take a few minutes.'):
            success = model.train_model(dataset_path)
        
        if success:
            st.success("Model trained successfully!")
            st.rerun()
        else:
            st.error("Failed to train model!")
            return None
    
    return load_model()

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection</h1>', unsafe_allow_html=True)
    
    # Load or train model
    model = train_model_if_needed()
    
    if model is None:
        st.error("Unable to load or train model. Please check your dataset.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        if hasattr(model, 'metrics') and model.metrics:
            st.metric("Accuracy", f"{model.metrics['accuracy']:.4f}")
            st.metric("Precision", f"{model.metrics['precision']:.4f}")
            st.metric("Recall", f"{model.metrics['recall']:.4f}")
            st.metric("F1-Score", f"{model.metrics['f1_score']:.4f}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This application uses Machine Learning to detect fake news articles.
        
        **How it works:**
        1. Text preprocessing (cleaning, tokenization)
        2. TF-IDF vectorization
        3. Logistic Regression classification
        
        **Model Training:**
        - Dataset: Kaggle Fake News Dataset
        - Algorithm: Logistic Regression
        - Features: TF-IDF vectors (5000 features)
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 News Article Analysis")
        
        # Text input
        user_input = st.text_area(
            "Enter a news headline or article:",
            height=150,
            placeholder="Type or paste your news text here..."
        )
        
        # Prediction button
        if st.button("🔍 Analyze News", type="primary"):
            if user_input.strip():
                with st.spinner('Analyzing...'):
                    try:
                        result = model.predict(user_input)
                        
                        # Display prediction
                        if result['prediction'] == 'Real':
                            st.markdown(f"""
                            <div class="prediction-box real-news">
                                ✅ REAL NEWS<br>
                                Confidence: {result['confidence']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box fake-news">
                                ❌ FAKE NEWS<br>
                                Confidence: {result['confidence']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show probability breakdown
                        st.subheader("📊 Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Category': ['Real News', 'Fake News'],
                            'Probability': [result['probabilities']['Real'], result['probabilities']['Fake']]
                        })
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#28a745', '#dc3545']
                        bars = ax.bar(prob_df['Category'], prob_df['Probability'], color=colors, alpha=0.7)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Confidence')
                        ax.set_ylim(0, 1)
                        
                        # Add value labels on bars
                        for bar, prob in zip(bars, prob_df['Probability']):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{prob:.2%}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("📝 Sample Texts")
        
        sample_texts = {
            "Real News Sample": "Government announces new education policy to improve literacy rates across the country.",
            "Fake News Sample": "Scientists discover miracle cure that doctors don't want you to know about.",
            "Technology News": "Tech company releases new smartphone with improved battery life and camera features.",
            "Health Misinformation": "Local man loses 50 pounds with this one weird trick that nutritionists hate."
        }
        
        for label, text in sample_texts.items():
            if st.button(f"Try: {label}", key=label):
                st.session_state.sample_text = text
        
        if 'sample_text' in st.session_state:
            st.text_area("Selected sample:", value=st.session_state.sample_text, height=100, key="sample_display")
    
    # File upload section
    st.markdown("---")
    st.header("📁 Batch Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file for batch analysis", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' in df.columns:
                st.success(f"File uploaded successfully! Found {len(df)} articles.")
                
                if st.button("🔍 Analyze All Articles"):
                    with st.spinner('Analyzing all articles...'):
                        predictions = []
                        for text in df['text']:
                            try:
                                result = model.predict(str(text))
                                predictions.append({
                                    'prediction': result['prediction'],
                                    'confidence': result['confidence']
                                })
                            except:
                                predictions.append({
                                    'prediction': 'Error',
                                    'confidence': 0.0
                                })
                        
                        # Add predictions to dataframe
                        pred_df = pd.DataFrame(predictions)
                        result_df = pd.concat([df, pred_df], axis=1)
                        
                        # Display results
                        st.subheader("📊 Batch Analysis Results")
                        st.dataframe(result_df)
                        
                        # Summary statistics
                        if len(pred_df[pred_df['prediction'] != 'Error']) > 0:
                            valid_predictions = pred_df[pred_df['prediction'] != 'Error']
                            real_count = len(valid_predictions[valid_predictions['prediction'] == 'Real'])
                            fake_count = len(valid_predictions[valid_predictions['prediction'] == 'Fake'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Articles", len(df))
                            with col2:
                                st.metric("Real News", real_count)
                            with col3:
                                st.metric("Fake News", fake_count)
                            
                            # Pie chart
                            if real_count + fake_count > 0:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                labels = ['Real News', 'Fake News']
                                sizes = [real_count, fake_count]
                                colors = ['#28a745', '#dc3545']
                                
                                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                                ax.set_title('Distribution of Predictions')
                                st.pyplot(fig)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="fake_news_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV file must contain a 'text' column with news articles.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()