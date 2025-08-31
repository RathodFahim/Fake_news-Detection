# Fake News Detection Web Application

A machine learning-powered web application that detects fake news articles using Natural Language Processing and Logistic Regression.

## 🚀 Features

- **Real-time Prediction**: Analyze news articles instantly
- **Interactive Web Interface**: Built with Streamlit
- **Batch Analysis**: Upload CSV files for bulk analysis
- **Model Metrics**: View accuracy, precision, recall, and F1-score
- **Visual Analytics**: Probability charts and prediction distributions
- **Sample Texts**: Pre-loaded examples for testing

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: Scikit-learn, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model**: Logistic Regression with TF-IDF vectorization

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fake_news_2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatic on first run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 📊 Dataset

### Option 1: Kaggle Dataset (Recommended)
1. Download the "Fake News" dataset from Kaggle
2. Save as `fake_news.csv` in the project directory
3. Ensure columns: `text` (news content) and `label` (0=Real, 1=Fake)

### Option 2: Sample Dataset
The application automatically creates a sample dataset for testing if no main dataset is found.

## 🏃‍♂️ Running the Application

### Method 1: Streamlit Web App
```bash
streamlit run app.py
```

### Method 2: Train Model Separately
```bash
python model_training.py
```

## 📱 Usage

### Web Interface
1. **Single Prediction**:
   - Enter news text in the text area
   - Click "Analyze News"
   - View prediction with confidence score

2. **Batch Analysis**:
   - Upload CSV file with 'text' column
   - Click "Analyze All Articles"
   - Download results with predictions

3. **Sample Testing**:
   - Use pre-loaded sample texts
   - Compare real vs fake news examples

### Model Performance
The sidebar displays:
- Accuracy score
- Precision score
- Recall score
- F1-score

## 🔧 Project Structure

```
fake_news_2/
├── app.py                 # Streamlit web application
├── model_training.py      # ML model training and evaluation
├── data_preprocessing.py  # Text preprocessing and feature extraction
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── fake_news.csv         # Main dataset (user-provided)
├── sample_fake_news.csv  # Sample dataset (auto-generated)
├── logistic_model.pkl    # Trained model (generated)
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer (generated)
└── model_metrics.pkl     # Model performance metrics (generated)
```

## 🧠 How It Works

1. **Text Preprocessing**:
   - Convert to lowercase
   - Remove punctuation and special characters
   - Tokenization
   - Remove stopwords

2. **Feature Extraction**:
   - TF-IDF vectorization (5000 features)
   - Transform text to numerical vectors

3. **Model Training**:
   - Logistic Regression classifier
   - 80/20 train-test split
   - Cross-validation for robust evaluation

4. **Prediction**:
   - Preprocess input text
   - Transform using trained vectorizer
   - Classify as Real (0) or Fake (1)
   - Return confidence scores

## 📈 Model Performance

Expected performance on the Kaggle dataset:
- **Accuracy**: ~92-95%
- **Precision**: ~90-94%
- **Recall**: ~89-93%
- **F1-Score**: ~90-93%

## 🎯 Sample Predictions

### Real News Examples:
- "Government announces new education policy"
- "Scientists discover breakthrough in renewable energy"
- "Stock market shows steady growth in technology sector"

### Fake News Examples:
- "Celebrity endorses miracle cure doctors don't want you to know"
- "Local man loses 50 pounds with this one weird trick"
- "Breaking: World leaders secretly controlled by aliens"

## 🔮 Future Enhancements

- **Deep Learning Models**: LSTM, BERT integration
- **Multi-language Support**: Detect fake news in multiple languages
- **Real-time News Feed**: Analyze live news articles
- **Advanced Visualizations**: Word clouds, feature importance
- **API Endpoint**: REST API for external integrations
- **User Feedback**: Allow users to correct predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Kaggle for the Fake News dataset
- Scikit-learn for machine learning tools
- Streamlit for the web framework
- NLTK for natural language processing

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description

---

**Built with ❤️ using Python and Machine Learning**