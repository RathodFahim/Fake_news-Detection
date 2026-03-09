"""
Model Training Module for Fake News Detection
Trains multiple classifiers, evaluates, and saves the best model.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from data_preprocessing import DataPreprocessor


MODELS = {
    "logistic": lambda: LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
    "passive_aggressive": lambda: PassiveAggressiveClassifier(max_iter=1000, C=0.5, class_weight="balanced", random_state=42),
    "naive_bayes": lambda: MultinomialNB(alpha=1.0),
    "linear_svc": lambda: CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)),
}


class FakeNewsModel:
    """Wrapper that trains, evaluates, saves, and serves a fake-news classifier."""

    def __init__(self, model_type="logistic"):
        if model_type not in MODELS:
            raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(MODELS.keys())}")
        self.model_type = model_type
        self.model = MODELS[model_type]()
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        self.metrics: dict = {}

    # ── training ────────────────────────────────────────────────────────
    def train(self, dataset_path="FakeNewsNet.csv"):
        """Train and evaluate the model. Returns True on success."""
        print(f"[1/5] Loading dataset from {dataset_path} …")
        df = self.preprocessor.load_and_preprocess(dataset_path)
        print(f"       {len(df)} samples  |  Fake {df['label'].sum()}  |  Real {(df['label']==0).sum()}")

        print("[2/5] Vectorising text (TF-IDF, bigrams) …")
        X, y = self.preprocessor.prepare_features(df)

        print("[3/5] Splitting 80 / 20 (stratified) …")
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)

        print(f"[4/5] Training {self.model_type} …")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Real", "Fake"], output_dict=True
            ),
        }

        print("\n── Results ──────────────────────────────────")
        print(f"  Accuracy : {self.metrics['accuracy']}")
        print(f"  Precision: {self.metrics['precision']}")
        print(f"  Recall   : {self.metrics['recall']}")
        print(f"  F1-Score : {self.metrics['f1_score']}")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        print("[5/5] Saving artefacts …")
        self._save()
        self.is_trained = True
        return True

    def _save(self):
        joblib.dump(self.model, "model.pkl")
        self.preprocessor.save("tfidf_vectorizer.pkl")
        joblib.dump(self.metrics, "model_metrics.pkl")
        # Also save as JSON for the GitHub Pages site
        with open("docs/model_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        print("  Saved: model.pkl, tfidf_vectorizer.pkl, model_metrics.pkl, docs/model_metrics.json")

    # ── loading ─────────────────────────────────────────────────────────
    def load(self):
        """Load previously-saved model artefacts."""
        self.model = joblib.load("model.pkl")
        self.preprocessor.load("tfidf_vectorizer.pkl")
        self.metrics = joblib.load("model_metrics.pkl")
        self.is_trained = True

    # ── prediction ──────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """Return prediction dict for a single text string."""
        if not self.is_trained:
            self.load()

        cleaned = self.preprocessor.clean_text(text)
        vec = self.preprocessor.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0] if hasattr(self.model, "predict_proba") else None

        if proba is not None:
            confidence = float(max(proba))
            probabilities = {"Real": float(proba[0]), "Fake": float(proba[1])}
        else:
            confidence = 1.0
            probabilities = {"Real": float(1 - pred), "Fake": float(pred)}

        return {
            "prediction": "Fake" if pred == 1 else "Real",
            "confidence": confidence,
            "probabilities": probabilities,
        }


def train_all_models(dataset_path="FakeNewsNet.csv"):
    """Train every available model and print a comparison table."""
    results = {}
    for name in MODELS:
        print(f"\n{'='*50}\n  MODEL: {name}\n{'='*50}")
        m = FakeNewsModel(name)
        m.train(dataset_path)
        results[name] = m.metrics

    print("\n\n══ Comparison ══════════════════════════════════")
    header = f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print(header)
    print("─" * len(header))
    for name, met in results.items():
        print(f"{name:<22} {met['accuracy']:>7.4f} {met['precision']:>7.4f} {met['recall']:>7.4f} {met['f1_score']:>7.4f}")
    return results


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    train_all_models()