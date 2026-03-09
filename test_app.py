#!/usr/bin/env python3
"""
Test script — verifies model loading, prediction, and API imports.
"""

import sys
import os


def test_imports():
    """Test all required imports."""
    print("Testing imports…")
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import joblib
        import plotly
        import flask
        import flask_cors
        import requests
        import bs4
        print("  ✅ All imports OK")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_model():
    """Test model train → predict cycle."""
    print("Testing model…")
    from model_training import FakeNewsModel

    model = FakeNewsModel("logistic")

    if os.path.exists("model.pkl"):
        model.load()
        print("  Loaded pre-trained model")
    else:
        print("  No pre-trained model; training on FakeNewsNet.csv …")
        os.makedirs("docs", exist_ok=True)
        assert model.train("FakeNewsNet.csv"), "Training failed"

    tests = [
        ("Government announces new education policy to improve literacy rates", "Real"),
        ("Celebrity endorses miracle cure doctors hate and want to hide from you", "Fake"),
        ("Apple reports record quarterly revenue driven by strong iPhone sales", "Real"),
        ("Breaking: Scientists confirm earth is hollow with underground cities", "Fake"),
    ]

    print("  Predictions:")
    for text, expected in tests:
        r = model.predict(text)
        match = "✅" if r["prediction"] == expected else "⚠️"
        print(f"    {match}  {r['prediction']:4s} ({r['confidence']:.1%})  ← {text[:50]}…")

    print("  ✅ Model test passed")
    return True


def test_api_import():
    """Verify the Flask app object can be created."""
    print("Testing API module…")
    try:
        from api import app  # noqa: F401
        print("  ✅ API module OK")
        return True
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return False


if __name__ == "__main__":
    print("═" * 50)
    print("  Fake News Detector — Test Suite")
    print("═" * 50)

    ok = test_imports() and test_model() and test_api_import()

    print("═" * 50)
    if ok:
        print("✅ All tests passed!")
        print("Run  streamlit run app.py  to start the dashboard")
        print("Run  python api.py         to start the REST API")
    else:
        print("❌ Some tests failed")
        sys.exit(1)