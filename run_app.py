#!/usr/bin/env python3
"""
Quick-start script — installs deps, trains model if needed, launches dashboard.
"""

import subprocess
import sys
import os


def install_requirements():
    print("📦 Installing requirements…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
    )
    print("   Done")


def ensure_model():
    if os.path.exists("model.pkl"):
        print("🤖 Pre-trained model found")
        return
    print("🤖 Training model on FakeNewsNet.csv …")
    os.makedirs("docs", exist_ok=True)
    from model_training import FakeNewsModel
    m = FakeNewsModel("logistic")
    m.train("FakeNewsNet.csv")


def main():
    print("═" * 44)
    print("  🛡️  Fake News Detector — Quick Start")
    print("═" * 44)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        install_requirements()

    ensure_model()

    print("\n🚀 Launching Streamlit dashboard…")
    print("   Press Ctrl+C to stop\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()