#!/usr/bin/env python3
"""
Quick start script for Fake News Detection App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install requirements!")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download NLTK data: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    print("Starting Fake News Detection App...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Failed to run app: {e}")

def main():
    print("Fake News Detection App Setup")
    print("=" * 40)
    
    # Check if requirements are installed
    try:
        import streamlit
        import sklearn
        import nltk
        print("Main packages already installed")
    except ImportError:
        if not install_requirements():
            return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    print("\nStarting web application...")
    print("The app will open in your default browser")
    print("Press Ctrl+C to stop the application")
    print("-" * 40)
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()