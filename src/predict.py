#!/usr/bin/env python3

import sys
import joblib
from pathlib import Path

# Define the project root and models directory
BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

# Load the TF-IDF vectorizer and the calibrated model
tfidf = joblib.load(MODELS / "tfidf_vectorizer.joblib")
model = joblib.load(MODELS / "final_fake_news_model.joblib")

def predict_texts(texts, threshold=0.7):
    """
    Classify a list of news‐style texts as 'real' or 'fake'.

    Parameters:
    - texts: list of str, each a news headline or article snippet.
    - threshold: float, cutoff on P(real) above which we label as 'real'.
    """
    # Transform raw texts into the TF-IDF feature space
    X = tfidf.transform(texts)
    # Get the probability assigned to the 'real' class
    probs = model.predict_proba(X)[:, 1]  

    for txt, p in zip(texts, probs):
        # Determine the final label based on the threshold
        label = 'real' if p >= threshold else 'fake'
        # Print both probabilities: P(real) and P(fake)
        print(
            f"📰 {txt}\n"
            f"    → Predicted: {label} "
            f"(p_real={p:.2f}, p_fake={(1-p):.2f})\n"
        )

if __name__ == "__main__":
    # Read command-line arguments (skip script name)
    inputs = sys.argv[1:]
    # If no inputs provided, print usage instructions
    if not inputs:
        print('Usage: python predict.py "Some news text" "Another headline"')
        sys.exit(1)

    # Run prediction on provided texts
    predict_texts(inputs)
