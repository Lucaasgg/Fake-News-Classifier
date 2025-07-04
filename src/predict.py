import sys
import joblib

def classify_texts(texts, threshold=0.8):
    """
    texts: list of strings
    threshold: cutoff for labeling 'real' vs 'fake'
    """
    # Load TF-IDF vectorizer and calibrated model
    tfidf      = joblib.load('../models/tfidf_vectorizer.joblib')
    calibrator = joblib.load('../models/calibrator.joblib')

    X     = tfidf.transform(texts)
    probs = calibrator.predict_proba(X)[:, 1]  # probability of 'real'
    preds = (probs >= threshold).astype(int)

    for txt, p, pr in zip(texts, probs, preds):
        label = 'real' if pr == 1 else 'fake'
        print(f"📰 {txt}\n    → Predicted: {label} (p_real={p:.2f}, p_fake={(1-p):.2f})\n")

if __name__ == "__main__":
    inputs = sys.argv[1:]
    if not inputs:
        print("Usage: python predict.py \"Some news text here\"")
        sys.exit(1)
    classify_texts(inputs, threshold=0.8)
