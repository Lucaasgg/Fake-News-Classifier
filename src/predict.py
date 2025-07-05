import argparse
import joblib
import os
import numpy as np


def load_model(model_path: str):
    """Load the calibrated model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model


def predict_texts(texts: list, model):
    """Return labels and probabilities for a list of raw texts using the calibrated pipeline."""
    # model is a CalibratedClassifierCV wrapping a Pipeline (tfidf + classifier)
    labels = model.predict(texts)
    # Determine index of 'real' class
    idx_real = list(model.classes_).index('real')
    probs_real = model.predict_proba(texts)[:, idx_real]
    return labels, probs_real


def main():
    parser = argparse.ArgumentParser(
        description="Classify news texts as real or fake using a pretrained calibrated model."
    )
    parser.add_argument(
        'texts', nargs='+', help='One or more texts (full article or headline) to classify.'
    )
    parser.add_argument(
        '--model', default='models/final_fake_news_model.joblib',
        help='Path to the trained calibrated model.'
    )
    args = parser.parse_args()

    model = load_model(args.model)
    labels, probs = predict_texts(args.texts, model)

    for text, label, prob in zip(args.texts, labels, probs):
        snippet = text if len(text) < 80 else text[:77] + '...'
        print(f"Input: {snippet}\n--> Predicted: {label} (P(real)={prob:.3f})\n")


if __name__ == '__main__':
    main()
