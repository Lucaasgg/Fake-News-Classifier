
import joblib

# Carga artefactos
tfidf = joblib.load('../models/tfidf_vectorizer.joblib')
model = joblib.load('../models/final_fake_news_model.joblib')

def predict(texts):
    """
    texts: lista de strings
    devuelve: lista de tuplas (label, prob_real)
    """
    X = tfidf.transform(texts)
    probs = model.predict_proba(X)
    preds = model.predict(X)
    out = []
    for p, prob in zip(preds, probs):
        label = 'real' if p==1 else 'fake'
        out.append((label, prob[p]))
    return out

if __name__ == "__main__":
    # Ejemplo de uso desde línea de comandos
    import sys
    inputs = sys.argv[1:]
    if not inputs:
        print("Usage: python predict.py \"Some news text here\" \"Another text\"")
        sys.exit(1)
    results = predict(inputs)
    for txt, (label, prob) in zip(inputs, results):
        print(f"{label} (p={prob:.2f}): {txt}")
