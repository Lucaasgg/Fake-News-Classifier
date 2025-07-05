# Fake News Classifier / Machine Learning Teamwork

A complete pipeline to detect real vs. fake news articles using TF-IDF, Logistic Regression, and probability calibration. Includes:

* **Exploratory Data Analysis (EDA)**
* **Model training** with hyperparameter tuning and calibration
* **Command‑line interface** (`predict.py`) for batch or single‑article prediction

---

## Repository Structure

```
ML_Teamwork/
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── processed/
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── final_fake_news_model.joblib
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_InspectModels.ipynb
├── src/
│   ├── train_and_calibrate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/ML_Teamwork.git
   cd ML_Teamwork
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Data Description

* `data/raw/Fake.csv`: \~23 502 fake news articles
* `data/raw/True.csv`: \~21 417 real news articles

Both files contain columns:

| Column  | Description               |
| ------- | ------------------------- |
| title   | Headline of the article   |
| text    | Body text of the article  |
| subject | Category/subject label    |
| date    | Publication date (string) |

Labels are automatically assigned during preprocessing.

---

## Exploratory Data Analysis

Explore the dataset using `notebooks/01_EDA.ipynb`:

* **Class balance**: \~52.3% fake vs. 47.7% real
* **Subject distribution**, **date distribution**, **length metrics**
* **Word clouds** for each class
* **Top n‑grams** and **PCA** on TF-IDF vectors

Visualizations help validate separability and guide feature engineering.

---

## Model Training & Calibration

Use `src/train_and_calibrate.py` to:

1. **Load** and label `title + text` as `content`.
2. **Train/test split** (80/20, stratified).
3. **Pipeline**: TF-IDF (5 000 features, unigrams + bigrams) + Logistic Regression.
4. **Hyperparameter tuning** via GridSearchCV (C ∈ {0.01,0.1,1,10}, ngram\_range ∈ {(1,1),(1,2)}) with an F1 scorer on the “real” class.
5. **Calibrate** probabilities with Platt scaling (`CalibratedClassifierCV(method='sigmoid', cv=5)`).
6. **Evaluation**: classification report, ROC AUC (\~0.9997), and optimal threshold (\~0.573).
7. **Outputs** saved to `models/`:

   * `models/tfidf_vectorizer.joblib`
   * `models/final_fake_news_model.joblib`

**Run**:

```bash
python src/train_and_calibrate.py
```

(This may take several minutes depending on CPU cores.)

---

## Predicting New Articles

Use `src/predict.py` for inference on new texts.

**Usage**:

```bash
python src/predict.py "Your full article or headline here"
```

**Output**:

```
Input: Your full article or ...
--> Predicted: real (P(real)=0.82)
```

`predict.py` loads the calibrated pipeline directly and dynamically finds the index for the “real” class. Supports multiple texts in one call.

---

## Requirements

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
wordcloud
joblib
```

(Full versions in `requirements.txt`.)

---

## Next Steps & Improvements

* Deploy as a REST API (e.g., using FastAPI or Flask).
* Monitor data drift & schedule periodic retraining.
* Experiment with transformer-based models (e.g., BERT) for deeper context.
* Add unit tests and continuous integration.

---
## Team Members
Machine learning teamwork done by Lucas Garcia Garcia , Jaime del Reguero Garcia , Alejandro Muñoz Suarez
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset this is the chosen teamwork.
