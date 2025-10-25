#!/usr/bin/env python3
"""
spam_classifier.py
A step-by-step spam classifier:
- Load dataset (CSV like the SMS Spam Collection, or provide your own)
- Clean text
- TF-IDF vectorize
- Train MultinomialNB (fast) or LinearSVC (stronger)
- Evaluate (accuracy, precision, recall, F1) + confusion matrix
- Save pipeline to disk
"""

import argparse
import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------
# Utility / Preprocessing
# ---------------------------
def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs/emails/numbers/punctuation, extra spaces."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)           # remove urls
    s = re.sub(r"\S+@\S+", " ", s)                   # remove emails
    s = re.sub(r"\d+", " ", s)                       # remove numbers
    s = re.sub(r"[^\w\s]", " ", s)                   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()               # collapse whitespace
    return s

def load_sms_like_csv(path: str) -> pd.DataFrame:
    """Loads spam.csv (handles common Kaggle or TSV versions automatically)."""
    import chardet
    import io

    # detect encoding
    with open(path, 'rb') as f:
        enc = chardet.detect(f.read(20000))['encoding']

    # try flexible read
    df = pd.read_csv(path, encoding=enc, sep=None, engine='python')
    
    # handle Kaggle format
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    # handle UCI or TSV format
    elif 'label' in df.columns and 'message' in df.columns:
        pass
    # handle possible weird spacing/tab headers
    else:
        df.columns = [c.strip().lower() for c in df.columns]
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        elif len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'message'})
        else:
            raise ValueError("Could not map columns. Expected 2 columns (label/message).")

    # clean up text
    df = df[['label', 'message']].dropna()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].replace({'0': 'ham', '1': 'spam'})
    df = df[df['label'].isin(['ham', 'spam'])]
    df['message_clean'] = df['message'].astype(str)
    return df

# ---------------------------
# Training + Evaluation
# ---------------------------
def train_and_evaluate(df: pd.DataFrame, classifier='nb', test_size=0.2, random_state=42, save_path='spam_pipeline.joblib'):
    """
    Train and evaluate.
    classifier: 'nb' (MultinomialNB) or 'svc' (LinearSVC with optional GridSearch)
    """
    X = df['message_clean'].values
    y = df['label'].map({'ham': 0, 'spam': 1}).values

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    if classifier == 'nb':
        print("[Step] Building pipeline: TF-IDF -> MultinomialNB")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))),
            ('clf', MultinomialNB())
        ])
        pipeline.fit(X_train, y_train)
    elif classifier == 'svc':
        print("[Step] Building pipeline: TF-IDF -> LinearSVC")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000)),
            ('clf', LinearSVC(max_iter=10000))
        ])
        # optional grid search for SVC (small grid to keep run-time reasonable)
        param_grid = {
            'tfidf__ngram_range': [(1,1), (1,2)],
            'tfidf__max_df': [0.75, 1.0],
            'clf__C': [0.1, 1, 5]
        }
        print("[Step] Running GridSearchCV (this may take a bit)...")
        gs = GridSearchCV(pipeline, param_grid, cv=4, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        pipeline = gs.best_estimator_
    else:
        raise ValueError("classifier must be 'nb' or 'svc'")

    # predict & evaluate
    y_pred = pipeline.predict(X_test)
    print("\n---- Evaluation ----")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # confusion matrix and save plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['ham','spam'])
    ax.set_yticklabels(['ham','spam'])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='white' if val > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to confusion_matrix.png")

    # save pipeline
    joblib.dump(pipeline, save_path)
    print(f"Saved trained pipeline to: {save_path}")
    return pipeline

# ---------------------------
# Inference helper
# ---------------------------
def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pipeline file found at {path}")
    return joblib.load(path)

def predict_one(pipeline, text: str):
    clean = clean_text(text)
    pred = pipeline.predict([clean])[0]
    return 'spam' if int(pred) == 1 else 'ham'

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Spam classifier: train and run a spam/ham model.")
    parser.add_argument('--data', type=str, default='spam.csv', help="Path to CSV dataset (spam.csv style)")
    parser.add_argument('--model', type=str, choices=['nb','svc'], default='nb', help="Model: 'nb' (fast) or 'svc' (slower, stronger)")
    parser.add_argument('--save', type=str, default='spam_pipeline.joblib', help="Where to save trained pipeline")
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    print("[Step 1] Loading dataset:", args.data)
    df = load_sms_like_csv(args.data)
    print("Dataset loaded: {} examples ({} spam, {} ham)".format(len(df), sum(df['label']=='spam'), sum(df['label']=='ham')))

    print("[Step 2] Training classifier:", args.model)
    pipeline = train_and_evaluate(df, classifier=args.model, test_size=args.test_size, save_path=args.save)

    print("\n[Step 3] Quick test (3 sample messages):")
    samples = [
        "Congratulations! You've won a free ticket. Call 0800-123-456 to claim.",
        "Hey, are we still meeting tomorrow for lunch?",
        "URGENT: Update your account information at http://fakebank.example/login"
    ]
    for s in samples:
        print("Text:", s)
        print("Predicted:", predict_one(pipeline, s))
        print("-" * 40)

if __name__ == "__main__":
    main()
