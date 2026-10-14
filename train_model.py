# cspell:ignore TfidfVectorizer LogisticRegression
from pathlib import Path
import json
import joblib  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

DATA_PATH = Path("data/news.csv")
MODEL_PATH = Path("models/fake_news_pipeline.joblib")
METRICS_PATH = Path("models/evaluation_metrics.json")


def find_column(columns, candidates):
    lookup = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found at data/news.csv. Read README.md for the required CSV format."
        )

    df = pd.read_csv(DATA_PATH)
    text_col = find_column(df.columns, ["text", "article", "content", "news", "headline", "title"])
    label_col = find_column(df.columns, ["label", "class", "target", "category", "status"])

    if text_col is None or label_col is None:
        raise ValueError(
            "Could not identify text and label columns. Use columns named 'text' and 'label'."
        )

    data = df[[text_col, label_col]].copy()
    data.columns = ["text", "label"]
    data = data.dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["label"] != "")]
    data = data.drop_duplicates(subset=["text", "label"])

    if data["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two different labels.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two rows for a stratified train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset_rows_after_cleaning": int(len(data)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": [str(label) for label in pipeline.classes_],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=pipeline.classes_
        ).tolist(),
        "classification_report": report,
        "note": "Scores apply only to this held-out split and dataset; they do not prove factual truth."
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Rows used: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows=true labels, columns=predicted labels):")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
