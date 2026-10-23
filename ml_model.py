
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


DATASET_PATH = Path(__file__).parent / "data" / "dataset.csv"

# Stores evaluation results for the dashboard.
MODEL_METRICS = {}


def train_model():
    global MODEL_METRICS

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    if "Disease" not in df.columns:
        raise ValueError(
            "dataset.csv must contain a Disease column."
        )

    df = df.dropna(subset=["Disease"]).copy()
    df["Disease"] = df["Disease"].astype(str).str.strip()

    symptom_columns = [
        column for column in df.columns
        if column.lower().startswith("symptom")
    ]

    if not symptom_columns:
        raise ValueError(
            "No symptom columns were found in dataset.csv."
        )

    all_symptoms = sorted({
        str(value).strip()
        for column in symptom_columns
        for value in df[column].dropna()
        if str(value).strip()
    })

    if not all_symptoms:
        raise ValueError(
            "The dataset contains no symptom values."
        )

    symptom_names = all_symptoms
    symptom_to_index = {
        symptom: index
        for index, symptom in enumerate(symptom_names)
    }

    X = pd.DataFrame(
        0, index=df.index, columns=symptom_names, dtype=int
    )

    for _, row in df.iterrows():
        for column in symptom_columns:
            value = row[column]

            if pd.notna(value):
                symptom = str(value).strip()

                if symptom in symptom_to_index:
                    X.at[row.name, symptom] = 1

    y = df["Disease"]

    class_counts = y.value_counts()
    can_stratify = (
        len(class_counts) > 1
        and class_counts.min() >= 2
        and int(len(y) * 0.25) >= len(class_counts)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    model = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate evaluation metrics using held-out test data.
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    disease_labels = sorted(y.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=disease_labels,
    )

    # Store dashboard-friendly results.
    MODEL_METRICS = {
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "symptom_count": int(len(symptom_names)),
        "disease_count": int(y.nunique()),
        "disease_labels": disease_labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=disease_labels,
            output_dict=True,
            zero_division=0,
        ),
        "evaluation_note": (
            "Evaluation on a held-out portion of the supplied "
            "educational dataset. Not clinical validation."
        ),
    }

    print(f"Dataset rows: {len(df)}")
    print(f"Unique symptoms: {len(symptom_names)}")
    print(f"Disease labels: {y.nunique()}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (weighted): {precision * 100:.2f}%")
    print(f"Recall (weighted): {recall * 100:.2f}%")
    print(f"F1-score (weighted): {f1 * 100:.2f}%")
    print(
        "These results are educational dataset metrics, "
        "not evidence of clinical accuracy."
    )

    # Preserve the existing return format.
    return model, symptom_names
