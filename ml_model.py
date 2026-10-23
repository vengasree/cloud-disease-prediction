from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    # Small illustrative dataset for testing the ML pipeline.
    # These examples are NOT medically validated.
    symptom_names = [
        "fever",
        "cough",
        "headache",
        "fatigue",
        "sore_throat",
        "body_pain",
    ]

    examples = [
        [1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ]

    # Arbitrary demonstration labels, not real disease diagnoses.
    labels = [
        "Example Group A",
        "Example Group B",
        "Example Group A",
        "Example Group B",
        "Example Group B",
        "Example Group A",
        "Example Group A",
        "Example Group B",
        "Example Group B",
        "Example Group A",
        "Example Group B",
        "Example Group A",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        examples,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Demo model test accuracy:", accuracy_score(y_test, predictions))

    return model, symptom_names


if __name__ == "__main__":
    train_model()