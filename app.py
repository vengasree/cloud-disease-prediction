
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

import ml_model
from ml_model import train_model

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Keep the existing trained-model workflow.
model, symptom_names = train_model()


def load_csv(filename):
    """Load a CSV file if it exists."""
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


description_df = load_csv("symptom_Description.csv")
precaution_df = load_csv("symptom_precaution.csv")


def find_disease_row(df, disease):
    """Find a disease row without case-sensitive matching."""
    if df.empty or "Disease" not in df.columns:
        return None

    matches = df[
        df["Disease"].astype(str).str.strip().str.casefold()
        == disease.strip().casefold()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptom_names=symptom_names,
        metrics=ml_model.MODEL_METRICS
    )


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/model-metrics")
def model_metrics():
    return jsonify({
        key: value
        for key, value in ml_model.MODEL_METRICS.items()
        if key != "classification_report"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    age = data.get("age")
    gender = str(data.get("gender", "")).strip()
    symptoms = data.get("symptoms", [])

    if not name:
        return jsonify({"error": "Please enter the patient name."}), 400

    try:
        age = int(age)
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter a valid age."}), 400

    if not 1 <= age <= 120:
        return jsonify({"error": "Age must be between 1 and 120."}), 400

    if gender not in {"Female", "Male", "Other", "Prefer not to say"}:
        return jsonify({"error": "Please select a valid gender option."}), 400

    if not isinstance(symptoms, list) or not symptoms:
        return jsonify({
            "error": "Please select at least one symptom."
        }), 400

    if any(not isinstance(s, str) for s in symptoms):
        return jsonify({"error": "Invalid symptom selection."}), 400

    symptom_lookup = {
        symptom.casefold(): symptom for symptom in symptom_names
    }
    selected = {s.strip().casefold() for s in symptoms}
    invalid = selected - set(symptom_lookup)

    if invalid:
        return jsonify({
            "error": "One or more selected symptoms are unsupported."
        }), 400

    # Build features in the exact order used to train the model.
    selected_names = {
        symptom_lookup[s] for s in selected
    }

    features = pd.DataFrame(
        [[int(symptom in selected_names) for symptom in symptom_names]],
        columns=symptom_names
    )

    prediction = str(model.predict(features)[0])

    return jsonify({
        "patient": {
            "name": name,
            "age": age,
            "gender": gender
        },
        "symptoms": sorted(selected_names),
        "prediction": prediction,
        "message": (
            "This is an educational model prediction, "
            "not a medical diagnosis."
        )
    })


@app.route("/disease-search", methods=["POST"])
def disease_search():
    data = request.get_json(silent=True) or {}
    query = str(data.get("disease", "")).strip()

    if not query:
        return jsonify({
            "error": "Please enter a disease name."
        }), 400

    # Search the disease names present in the training dataset.
    disease_names = sorted({
        str(name).strip()
        for name in model.named_steps["randomforestclassifier"].classes_
    }, key=str.casefold)

    exact_match = next(
        (name for name in disease_names
         if name.casefold() == query.casefold()),
        None
    )

    if exact_match:
        matches = [exact_match]
    else:
        matches = [
            name for name in disease_names
            if query.casefold() in name.casefold()
        ][:10]

    if not matches:
        return jsonify({
            "error": (
                "No matching disease was found in the model's dataset. "
                "Try a disease name from the available dataset."
            )
        }), 404

    results = []

    for disease in matches:
        description_row = find_disease_row(description_df, disease)
        precaution_row = find_disease_row(precaution_df, disease)

        description = ""
        if description_row is not None and "Description" in description_row.index:
            value = description_row["Description"]
            if pd.notna(value):
                description = str(value).strip()

        precautions = []
        if precaution_row is not None:
            for column in precaution_df.columns:
                if column.casefold().startswith("precaution"):
                    value = precaution_row[column]
                    if pd.notna(value) and str(value).strip():
                        precautions.append(str(value).strip())

        results.append({
            "disease": disease,
            "description": description or
                "No description is available in the supplied dataset.",
            "precautions": precautions
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
