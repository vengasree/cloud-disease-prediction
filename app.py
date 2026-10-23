
from flask import Flask, jsonify, render_template, request
from ml_model import train_model

app = Flask(__name__)

model, symptom_names = train_model()


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptom_names=symptom_names
    )


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])

    if not isinstance(symptoms, list) or not symptoms:
        return jsonify({
            "error": "Please select at least one symptom."
        }), 400

    if any(not isinstance(s, str) for s in symptoms):
        return jsonify({
            "error": "Invalid symptom selection."
        }), 400

    # Match the submitted symptoms against the dataset vocabulary.
    selected = {s.strip().lower() for s in symptoms}
    symptom_lookup = {
        symptom.lower(): symptom for symptom in symptom_names
    }

    invalid = selected - set(symptom_lookup)

    if invalid:
        return jsonify({
            "error": (
                "Some selected symptoms are not supported by the "
                "dataset model. Please update the website symptom "
                "options to match the dataset."
            )
        }), 400

    features = [
        1 if symptom.lower() in selected else 0
        for symptom in symptom_names
    ]

    prediction = model.predict([features])[0]

    return jsonify({
        "message": (
            f"Dataset model output: {prediction}. "
            "This is an educational prediction only, not a diagnosis."
        )
    })


if __name__ == "__main__":
    app.run(debug=True)
