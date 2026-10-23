from flask import Flask, jsonify, render_template, request
from ml_model import train_model

app = Flask(__name__)

model, symptom_names = train_model()


@app.route("/")
def home():
    return render_template("index.html")


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

    allowed_symptoms = set(symptom_names)

    if any(
        not isinstance(symptom, str) or symptom not in allowed_symptoms
        for symptom in symptoms
    ):
        return jsonify({
            "error": "One or more symptoms are invalid."
        }), 400

    # Convert selected symptoms into the feature format expected by the model.
    features = [
        1 if symptom in symptoms else 0
        for symptom in symptom_names
    ]

    prediction = model.predict([features])[0]

    return jsonify({
        "message": (
            f"Demo model output: {prediction}. "
            "This is an illustrative result, not a medical diagnosis."
        )
    })


if __name__ == "__main__":
    app.run(debug=True)
