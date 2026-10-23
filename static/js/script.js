const form = document.getElementById("prediction-form");
const result = document.getElementById("result");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const symptomsSelect = document.getElementById("symptoms");

    const symptoms = Array.from(symptomsSelect.selectedOptions)
        .map(option => option.value);

    if (symptoms.length === 0) {
        result.textContent = "Please select at least one symptom.";
        return;
    }

    result.textContent = "Processing your request...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ symptoms: symptoms })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction request failed.");
        }

        result.textContent = data.message;

    } catch (error) {
        result.textContent =
            "Unable to get a prediction. Please try again.";
        console.error(error);
    }
});