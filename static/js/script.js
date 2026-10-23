
const form = document.getElementById("symptom-form");
const result = document.getElementById("result");
const predictButton = document.getElementById("predict-button");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const symptoms = Array.from(
        form.querySelectorAll('input[name="symptoms"]:checked')
    ).map((checkbox) => checkbox.value);

    result.hidden = false;
    result.classList.remove("error");

    if (symptoms.length === 0) {
        result.textContent = "Please select at least one symptom.";
        result.classList.add("error");
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Processing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ symptoms })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unable to process the request.");
        }

        result.textContent = data.message || "No result was returned.";
    } catch (error) {
        result.textContent = error.message || "Something went wrong. Please try again.";
        result.classList.add("error");
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Check Prediction →";
    }
});
