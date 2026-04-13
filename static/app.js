const form = document.getElementById("predictForm");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submitBtn");

const resultSection = document.getElementById("result");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const reviewText = document.getElementById("reviewText");
const confidenceBar = document.getElementById("confidenceBar");
const errorText = document.getElementById("errorText");

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    preview.hidden = true;
    preview.removeAttribute("src");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.hidden = false;
  };
  reader.readAsDataURL(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorText.textContent = "";

  const file = imageInput.files?.[0];
  if (!file) {
    errorText.textContent = "Please select an image.";
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const body = new FormData();
    body.append("image", file);

    const response = await fetch("/predict", {
      method: "POST",
      body,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }

    const confidence = Number(payload.confidence || 0);
    predictionText.textContent = `Prediction: ${payload.prediction}`;
    confidenceText.textContent = `Confidence: ${confidence.toFixed(2)}`;
    reviewText.textContent = payload.label || "";
    confidenceBar.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
    resultSection.hidden = false;
  } catch (error) {
    errorText.textContent = error.message || "Something went wrong.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});
