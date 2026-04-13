import os

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model_utils import Predictor


DEFAULT_MODEL_PATH = "artifacts/convnext_mvp.pt"
CONFIDENCE_REVIEW_THRESHOLD = 0.70

app = Flask(__name__)

_predictor = None
_model_load_error = None


def get_predictor():
    global _predictor, _model_load_error

    if _predictor is not None:
        return _predictor

    if _model_load_error is not None:
        raise RuntimeError(_model_load_error)

    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        candidates = [
            DEFAULT_MODEL_PATH,
            "artifacts/convnext_mvp_new_new.pt",
        ]
        existing = [p for p in candidates if os.path.exists(p)]
        model_path = existing[0] if existing else DEFAULT_MODEL_PATH

    try:
        _predictor = Predictor(model_path)
        return _predictor
    except Exception as exc:  # noqa: BLE001
        _model_load_error = f"path={model_path} | {exc}"
        raise RuntimeError(_model_load_error) from exc


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Invalid image"}), 400

    try:
        predictor = get_predictor()
    except RuntimeError as exc:
        print(f"Error loading model: {exc}")
        return (
            jsonify(
                {
                    "error": "Model is not available. Train first or set MODEL_PATH.",
                    "detail": str(exc),
                }
            ),
            503,
        )

    result = predictor.predict(image)

    confidence = result["confidence"]
    response = {
        "prediction": result["prediction"],
        "confidence": confidence,
    }

    if confidence < CONFIDENCE_REVIEW_THRESHOLD:
        response["label"] = "Needs further evaluation"

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
