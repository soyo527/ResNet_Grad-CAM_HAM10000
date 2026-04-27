from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

PREDICTORS: dict[str, SkinLesionPredictor] = {}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARCHITECTURES = ("resnet18", "resnet50", "resnet101")


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_rgb_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def get_predictor(architecture: str) -> SkinLesionPredictor:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if architecture not in PREDICTORS:
        from inference import SkinLesionPredictor

        PREDICTORS[architecture] = SkinLesionPredictor(architecture=architecture)
    return PREDICTORS[architecture]


def read_upload():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        raise ValueError("Please upload an image.")
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Supported formats: JPG, PNG, BMP, WEBP.")
    return uploaded.read()


@app.get("/")
def index():
    return render_template("index.html", architectures=ARCHITECTURES, result=None, error=None)


@app.post("/")
def classify_page():
    architecture = request.form.get("architecture", "resnet50")
    try:
        image_bytes = read_upload()
        image = load_rgb_image(image_bytes)
        result = get_predictor(architecture).predict(image)
        result["source_image"] = encode_png(image)
        return render_template("index.html", architectures=ARCHITECTURES, result=result, error=None)
    except Exception as exc:
        return render_template("index.html", architectures=ARCHITECTURES, result=None, error=str(exc)), 400


@app.post("/api/predict")
def classify_api():
    architecture = request.form.get("architecture", "resnet50")
    try:
        image = load_rgb_image(read_upload())
        return jsonify(get_predictor(architecture).predict(image))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
