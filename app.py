
import os
import json
import uuid
import logging

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model


# =========================================================
# 1. LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cropcare-backend")


# =========================================================
# 2. CPU CONFIGURATION FOR RENDER
# =========================================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass


# =========================================================
# 3. FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# =========================================================
# 4. PROJECT PATHS
# =========================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(MODEL_DIR, "best_efficientnetb0.keras")
)

CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH",
    os.path.join(MODEL_DIR, "class_names.json")
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

INDEX_PATH = os.path.join(BASE_DIR, "index.html")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# 5. MODEL CONFIGURATION
# =========================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 50

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "bmp", "webp"
}


# =========================================================
# 6. LOAD CLASS NAMES
# =========================================================

logger.info("Loading disease classes...")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if not isinstance(class_names, list):
    raise ValueError("class_names.json must contain a list")

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(class_names)}"
    )

if len(set(class_names)) != NUM_CLASSES:
    raise ValueError("Duplicate class names detected")

logger.info("Loaded %d classes", len(class_names))


# =========================================================
# 7. DATA AUGMENTATION
# MUST MATCH TRAINING ARCHITECTURE
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10)
], name="data_augmentation")


# =========================================================
# 8. BUILD EFFICIENTNETB0 MODEL
# =========================================================

def build_model():

    logger.info("Building EfficientNetB0...")

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


# =========================================================
# 9. LOAD TRAINED WEIGHTS
# =========================================================

logger.info("Loading trained model weights...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = build_model()

# This assumes the checkpoint is weights-only
model.load_weights(MODEL_PATH)

logger.info("Model weights loaded successfully")
logger.info("Input shape: %s", model.input_shape)
logger.info("Output shape: %s", model.output_shape)

if model.output_shape[-1] != len(class_names):
    raise ValueError("Model output count and class count mismatch")


# =========================================================
# 10. FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# 11. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        resampling = getattr(
            Image,
            "Resampling",
            Image
        )

        image = image.resize(
            IMG_SIZE,
            resampling.BILINEAR
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Keep 0-255 pixel values.
    # Standard EfficientNetB0 includes its own rescaling layer.

    return image_array


# =========================================================
# 12. PREDICTION
# =========================================================

def predict_disease(image_path):

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    logger.info(
        "Prediction: %s | Confidence: %.2f%%",
        predicted_class,
        confidence * 100
    )

    return predicted_class, confidence


# =========================================================
# 13. HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    if os.path.exists(INDEX_PATH):
        return send_from_directory(BASE_DIR, "index.html")

    return jsonify({
        "status": "API is running",
        "model": "EfficientNetB0",
        "classes": NUM_CLASSES
    })


# =========================================================
# 14. HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "API is running",
        "model_loaded": model is not None,
        "model": "EfficientNetB0",
        "classes": len(class_names),
        "input_size": "224x224"
    })


# =========================================================
# 15. GET CLASSES
# =========================================================

@app.route("/classes", methods=["GET"])
def get_classes():

    return jsonify({
        "total_classes": len(class_names),
        "classes": class_names
    })


# =========================================================
# 16. PREDICTION API
# RESPONSE FORMAT MATCHES OLD MODEL
# =========================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    logger.info("Prediction API called")

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type",
            "allowed": list(ALLOWED_EXTENSIONS)
        }), 400

    file_path = None

    try:

        original_name = secure_filename(file.filename)

        unique_name = (
            f"{uuid.uuid4().hex}_{original_name}"
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(file_path)

        logger.info("Image uploaded: %s", unique_name)

        # Verify uploaded image
        with Image.open(file_path) as image:
            image.verify()

        # Run prediction
        predicted_class, confidence = predict_disease(
            file_path
        )

        # Generate image URL
        file_url = (
            f"{request.host_url.rstrip('/')}"
            f"/static/uploads/{unique_name}"
        )

        # SAME RESPONSE FIELD NAMES AS OLD MODEL
        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "image_url": file_url
        }), 200

    except (UnidentifiedImageError, OSError):

        return jsonify({
            "error": "Uploaded file is not a valid image"
        }), 400

    except Exception:

        logger.exception("Prediction failed")

        return jsonify({
            "error": "Image prediction failed"
        }), 500

    finally:

        if file_path and os.path.exists(file_path):

            try:
                os.remove(file_path)

            except OSError:
                logger.warning("Could not delete temporary image")


# =========================================================
# 17. FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image exceeds the 8 MB upload limit"
    }), 413


# =========================================================
# 18. RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    logger.info("Starting CropCare API...")
    logger.info("Model: EfficientNetB0")
    logger.info("Classes: %d", NUM_CLASSES)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
