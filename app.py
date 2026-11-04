
import os
import json
import uuid
import logging

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("plant-disease-api")


# =========================================================
# CPU CONFIGURATION
# =========================================================

# Render free tier uses CPU inference.
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        BASE_DIR,
        "models",
        "best_efficientnetb0.keras"
    )
)

CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH",
    os.path.join(
        BASE_DIR,
        "models",
        "class_names.json"
    )
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

EXPECTED_CLASSES = 50

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# LOAD CLASS NAMES
# =========================================================

logger.info("Loading disease classes...")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:

    class_names = json.load(file)


if not isinstance(class_names, list):
    raise ValueError(
        "class_names.json must contain a JSON list."
    )

if len(class_names) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"found {len(class_names)}"
    )

logger.info("Loaded %d classes", len(class_names))


# =========================================================
# CUSTOM OBJECTS
# =========================================================

def silu(x):

    return x * tf.nn.sigmoid(x)


# =========================================================
# LOAD MODEL
# =========================================================

model = None

try:

    logger.info("Loading EfficientNetB0 model...")
    logger.info("Model path: %s", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "silu": silu
        }
    )

    logger.info("Model loaded successfully")

    logger.info("Input shape: %s", model.input_shape)

    logger.info("Output shape: %s", model.output_shape)

    if model.output_shape[-1] != EXPECTED_CLASSES:

        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but {EXPECTED_CLASSES} classes were configured."
        )

except Exception:

    logger.exception("Model loading failed")

    # Keep the application alive so health checks can report
    # the loading problem. Prediction returns an error.
    model = None


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image = image.resize(
            IMG_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# PREDICTION
# =========================================================

def predict_disease(image_path):

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    if predictions.shape != (1, EXPECTED_CLASSES):

        raise ValueError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# =========================================================
# HOME / HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message": "Plant Disease Detection API is running",

        "model": "EfficientNetB0",

        "classes": len(class_names),

        "model_loaded": model is not None,

        "input_size": "224x224",

        "device": "CPU"

    })


# =========================================================
# GET CLASSES
# =========================================================

@app.route("/classes", methods=["GET"])
def get_classes():

    return jsonify({

        "success": True,

        "total_classes": len(class_names),

        "classes": class_names

    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    logger.info("Prediction API called")

    if model is None:

        return jsonify({

            "success": False,

            "error": "Model not loaded. Check Render logs."

        }), 503

    if "file" not in request.files:

        return jsonify({

            "success": False,

            "error": "No file uploaded. Use form-data key 'file'."

        }), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify({

            "success": False,

            "error": "No file selected"

        }), 400

    if not allowed_file(file.filename):

        return jsonify({

            "success": False,

            "error": "Invalid file type",

            "allowed": list(ALLOWED_EXTENSIONS)

        }), 400

    file_path = None

    try:

        # Generate a unique filename
        original_name = secure_filename(file.filename)

        unique_name = (
            f"{uuid.uuid4().hex}_{original_name}"
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        # Save uploaded file
        file.save(file_path)

        logger.info("Image uploaded successfully")

        # Verify image
        with Image.open(file_path) as image:
            image.verify()

        # Prediction
        logger.info("Running disease prediction...")

        predicted_class, confidence = predict_disease(
            file_path
        )

        logger.info(
            "Prediction: %s | Confidence: %.2f%%",
            predicted_class,
            confidence * 100
        )

        return jsonify({

            "success": True,

            "predicted_class": predicted_class,

            "confidence": round(
                confidence * 100,
                2
            )

        }), 200

    except (UnidentifiedImageError, OSError):

        return jsonify({

            "success": False,

            "error": "Uploaded file is not a valid image"

        }), 400

    except Exception:

        logger.exception("Prediction failed")

        return jsonify({

            "success": False,

            "error": "Image prediction failed"

        }), 500

    finally:

        # Remove temporary image
        if file_path and os.path.exists(file_path):

            try:
                os.remove(file_path)

            except OSError:
                logger.warning(
                    "Could not remove temporary image"
                )


# =========================================================
# FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": "Image exceeds the 8 MB upload limit"

    }), 413


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    logger.info("Starting Plant Disease Detection API")

    logger.info("Classes: %d", len(class_names))

    logger.info("Port: %d", port)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
