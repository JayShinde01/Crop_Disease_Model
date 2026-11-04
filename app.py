
import os
import json
import uuid
import logging

import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model


# =========================================================
# 1. LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("plant-disease-api")


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

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# =========================================================
# 4. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        MODEL_DIR,
        "best_efficientnetb0.keras"
    )
)

CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH",
    os.path.join(
        MODEL_DIR,
        "class_names.json"
    )
)

INDEX_PATH = os.path.join(BASE_DIR, "index.html")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

NUM_CLASSES = 50

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


# =========================================================
# 5. LOAD CLASS NAMES
# =========================================================

logger.info("Loading disease classes...")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


if not isinstance(class_names, list):
    raise ValueError(
        "class_names.json must contain a JSON list."
    )

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}"
    )

if len(set(class_names)) != NUM_CLASSES:
    raise ValueError("Duplicate class names detected.")

logger.info("Loaded %d classes", len(class_names))


# =========================================================
# 6. DATA AUGMENTATION
# Must match training architecture
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.15),

    layers.RandomZoom(0.15),

    layers.RandomContrast(0.10)

], name="data_augmentation")


# =========================================================
# 7. RECREATE EFFICIENTNETB0 ARCHITECTURE
# =========================================================

def build_model():

    logger.info("Building EfficientNetB0 architecture...")

    # weights=None avoids downloading ImageNet weights.
    # The complete trained weights will be loaded below.

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

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
# 8. LOAD TRAINED WEIGHTS
# =========================================================

logger.info("Loading EfficientNetB0 weights...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}"
    )

model = build_model()

# IMPORTANT:
# This is a weights-only checkpoint.
# Do not use tf.keras.models.load_model() here.

model.load_weights(MODEL_PATH)

logger.info("Model weights loaded successfully!")

logger.info("Model input shape: %s", model.input_shape)

logger.info("Model output shape: %s", model.output_shape)

if model.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} outputs, "
        f"but model has {model.output_shape[-1]}"
    )


# =========================================================
# 9. FILE VALIDATION
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# 10. IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        # Pillow compatibility
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

    # EfficientNetB0 includes its own rescaling layer.
    # Keep pixel values in the 0-255 range,
    # matching the training pipeline.

    return image_array


# =========================================================
# 11. PREDICTION FUNCTION
# =========================================================

def predict_disease(image_path):

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    if predictions.shape != (1, NUM_CLASSES):
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
# 12. FRONTEND ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    if os.path.exists(INDEX_PATH):
        return send_from_directory(
            BASE_DIR,
            "index.html"
        )

    return jsonify({
        "success": True,
        "message": "Plant Disease Detection API is running",
        "model": "EfficientNetB0",
        "classes": NUM_CLASSES
    })


# =========================================================
# 13. HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

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
# 14. GET ALL CLASSES
# =========================================================

@app.route("/classes", methods=["GET"])
def get_classes():

    return jsonify({

        "success": True,

        "total_classes": len(class_names),

        "classes": class_names

    })


# =========================================================
# 15. PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    logger.info("Prediction API called")

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

        # Generate unique filename
        original_name = secure_filename(file.filename)

        unique_name = (
            f"{uuid.uuid4().hex}_{original_name}"
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        # Save image
        file.save(file_path)

        logger.info("Image uploaded successfully")

        # Verify image
        with Image.open(file_path) as image:
            image.verify()

        # Predict
        logger.info("Running prediction...")

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

        # Delete temporary image
        if file_path and os.path.exists(file_path):

            try:
                os.remove(file_path)

            except OSError:
                logger.warning(
                    "Could not remove temporary image"
                )


# =========================================================
# 16. FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": "Image exceeds the 8 MB upload limit"

    }), 413


# =========================================================
# 17. RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    logger.info("Starting Plant Disease Detection API")

    logger.info("Model: EfficientNetB0")

    logger.info("Classes: %d", NUM_CLASSES)

    logger.info("Device: CPU")

    logger.info("Port: %d", port)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
