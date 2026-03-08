from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # limit upload to 10MB

model = tf.keras.models.load_model("model/plant_disease_model.h5")

import json

with open("model/class_names.json") as f:
    CLASS_NAMES = json.load(f)

def preprocess(img):

    img = img.convert("RGB")

    # Handle large images
    img.thumbnail((512, 512))

    # Resize to model input size
    img = img.resize((128,128))

    img = np.array(img, dtype=np.float32)

    # Normalize
    img = img / 255.0

    # Ensure shape is correct
    img = np.expand_dims(img, axis=0)

    return img

@app.route('/api/detect_disease', methods=['POST'])
def detect_disease():

    try:

        if 'file' not in request.files:
            return jsonify({"error":"No file uploaded"}),400

        file = request.files['file']

        image = Image.open(file).convert("RGB")

        img = preprocess(image)
        print("Image shape:", img.shape)
        prediction = model.predict(img)

        index = np.argmax(prediction)

        disease = CLASS_NAMES[index]

        confidence = float(np.max(prediction))*100
        
        if confidence < 60:
          disease = "Unknown / Low confidence image"

        return jsonify({
            "disease": disease,
            "confidence": confidence
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error":"Image processing failed"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    