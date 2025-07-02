import os
import warnings
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request

warnings.filterwarnings('ignore')

# Load MobileNetV2 base model without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Build the model using Functional API
inputs = keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)                         # Preprocessing layer
x = base_model(x, training=False)                    # Feature extraction
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)  # 5 rice classes
model = keras.Model(inputs, outputs)

# Optional: Load trained weights if available
# model.load_weights("rice_weights.h5")

# Label mapping (index to label)
df_labels = {
    'arborio': 0,
    'basmati': 1,
    'ipsala': 2,
    'jasmine': 3,
    'karacadag': 4
}
index_to_label = {v: k for k, v in df_labels.items()}

# Flask app setup
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload (details) page
@app.route('/details')
def details():
    return render_template('details.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Create uploads directory if not exist
    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, '../static/uploads')
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, file.filename)
    file.save(upload_path)

    # Read and preprocess image
    img = cv2.imread(upload_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predict rice type
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    label = index_to_label[predicted_index]

    # Show result with image
    return render_template('results.html', prediction_text=label, image_file=f"uploads/{file.filename}")

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
