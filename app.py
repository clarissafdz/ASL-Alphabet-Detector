# app.py
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model (make sure to provide the correct path and model name)
model = tf.keras.models.load_model('model_edged.h5')

# Define a mapping from class indices to ASL letters
class_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                   23: 'X', 24: 'Y', 25: 'Z'}

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((100, 100))  # Resize to 100x100 pixels
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)  # Add a singleton dimension for the channel
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = image / 255.0  # Normalize to the range [0, 1]
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame part in the request'}), 400

    file = request.files['frame']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Read the image and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        prediction = model.predict(preprocess_image(image))
        predicted_class = np.argmax(prediction, axis=1)[0]
        asl_letter = class_to_letter.get(predicted_class, 'Unknown')

        return jsonify({'prediction': asl_letter})

if __name__ == '__main__':
    app.run(debug=True)
